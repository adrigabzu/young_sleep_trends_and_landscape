""" 
Created Date: Tuesday, March 25th 2025
Author: Adrian G. Zucco
Decription: 
    This script generates co-occurrence matrix and GloVe embeddings from synthetic longitudinal data
    using PyTorch.

Code inspired by implementations in:
https://github.com/2014mchidamb/TorchGlove/blob/master/glove.py
https://github.com/noaRricky/pytorch-glove/blob/master/glove.py
"""

# %%
import polars as pl
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import Counter

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from gensim.models import KeyedVectors
import pickle


# %%
def process_sequence(sequence):
    local_coocs = Counter()
    seq_length = len(sequence)

    for i, center_idx in enumerate(sequence):
        start = max(0, i - context_size)
        end = min(seq_length, i + context_size + 1)

        for j in range(start, end):
            if i != j:
                context_idx = sequence[j]
                local_coocs[(center_idx, context_idx)] += 1.0 / abs(i - j)

    return local_coocs


# %%  Set parameters
context_size = 100

df = pl.read_csv("../data/synthetic_data/mock_longdata.csv")
print("Column names:", df.columns)
# Column names: ['PNR', 'year', 'code', 'type']

# %%
# Group by PNR and collect codes into lists
life_courses = df.group_by("PNR").agg(pl.col("code").alias("codes"))

# Convert to a nested list where each sublist contains the codes for one patient
sentences = life_courses["codes"].to_list()

# For debugging, print the first few lists
print(f"Number of individuals: {len(sentences)}")
print(
    f"First patient's codes (first 5): {sentences[0][:5] if len(sentences[0]) >= 5 else sentences[0]}"
)

# %%
individuals = len(sentences)
vocab = df["code"].unique().to_numpy()
vocab_size = len(vocab)
# Create word to index mapping
w_to_i = {word: ind for ind, word in enumerate(vocab)}
# Precompute indices for each sequence to avoid repeated dictionary lookups
indexed_sequences = [[w_to_i[code] for code in seq] for seq in sentences]

################## CO-OCCURRENCE MATRIX CONSTRUCTION ####################
# %%
# Time the co-occurrence matrix construction
start_time = time.time()

# Process each sequence using the process_sequence function
co_occurrences = Counter()
for indexed_sequence in tqdm(indexed_sequences, desc="Processing sequences"):
    local_coocs = process_sequence(indexed_sequence)
    co_occurrences.update(local_coocs)

# Create sparse matrix from counter
rows, cols, data = [], [], []
for (i, j), value in tqdm(co_occurrences.items(), desc="Building sparse matrix"):
    rows.append(i)
    cols.append(j)
    data.append(value)

comat = sp.csr_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size))
# co_oc_nonzero = np.transpose(np.nonzero(comat))

# Save sparse matrix and co_oc_nonzero array

with open("../results/cooccurrence_matrix.pkl", "wb") as f:
    pickle.dump(comat, f)

end_time = time.time()
print(f"Co-occurrence matrix 4 construction time: {end_time - start_time:.2f} seconds")

# %%
class GloveDataset(Dataset):
    """
    PyTorch Dataset for GloVe training.
    Extracts data directly from the non-zero entries of a SciPy CSR matrix.
    """
    def __init__(self, comat: sp.csr_matrix):
        """
        Args:
            comat (sp.csr_matrix): The co-occurrence matrix (CSR format).
                                   Values should ideally be float32.
        """
        if not sp.isspmatrix_csr(comat):
            raise TypeError("Input must be a SciPy CSR matrix.")

        # Extract non-zero indices (rows, cols) and data
        rows, cols = comat.nonzero()
        values = comat.data

        self.num_nonzero = comat.nnz

        # Store indices as int64 (PyTorch default for LongTensor)
        self.focal_indices = rows.astype(np.int64)
        self.context_indices = cols.astype(np.int64)

        # Ensure values are float32 for model compatibility
        if values.dtype != np.float32:
            print(f"Warning: Co-occurrence matrix values dtype was {values.dtype}. Casting to float32.")
            self.cooc_values = values.astype(np.float32)
        else:
            self.cooc_values = values

        # Sanity check (optional but good)
        if not (len(self.focal_indices) == self.num_nonzero and
                len(self.context_indices) == self.num_nonzero and
                len(self.cooc_values) == self.num_nonzero):
            raise ValueError("Mismatch in lengths derived from sparse matrix components.")


    def __len__(self):
        """Returns the number of non-zero co-occurrence pairs."""
        return self.num_nonzero

    def __getitem__(self, idx):
        """Returns the idx-th training sample (focal_idx, context_idx, value)."""
        if idx < 0 or idx >= self.num_nonzero:
            raise IndexError(f"Index {idx} out of bounds for {self.num_nonzero} elements.")
        return (
            self.focal_indices[idx],
            self.context_indices[idx],
            self.cooc_values[idx]
        )


################## Train GloVe model ####################
# %%
# Set GloVe parameters
embed_size = 50  # Embedding dimension
x_max = 100  # Maximum co-occurrence value for weighting function
alpha = 0.75  # Parameter in weighting function
batch_size = 1024  # Batch size for training
learning_rate = 0.05  # Learning rate for optimizer
num_epochs = 10  # Number of training epochs
num_workers = 6

# %%
# Initialize embeddings using nn.Embedding for better efficiency
focal_embeddings = nn.Embedding(vocab_size, embed_size)
context_embeddings = nn.Embedding(vocab_size, embed_size)
focal_biases = nn.Embedding(vocab_size, 1)
context_biases = nn.Embedding(vocab_size, 1)


# Initialize weights with uniform distribution
for params in [
    focal_embeddings.weight,
    context_embeddings.weight,
    focal_biases.weight,
    context_biases.weight,
]:
    nn.init.uniform_(params, a=-1, b=1)
# Move to double precision for numerical stability
focal_embeddings = focal_embeddings.float()
context_embeddings = context_embeddings.float()
focal_biases = focal_biases.float()
context_biases = context_biases.float()

# Set up optimizer with all parameters
params = (
    list(focal_embeddings.parameters())
    + list(context_embeddings.parameters())
    + list(focal_biases.parameters())
    + list(context_biases.parameters())
)
optimizer = optim.Adam(params, lr=learning_rate)

# Weight function for co-occurrence values
def weight_func(cooccurrence_count):
    weight_factor = torch.pow(
        torch.tensor(cooccurrence_count / x_max, dtype=torch.float32), alpha
    )
    return torch.minimum(weight_factor, torch.ones_like(weight_factor))


# Function to create a batch of training data from co-occurrence matrix
# def create_batch(batch_size):
#     indices = np.random.choice(
#         len(co_oc_nonzero), size=min(batch_size, len(co_oc_nonzero)), replace=False
#     )

#     focal_indices_np = co_oc_nonzero[indices, 0]
#     context_indices_np = co_oc_nonzero[indices, 1]

#     # Efficiently get values from sparse matrix
#     cooc_values_np = comat[
#         focal_indices_np, context_indices_np
#     ].A1  # .A1 flattens the result

#     # Convert to PyTorch tensors (use float32!)
#     focal_indices = torch.LongTensor(focal_indices_np)
#     context_indices = torch.LongTensor(context_indices_np)
#     # Use FloatTensor for consistency
#     cooc_values = torch.FloatTensor(cooc_values_np)

#     return focal_indices, context_indices, cooc_values

# --- Instantiate Dataset and DataLoader (using the modified Dataset) ---
print("Creating Dataset and DataLoader...")
# Pass the sparse matrix directly to the Dataset constructor
glove_dataset = GloveDataset(comat) # <-- Key change here

pin_memory = torch.cuda.is_available() and num_workers > 0
dataloader = DataLoader(
    glove_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)
print(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")


# %%
# Train the GloVe model
print("Starting GloVe training...")
losses = []
# Determine device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.set_num_threads(num_workers)

# Move models to device
focal_embeddings = focal_embeddings.to(device)
context_embeddings = context_embeddings.to(device)
focal_biases = focal_biases.to(device)
context_biases = context_biases.to(device)


for epoch in range(num_epochs):
    start_time = time.time()
    # num_batches = int(np.ceil(len(co_oc_nonzero) / batch_size))
    epoch_loss = 0.0

    # for batch in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
    for batch_idx, (focal_idx, context_idx, cooc_vals) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        # Create batch
        # focal_idx, context_idx, cooc_vals = create_batch(batch_size)

        # Move to device
        focal_idx = focal_idx.to(device)
        context_idx = context_idx.to(device)
        cooc_vals = cooc_vals.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        focal_embed = focal_embeddings(focal_idx)
        context_embed = context_embeddings(context_idx)
        focal_bias = focal_biases(focal_idx).squeeze()
        context_bias = context_biases(context_idx).squeeze()

        # Calculate embeddings product (dot product)
        embedding_products = torch.sum(focal_embed * context_embed, dim=1)

        # Calculate log co-occurrences
        log_coocs = torch.log(cooc_vals)

        # Calculate squared error
        distance_expr = torch.square(
            embedding_products + focal_bias + context_bias - log_coocs
        )

        # Apply weighting function
        weights = weight_func(cooc_vals)
        weighted_errors = weights * distance_expr

        # Calculate loss
        loss = torch.mean(weighted_errors)
        epoch_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Calculate average loss for epoch
    # avg_loss = epoch_loss / num_batches
    avg_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0
    losses.append(avg_loss)

    end_time = time.time()
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Time: {end_time - start_time:.2f}s"
    )


print("GloVe training complete.")

# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GloVe Training Loss")
plt.grid(True)
plt.savefig("../results/glove_training_loss.png")
plt.close()

# %%
# Create final embeddings by averaging focal and context embeddings
final_embeddings = np.zeros((vocab_size, embed_size))
for i in range(vocab_size):
    # combined = (focal_embeddings.weight[i].detach().cpu().numpy() +
    #             context_embeddings.weight[i].detach().cpu().numpy()) / 2
    combined = (
        focal_embeddings.weight[i].detach().cpu().numpy()
        + context_embeddings.weight[i].detach().cpu().numpy()
    )
    final_embeddings[i] = combined.reshape(-1)

# Save embeddings
np.save("../results/code_glove_embeddings.npy", final_embeddings)

# Create embedding dictionary mapping from codes to vectors
embedding_dict = {code: final_embeddings[w_to_i[code]] for code in vocab}

# %%
# # --- Save Embeddings in Gensim Word2Vec Text Format ---

# Define output file path for Gensim format
gensim_output_file = "../results/code_glove_embeddings.txt"

print(f"Saving embeddings in Gensim format to {gensim_output_file}...")
start_time = time.time()

# Create inverse mapping: index -> word
i_to_w = {i: w for w, i in w_to_i.items()}

with open(gensim_output_file, "w", encoding="utf-8") as f:
    # Write header: number_of_vectors dimension
    f.write(f"{vocab_size} {embed_size}\n")

    # Write word vectors
    for i in range(vocab_size):
        word = i_to_w[i]
        vector = final_embeddings[i]
        vector_str = " ".join(map(str, vector))  # Convert vector components to string
        f.write(f"{word} {vector_str}\n")

end_time = time.time()
print(f"Saved embeddings in {end_time - start_time:.2f} seconds.")
