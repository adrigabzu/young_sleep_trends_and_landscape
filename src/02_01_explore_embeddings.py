#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Author  :   Adrian G. Zucco
@Contact :   adrigabzu@sund.ku.dk
Decription: 
    This script loads Word2vec embeddings and computes a cosine distance matrix.
    It then applies dimensionality reduction using PaCMAP.
'''

# %% Import modules
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
from omniplot import plot as op

import pacmap

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import os
import multiprocessing

pd.set_option("display.max_rows", 100)
print("## Number of CPU cores: ", multiprocessing.cpu_count())


# %% Import model
model_name = "w2v_embeddings_embeddings"
# Load using txt format
model = KeyedVectors.load_word2vec_format(
    f"../results/{model_name}.txt"
)

# %%
vocab = model.index_to_key
vocab_size = len(vocab)
vector_size = model.vector_size

print("Vocabulary size: ", vocab_size)
print("Vector size: ", vector_size)

# %% ############################# Create cosine distance matrix

# Create cosine distance matrix from model.vectors
cosine_sim = cosine_similarity(model.vectors)

# label cosine similarity matrix using labels at model.index_to_key
cosine_sim_df = pd.DataFrame(
    cosine_sim, index=model.index_to_key, columns=model.index_to_key
)

# %% ############### DIMENSIONALITY REDUCTION, PACMAP computation ####################
# Set seed
seed = 1234
neigh_nr = 20
word_vectors = model.vectors
pacmap_model = pacmap.PaCMAP(
    n_components=2,
    n_neighbors=neigh_nr,
    verbose=True,
    distance="angular",
    random_state=seed,
)

reduced_vectors = pacmap_model.fit_transform(word_vectors)

# %% Create dataframe
reduced_vec_df = pd.DataFrame(reduced_vectors, columns=["dim1", "dim2"])
reduced_vec_df["term"] = model.index_to_key
