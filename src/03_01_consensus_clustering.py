#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Author  :   Adrian G. Zucco
@Contact :   adrigabzu@sund.ku.dk
Decription: 
    This script performs multiple clustering analyses on sleep data using a fixed number
    of clusters (k=5). It loads a distance matrix and corresponding dataframe from the
    clustering_data directory and applies various clustering algorithms.
    The goal is to identify robust patterns across different clustering methods.
    
'''

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from typing import List
from itertools import combinations
import time

import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Define constants
N_CLUSTERS = 5
DATA_DIR = "../data/clustering_data"
DISTANCE_MATRIX_PATH = os.path.join(DATA_DIR, "distance_matrix.parquet")
FEATURES_PATH = os.path.join(DATA_DIR, "top_codes_clustered.parquet")


# %%
def load_data(distance_matrix_path=None, features_path=None):
    """
    Load distance matrix and dataframe from parquet files.

    Args:
        distance_matrix_path: Path to distance matrix parquet file
        features_path: Path to features dataframe parquet file

    Returns:
        tuple: (distance_matrix, dataframe)
    """
    distance_matrix_path = distance_matrix_path or DISTANCE_MATRIX_PATH
    features_path = features_path or FEATURES_PATH

    distance_matrix = pd.read_parquet(distance_matrix_path)
    df = pd.read_parquet(features_path)

    print(f"Loaded distance matrix with shape: {distance_matrix.shape}")
    print(f"Loaded dataframe with shape: {df.shape}")

    return distance_matrix, df


def run_clustering_algorithms(distance_matrix, df):
    """
    Apply multiple clustering algorithms with k=5, using distance matrix when possible,
    otherwise using dim1 and dim2 coordinates.

    Args:
        distance_matrix: Precomputed distance matrix
        df: DataFrame with features (must contain 'dim1' and 'dim2' columns)

    Returns:
        dict: Dictionary of clustering results
    """
    # Initialize dictionary to store cluster assignments
    cluster_results = {}

    # Methods using distance matrix
    clustering_methods = {
        "complete": {"linkage": "complete"},
        "average": {"linkage": "average"},
        # 'ward': {'linkage': 'ward'}, # Ward's method requires Euclidean distance
        "single": {"linkage": "single"},
    }

    print("\nRunning distance matrix-based clustering methods:")
    for method_name, params in clustering_methods.items():
        start_time = time.time()
        print(f"\nStarting {method_name} linkage clustering...", end="", flush=True)

        agglo = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, metric="precomputed", **params
        )
        labels = agglo.fit_predict(distance_matrix)
        cluster_results[method_name] = labels
        df[f"cluster_{method_name}"] = labels

        elapsed = time.time() - start_time
        print(f" Done! (Time: {elapsed:.2f}s)")

    # Methods using dim1/dim2 coordinates
    coords = df[["dim1", "dim2"]].values

    print("\nRunning coordinate-based clustering methods:")

    # K-Means
    start_time = time.time()
    print("\nStarting K-means clustering...", end="", flush=True)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    cluster_results["kmeans"] = labels
    df["cluster_kmeans"] = labels
    elapsed = time.time() - start_time
    print(f" Done! (Time: {elapsed:.2f}s)")

    # # MiniBatch K-Means
    # start_time = time.time()
    # print("Starting MiniBatch K-means clustering...", end='', flush=True)
    # minibatch = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=256)
    # labels = minibatch.fit_predict(coords)
    # cluster_results['minibatch_kmeans'] = labels
    # df['cluster_minibatch_kmeans'] = labels
    # elapsed = time.time() - start_time
    # print(f" Done! (Time: {elapsed:.2f}s)")

    # Gaussian Mixture
    start_time = time.time()
    print("Starting Gaussian Mixture clustering...", end="", flush=True)
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=42, n_init=10)
    labels = gmm.fit_predict(coords)
    cluster_results["gmm"] = labels
    df["cluster_gmm"] = labels
    elapsed = time.time() - start_time
    print(f" Done! (Time: {elapsed:.2f}s)")

    return cluster_results


def add_co_occurrence_frequency(
    df: pd.DataFrame, cluster_cols: List[str], ref_col: str
) -> pd.DataFrame:
    """
    Calculates the co-occurrence frequency for each element and adds it as a
    new column to the DataFrame.

    The frequency for an element 'e' is the average frequency across all pairs
    (e, e') where e' is in the same reference cluster as e (defined by
    ref_col). The frequency for a single pair (e, e') is the proportion of
    other clustering columns (in cluster_cols, excluding ref_col) where e
    and e' are also clustered together.

    Args:
        df: Input DataFrame with elements and clustering results.
        cluster_cols: List of column names containing clustering results.
        ref_col: The name of the column to use as the reference clustering.

    Returns:
        The original DataFrame with a new column added, named
        '<ref_col>_co_occurrence_freq', containing the calculated frequency
        for each element. Elements in reference clusters of size 1 will have
        a frequency of NaN.
    """
    # --- Input Validation ---
    if ref_col not in df.columns:
        raise ValueError(f"Reference column '{ref_col}' not found in DataFrame.")
    if ref_col not in cluster_cols:
        print(
            f"Warning: Reference column '{ref_col}' was not found in "
            f"'cluster_cols'. It will be implicitly added for processing, "
            f"but ensure 'cluster_cols' contains all columns intended for "
            f"comparison."
        )
        # Add it if missing, although it won't be used for comparison itself
        # cluster_cols = [ref_col] + cluster_cols

    missing_cols = [col for col in cluster_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns specified in 'cluster_cols' not found in DataFrame: "
            f"{missing_cols}"
        )

    # Identify columns to compare against (exclude the reference column)
    other_cols = [col for col in cluster_cols if col != ref_col]

    if not other_cols:
        print(
            f"Warning: No other cluster columns provided besides the "
            f"reference column '{ref_col}'. Co-occurrence frequency cannot "
            f"be calculated based on other algorithms. Returning DataFrame "
            f"with NaN frequency column."
        )
        df[f"{ref_col}_co_occurrence_freq"] = np.nan
        return df

    num_other_algos = len(other_cols)

    # --- Calculation ---
    # Initialize Series to store the frequency for each element (index)
    element_frequencies = pd.Series(np.nan, index=df.index, dtype=float)

    # Group by the reference cluster
    grouped = df.groupby(ref_col)

    # Iterate through each cluster defined in the reference column
    for _, group in grouped:
        element_indices = group.index.tolist()

        print(element_indices)

        # Need at least 2 elements in the reference cluster to form pairs
        if len(element_indices) < 2:
            # For elements in clusters of size 1, frequency is undefined (NaN)
            # They are already initialized to NaN, so no action needed here.
            continue

        # Store sum of frequencies and count of pairs for each element in the group
        group_freq_sum = {idx: 0.0 for idx in element_indices}
        group_pair_count = {idx: 0 for idx in element_indices}

        # Generate all unique pairs of elements within this reference cluster
        for idx1, idx2 in combinations(element_indices, 2):
            co_cluster_count = 0
            # Check this pair against each of the other algorithms
            for algo_col in other_cols:
                if df.loc[idx1, algo_col] == df.loc[idx2, algo_col]:
                    co_cluster_count += 1

            # Calculate the frequency for this specific pair
            pair_frequency = co_cluster_count / num_other_algos

            # Add this pair's frequency to the sum for both elements
            group_freq_sum[idx1] += pair_frequency
            group_pair_count[idx1] += 1
            group_freq_sum[idx2] += pair_frequency
            group_pair_count[idx2] += 1

        # Calculate the average frequency for each element in the group
        for idx in element_indices:
            if group_pair_count[idx] > 0:
                element_frequencies.loc[idx] = (
                    round(group_freq_sum[idx] / group_pair_count[idx],3)
                )
            # else: remains NaN (shouldn't happen if len(indices) >= 2)

    # --- Add column to DataFrame ---
    output_col_name = f"{ref_col}_co_occurrence_freq"
    df[output_col_name] = element_frequencies

    return df

# %%
# if __name__ == "__main__":

"""Main function to run the clustering analysis."""
print("Loading data...")
distance_matrix, df = load_data()

# %%
print("\nRunning clustering algorithms...")
cluster_results = run_clustering_algorithms(1 - distance_matrix, df)

# %%
# Define cluster columns and the reference column
all_cluster_columns = [
    "cluster",
    "cluster_complete",
    "cluster_single",
    "cluster_kmeans",
    "cluster_gmm",
]
reference_column = "cluster"

# Calculate co-occurrence frequencies
df_summary = add_co_occurrence_frequency(
    df[["code"] + all_cluster_columns], all_cluster_columns, reference_column
)

# %%
# Save as parquet file
df_summary.to_parquet(
    os.path.join(DATA_DIR, f"consensus_clustering.parquet"),
    index=False,
)

# Save as Excel file
df_summary.to_excel(
    os.path.join(DATA_DIR, f"consensus_clustering.xlsx"),
    index=False,
)

# %%
print(f"\n--- DataFrame with '{reference_column}_co_occurrence_freq' ---")
print(df_summary.head())
print("-" * 30)

# Display elements sorted by frequency
print("\n--- Elements sorted by Co-occurrence Frequency (Descending) ---")
print(
    df_summary.sort_values(
        f"{reference_column}_co_occurrence_freq", ascending=False
    ).head(10)
)
print("-" * 30)

# Check for NaNs (should correspond to elements in single-member reference clusters if any)
print("\n--- Elements with NaN Frequency (if any) ---")
print(df_summary[df_summary[f"{reference_column}_co_occurrence_freq"].isna()])
print("-" * 30)
