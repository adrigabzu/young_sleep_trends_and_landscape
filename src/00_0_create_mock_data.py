#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Author  :   Adrian G. Zucco
@Contact :   adrigabzu@sund.ku.dk
Decription: 
    This script generates longitudinal mock data.
'''

# %%
import polars as pl
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# Set the random seed for reproducibility
random.seed(123)
np.random.seed(123)

# Source of ATC and ICD-10 codes
codes_to_sample = pl.read_csv('../metadata/annotation_codes.tsv', separator='\t')
synth_data_path = "../data/synthetic_data/"

# Check if folder exists, if not create it
Path(synth_data_path).mkdir(parents=True, exist_ok=True)

# %%
########## Parameters for generating synthetic longitudinal data #########
num_patients = 2000
min_obs = 10
max_obs = 45
min_date = datetime.strptime("1980-01-01", "%Y-%m-%d")
max_date = datetime.strptime("2021-12-31", "%Y-%m-%d")


# Add a weight column so that certain codes are more likely to be sampled
adv_codes = [f"adv{i}" for i in range(1, 15)]
codes_to_sample = codes_to_sample.with_columns(
    pl.when(pl.col("code").is_in(adv_codes))
    .then(pl.lit(100))
    .otherwise(pl.lit(1))
    .alias("weight")
)

# Function to generate random date between two dates
def random_date(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

# Generate list of patients
patients = [f"P{i}" for i in range(1, num_patients + 1)]

# For efficient sampling, convert to Python lists
codes = codes_to_sample["code"].to_list()
types = codes_to_sample["type"].to_list()
weights = codes_to_sample["weight"].to_list()

# %%
# Generate data more efficiently by creating records first, then building the dataframe
records = []

for patient in patients:
    # Random sample of observations
    num_observations = random.randint(min_obs, max_obs)

    for _ in range(num_observations):
        # Sample a row from medical_codes based on weight
        idx = random.choices(range(len(codes)), weights=weights)[0]
        
        # Add record
        records.append({
            "PNR": patient,
            "year" : random_date(min_date, max_date).year,
            "code": codes[idx],
            "type": types[idx]
        })

# Create dataframe from records
df = pl.DataFrame(records)

# Display summary statistics (equivalent to skim in R)
print(df.describe())

# Save the synthetic data to a CSV file
df.write_csv(f"{synth_data_path}mock_longdata.csv")
