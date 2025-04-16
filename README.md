# Life course embeddings for exploring sleep problems from late adolescence to adulthood 

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- Optional: Add a DOI badge if you have one for the code/paper -->
<!-- [![DOI](https://zenodo.org/badge/DOI/YOUR_ZENODO_DOI.svg)](https://doi.org/YOUR_ZENODO_DOI) -->

Code supporting the article:

**"Exploring nationwide patterns of sleep problems from late adolescence to adulthood using machine learning"**

By Adrian G. Zucco*, Henning Johannes Drews, Jeroen F. Uleman, Samir Bhatt, Naja Hulvej Rod.

*Corresponding author: adrigabzu@sund.ku.dk

<!-- 
**Paper Link:** [Link to Paper, e.g., journal URL or preprint]
**DOI:** [DOI of the paper] -->

---

## Abstract

Sleep problems among young adults pose a major public health challenge. Based on nationwide health surveys and registers from Denmark, we investigated patterns of sleep problems from late adolescence to adulthood and explored early life-course determinants. We generated life course embeddings using unsupervised machine learning on data from 2.2 million individuals born 1980-2015. We used this landscape to identify neighboring factors to sleep problems. We observed a substantial increase in self-reported sleep problems among individuals aged 15-45, from 34% to 49% between 2010 and 2021, and a tenfold increase in melatonin use. We also found five distinct clusters of sleep-related prescriptions, diagnoses and procedures with age-specific incidence patterns. Specific childhood adversities, such as sibling psychiatric illness, foster care, and parental divorce were shared factors across multiple sleep disorders such as insomnia and nightmares. These findings underscore the complex interplay between medical and psychosocial factors in sleep.

## Setup

1.  **Clone the repository**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    This repository uses `pip` for package management. Install the required packages using:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The analysis pipeline is organized numerically in the `src/` directory. The scripts were designed to be run interactively and explore the code step by step. The following is a high-level overview of the main steps in the analysis pipeline:

1.  **Generate Synthetic Data (Optional)**
    If you do not have access to the original data, you can generate synthetic data for testing the scripts.

2.  **Train Embedding Models**

3.  **Explore Embeddings**

4.  **Perform Clustering**

## Citation

> Not yet available

