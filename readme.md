# 🎬 Recommendation Systems for Personalized Content Discovery: The Netflix Prize Challenge

**Cult Open Projects 2026** | **Author:** Rahul, IIT Roorkee

---

## 📑 Table of Contents

1. [Project Overview & Motivation](#1-project-overview--motivation)
2. [Repository Structure](#2-repository-structure)
3. [Instructions to Reproduce](#3-instructions-to-reproduce)
4. [Exploratory Data Analysis (EDA)](#4-exploratory-data-analysis-eda)
5. [Methodology & Data Engineering](#5-methodology--data-engineering)
6. [Model Architectures](#6-model-architectures)
7. [Quantitative Results](#7-quantitative-results)
8. [Qualitative Analysis & Success Cases](#8-qualitative-analysis--success-cases)
9. [Future Work](#9-future-work)

---

## 1. Project Overview & Motivation

Every day, billions of users interact with recommendation systems that dictate the media they consume. The core challenge of the famous **Netflix Prize dataset** lies in **data sparsity**. With over 480,000 users and 17,770 movies, the user-item interaction matrix is overwhelmingly empty (98.8% missing data).

This project solves two distinct machine learning problems:

1. **Rating Prediction:** Accurately forecasting a user's exact 1-5 star rating for unseen content using **Root Mean Squared Error (RMSE)**.
2. **Relevance Ranking:** Identifying the top movies that a user is practically guaranteed to enjoy (rating ≥ 3.5) and presenting them as a sorted Top-10 list using **Mean Average Precision at 10 (MAP@10)**.

---

## 2. Repository Structure

This repository fulfills all Deliverable 2 requirements for the Cult Open Projects challenge.

- `netflix-prize-data/` : (User must create this and add raw Kaggle data)
- `phase1_eda.ipynb` : Data Processing & Parsing Pipeline
- `phase2.ipynb` : Baseline Model Training (Grid Search & SVD Matrix Factorization)
- `phase3.ipynb` : Deep Learning (NCF) & Recommendation Generation Module
- `requirements.txt` : Python Dependencies
- `README.md` : Comprehensive Project Documentation

---

## 3. Instructions to Reproduce

### Environment Setup

Clone this repository to your local machine or a cloud environment. A GPU is highly recommended for executing Phase 3 (PyTorch). Ensure you have Python 3.10+ installed.

git clone https://github.com/MelliCorleone1972/Recommendation-Systems-for-Personalized-Content-Discovery.git
cd Recommendation-Systems-for-Personalized-Content-Discovery

Install the required dependencies. _(Note: `numpy<2` is strictly required to prevent C-API compilation errors with the `scikit-surprise` library)._
pip install "numpy<2" pandas fastparquet scikit-surprise torch scikit-learn matplotlib seaborn

### Data Setup

1. Download the raw [Netflix Prize dataset from Kaggle](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
2. Create a folder named `netflix-prize-data/` in the root directory.
3. Extract and place the files (`combined_data_1.txt` through `combined_data_4.txt`, `movie_titles.csv`, and `probe.txt`) inside this folder.

### Execution Pipeline

Execute the Jupyter Notebooks sequentially to reproduce the findings:

1. **`phase1_eda.ipynb`**: Parses 100M+ raw text lines, handles memory-efficient downcasting, strictly isolates the `probe.txt` test partition to prevent data leakage, and outputs compressed `.parquet` matrices.
2. **`phase2.ipynb`**: Filters the matrix to mitigate sparsity, performs a 3-fold cross-validation Grid Search, trains an optimized SVD baseline, and computes evaluation metrics.
3. **`phase3.ipynb`**: Maps embedding tokens, trains a PyTorch Neural Collaborative Filtering (NCF) network, and generates qualitative Top-10 Success/Failure analyses.

---

## 4. Exploratory Data Analysis (EDA)

An extensive analysis was conducted on the 100.4 million ratings:

- **Sparsity Characteristics:** The overall dataset has a density of only 1.17%.
- **The "Cold-Start" Problem:** A significant portion of the user base rated fewer than 5 movies. Conversely, a small subset of "power users" dominated the interaction logs.
- **Content Popularity:** A fraction of blockbusters contains hundreds of thousands of ratings, while thousands of niche movies contain very few.
- **Rating Quality vs. Popularity:** A "quality floor" exists for popular movies, whereas niche movies exhibit extreme rating variance.

---

## 5. Methodology & Data Engineering

Because mapping a 100-million row matrix for users with only 1 or 2 ratings causes mathematical breakdowns, a **Smart Sampling** methodology was implemented.

The dataset was filtered to exclusively train on **"active users"** (those with ≥ 50 ratings). This established a mathematically dense signal. A manageable subset of 2 million ratings was sampled from this active user base to ensure computational efficiency, while testing was performed strictly on the hold-out `probe.txt` set.

---

## 6. Model Architectures

### Model A: Tuned Singular Value Decomposition (SVD)

The baseline model utilizes SVD to compress the sparse user-item matrix into lower-dimensional latent factor matrices, capturing linear relationships via dot products.

- **Optimal Parameters (Grid Search):** `n_factors` = 50, `n_epochs` = 20, `lr_all` = 0.005, `reg_all` = 0.05.

### Model B: Neural Collaborative Filtering (NCF)

To capture complex, non-linear interactions, an NCF architecture was engineered using **PyTorch**.

- **Architecture:** 32-dimensional user/movie embeddings concatenated and passed through a Multi-Layer Perceptron (64 → 32 → 1). Optimized via MSE loss using Adam.

---

## 7. Quantitative Results

Evaluated on prediction accuracy (RMSE) and ranking precision (MAP@10) with a relevance threshold of ≥ 3.5 stars.

| Model Architecture                       | RMSE (Lower is Better) | MAP@10 (Higher is Better) |
| :--------------------------------------- | :--------------------: | :-----------------------: |
| Global Mean (Baseline Guess)             |         1.0628         |             —             |
| User-Based CF                            |         1.0929         |          0.8514           |
| Item-Based CF                            |         1.1585         |          0.8238           |
| **SVD Matrix Factorization (Tuned)**     |       **1.0409**       |    **0.5573 (55.73%)**    |
| **Neural Collaborative Filtering (NCF)** |       **1.0723**       |    **0.5560 (55.60%)**    |

**Key Insight:** By filtering for active users, the MAP@10 precision was successfully raised from a random baseline of ~17% up to **55.7%**. The optimized linear dot-product space learned by SVD remains highly competitive against deep learning frameworks for explicit numerical rating boundaries.

---

## 8. Qualitative Analysis & Success Cases

To qualify the MAP@10 performance, a success/failure analysis was conducted via the Recommendation Generation Module. For **User 69403**, the NCF model demonstrated flawless precision (100% success rate on the Top-10 list).

**Recommendation Examples (User 69403):**

- 🎯 _Mortal Kombat_ — Predicted: 4.88 | Actual: 5.0
- 🎯 _Lord of the Rings: The Fellowship of the Ring_ — Predicted: 4.54 | Actual: 4.0
- 🎯 _X-Men_ — Predicted: 4.21 | Actual: 4.0

---

## 9. Future Work

1. **Bayesian Personalized Ranking (BPR):** Transitioning the neural network's loss function to explicitly optimize the MAP@10 metric.
2. **Generalized Matrix Factorization (GMF):** Creating a hybrid architecture that fuses SVD dot-products with non-linear MLP layers.
3. **Temporal Dynamics:** Incorporating a time-decay factor into user embeddings to adapt mathematically as user tastes evolve over time.
