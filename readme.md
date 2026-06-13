# Recommendation Systems for Personalized Content Discovery

**Cult Open Projects 2026 — Project 1: Netflix Prize Dataset**
**Author:** Rahul, IIT Roorkee

An end-to-end machine learning pipeline that scales from classical Matrix Factorization baselines up to non-linear Deep Learning architectures, evaluated on the 100-million-row Netflix Prize Dataset. This repository fulfills all deliverables for the Cult Open Projects 2026 challenge.

---

## 📂 Repository Structure (Deliverables)

- **`phase1_eda.ipynb`** _(Data Processing Pipeline)_: Parses raw text streams, handles memory-efficient downcasting, isolates the test probe partition, and outputs compressed `.parquet` matrices.
- **`phase2.ipynb`** _(Baseline Model Training & Evaluation)_: Filters active users to mitigate data sparsity, performs a 3-fold cross-validation Grid Search, trains an optimized SVD Matrix Factorization model, and computes RMSE/MAP@10 metrics.
- **`phase3.ipynb`** _(Deep Learning & Recommendation Generation)_: Implements a PyTorch custom dataset loader, trains a deep Neural Collaborative Filtering (NCF) Multi-Layer Perceptron network, and outputs Top-10 personalized qualitative recommendations.
- **`requirements.txt`**: Complete list of Python dependencies.

---

## 🚀 Instructions to Reproduce Results

### 1. Environment Setup & Dependency Installation

Clone this repository to your local machine or a cloud environment (a GPU is highly recommended for Phase 3). Ensure you have Python 3.10+ installed.

```bash
git clone [https://github.com/YOUR_USERNAME/Recommendation-Systems-for-Personalized-Content-Discovery.git](https://github.com/YOUR_USERNAME/Recommendation-Systems-for-Personalized-Content-Discovery.git)
cd Recommendation-Systems-for-Personalized-Content-Discovery

Install the required dependencies. (Note: numpy<2 is strictly required to prevent C-API compilation errors with the scikit-surprise library).
Bash

pip install "numpy<2" pandas fastparquet scikit-surprise torch scikit-learn matplotlib seaborn

2. Data Setup

    Download the raw Netflix Prize dataset from Kaggle.

    Create a folder named netflix-prize-data/ in the root directory of this repository.

    Place the downloaded files (combined_data_1.txt through combined_data_4.txt, movie_titles.csv, and probe.txt) inside this folder.

3. Running the Application

To reproduce the exact findings of the technical report, execute the Jupyter Notebooks sequentially:

    Run phase1_eda.ipynb: This will generate train.parquet, test.parquet, and movies.parquet.

    Run phase2.ipynb: This will execute the Grid Search, train the SVD baseline, and output the baseline metrics.

    Run phase3.ipynb: This will map the embedding tokens, train the PyTorch NCF model, and print the qualitative Top-10 Success/Failure user analysis.

📊 Evaluation & Model Results

Models were evaluated based on their ability to predict exact star ratings (RMSE) and their ability to rank highly relevant, ≥3.5-star movies in a user's Top 10 list (MAP@10).
Model Architecture	RMSE	MAP@10
Global Mean (Baseline Guess)	1.0628	—
SVD Matrix Factorization (Tuned)	1.0409	0.5573 (55.73%)
Neural Collaborative Filtering (NCF)	1.0723	0.5560 (55.60%)
Key Project Insights

    The Sparsity Floor: Standard collaborative filtering breaks down on uniform random samples due to "cold-start" profiles. Stratifying the training matrix to track active users (≥50 ratings) expanded recommendation ranking precision (MAP@10) from ~17% to over 55%.

    Linear Efficiency: The optimized linear dot-product space learned by SVD remains highly competitive against deep learning MLP frameworks for explicit numerical rating boundaries, requiring far less computational overhead.
```
