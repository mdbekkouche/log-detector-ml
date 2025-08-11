# Log Detector ML

**Log Detector ML** is a Python-based framework for implementing and evaluating a wide range of **machine learning approaches** and **baseline models** for **log-based anomaly detection**.  
It supports both **supervised** and **unsupervised** methods, as well as classical ML and deep learning architectures.

This tool has been used for the experiments in the following works:

- **Bekkouche M., Meski M., Khodja Y., Benslimane S.M.** (2025) *Supervised Machine Learning Approaches for Log-Based Anomaly Detection: A Case Study on the Spirit Dataset.*  
  Submitted to **The Tunisian Algerian Conference on Applied Computing (TACC 2025)**.

- **Bekkouche M., Meski M., Khodja Y., Benslimane S.M.** (2025) *Log-based anomaly detection using BiLSTM-Autoencoder.*  
  Submitted to **7th International Conference on Networking and Advanced Systems (ICNAS 2025)**.

- **Bekkouche M., Benslimane S.M.** (2025) *Improving Anomaly Detection in the HDFS Dataset with Novel Machine Learning Models and Techniques.*  
  Submitted to **Computer Science Journal of Moldova**.

The present tool is developed based on the tool proposed in the article:  
> **Experience Report: System Log Analysis for Anomaly Detection**.(2016)  
> [PDF link](https://jiemingzhu.github.io/pub/slhe_issre2016.pdf)

---

## Features

- Multiple **baseline models**:
  - Decision Tree, Random Forest, Logistic Regression
  - SVM (with and without SMOTE)
  - Isolation Forest, One-Class SVM
  - KMeans clustering variants
  - PCA-based anomaly detection
  - XGBoost

- **Deep learning models** (PyTorch):
  - Autoencoder
  - BiLSTM Autoencoder
  - LSTM Autoencoder
  - Variational Autoencoder (VAE)
  - DeepLog

- **Embedding methods**:
  - TF-IDF
  - Word2Vec

- **Datasets supported**:
  - Spirit Dataset
  - HDFS Dataset
  - Custom CSV-based logs

- **Feature engineering**:
  - Time-based features
  - Incremental PCA
  - Invariants Mining

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/<your-username>/log-detector-ml.git
cd log-detector-ml
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Dataset
Place your dataset (e.g., `Spirit5M.log_structured.csv`, `HDFS.log_structured.csv`) inside the `data/` folder or specify its path in the scripts.

---

### 2. Run a Baseline Model

**Example: Decision Tree on TF-IDF features**

```bash
python demo\DecisionTree_demo.py
```

**Example: Decision Tree with SMOTE oversampling**

```bash
python demo\DecisionTree_Smote_demo.py
```

---

### 3. Run a Deep Learning Model (PyTorch)

**Example: BiLSTM Autoencoder with Word2Vec embeddings**

```bash
python demo\BLSTMAutoencoder_Word2Vec_demo.py
```

**Example: LSTM Autoencoder with TF-IDF embeddings**

```bash
python demo\LSTMAutoencoder_demo.py
```

---

### 4. Unsupervised Anomaly Detection

**Example: Isolation Forest on TF-IDF features**

```bash
python demo\IsolationForest_demo.py
```

**Example: KMeans + Isolation Forest hybrid**

```bash
python demo\KMeansIF_demo.py
```

---

### 5. Feature Engineering and Dimensionality Reduction

**Example: PCA on TF-IDF features**

```bash
python demo\PCA_demo.py
```

**Example: Incremental PCA with Word2Vec**

```bash
python demo\IncrementalPCA_Word2Vec_demo.py
```

---

### 6. Available Scripts

| Script Name                         | Approach                  | Feature Type |
| ----------------------------------- | ------------------------- | ------------ |
| DecisionTree\_demo.py               | Decision Tree             | TF-IDF       |
| DecisionTree\_Word2Vec\_demo.py     | Decision Tree             | Word2Vec     |
| DecisionTree\_Smote\_demo.py        | Decision Tree + SMOTE     | TF-IDF       |
| BLSTMAutoencoder\_demo.py           | BiLSTM Autoencoder        | TF-IDF       |
| BLSTMAutoencoder\_Word2Vec\_demo.py | BiLSTM Autoencoder        | Word2Vec     |
| LSTMAutoencoder\_demo.py            | LSTM Autoencoder          | TF-IDF       |
| LSTMAutoencoder\_Word2Vec\_demo.py  | LSTM Autoencoder          | Word2Vec     |
| DeepLog\_demo.py                    | DeepLog                   | Sequential   |
| XGBoost\_demo.py                    | XGBoost                   | TF-IDF       |
| XGBoost\_Word2Vec\_demo.py          | XGBoost                   | Word2Vec     |
| IsolationForest\_demo.py            | Isolation Forest          | TF-IDF       |
| IsolationForest\_Word2Vec\_demo.py  | Isolation Forest          | Word2Vec     |
| KMeansIF\_demo.py                   | KMeans + Isolation Forest | TF-IDF       |
| KMeansIF\_Word2Vec\_demo.py         | KMeans + Isolation Forest | Word2Vec     |
| PCA\_demo.py                        | PCA                       | TF-IDF       |
| PCA\_Word2Vec\_demo.py              | PCA                       | Word2Vec     |
| ...                                 | ...                       | ...          |

*(See repository for the full list of scripts.)*

---

## 📋 Requirements

Dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```
