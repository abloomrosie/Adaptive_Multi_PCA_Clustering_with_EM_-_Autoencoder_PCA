# Adaptive_Multi_PCA_Clustering_with_EM_&_Autoencoder_PCA
 # Multi-PCA Clustering with Expectation-Maximization & Autoencoder PCA Approximation

## Overview
This project explores scenarios where:
- The relationship between features is **non-linear**.
- The dataset contains **multiple clusters**, each exhibiting different feature relationships.

To address these challenges, we:
1. **Develop a method for combining multiple Principal Component Analyses (PCA) using the Expectation-Maximization (EM) algorithm**.
2. **Implement additional constraints on an Autoencoder model (1-layer encoder, 1-layer decoder) in PyTorch to mimic PCA**.

## Dataset
We use the **UCI Vehicle Dataset**, which can be found [here](https://github.com/milaan9/Clustering-Datasets/blob/master/01.%20UCI/vehicle.csv).  
This dataset consists of vehicle silhouette images characterized by shape features.

## Methods

### 1. Multi-PCA with Expectation-Maximization (EM)
- Clustering is performed using an **EM algorithm** that iteratively estimates:
  - **Cluster assignments** for each data point.
  - **PCA subspaces** for each cluster.
- This method allows each cluster to have a different PCA representation.

### 2. Autoencoder PCA Approximation (PyTorch)
- We implement an **Autoencoder** with:
  - **1-layer encoder** to compress data.
  - **1-layer decoder** to reconstruct data.
- Additional constraints are applied to force the **Autoencoder to behave like PCA**.

## Installation
### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas scikit-learn torch torchvision matplotlib seaborn

