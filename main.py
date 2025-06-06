# main.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import os

def run_clustering(input_csv: str, output_csv: str):
    # Step 1: Load dataset
    try:
        data = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[SKIP] {input_csv} not found. Skipping...")
        return

    print(f"[INFO] Processing '{input_csv}' with shape {data.shape}")

    # Step 2: Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Step 3: Determine number of clusters
    n_features = X_scaled.shape[1]
    n_clusters = 4 * n_features - 1
    print(f"[INFO] Clustering into {n_clusters} clusters")

    # Step 4: Cluster
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_scaled)

    # Step 5: Save to CSV
    df = pd.DataFrame({
    'id': np.arange(len(labels)),
    'label': labels
    })
    df.to_csv(output_csv, index=False)
    print(f"[DONE] Saved labels to '{output_csv}'")

if __name__ == "__main__":
    run_clustering("public_data.csv", "public_submission.csv")
    run_clustering("private_data.csv", "private_submission.csv")