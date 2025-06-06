# main.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
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

    # Step 4: Choose algorithm
    if input_csv == "public_data.csv":
        print("[INFO] Using GMM for public data")
        model = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        print("[INFO] Using KMeans for private data")
        model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        labels = model.fit_predict(X_scaled)

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