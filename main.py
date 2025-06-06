# main.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

def run_clustering(input_csv: str, output_csv: str):
    # Step 1: Load dataset
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[SKIP] {input_csv} not found. Skipping...")
        return

    print(f"[INFO] Loaded '{input_csv}' with shape {df.shape}")

    # Step 2: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Step 3: PCA — preserve 95% of variance
    n_features = X_scaled.shape[1]
    print(f"[INFO] Performing PCA on {n_features}-dimensional data (retain 95% variance)")
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    comp = X_pca.shape[1]
    explained = np.sum(pca.explained_variance_ratio_)
    print(f"[INFO] PCA reduced to {comp} components, explained variance = {explained:.4f}")

    # Step 4: Determine k = 4n – 1 (based on original dims)
    k = 4 * n_features - 1
    print(f"[INFO] Clustering into {k} clusters (4 × {n_features} - 1)")

    # Step 5: KMeans with more inits
    print("[INFO] Fitting KMeans (n_init=20, random_state=42)")
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)

    # Step 6: Save to CSV (id,label)
    out_df = pd.DataFrame({
        'id': np.arange(len(labels)),
        'label': labels
    })
    out_df.to_csv(output_csv, index=False)
    print(f"[DONE] Saved labels to '{output_csv}'\n")

if __name__ == "__main__":
    run_clustering("public_data.csv",  "public_submission.csv")
    run_clustering("private_data.csv", "private_submission.csv")