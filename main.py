# main.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os

def run_clustering(input_csv: str, output_csv: str):
    # Step 1: Load dataset
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"[SKIP] {input_csv} not found. Skipping...")
        return

    print(f"[INFO] Loaded '{input_csv}' with shape {df.shape}")

    # Step 2: Preprocess (scale → PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Perform PCA to capture most variance.
    # We choose n_components = min(n_features, 10) to denoise.
    n_features = X_scaled.shape[1]
    n_components = min(n_features, 10)
    print(f"[INFO] Running PCA: reducing {n_features} dims → {n_components} dims")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    explained = np.sum(pca.explained_variance_ratio_)
    print(f"[INFO] PCA explained variance (sum): {explained:.4f}")

    # Step 3: Number of clusters = 4n - 1 (original dims)
    k = 4 * n_features - 1
    print(f"[INFO] Clustering into {k} clusters (4 × {n_features} - 1)")

    # Step 4: Fit GaussianMixture with “tied” covariance (one shared covariance matrix)
    print("[INFO] Fitting GMM (covariance_type='tied', n_init=5)")
    gmm = GaussianMixture(
        n_components=k,
        covariance_type='tied',
        n_init=5,
        random_state=42
    )
    labels = gmm.fit_predict(X_pca)

    # Step 5: Save to CSV in (id, label) format
    out_df = pd.DataFrame({
        'id': np.arange(len(labels)),
        'label': labels
    })
    out_df.to_csv(output_csv, index=False)
    print(f"[DONE] Saved labels to '{output_csv}' (shape {out_df.shape})\n")

if __name__ == "__main__":
    run_clustering("public_data.csv",  "public_submission.csv")
    run_clustering("private_data.csv", "private_submission.csv")