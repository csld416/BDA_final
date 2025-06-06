import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os

def cluster_and_save(input_file, output_file, n_clusters):
    if not os.path.exists(input_file):
        print(f"[SKIP] {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"[INFO] Loaded {input_file} with shape {df.shape}")

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # PCA to reduce noise
    pca = PCA(n_components=min(X_scaled.shape[1], 3))
    X_pca = pca.fit_transform(X_scaled)

    # GMM clustering
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_pca)

    # Save results
    pd.DataFrame({
        'id': np.arange(len(labels)),
        'label': labels
    }).to_csv(output_file, index=False)
    print(f"[DONE] Saved results to {output_file}")

if __name__ == "__main__":
    cluster_and_save("public_data.csv", "public_submission.csv", n_clusters=15)
    cluster_and_save("private_data.csv", "private_submission.csv", n_clusters=23)