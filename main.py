import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import os

def run_spectral(input_csv: str, output_csv: str, cluster_count: int):
    if not os.path.exists(input_csv):
        print(f"[SKIP] {input_csv} not found.")
        return

    df = pd.read_csv(input_csv)

    # Extract only the important features: S2, S3, S4 (i.e., columns 2, 3, 4)
    try:
        X = df.iloc[:, [2, 3, 4]]  # assuming no header renaming
    except:
        raise ValueError("Ensure the input has at least 5 columns including ID, S1, S2, S3, S4")

    print(f"[INFO] Clustering {input_csv} → {cluster_count} clusters using SpectralClustering on S2, S3, S4")

    # Scale the features
    X_scaled = StandardScaler().fit_transform(X)

    # Run Spectral Clustering
    model = SpectralClustering(
        n_clusters=cluster_count,
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans',
        random_state=42
    )
    labels = model.fit_predict(X_scaled)

    # Output to CSV
    pd.DataFrame({
        'id': np.arange(len(labels)),
        'label': labels
    }).to_csv(output_csv, index=False)

    print(f"[DONE] Saved {output_csv} (shape: {len(labels)} rows)")

if __name__ == "__main__":
    run_spectral("public_data.csv", "public_submission.csv", 15)
    run_spectral("private_data.csv", "private_submission.csv", 23)