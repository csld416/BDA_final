import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('public_data.csv')
    X = df.iloc[:, :4].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def run():
    X = load_data()
    output_dir = Path('submission_hdbscan_gridsearch')
    output_dir.mkdir(exist_ok=True)

    umap_n_neighbors = [5, 10, 30]
    umap_min_dist = [0.0, 0.1, 0.5]
    hdb_min_cluster_size = [5, 10, 15]
    hdb_min_samples = [5, 7, 10]

    i = 0
    for n_neighbors in umap_n_neighbors:
        for min_dist in umap_min_dist:
            for min_cluster_size in hdb_min_cluster_size:
                for min_samples in hdb_min_samples:
                    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
                    X_reduced = reducer.fit_transform(X)
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                    labels = clusterer.fit_predict(X_reduced)

                    filename = f"{i:03d}_n{n_neighbors}_d{min_dist}_cs{min_cluster_size}_s{min_samples}.csv"
                    submission = pd.DataFrame({'id': np.arange(len(labels)), 'label': labels})
                    submission.to_csv(output_dir / filename, index=False)
                    print(f"[{i:03d}] Saved {filename}")
                    i += 1

if __name__ == '__main__':
    run()
