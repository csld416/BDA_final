import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PUBLIC_DATA_PATH = 'public_data.csv'
PRIVATE_DATA_PATH = 'private_data.csv'
OUTPUT_DIR = Path('submission_umap_gmm')
OUTPUT_DIR.mkdir(exist_ok=True)

PUBLIC_N_DIM = 4
PRIVATE_N_DIM = 6
PUBLIC_N_CLUSTERS = 15  # 4 * 4 - 1
PRIVATE_N_CLUSTERS = 23  # 4 * 6 - 1

def load_and_preprocess(file_path, n_dim):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :n_dim].values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled

def embed_features(X, n_components=5):
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    X_embedded = reducer.fit_transform(X)
    return X_embedded

def hybrid_features(X_orig, X_embed):
    return np.hstack((X_orig, X_embed))

def run_clustering(X_hybrid, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(X_hybrid)
    return labels

def save_submission(labels, path):
    df = pd.DataFrame({'cluster': labels})
    df.to_csv(path, index=False)
    print(f'[DONE] Saved submission to {path}')

def process(file_path, n_dim, n_clusters, label):
    print(f'[INFO] Processing {label} dataset...')
    X = load_and_preprocess(file_path, n_dim)
    X_umap = embed_features(X, n_components=5)
    X_combined = hybrid_features(X, X_umap)
    labels = run_clustering(X_combined, n_clusters)
    save_submission(labels, OUTPUT_DIR / f'{label}_submission.csv')

def main():
    process(PUBLIC_DATA_PATH, PUBLIC_N_DIM, PUBLIC_N_CLUSTERS, 'public')
    process(PRIVATE_DATA_PATH, PRIVATE_N_DIM, PRIVATE_N_CLUSTERS, 'private')
    print('[INFO] All processing complete.')

if __name__ == '__main__':
    main()