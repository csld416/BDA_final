import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PUBLIC_DATA_PATH = 'public_data.csv'
PRIVATE_DATA_PATH = 'private_data.csv'
OUTPUT_DIR = Path('submission_hdbscan_umap')
OUTPUT_DIR.mkdir(exist_ok=True)
PUBLIC_N_DIM = 4
PRIVATE_N_DIM = 6

def load_and_preprocess_data(file_path, n_dim):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :n_dim].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df

def reduce_dimensions(X, n_components=2, n_neighbors=15):
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
    X_reduced = reducer.fit_transform(X)
    return X_reduced

def cluster_data(X, min_cluster_size=10, min_samples=5):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    labels = clusterer.fit_predict(X)
    return labels

def visualize_clusters(X_2d, labels, title, save_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=labels, palette='deep', s=50)
    plt.title(title)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Cluster')
    plt.savefig(save_path)
    plt.close()

def visualize_pairwise(df, dim1, dim2, labels, title, save_path):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.iloc[:, dim1], y=df.iloc[:, dim2], hue=labels, palette='deep', s=50)
    plt.title(title)
    plt.xlabel(f'Dimension {dim1 + 1}')
    plt.ylabel(f'Dimension {dim2 + 1}')
    plt.legend(title='Cluster')
    plt.savefig(save_path)
    plt.close()

def save_submission(labels, output_path):
    submission = pd.DataFrame({'id': np.arange(len(labels)), 'label': labels})
    submission.to_csv(output_path, index=False)
    print(f'Saved submission to {output_path}')

def main():
    print('Processing public dataset...')
    X_public, df_public = load_and_preprocess_data(PUBLIC_DATA_PATH, PUBLIC_N_DIM)
    X_public_2d = reduce_dimensions(X_public, n_components=2)
    labels_public = cluster_data(X_public, min_cluster_size=10, min_samples=5)
    visualize_clusters(X_public_2d, labels_public, 'Public Dataset Clusters (UMAP)', OUTPUT_DIR / 'public_umap_clusters.png')
    visualize_pairwise(df_public, 1, 2, labels_public, 'Public Dataset: Dimension 2 vs 3', OUTPUT_DIR / 'public_dim2_vs_dim3.png')
    save_submission(labels_public, OUTPUT_DIR / 'public_submission.csv')

    print('Processing private dataset...')
    X_private, df_private = load_and_preprocess_data(PRIVATE_DATA_PATH, PRIVATE_N_DIM)
    X_private_2d = reduce_dimensions(X_private, n_components=2)
    labels_private = cluster_data(X_private, min_cluster_size=10, min_samples=5)
    visualize_clusters(X_private_2d, labels_private, 'Private Dataset Clusters (UMAP)', OUTPUT_DIR / 'private_umap_clusters.png')
    visualize_pairwise(df_private, 1, 2, labels_private, 'Private Dataset: Dimension 2 vs 3', OUTPUT_DIR / 'private_dim2_vs_dim3.png')
    save_submission(labels_private, OUTPUT_DIR / 'private_submission.csv')

    print('Processing complete. Check submission_hdbscan_umap folder for outputs.')

if __name__ == '__main__':
    main()