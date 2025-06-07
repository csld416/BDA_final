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


# Load public data and keep only S2 and S3
df = pd.read_csv("public_data.csv")
X = df.iloc[:, [2, 3]].values  # This selects S2 and S3

# Scale
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

# Cluster
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=14, min_samples=7)
labels = clusterer.fit_predict(X_scaled)

# Save
submission = pd.DataFrame({'id': df['id'], 'label': labels})
submission.to_csv("public_submission.csv", index=False)
