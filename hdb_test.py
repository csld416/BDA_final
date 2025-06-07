import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan

# Load public data
df = pd.read_csv("public_data.csv")
X = df.iloc[:, :4].values
X_scaled = StandardScaler().fit_transform(X)

# Run HDBSCAN directly on the 4D data
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
labels = clusterer.fit_predict(X_scaled)

# Save output
submission = pd.DataFrame({'id': np.arange(len(labels)), 'label': labels})
submission.to_csv('public_submission.csv', index=False)
