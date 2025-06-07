import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan
from pathlib import Path

# Load public data
df = pd.read_csv("public_data.csv")
X = df.iloc[:, :4].values
X_scaled = StandardScaler().fit_transform(X)

# Output directory
output_dir = Path("submission_hdbscan_raw")
output_dir.mkdir(exist_ok=True)

# Parameter grid
min_cluster_sizes = [5, 8, 10, 12, 15]
min_samples_list = [3, 5, 7, 10]

i = 0
for cs in min_cluster_sizes:
    for s in min_samples_list:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cs, min_samples=s)
        labels = clusterer.fit_predict(X_scaled)

        filename = f"{i:03d}_cs{cs}_s{s}.csv"
        submission = pd.DataFrame({'id': np.arange(len(labels)), 'label': labels})
        submission.to_csv(output_dir / filename, index=False)
        print(f"[{i:03d}] Saved {filename}")
        i += 1
