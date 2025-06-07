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
output_dir = Path("submission_hdbscan_fine")
output_dir.mkdir(exist_ok=True)

# Fine-tuned grid near best config
min_cluster_sizes = [13, 14, 15, 16, 17]
min_samples_list = [6, 7, 8]
metrics = ['euclidean', 'manhattan', 'cosine']

i = 0
for cs in min_cluster_sizes:
    for s in min_samples_list:
        for metric in metrics:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cs, min_samples=s, metric=metric)
                labels = clusterer.fit_predict(X_scaled)
                filename = f"{i:03d}_cs{cs}_s{s}_m{metric}.csv"
                submission = pd.DataFrame({'id': np.arange(len(labels)), 'label': labels})
                submission.to_csv(output_dir / filename, index=False)
                print(f"[{i:03d}] Saved {filename}")
                i += 1
            except Exception as e:
                print(f"[{i:03d}] Failed cs={cs}, s={s}, metric={metric} → {e}")
                i += 1
