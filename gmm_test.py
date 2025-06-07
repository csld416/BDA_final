import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from pathlib import Path

# Load data
df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:5].values  # Columns S1 to S4
X_scaled = StandardScaler().fit_transform(X)

# Grid parameters
n_components_list = [13, 14, 15, 16, 17]
covariance_types = ['full', 'tied', 'diag', 'spherical']
reg_covar_list = [1e-6, 1e-5]

# Output directory
output_dir = Path("submission_gmm_grid")
output_dir.mkdir(exist_ok=True)

i = 0
for n in n_components_list:
    for cov in covariance_types:
        for reg in reg_covar_list:
            try:
                gmm = GaussianMixture(n_components=n,
                                      covariance_type=cov,
                                      reg_covar=reg,
                                      random_state=42)
                labels = gmm.fit_predict(X_scaled)

                filename = f"{i:03d}_n{n}_cov{cov}_reg{reg}.csv"
                submission = pd.DataFrame({'id': df['id'], 'label': labels})
                submission.to_csv(output_dir / filename, index=False)
                print(f"[{i:03d}] Saved {filename}")
                i += 1
            except Exception as e:
                print(f"[{i:03d}] Failed n={n}, cov={cov}, reg={reg} → {e}")
                i += 1
