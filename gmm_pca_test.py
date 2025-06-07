import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pathlib import Path

# Load and preprocess public data
df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:5].values  # S1 to S4
X_scaled = StandardScaler().fit_transform(X)

# Reduce dimensions using PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Parameters to explore
n_components_list = [12, 13, 14]
reg_covar_list = [1e-6, 5e-6, 1e-5, 2e-5]
covariance_types = ['tied', 'full']

# Output directory
output_dir = Path("submission_pca_gmm")
output_dir.mkdir(exist_ok=True)

# Run grid search
i = 0
for n in n_components_list:
    for cov in covariance_types:
        for reg in reg_covar_list:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=cov,
                    reg_covar=reg,
                    random_state=42
                )
                labels = gmm.fit_predict(X_pca)
                filename = f"{i:03d}_n{n}_cov{cov}_reg{reg}_pca.csv"
                submission = pd.DataFrame({'id': df['id'], 'label': labels})
                submission.to_csv(output_dir / filename, index=False)
                print(f"[{i:03d}] Saved {filename}")
                i += 1
            except Exception as e:
                print(f"[{i:03d}] Failed: n={n}, cov={cov}, reg={reg} → {e}")
                i += 1
