import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Load data
df = pd.read_csv("public_data.csv")
X = df.iloc[:, 1:5].values  # Columns S1 to S4 (excluding 'id')

# Scale the data
X_scaled = StandardScaler().fit_transform(X)

# Run Gaussian Mixture Model with 15 clusters
gmm = GaussianMixture(n_components=15, covariance_type='full', random_state=42)
labels = gmm.fit_predict(X_scaled)

# Save to public_submission.csv
submission = pd.DataFrame({
    'id': df['id'],
    'label': labels
})
submission.to_csv("public_submission.csv", index=False)

print("✅ GMM clustering complete. Output saved to public_submission.csv.")
