import pandas as pd
from grader import score
from pathlib import Path

# Paths
SUB_DIR = Path("submission_hdbscan_gridsearch")
LOG_FILE = Path("score_log.csv")

# Prepare result list
records = []

for file in sorted(SUB_DIR.glob("*.csv")):
    try:
        submission = pd.read_csv(file).sort_values("id").reset_index(drop=True)
        labels_pred = submission["label"].tolist()
        s = score(labels_pred)
        print(f"{file.name}: {s:.4f}")
        records.append({"file": file.name, "score": s})
    except Exception as e:
        print(f"Failed to score {file.name}: {e}")
        records.append({"file": file.name, "score": -1.0})

# Save all results
df = pd.DataFrame(records).sort_values(by="score", ascending=False)
df.to_csv(LOG_FILE, index=False)
print(f"✓ Score log saved to {LOG_FILE}")
