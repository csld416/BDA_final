import pandas as pd
import shutil
from grader import score
from pathlib import Path

# Setup
input_dir = Path("submission_hdbscan_raw")
log_file = Path("score_log.csv")
public_file = Path("public_submission.csv")

# Results storage
records = []

# Loop through each CSV file
for file in sorted(input_dir.glob("*.csv")):
    try:
        shutil.copy(file, public_file)  # overwrite public_submission.csv
        submission = pd.read_csv(public_file).sort_values("id").reset_index(drop=True)
        labels_pred = submission["label"].tolist()
        s = score(labels_pred)
        print(f"{file.name}: {s:.4f}")
        records.append({"file": file.name, "score": s})
    except Exception as e:
        print(f"❌ Failed to score {file.name}: {e}")
        records.append({"file": file.name, "score": -1.0})

# Save score log
df = pd.DataFrame(records).sort_values(by="score", ascending=False)
df.to_csv(log_file, index=False)
print(f"\n✅ All scores saved to {log_file}")
print(f"🏆 Best score: {df.iloc[0]['score']:.4f} ({df.iloc[0]['file']})")
