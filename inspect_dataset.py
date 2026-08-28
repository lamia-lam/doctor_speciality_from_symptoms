from pathlib import Path
import pandas as pd

files = list(Path("data/raw").glob("*.csv"))
if not files:
    raise FileNotFoundError("No CSV found in data/raw")

csv_path = max(files, key=lambda p: p.stat().st_size)
df = pd.read_csv(csv_path)

print("File:", csv_path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:20])
print(df.head())
print(df.dtypes.value_counts())