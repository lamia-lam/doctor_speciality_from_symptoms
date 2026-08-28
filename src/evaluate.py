from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/specialty_dataset.csv")
MODEL_PATH = Path("models/specialty_classifier.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
symptom_columns = bundle["symptom_columns"]

# For a formal evaluation pipeline, persist split IDs during training.
# This compact demo example redraws a deterministic split for visualization.
df = pd.read_csv(DATA_PATH)
x = df[symptom_columns]
y = df["specialty"]

_, x_test, _, y_test = train_test_split(
    x,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y,
)

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_estimator(
    model,
    x_test,
    y_test,
    xticks_rotation=90,
    ax=ax,
    colorbar=False,
)
fig.tight_layout()

output = Path("reports/confusion_matrix.png")
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
print(f"Saved {output}")