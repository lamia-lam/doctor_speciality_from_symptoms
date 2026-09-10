"""
generate_thesis_figures.py
Generates all ML result figures needed for the Results & Discussion chapter.
Run from the doctor_speciality_from_symptoms/ directory:
    python src/generate_thesis_figures.py

Outputs written to reports/thesis_figures/:
    01_model_comparison_bar.png
    02_confusion_matrix_lr.png
    03_classification_report_heatmap.png
    04_top_specialties_f1.png
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("data/processed/specialty_dataset.csv")
MODEL_PATH = Path("models/specialty_classifier.joblib")
OUT_DIR    = Path("reports/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ── Colour palette (professional / thesis-friendly) ──────────────────────────
C_RF  = "#2563EB"   # blue  – Random Forest
C_LR  = "#16A34A"   # green – Logistic Regression  (deployed)
C_BNB = "#9333EA"   # purple – BernoulliNB

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "figure.dpi": 150,
})

# ── Load data & rebuild splits ────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
excluded = {"disease", "specialty"}
symptom_columns = [c for c in df.columns if c not in excluded]
X = df[symptom_columns].astype("uint8")
y = df["specialty"].astype(str)

# Same split as train.py  (70 / 15 / 15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

# ── Train all three models, evaluate on VALIDATION set (matches train.py) ─────
print("Training models …")
models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample"),
    "Logistic Regression":  LogisticRegression(solver="saga", max_iter=1000, random_state=RANDOM_STATE),
    "BernoulliNB":          BernoulliNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)          # ← validation set, same as train.py
    results[name] = {
        "model":    model,
        "y_pred":   y_pred,
        "accuracy": accuracy_score(y_val, y_pred),
        "macro_p":  precision_score(y_val, y_pred, average="macro", zero_division=0),
        "macro_r":  recall_score(y_val, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
        "w_f1":     f1_score(y_val, y_pred, average="weighted", zero_division=0),
    }
    print(f"  {name}: acc={results[name]['accuracy']:.4f}  macro_f1={results[name]['macro_f1']:.4f}")

# ── Auto-select best model by validation macro_f1 ────────────────────────────
best_name = max(results, key=lambda n: results[n]["macro_f1"])
print(f"\n★  Best model on validation set: {best_name}  "
      f"(macro_f1={results[best_name]['macro_f1']:.4f})")

# Refit best model on train+val, evaluate on held-out test set (matches train.py)
# Used only for confusion matrix & per-class figures — NOT the comparison table.
X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)
best_model_final = results[best_name]["model"].__class__(**results[best_name]["model"].get_params())
best_model_final.fit(X_train_final, y_train_final)
best_test_pred = best_model_final.predict(X_test)
print(f"  {best_name} (refitted, test set): acc={accuracy_score(y_test, best_test_pred):.4f}  "
      f"macro_f1={f1_score(y_test, best_test_pred, average='macro', zero_division=0):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Model Comparison Grouped Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Figure 1: Model Comparison Bar Chart …")

metrics_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]
metric_keys    = ["accuracy", "macro_p", "macro_r", "macro_f1", "w_f1"]
colors         = [C_RF, C_LR, C_BNB]
model_names    = list(results.keys())

x_pos = np.arange(len(metrics_labels))
width = 0.22

fig, ax = plt.subplots(figsize=(13, 6))

for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][k] for k in metric_keys]
    bars = ax.bar(x_pos + i * width, vals, width, label=name, color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold"
        )

ax.set_xticks(x_pos + width)
ax.set_xticklabels(metrics_labels, fontsize=11)
ax.set_ylim(0.88, 1.01)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Random Forest vs Logistic Regression vs BernoulliNB\n(Validation Set: 15% stratified split)", fontsize=13, fontweight="bold", pad=14)

# Dynamically mark the best model in the legend
color_map = dict(zip(model_names, colors))
patches = []
for n, c in color_map.items():
    label = f"{n} ★ (Best)" if n == best_name else n
    patches.append(mpatches.Patch(color=c, label=label))
ax.legend(handles=patches, fontsize=10, loc="lower right")

ax.axhline(0.95, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(len(metrics_labels) - 0.1, 0.951, "0.95 baseline", fontsize=8, color="gray")

fig.tight_layout()
out = OUT_DIR / "01_model_comparison_bar.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Confusion Matrix (Best Model)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating Figure 2: Confusion Matrix ({best_name}) …")

# Uses best_model_final (refitted on train+val) evaluated on the untouched test set
fig, ax = plt.subplots(figsize=(14, 13))
disp = ConfusionMatrixDisplay.from_estimator(
    best_model_final, X_test, y_test,
    xticks_rotation=90,
    ax=ax,
    colorbar=True,
    cmap="Blues",
)
ax.set_title(
    f"Confusion Matrix — {best_name} (Best Model ★, Deployed)\n22 Doctor Specialties  |  Test Set",
    fontsize=13, fontweight="bold", pad=16
)
fig.tight_layout()
out = OUT_DIR / "02_confusion_matrix_best.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Per-Class F1 Score Horizontal Bar (Best Model)
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating Figure 3: Per-class F1 Score Heatmap ({best_name}) …")

# Uses best_model_final predictions on test set for per-class breakdown
best_pred   = best_test_pred
report_dict = classification_report(y_test, best_pred, output_dict=True, zero_division=0)

classes   = sorted(y.unique().tolist())
f1_scores = [report_dict.get(c, {}).get("f1-score", 0.0) for c in classes]
prec_scores = [report_dict.get(c, {}).get("precision", 0.0) for c in classes]
rec_scores  = [report_dict.get(c, {}).get("recall", 0.0) for c in classes]

y_idx = np.arange(len(classes))
fig, ax = plt.subplots(figsize=(11, 9))

bar_colors = ["#16A34A" if v >= 0.95 else "#F59E0B" if v >= 0.85 else "#EF4444" for v in f1_scores]
bars = ax.barh(y_idx, f1_scores, color=bar_colors, alpha=0.85, edgecolor="white")

for bar, val in zip(bars, f1_scores):
    ax.text(
        min(val + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold"
    )

ax.set_yticks(y_idx)
ax.set_yticklabels(classes, fontsize=10)
ax.set_xlim(0, 1.1)
ax.set_xlabel("F1 Score", fontsize=12)
ax.set_title(
    f"Per-Specialty F1 Score — {best_name} (Best Model ★, Deployed)\nGreen ≥ 0.95 | Amber ≥ 0.85 | Red < 0.85",
    fontsize=13, fontweight="bold", pad=14
)
ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

green_patch  = mpatches.Patch(color="#16A34A", alpha=0.85, label="F1 ≥ 0.95 (Excellent)")
amber_patch  = mpatches.Patch(color="#F59E0B", alpha=0.85, label="F1 ≥ 0.85 (Good)")
red_patch    = mpatches.Patch(color="#EF4444", alpha=0.85, label="F1 < 0.85 (Needs improvement)")
ax.legend(handles=[green_patch, amber_patch, red_patch], fontsize=9, loc="lower right")

fig.tight_layout()
out = OUT_DIR / "03_per_specialty_f1.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Summary Metrics Table (saved as image)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Figure 4: Summary Metrics Table …")

table_data = []
for name in model_names:
    r = results[name]
    tag = " ★ (Best / Deployed)" if name == best_name else ""
    table_data.append([
        name + tag,
        f"{r['accuracy']:.4f}",
        f"{r['macro_p']:.4f}",
        f"{r['macro_r']:.4f}",
        f"{r['macro_f1']:.4f}",
        f"{r['w_f1']:.4f}",
    ])

col_labels = ["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]

fig, ax = plt.subplots(figsize=(16, 2.5))
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1, 2.2)

# Style header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor("#1E3A5F")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Highlight best model row dynamically
best_row_idx = model_names.index(best_name) + 1   # +1 for header row
for j in range(len(col_labels)):
    tbl[best_row_idx, j].set_facecolor("#DCFCE7")

ax.set_title("Table: Model Performance Comparison (Validation Set)", fontsize=13, fontweight="bold", pad=10, y=0.95)
fig.tight_layout()
out = OUT_DIR / "04_model_comparison_table.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Classification Report (text) saved as formatted image
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating Figure 5: Classification Report ({best_name}) …")

# Uses best_model_final (refitted) on test set — matches train.py's final evaluation
report_text = classification_report(y_test, best_test_pred, zero_division=0)
report_filename = best_name.lower().replace(" ", "_")
Path(f"reports/classification_report_{report_filename}.txt").write_text(report_text, encoding="utf-8")

fig, ax = plt.subplots(figsize=(11, 10))
ax.axis("off")
ax.text(
    0.01, 0.99,
    f"Classification Report — {best_name} (Best Model ★, Deployed | Test Set)\n\n" + report_text,
    transform=ax.transAxes,
    fontsize=8.5,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(facecolor="#F8FAFC", edgecolor="#CBD5E1", boxstyle="round,pad=0.6"),
)
fig.tight_layout()
out = OUT_DIR / f"05_classification_report_{report_filename}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {out}")

print("\n✅  All thesis figures saved to:", OUT_DIR.resolve())
print("\nFiles generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
