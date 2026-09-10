from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


DATA_PATH = Path("data/processed/specialty_dataset.csv")
MODEL_PATH = Path("models/specialty_classifier.joblib")
REPORT_PATH = Path("reports/model_comparison.csv")
RANDOM_STATE = 42


def evaluate(model, x, y) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro"),
        "weighted_f1": f1_score(y, prediction, average="weighted"),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    if "specialty" not in df.columns:
        raise ValueError("Prepared dataset must contain specialty column")

    excluded = {"disease", "specialty"}
    symptom_columns = [c for c in df.columns if c not in excluded]

    x = df[symptom_columns].astype("uint8")
    y = df["specialty"].astype(str)

    # 70% train, 15% validation, 15% test.
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
        
    )
    

    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    models = {
        "bernoulli_nb": BernoulliNB(),
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
    }

    rows: list[dict] = []
    fitted = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_validation, y_validation)
        fitted[name] = model
        rows.append({"model": name, **metrics})
        print(name, metrics)

    comparison = pd.DataFrame(rows).sort_values(
        "macro_f1",
        ascending=False,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(REPORT_PATH, index=False)
    print("\nValidation comparison:")
    print(comparison.to_string(index=False))

    # Logistic Regression generalizes significantly better for real-world user queries
    # where only 1 or 2 symptoms are provided (avoiding decision-tree sparse-feature bias).
    # Auto-select the best model by validation macro_f1.
    best_name = comparison.iloc[0]["model"]   # already sorted descending by macro_f1
    print(f"\n★  Best model on validation set: {best_name}")

    # Refit the selected algorithm on train + validation data only after
    # model selection is finished.
    x_train_final = pd.concat([x_train, x_validation], axis=0)
    y_train_final = pd.concat([y_train, y_validation], axis=0)

    best_model = models[best_name]
    best_model.fit(x_train_final, y_train_final)

    test_prediction = best_model.predict(x_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "macro_f1": f1_score(y_test, test_prediction, average="macro"),
        "weighted_f1": f1_score(
            y_test,
            test_prediction,
            average="weighted",
        ),
    }

    print("\nFinal untouched-test metrics:")
    print(test_metrics)

    report = classification_report(
        y_test,
        test_prediction,
        zero_division=0,
    )
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "symptom_columns": symptom_columns,
        "specialties": sorted(y.unique().tolist()),
        "test_metrics": test_metrics,
        "random_state": RANDOM_STATE,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()