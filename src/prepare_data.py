from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
MAPPING_PATH = Path("config/disease_specialty_mapping.csv")
OUTPUT_PATH = Path("data/processed/specialty_dataset.csv")


def normalize_name(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def find_dataset_csv() -> Path:
    files = list(RAW_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            "No CSV files found under data/raw"
        )

    return max(files, key=lambda p: p.stat().st_size)


def find_target_column(df: pd.DataFrame) -> str:
    normalized = {
        normalize_name(c): c
        for c in df.columns
    }

    for candidate in (
        "disease",
        "prognosis",
        "diagnosis",
        "label",
    ):
        if candidate in normalized:
            return normalized[candidate]

    object_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if len(object_columns) == 1:
        return object_columns[0]

    raise ValueError(
        "Could not determine disease target column. "
        f"Columns: {df.columns.tolist()}"
    )


def main() -> None:

    # ---------------------------------------------------------
    # 1. Read raw dataset
    # ---------------------------------------------------------

    source = find_dataset_csv()
    print(f"Reading {source}")

    df = pd.read_csv(source)

    target_column = find_target_column(df)

    df = df.rename(
        columns={target_column: "disease"}
    )

    df["disease"] = (
        df["disease"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # 2. Fix clear disease spelling errors
    # ---------------------------------------------------------

    disease_fixes = {
        "Diabetes": "Diabetes",
        "Hypertension": "Hypertension",
        "Peptic ulcer diseae":
            "Peptic ulcer disease",
        "Osteoarthristis":
            "Osteoarthritis",
        "(vertigo) Paroymsal  Positional Vertigo":
            "Paroxysmal positional vertigo",
        "Dimorphic hemmorhoids(piles)":
            "Dimorphic hemorrhoids (piles)",
    }

    df["disease"] = df["disease"].replace(
        disease_fixes
    )

    # ---------------------------------------------------------
    # 3. Read disease -> specialty mapping
    # ---------------------------------------------------------

    mapping = pd.read_csv(
        MAPPING_PATH
    )

    required_mapping_columns = {
        "disease",
        "specialty",
    }

    if not required_mapping_columns.issubset(
        mapping.columns
    ):
        raise ValueError(
            "Mapping CSV must contain columns: "
            "disease,specialty"
        )

    mapping["disease"] = (
        mapping["disease"]
        .astype(str)
        .str.strip()
    )

    mapping["specialty"] = (
        mapping["specialty"]
        .astype(str)
        .str.strip()
    )

    mapping = mapping.drop_duplicates(
        subset=["disease"],
        keep="last",
    )

    # ---------------------------------------------------------
    # 4. Current dataset is already one-hot encoded
    # ---------------------------------------------------------
    #
    # Example:
    #
    # disease | headache | dizziness | fever | specialty
    # ---------------------------------------------------------
    #
    # Therefore we DON'T convert cell values such as
    # 0 and 1 into symptom names.
    #
    # We simply keep the existing symptom columns.
    # ---------------------------------------------------------

    symptom_source_columns = [
        c for c in df.columns
        if c not in {"disease", "specialty"}
    ]

    processed = df[
        ["disease"] + symptom_source_columns
    ].copy()

    # Make sure every symptom column is binary.
    processed[symptom_source_columns] = (
        processed[symptom_source_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
        .clip(0, 1)
        .astype("uint8")
    )

    symptom_columns = symptom_source_columns

    # ---------------------------------------------------------
    # 5. Disease -> specialty mapping
    # ---------------------------------------------------------

    def disease_key(value: str) -> str:
        return str(value).strip().lower()

    processed["_disease_key"] = (
        processed["disease"]
        .map(disease_key)
    )

    mapping["_disease_key"] = (
        mapping["disease"]
        .map(disease_key)
    )

    all_diseases = set(
        processed["_disease_key"].unique()
    )

    mapped_diseases = set(
        mapping["_disease_key"].unique()
    )

    unmapped = sorted(
        all_diseases - mapped_diseases
    )

    print(
        f"Raw rows: {len(df):,}"
    )

    print(
        f"Unique diseases: {len(all_diseases):,}"
    )

    print(
        f"Mapped diseases: "
        f"{len(all_diseases & mapped_diseases):,}"
    )

    print(
        f"Unmapped diseases: "
        f"{len(unmapped):,}"
    )

    # ---------------------------------------------------------
    # 6. Stop if diseases are missing from mapping
    # ---------------------------------------------------------

    if unmapped:

        report_path = Path(
            "reports/unmapped_diseases.txt"
        )

        report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        report_path.write_text(
            "\n".join(unmapped),
            encoding="utf-8",
        )

        raise ValueError(
            f"Unmapped diseases found: {unmapped}"
        )

    # ---------------------------------------------------------
    # 7. Merge specialty
    # ---------------------------------------------------------

    processed = processed.merge(
        mapping[
            ["_disease_key", "specialty"]
        ],
        on="_disease_key",
        how="inner",
    )

    processed = processed.drop(
        columns=["_disease_key"]
    )

    # ---------------------------------------------------------
    # 8. Keep ALL original rows
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # Your current expanded dataset has 1,100 rows.
    #
    # We do NOT collapse these rows to one row
    # per disease.
    # ---------------------------------------------------------

    duplicates_removed = 0

    # ---------------------------------------------------------
    # 9. Arrange columns
    # ---------------------------------------------------------

    ordered_columns = (
        ["disease"]
        + sorted(symptom_columns)
        + ["specialty"]
    )

    processed = processed[
        ordered_columns
    ]

    # ---------------------------------------------------------
    # 10. Save processed dataset
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # 11. Print summary
    # ---------------------------------------------------------

    print(
        f"Exact duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Prepared rows: "
        f"{len(processed):,}"
    )

    print(
        f"Symptom features: "
        f"{len(symptom_columns):,}"
    )

    print(
        f"Specialties: "
        f"{processed['specialty'].nunique():,}"
    )

    print(
        "\nSpecialty distribution:"
    )

    print(
        processed["specialty"].value_counts()
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()