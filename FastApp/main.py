from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.symptom_parser import SymptomParser


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




import pandas as pd
import numpy as np

MODEL_PATH = "models/specialty_classifier.joblib"
ALIASES_PATH = "config/symptom_aliases.json"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data/processed/specialty_dataset.csv"

bundle = joblib.load(MODEL_PATH)
classifier = bundle["model"]
symptom_columns = bundle["symptom_columns"]

parser = SymptomParser(
    model_path=MODEL_PATH,
    aliases_path=ALIASES_PATH,
)

# Pre-load disease symptom matrix for condition suggestion
disease_names: list[str] = []
disease_matrix: np.ndarray = np.zeros((0, len(symptom_columns)))
disease_specialty_map: dict[str, str] = {}

if DATASET_PATH.exists():
    df_data = pd.read_csv(DATASET_PATH)
    disease_group = df_data.groupby("disease")[symptom_columns].max()
    disease_names = disease_group.index.tolist()
    disease_matrix = disease_group.values
    for _, r in df_data[["disease", "specialty"]].drop_duplicates().iterrows():
        disease_specialty_map[r["disease"]] = r["specialty"]

print("Model & disease profiles loaded successfully!")



@app.get("/classify")
def classify(text):
    print(text)

    # 1. Parse the user's text
    matches = parser.parse(text)

    print("Detected symptoms:")
    for match in matches:
        print(match)

    # 2. Convert detected symptoms to feature vector
    vector = parser.to_feature_vector(matches)

    # 3. Predict probabilities
    probabilities = classifier.predict_proba([vector])[0]

    # 4. Get specialty names
    classes = classifier.classes_

    # 5. Rank specialties
    ranked = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    # 6. Top 3 predictions
    top_specialties = [
        {
            "specialty": str(specialty),
            "probability": round(float(probability) * 100, 2)
        }
        for specialty, probability in ranked[:3]
    ]

    # 7. Identify top candidate diseases / conditions matching the symptoms
    possible_conditions: list[str] = []
    if len(disease_names) > 0 and vector.sum() > 0:
        # Calculate symptom overlap count for each disease
        overlaps = disease_matrix @ vector
        top_specialty_name = top_specialties[0]["specialty"] if top_specialties else ""
        
        # Sort diseases by highest overlap count, preferring the predicted specialty
        scored_diseases = []
        for i, disease in enumerate(disease_names):
            score = overlaps[i]
            if score > 0:
                bonus = 1.5 if disease_specialty_map.get(disease) == top_specialty_name else 1.0
                scored_diseases.append((disease, score * bonus))
        
        scored_diseases.sort(key=lambda x: x[1], reverse=True)
        possible_conditions = [d[0] for d in scored_diseases[:2]]

    # 8. Convert SymptomMatch objects to JSON-safe dictionaries
    detected_symptoms = [
        {
            "canonical": str(match.canonical),
            "state": str(match.state),
            "source_text": str(match.source_text),
            "method": str(match.method),
            "score": float(match.score)
        }
        for match in matches
    ]

    
    return {
        "text": str(text),
        "detected_symptoms": detected_symptoms,
        "possible_conditions": possible_conditions,
        "top_specialties": top_specialties
    }