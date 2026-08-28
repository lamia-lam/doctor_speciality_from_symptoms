import json
from pathlib import Path
import joblib

bundle = joblib.load("models/specialty_classifier.joblib")
symptoms = bundle["symptom_columns"]

aliases = {}
for symptom in symptoms:
    human_name = symptom.replace("_", " ").strip()
    aliases[symptom] = sorted({symptom, human_name})

output = Path("config/symptom_aliases.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(
    json.dumps(aliases, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Created {output} with {len(aliases)} canonical symptoms")
