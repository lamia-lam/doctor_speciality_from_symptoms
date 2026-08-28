import joblib

from symptom_parser import SymptomParser

MODEL_PATH = "models/specialty_classifier.joblib"
ALIASES_PATH = "config/symptom_aliases.json"

bundle = joblib.load(MODEL_PATH)
classifier = bundle["model"]

parser = SymptomParser(
    model_path=MODEL_PATH,
    aliases_path=ALIASES_PATH,
)

text = (
    "My head has been hurting badly and I feel dizzy. "
    "I do not have fever."
)

matches = parser.parse(text)
#print(matches)
vector = parser.to_feature_vector(matches)
#print(vector)

probabilities = classifier.predict_proba([vector])[0]
classes = classifier.classes_

ranked = sorted(
    zip(classes, probabilities),
    key=lambda item: item[1],
    reverse=True,
)

print("Detected symptoms:")
for match in matches:
    print(match)

print("\nTop specialties:")
for specialty, probability in ranked[:3]:
    print(f"{specialty}: {probability:.1%}")
# git check