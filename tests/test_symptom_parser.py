from src.symptom_parser import SymptomParser

MODEL_PATH = "models/specialty_classifier.joblib"
ALIASES_PATH = "config/symptom_aliases.json"

parser = SymptomParser(MODEL_PATH, ALIASES_PATH)


def state_map(text: str) -> dict[str, str]:
    return {
        item.canonical: item.state
        for item in parser.parse(text)
    }


def test_headache_present():
    result = state_map("I have a headache")
    assert result["headache"] == "PRESENT"


def test_fever_negated():
    result = state_map("I do not have fever")
    assert result["fever"] == "ABSENT"


def test_multiple_symptoms():
    
    
    
    result = state_map("I have a headache and dizziness")
    assert result["headache"] == "PRESENT"
    assert result["dizziness"] == "PRESENT"
