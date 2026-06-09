"""
Predict the structural class of a protein sequence.
"""

from pathlib import Path
import sys
import pandas as pd
import joblib

from features import extract_features


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "protein_structure_model.pkl"


def predict_structure(sequence: str):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run this first: python src/train_model.py"
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_columns = saved["feature_columns"]

    features = extract_features(sequence)
    X = pd.DataFrame([features])

    # Match training feature order
    X = X[feature_columns]

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    labels = model.classes_
    confidence = dict(zip(labels, probabilities))

    return prediction, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "MKTAYIAKQRQISFVKSHFSRQDILDLIC"')
        sys.exit(1)

    sequence = sys.argv[1]
    prediction, confidence = predict_structure(sequence)

    print(f"Predicted structure class: {prediction}")
    print("Confidence scores:")
    for label, score in confidence.items():
        print(f"  {label}: {score:.2f}")
