"""
Train the protein structure classification model.
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from features import build_feature_table


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_proteins.csv"
MODEL_PATH = ROOT / "models" / "protein_structure_model.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = build_feature_table(df)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model training complete.")
    print(f"Test accuracy: {accuracy:.2f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))

    # Cross-validation is small because this sample dataset is small.
    cv_scores = cross_val_score(model, X, y, cv=3)
    print(f"3-fold cross-validation accuracy: {cv_scores.mean():.2f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns)
        },
        MODEL_PATH
    )

    print()
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
