"""Train and evaluate the TF-IDF/logistic-regression sentiment model."""

import csv
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report

from sentiment_analyzer import build_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"


def load_split(path: Path) -> tuple[list[str], list[str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    if not rows:
        raise ValueError(f"No examples found in {path}")
    return [row["text"] for row in rows], [row["sentiment"] for row in rows]


def main() -> None:
    train_texts, train_labels = load_split(DATA_DIRECTORY / "train.csv")
    test_texts, test_labels = load_split(DATA_DIRECTORY / "test.csv")

    model = build_pipeline()
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Test accuracy: {accuracy_score(test_labels, predictions):.3f}")
    print(classification_report(test_labels, predictions, zero_division=0))
    print(f"Saved model to {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
