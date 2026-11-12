"""Trainable sentiment pipeline and command-line review predictor."""

import argparse
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def predict_sentiment(text: str, model_path: Path = MODEL_PATH) -> str:
    if not text.strip():
        raise ValueError("Review text must not be empty")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run prepare_data.py and train_model.py first."
        )
    model = joblib.load(model_path)
    return str(model.predict([text])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict review sentiment.")
    parser.add_argument("text", help="English review text to classify")
    arguments = parser.parse_args()
    print(predict_sentiment(arguments.text))


if __name__ == "__main__":
    main()
