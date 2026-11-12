"""Download the UCI review-sentence dataset and create reproducible CSV splits."""

import csv
import io
import random
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)
SOURCE_FILES = {
    "amazon": "sentiment labelled sentences/amazon_cells_labelled.txt",
    "imdb": "sentiment labelled sentences/imdb_labelled.txt",
    "yelp": "sentiment labelled sentences/yelp_labelled.txt",
}
LABELS = {0: "negative", 1: "positive"}
TEST_SIZE = 0.2
RANDOM_STATE = 42


@dataclass(frozen=True)
class Review:
    text: str
    sentiment: str
    source: str


def load_reviews(archive_bytes: bytes) -> list[Review]:
    """Parse the three tab-separated sources from the UCI archive."""
    reviews: list[Review] = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for source, filename in SOURCE_FILES.items():
            with archive.open(filename) as dataset_file:
                contents = dataset_file.read().decode("cp1252")
                for raw_line in contents.split("\n"):
                    raw_line = raw_line.rstrip("\r")
                    if not raw_line.strip():
                        continue
                    text, raw_label = raw_line.rsplit("\t", maxsplit=1)
                    label = int(raw_label)
                    if label not in LABELS:
                        raise ValueError(f"Unexpected label {label} in {filename}")
                    reviews.append(Review(text.strip(), LABELS[label], source))

    if len(reviews) != 3000:
        raise ValueError(f"Expected 3,000 reviews, found {len(reviews)}")
    source_label_counts = Counter(
        (review.source, review.sentiment) for review in reviews
    )
    unexpected_counts = {
        key: count
        for key, count in source_label_counts.items()
        if count != 500
    }
    if len(source_label_counts) != 6 or unexpected_counts:
        raise ValueError(
            "Expected 500 examples for each source/sentiment pair; "
            f"found {dict(source_label_counts)}"
        )
    return reviews


def split_reviews(
    reviews: list[Review],
) -> tuple[list[Review], list[Review]]:
    """Create an 80/20 split stratified by both sentiment and source."""
    indices = list(range(len(reviews)))
    strata = [f"{review.source}:{review.sentiment}" for review in reviews]
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )
    return (
        [reviews[index] for index in train_indices],
        [reviews[index] for index in test_indices],
    )


def write_csv(path: Path, reviews: list[Review]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=("text", "sentiment", "source"))
        writer.writeheader()
        writer.writerows(asdict(review) for review in reviews)


def count_labels(reviews: list[Review]) -> dict[str, int]:
    counts = Counter(review.sentiment for review in reviews)
    return {label: counts[label] for label in ("negative", "positive")}


def main() -> None:
    request = urllib.request.Request(
        DATASET_URL, headers={"User-Agent": "week4-sentiment-analyzer/1.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        reviews = load_reviews(response.read())

    train_reviews, test_reviews = split_reviews(reviews)
    write_csv(DATA_DIRECTORY / "train.csv", train_reviews)
    write_csv(DATA_DIRECTORY / "test.csv", test_reviews)

    print(f"Total: {len(reviews)}")
    print(f"Train: {len(train_reviews)} {count_labels(train_reviews)}")
    print(f"Test:  {len(test_reviews)} {count_labels(test_reviews)}")
    print(f"Saved splits to {DATA_DIRECTORY.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
