"""Evaluate the trained model on hand-written English challenge examples."""

from collections import defaultdict
from dataclasses import dataclass

import joblib

from sentiment_analyzer import MODEL_PATH


@dataclass(frozen=True)
class ChallengeExample:
    category: str
    text: str
    expected: str


EXAMPLES = (
    ChallengeExample("negation", "The acting was not good.", "negative"),
    ChallengeExample("negation", "I wasn't impressed by the performance.", "negative"),
    ChallengeExample("negation", "This isn't a film I'd recommend.", "negative"),
    ChallengeExample("negation", "The plot was not engaging at all.", "negative"),
    ChallengeExample("negation", "The camera quality is not bad at all.", "positive"),
    ChallengeExample(
        "sarcasm",
        "Oh wonderful, another two hours I'll never get back.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "Fantastic, the elevator was broken and the Wi‑Fi failed again.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "Great, my phone battery died in under an hour.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "Lovely, the train was delayed and then cancelled.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "Perfect, I got charged extra for a service I never asked for.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "Beautiful cinematography, completely wasted on a terrible script.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The cast was excellent, but the ending collapsed into nonsense.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The room was spotless, but the staff were rude and dismissive.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The soundtrack was stunning, and the final scene was perfect.",
        "positive",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The food was disappointing, but the desserts were unforgettable.",
        "positive",
    ),
    ChallengeExample("intensity", "It was fine.", "positive"),
    ChallengeExample("intensity", "It was okay.", "positive"),
    ChallengeExample("intensity", "It was good.", "positive"),
    ChallengeExample("intensity", "It was excellent.", "positive"),
    ChallengeExample("intensity", "It was extraordinary.", "positive"),
    ChallengeExample(
        "domain_shift",
        "This coffee grinder is sturdy and quiet, and it makes a perfect espresso.",
        "positive",
    ),
    ChallengeExample(
        "domain_shift",
        "The blender leaked on the first use and the motor smelled burnt.",
        "negative",
    ),
    ChallengeExample(
        "domain_shift",
        "The tacos were fresh, flavorful, and worth every bite.",
        "positive",
    ),
    ChallengeExample(
        "domain_shift",
        "The hotel room smelled musty and the shower was freezing cold.",
        "negative",
    ),
    ChallengeExample(
        "domain_shift",
        "The wireless keyboard feels premium and responsive for daily use.",
        "positive",
    ),
)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run prepare_data.py and train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    category_results: dict[str, list[bool]] = defaultdict(list)

    for example in EXAMPLES:
        prediction = str(model.predict([example.text])[0])
        correct = prediction == example.expected
        category_results[example.category].append(correct)
        status = "PASS" if correct else "FAIL"
        print(
            f"{status:4} [{example.category}] "
            f"expected={example.expected:8} predicted={prediction:8} "
            f"{example.text}"
        )

    print("\nAccuracy by challenge category:")
    for category in [
        "negation",
        "sarcasm",
        "mixed_sentiment",
        "intensity",
        "domain_shift",
    ]:
        results = category_results[category]
        accuracy = sum(results) / len(results) if results else 0.0
        print(f"{category}: {sum(results)}/{len(results)} ({accuracy:.2%})")

    overall_correct = sum(sum(results) for results in category_results.values())
    print(f"Overall: {overall_correct}/{len(EXAMPLES)} ({overall_correct / len(EXAMPLES):.2%})")


if __name__ == "__main__":
    main()
