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
    ChallengeExample("negation", "The movie was not good.", "negative"),
    ChallengeExample("negation", "The camera quality is not bad at all.", "positive"),
    ChallengeExample("negation", "I can't say I didn't enjoy this film.", "positive"),
    ChallengeExample("negation", "I expected to hate this, but I don't.", "positive"),
    ChallengeExample(
        "sarcasm",
        "What a brilliant idea to put the only exit behind the broken door.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "The service was so fast that I grew old waiting for dessert.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "The soundtrack was delightful if you enjoy nails on a chalkboard.",
        "negative",
    ),
    ChallengeExample(
        "sarcasm",
        "Perfect, another update that deleted all my saved files.",
        "negative",
    ),
    ChallengeExample("idiom", "This phone is a steal at this price.", "positive"),
    ChallengeExample("idiom", "The ending fell flat.", "negative"),
    ChallengeExample(
        "idiom", "The plot kept me on the edge of my seat.", "positive"
    ),
    ChallengeExample(
        "idiom",
        "The customer support left a bad taste in my mouth.",
        "negative",
    ),
    ChallengeExample("slang", "This movie absolutely slaps.", "positive"),
    ChallengeExample("slang", "The food was mid at best.", "negative"),
    ChallengeExample("slang", "That laptop is fire.", "positive"),
    ChallengeExample("slang", "The hotel staff were rude as hell.", "negative"),
    ChallengeExample(
        "mixed_sentiment",
        "The acting was excellent, but the story was a complete mess.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The first half drags, but the finale is brilliant.",
        "positive",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The interface is beautiful, but the app crashes constantly.",
        "negative",
    ),
    ChallengeExample(
        "mixed_sentiment",
        "The room was tiny, but the location was perfect.",
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
    for category, results in sorted(category_results.items()):
        print(f"{category}: {sum(results)}/{len(results)}")
    print(f"Overall: {sum(map(sum, category_results.values()))}/{len(EXAMPLES)}")


if __name__ == "__main__":
    main()
