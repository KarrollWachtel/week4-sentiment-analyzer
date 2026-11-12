# Dataset preparation

- **Dataset:** UCI Sentiment Labelled Sentences (Amazon, IMDb, and Yelp).
- **Total:** 3,000 labeled English review sentences.
- **Training split:** 2,400 (80%).
  - Negative: 1,200
  - Positive: 1,200
- **Test split:** 600 (20%).
  - Negative: 300
  - Positive: 300
- **Imbalance check:** There is no class imbalance; positive and negative
  examples are exactly balanced in the full dataset and both splits. A model
  that always predicts one class would score only 50% on this dataset.
- **Split design:** Seeded, stratified by both class and source domain, so all
  six source/class groups are represented at the same proportions in train
  and test.
- **Provenance/license:** Kotzias et al., [UCI dataset 331](https://doi.org/10.24432/C57604),
  CC BY 4.0. See the README for attribution.

# Baseline model and challenge findings

- **Model:** TF-IDF word unigrams/bigrams with logistic regression.
- **Held-out test accuracy:** 80.5% (483/600); precision/recall/F1 were about
  0.81/0.79/0.80 for negative and 0.80/0.82/0.81 for positive.
- **Challenge set:** 20 hand-written examples, four each for negation,
  sarcasm, idioms, slang, and mixed sentiment. The model got 11/20 right
  (55%). This is a targeted diagnostic set, not a representative accuracy
  estimate.

| English pattern | Correct | Observed misses |
|---|---:|---|
| Negation | 2/4 | Classified “The camera quality is not bad at all” as negative; failed to resolve “I expected to hate this, but I don't.” |
| Sarcasm | 2/4 | Read “The soundtrack was delightful if you enjoy nails on a chalkboard” and “Perfect, another update that deleted all my saved files” as positive. |
| Idioms | 3/4 | Read “The plot kept me on the edge of my seat” as negative. |
| Slang | 2/4 | Missed positive slang in “This movie absolutely slaps” and “That laptop is fire.” |
| Mixed sentiment | 2/4 | Predicted negative for a positive-ending contrast (“The first half drags, but the finale is brilliant”) and positive for a negative-ending contrast (“The interface is beautiful, but the app crashes constantly”). |

These failures are consistent with a bag-of-words model: it learns local word
associations but does not robustly resolve negation scope, sarcasm, idiomatic
phrases, newer slang, or which clause carries the overall judgment. The small,
curated set only supports these example-level observations; broader challenge
tests are needed before making general claims about deployment performance.
