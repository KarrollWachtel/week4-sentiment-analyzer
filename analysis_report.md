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
- **Confusion matrix:** rows are actual labels and columns are predicted labels;
  for the held-out test set this was `[[237, 63], [54, 246]]` in the order
  `[negative, positive]`.
- **What the model actually sees:** "With a bag-of-words or TF-IDF representation, it sees which words are present and how often — and nothing about the order they came in."
- **Challenge set:** 25 hand-written examples, five each for negation,
  sarcasm, mixed sentiment, intensity, and domain shift. The model got 15/25
  right (60%). This is a targeted diagnostic set, not a representative accuracy
  estimate.

| Category | Correct | Accuracy |
|---|---:|---:|
| Negation | 3/5 | 60.0% |
| Sarcasm | 3/5 | 60.0% |
| Mixed sentiment | 3/5 | 60.0% |
| Intensity | 2/5 | 40.0% |
| Domain shift | 4/5 | 80.0% |

Overall, the model is strong on the held-out review set (80.5%) but much weaker on
hard linguistic phenomena in the challenge set (60.0%). The biggest drop in
performance comes from **intensity**, which was the hardest category. Mild
positive phrases like “It was fine.” and “It was okay.” were often labelled as
negative, and even “It was extraordinary.” was sometimes misread. This is a
classic failure of a bag-of-words model: it learns that some words are associated
with positive or negative sentiment, but not that “fine,” “good,” “excellent,”
and “extraordinary” represent different degrees along the same positive scale.
The model sees words and counts, not graded meaning.

Negation was close behind, and it is exactly the issue that reveals why the model
fails structurally. In a bag-of-words setup, “not good” and “good not” are the
same input because the order of words is discarded before the model ever sees the
text. The token **not** is present in both cases, and no feature records what it
modifies. The model is not being stupid; it was never given the information that
would let it distinguish a negated adjective from a plain one.

The linguistic phenomenon that hurt most was **intensity**, because the model has
no mechanism for representing strength or degree. The next most damaging issue was
**negation**, where the model lacks word-order information and therefore cannot
learn that “not good” is fundamentally different from “good.” Sarcasm and mixed
sentiment also failed, but they are still partly anchored in different polarities
and contrastive phrasing; intensity is the hardest because there is no explicit
feature for scale.

# What I would change to fix negation specifically

To fix negation, I would need a representation that preserves some word order.
The most direct and realistic change would be to add **bigrams** as features, so
that “not good” becomes its own feature distinct from “not” and “good” on their
own. In a TF-IDF model, bigrams would let the system learn patterns such as
“not good,” “was not,” “didn't enjoy,” or “not bad,” instead of treating those
words as independent tokens with no relation to one another. This is a better fit
than unigrams alone because the model would at least capture local negation
groups.

I would try **TF-IDF with bigrams** first, because it is the smallest change to the
existing pipeline and still preserves local ordering. The trade-off is that it
increases sparsity and feature dimensionality, which can make the model less
stable and may require more training data. A stronger but more expensive option
would be a **sequence model** (for example, an LSTM or transformer) that reads the
sentence in order and learns dependencies directly. That would probably handle
negation and sarcasm better, but it costs more compute, more memory, and more
training time, and it is less interpretable than a logistic-regression TF-IDF
model.

# Would I deploy this?

I would not deploy this version in a production customer-routing system without
more safeguards. If a company used this analyser to send angry customers to a
human, the sarcastic ones would likely be misclassified as positive and routed
away from the queue they actually need. Sarcasm is especially dangerous here,
because the text is full of positive words like “wonderful,” “great,” or
“brilliant” while the intended meaning is negative. In practice, that means the
system could under-serve exactly the people most likely to be upset, which is not
just a modeling problem but a customer-experience risk. For a live support flow,
this model would need a human review step, better sarcasm handling, and more
robust sentiment logic before it could be trusted.
