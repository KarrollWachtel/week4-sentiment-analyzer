# Week 4: Sentiment Analyzer

A reproducible baseline sentiment classifier for English review sentences.
It uses TF-IDF word unigrams/bigrams and logistic regression.

## Setup and run

```powershell
python -m pip install -r requirements.txt
python prepare_data.py
python train_model.py
python sentiment_analyzer.py "The food was excellent and the staff were kind."
python challenge_set.py
```

`prepare_data.py` downloads the three-domain UCI dataset and writes
`data/processed/train.csv` and `data/processed/test.csv`. The split is
80/20, seeded, and stratified by both sentiment and source (Amazon, IMDb,
and Yelp). Each CSV contains `text`, `sentiment`, and `source`.

The trained model is written to `models/sentiment_pipeline.joblib`, which is
excluded from version control. The dataset CSVs are retained for reproducibility.

## Data attribution

The data is the [UCI Sentiment Labelled Sentences dataset](https://doi.org/10.24432/C57604),
created by Dimitrios Kotzias et al. It contains 3,000 sentences drawn from
product, movie, and restaurant reviews: 500 positive and 500 negative examples
from each of Amazon, IMDb, and Yelp. It is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Baseline and limitations

The model is a baseline, not a general English sentiment understanding system.
Its word/ngram features do not reliably represent sarcasm, complex negation,
context-dependent meaning, or text unlike the three review sources. The
stratified random test split includes all three sources in both sets, so it
measures held-out sentences from familiar domains rather than performance on a
new domain. `challenge_set.py` runs a small hand-written suite covering
negation, sarcasm, idioms, slang, and mixed sentiment; its results are reported
in `analysis_report.md`. Treat this challenge set as diagnostic examples, not a
representative estimate of production performance.
