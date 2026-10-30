# Sentiment Pipeline Model

## What this file is

[`sentiment_pipeline.joblib`](./sentiment_pipeline.joblib) is a serialized,
fitted scikit-learn pipeline used by the app's local binary sentiment
classifier. It is a binary artifact, not a text or source file; do not edit it
in a text editor. This document describes the saved model in a readable form.

## Saved pipeline

The pipeline runs these stages in order:

1. **TF-IDF text vectorizer**
   - Applies `mark_negation` before tokenization so words within negated
     phrases receive distinct tokens (for example, `not_good`).
   - Uses unigrams and bigrams, with a vocabulary of 10,000 features.
   - Uses English stop words while retaining sentiment-critical words.
   - Applies sublinear term frequency, inverse document frequency, and L2
     normalization.
2. **Logistic regression classifier**
   - Predicts `negative` or `positive`.
   - Uses `lbfgs`, `C=3.0`, `max_iter=1000`, and random seed `42`.
   - Supports class probability estimates used by the baseline's confidence
     output.

The saved `C=3.0` is the best value selected from the training script's
cross-validation grid (`1.0`, `1.5`, `2.0`, `2.5`, and `3.0`).

## Load and retrain

The app loads the artifact through `src.predict.load_model`. To retrain it from
the repository root using the default dataset and training settings:

```bash
python src/train_model.py
```

## scikit-learn version compatibility

The artifact records scikit-learn version `1.9.1`. In the environment used to
inspect it, scikit-learn `1.7.2` emitted an `InconsistentVersionWarning` while
loading. Basic class prediction ran, but probability prediction failed because
the deserialized classifier lacks the `multi_class` attribute expected by
scikit-learn 1.7.2. Since the app uses probabilities, inference with this
artifact is currently broken in that environment. Scikit-learn does not
guarantee compatibility across versions; use version `1.9.1` or retrain the
model in the target runtime environment.
