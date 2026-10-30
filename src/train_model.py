"""Training script for the TF-IDF + Logistic Regression sentiment classifier.

This script:
1. Loads and validates the dataset using `data_utils`.
2. Splits data into training and held-out test sets using stratified sampling.
   If supplementary linguistic rows are present, they are strictly kept in the
   training set, ensuring the held-out test set consists 100% of real reviews.
3. Tunes TF-IDF and Logistic Regression hyperparameters using 5-fold Stratified
   Cross-Validation strictly on the training set (preventing data leakage).
4. Evaluates the fitted pipeline on the held-out test data (accuracy, precision,
   recall, F1, confusion matrix, Brier score, log loss).
5. Tests real predictions on core sentiment benchmark expressions and evaluation phrases.
6. Saves the fitted pipeline to disk using joblib.
"""

from typing import Dict, Any, Tuple, Optional
import argparse
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Ensure repo root is on sys.path so this script can be executed both directly
# (e.g. `python src/train_model.py`) and as a module (`python -m src.train_model`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    log_loss,
)

from src.data_utils import load_and_validate_data, split_data, mark_negation


# Words with crucial sentiment or polarity impact that standard English stop word lists
# strip out by default. Preserving them is critical so that negations ("not good", "no help")
# and intensity modifiers ("very good", "too bad", "much better") are retained in the TF-IDF vocabulary.
SENTIMENT_PRESERVED_WORDS = {
    "not", "no", "never", "neither", "nor", "none",
    "very", "too", "much", "more", "most", "well",
    "least", "less", "hardly", "cannot", "cant",
    "couldnt", "without", "against", "but", "however",
    "better", "best", "good", "bad",
}

# The customized stop words list: default English stop words minus sentiment-critical words
SENTIMENT_STOP_WORDS = sorted(list(ENGLISH_STOP_WORDS - SENTIMENT_PRESERVED_WORDS))


def build_pipeline(
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 2.5,
    random_state: int = 42,
    max_iter: int = 1000,
    use_negation_marking: bool = True,
) -> Pipeline:
    """Build a scikit-learn Pipeline combining TF-IDF vectorization and Logistic Regression.

    Args:
        max_features: Maximum vocabulary size cap for the TF-IDF vectorizer.
        ngram_range: Lower and upper boundary of n-grams (1, 2 captures single words & word pairs).
        C: Inverse of regularization strength for Logistic Regression.
        random_state: Random state for reproducible Logistic Regression optimization.
        max_iter: Maximum number of iterations for the solver to converge.
        use_negation_marking: Whether to apply Potts/Pang&Lee mark_negation preprocessor.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    preprocessor = mark_negation if use_negation_marking else None

    tfidf = TfidfVectorizer(
        preprocessor=preprocessor,
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words=SENTIMENT_STOP_WORDS,
        sublinear_tf=True,  # replaces tf with 1 + log(tf), dampening outlier word counts
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        [
            ("tfidf", tfidf),
            ("classifier", classifier),
        ]
    )

    return pipeline


def tune_hyperparameters(
    X_train: pd.Series,
    y_train: pd.Series,
    random_state: int = 42,
    cv_splits: int = 5,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Tune Logistic Regression and TF-IDF hyperparameters via Stratified CV on training data only.

    The held-out test split is strictly excluded to prevent data leakage and optimistic bias.

    Args:
        X_train: Training review text series.
        y_train: Training sentiment label series.
        random_state: Random seed for stratified folding.
        cv_splits: Number of cross-validation folds.

    Returns:
        Tuple of (best fitted Pipeline, dict of tuning search results).
    """
    base_pipeline = build_pipeline(random_state=random_state)

    param_grid = {
        "classifier__C": [1.0, 1.5, 2.0, 2.5, 3.0],
        "tfidf__sublinear_tf": [True],
    }

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=skf,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    tuning_summary = {
        "best_params": grid_search.best_params_,
        "best_cv_f1": float(grid_search.best_score_),
        "cv_results": [
            {
                "params": params,
                "mean_f1": round(float(mean), 4),
                "std_f1": round(float(std), 4),
            }
            for params, mean, std in zip(
                grid_search.cv_results_["params"],
                grid_search.cv_results_["mean_test_score"],
                grid_search.cv_results_["std_test_score"],
            )
        ],
    }

    return grid_search.best_estimator_, tuning_summary


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate fitted pipeline on held-out test data.

    Computes standard discriminative metrics as well as probabilistic calibration metrics
    (Brier score loss and log loss) to evaluate probability reliability.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Test review text series.
        y_test: True test sentiment labels.

    Returns:
        Dictionary of evaluation metrics and report strings.
    """
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="positive", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="positive", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="positive", zero_division=0))
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    # Probability calibration metrics
    classes = list(pipeline.classes_)
    pos_idx = classes.index("positive") if "positive" in classes else 1
    y_test_bin = (y_test == "positive").astype(int)
    pos_probs = y_prob[:, pos_idx]

    brier = float(brier_score_loss(y_test_bin, pos_probs))
    loss = float(log_loss(y_test_bin, pos_probs))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "brier_score": brier,
        "log_loss": loss,
        "test_sample_count": len(y_test),
    }


def train_and_save_model(
    data_path: str = "data/sample_reviews.csv",
    model_output_path: str = "models/sentiment_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
    tune_cv: bool = True,
) -> Dict[str, Any]:
    """Load data, train pipeline on train set only, evaluate on test set, and save artifact.

    Args:
        data_path: Path to dataset CSV.
        model_output_path: Destination path for serialized pipeline.
        test_size: Fraction of samples held out for evaluation.
        random_state: Random seed for reproducibility.
        max_features: TF-IDF feature cap.
        tune_cv: Whether to tune hyperparameters using Stratified CV on train split.

    Returns:
        Evaluation metrics dictionary.
    """
    print(f"--> [1/5] Loading and validating data from: {data_path}")
    df = load_and_validate_data(data_path)
    total_reviews = len(df)
    class_counts = df["sentiment"].value_counts().to_dict()
    print(f"    Loaded {total_reviews} valid reviews: {class_counts}")

    print(f"--> [2/5] Splitting data into train/test sets (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"    Train size: {len(X_train)} | Test size: {len(X_test)}")

    if tune_cv:
        print(f"--> [3/5] Tuning hyperparameters via Stratified 5-Fold CV on training split ONLY...")
        pipeline, tuning_summary = tune_hyperparameters(X_train, y_train, random_state=random_state)
        print(f"    Best hyperparameters found: {tuning_summary['best_params']}")
        print(f"    Best 5-fold CV weighted F1: {tuning_summary['best_cv_f1']:.4f}")
    else:
        print(f"--> [3/5] Building TF-IDF + Logistic Regression pipeline (C=2.5, ngram_range=(1,2))...")
        pipeline = build_pipeline(
            max_features=max_features,
            ngram_range=(1, 2),
            C=2.5,
            random_state=random_state,
        )
        print(f"--> [4/5] Fitting pipeline ONLY on training data...")
        pipeline.fit(X_train, y_train)
        tuning_summary = {"best_params": {"classifier__C": 2.5}, "best_cv_f1": None}

    print(f"--> [5/5] Evaluating on held-out test data (never seen during training or tuning)...")
    metrics = evaluate_model(pipeline, X_test, y_test)
    metrics["best_params"] = tuning_summary["best_params"]
    metrics["best_cv_f1"] = tuning_summary["best_cv_f1"]

    # Ensure parent output directory exists and save model artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"==> Model successfully saved to: {model_output_path}")

    # Print evaluation metrics
    print("\n" + "=" * 60)
    print("                MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Samples: {metrics['test_sample_count']} (100% Real Consumer Reviews)")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f} (positive class)")
    print(f"Recall:       {metrics['recall']:.4f} (positive class)")
    print(f"F1-Score:     {metrics['f1']:.4f} (positive class)")
    print(f"Brier Score:  {metrics['brier_score']:.4f} (lower is better, 0 = perfect)")
    print(f"Log Loss:     {metrics['log_loss']:.4f} (lower is better)")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix [rows=true, cols=pred] (labels: ['negative', 'positive']):")
    print(metrics["confusion_matrix"])
    print("=" * 60)

    # Benchmark prediction check on evaluation phrases
    eval_phrases = [
        ("decent product", "positive"),
        ("much better than before", "positive"),
        ("not bad", "positive"),
        ("average quality", "negative"),
        ("excellent product", "positive"),
        ("not as good as expected", "negative"),
    ]
    classes = list(pipeline.classes_)
    pos_idx = classes.index("positive") if "positive" in classes else 1
    print("\nRequired Evaluation Phrases Predictions:")
    print("-" * 65)
    for phrase, expected in eval_phrases:
        pred_label = pipeline.predict([phrase])[0]
        prob_vec = pipeline.predict_proba([phrase])[0]
        conf = prob_vec[classes.index(pred_label)] * 100
        pos_prob = prob_vec[pos_idx] * 100
        neg_prob = prob_vec[1 - pos_idx] * 100
        status = "MATCH" if pred_label == expected else "MISMATCH"
        print(
            f"  {phrase:28} -> {pred_label:8} (conf: {conf:5.1f}%, pos: {pos_prob:5.1f}%, neg: {neg_prob:5.1f}%) [{status}]"
        )
    print("-" * 65)

    # Target expressions check
    target_expressions = [
        "decent", "better", "excellent", "satisfactory", "amazing",
        "disappointing", "mediocre", "wonderful", "poor", "not bad", "could be better"
    ]
    print("\nTarget Sentiment Expressions Predictions:")
    print("-" * 65)
    for expr in target_expressions:
        pred_label = pipeline.predict([expr])[0]
        prob_vec = pipeline.predict_proba([expr])[0]
        conf = prob_vec[classes.index(pred_label)] * 100
        pos_prob = prob_vec[pos_idx] * 100
        neg_prob = prob_vec[1 - pos_idx] * 100
        print(
            f"  {expr:20} -> {pred_label:8} (conf: {conf:5.1f}%, pos: {pos_prob:5.1f}%, neg: {neg_prob:5.1f}%)"
        )
    print("-" * 65 + "\n")

    return metrics


def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression Sentiment Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sample_reviews.csv",
        help="Path to labeled CSV dataset (default: data/sample_reviews.csv)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/sentiment_pipeline.joblib",
        help="Path where trained pipeline will be saved (default: models/sentiment_pipeline.joblib)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset to include in test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible split and model training (default: 42)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum TF-IDF features (default: 10000)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip grid search CV tuning and use default C=2.5",
    )

    args = parser.parse_args()

    train_and_save_model(
        data_path=args.data_path,
        model_output_path=args.model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
        tune_cv=not args.no_tune,
    )


if __name__ == "__main__":
    main()
