"""Training script for the TF-IDF + Logistic Regression sentiment classifier.

This script:
1. Loads and validates the dataset using `data_utils`.
2. Splits data into training and held-out test sets (stratified).
3. Constructs a scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
4. Fits the pipeline ONLY on the training data to prevent data leakage.
5. Evaluates the model on the test data (accuracy, precision, recall, F1, confusion matrix).
6. Saves the fitted pipeline to disk using joblib.
"""

from typing import Dict, Any, Tuple
import argparse
import os
import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure repo root is on sys.path so this script can be executed both directly
# (e.g. `python src/train_model.py`) and as a module (`python -m src.train_model`).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.data_utils import load_and_validate_data, split_data


def build_pipeline(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """Build a scikit-learn Pipeline combining TF-IDF vectorization and Logistic Regression.

    Args:
        max_features: Maximum vocabulary size cap for the TF-IDF vectorizer.
        ngram_range: Lower and upper boundary of n-grams (1, 2 captures single words & word pairs).
        random_state: Random state for reproducible Logistic Regression optimization.
        max_iter: Maximum number of iterations for the solver to converge.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        sublinear_tf=True,  # replaces tf with 1 + log(tf), dampening outlier word counts
    )

    classifier = LogisticRegression(
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


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate fitted pipeline on held-out test data.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        X_test: Test review text series.
        y_test: True test sentiment labels.

    Returns:
        Dictionary of evaluation metrics and report strings.
    """
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="positive", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="positive", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="positive", zero_division=0))
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "test_sample_count": len(y_test),
    }


def train_and_save_model(
    data_path: str = "data/sample_reviews.csv",
    model_output_path: str = "models/sentiment_pipeline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
) -> Dict[str, Any]:
    """Load data, train pipeline on train set only, evaluate on test set, and save artifact.

    Args:
        data_path: Path to dataset CSV.
        model_output_path: Destination path for serialized pipeline.
        test_size: Fraction of samples held out for evaluation.
        random_state: Random seed for reproducibility.
        max_features: TF-IDF feature cap.

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

    print(f"--> [3/5] Building TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline(
        max_features=max_features,
        ngram_range=(1, 2),
        random_state=random_state,
    )

    print(f"--> [4/5] Fitting pipeline ONLY on training data (preventing data leakage)...")
    pipeline.fit(X_train, y_train)

    print(f"--> [5/5] Evaluating on held-out test data...")
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Ensure parent output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"==> Model successfully saved to: {model_output_path}")

    # Print evaluation metrics
    print("\n" + "=" * 50)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test Samples: {metrics['test_sample_count']}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f} (positive class)")
    print(f"Recall:       {metrics['recall']:.4f} (positive class)")
    print(f"F1-Score:     {metrics['f1']:.4f} (positive class)")
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix [rows=true, cols=pred] (labels: ['negative', 'positive']):")
    print(metrics["confusion_matrix"])
    print("=" * 50 + "\n")

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
        default=5000,
        help="Maximum TF-IDF features (default: 5000)",
    )

    args = parser.parse_args()

    train_and_save_model(
        data_path=args.data_path,
        model_output_path=args.model_path,
        test_size=args.test_size,
        random_state=args.random_state,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
