"""Inference and sentiment prediction utilities.

This module provides functions to:
1. Safely load the saved scikit-learn pipeline from disk.
2. Predict sentiment ('positive' or 'negative') for single reviews or lists/batches of reviews.
3. Compute calibrated confidence scores via class probabilities.
4. Validate input text and provide beginner-friendly, descriptive error messages.
"""

from typing import Union, List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import joblib
import pandas as pd

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.pipeline import Pipeline
from src.data_utils import mark_negation

DEFAULT_MODEL_PATH = "models/sentiment_pipeline.joblib"


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a trained scikit-learn pipeline from disk.

    Args:
        model_path: Path to the .joblib artifact.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file is corrupted or not a valid Pipeline.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            f"Please run model training first: 'python src/train_model.py'"
        )

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to load model artifact from '{model_path}': {exc}"
        ) from exc

    if not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            f"Object loaded from '{model_path}' is not a valid classifier pipeline."
        )

    return pipeline


def predict_sentiment(
    texts: Union[str, List[str]],
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline: Optional[Pipeline] = None,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Predict sentiment ('positive' or 'negative') for one review or a list of reviews.

    Args:
        texts: A single review string or a list of review strings.
        model_path: Path to model file if pipeline is not provided directly.
        pipeline: Optional pre-loaded Pipeline instance (avoids reloading on repeated calls).

    Returns:
        If a single string was passed:
            A dict: {
                "text": str,
                "sentiment": "positive" | "negative",
                "confidence": float,
                "probabilities": {"positive": float, "negative": float}
            }
        If a list of strings was passed:
            A list of such dicts.

    Raises:
        ValueError: If input is empty, None, or contains non-string elements.
        FileNotFoundError: If model file cannot be found.
    """
    if pipeline is None:
        pipeline = load_model(model_path)

    is_single_input = isinstance(texts, str)
    raw_list: List[str] = [texts] if is_single_input else texts

    if not isinstance(raw_list, (list, tuple)):
        raise ValueError(
            f"Input texts must be a string or list of strings. Received: {type(texts).__name__}"
        )

    if len(raw_list) == 0:
        raise ValueError("Cannot predict on an empty list of reviews.")

    # Validate each review string
    cleaned_texts: List[str] = []
    for idx, item in enumerate(raw_list):
        if item is None:
            raise ValueError(f"Review at index {idx} is None. Reviews must be non-empty strings.")
        if not isinstance(item, str):
            raise ValueError(f"Review at index {idx} is not a string (type: {type(item).__name__}).")
        
        stripped = item.strip()
        if len(stripped) == 0:
            raise ValueError(f"Review at index {idx} is empty or contains only whitespace.")
        
        cleaned_texts.append(stripped)

    # Model inference
    predictions = pipeline.predict(cleaned_texts)
    probabilities = pipeline.predict_proba(cleaned_texts)
    classes = list(pipeline.classes_)

    results: List[Dict[str, Any]] = []
    for text, pred_label, probs in zip(cleaned_texts, predictions, probabilities):
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
        confidence = float(prob_dict.get(pred_label, max(probs)))

        results.append(
            {
                "text": text,
                "sentiment": str(pred_label),
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
            }
        )

    return results[0] if is_single_input else results


def predict_batch_dataframe(
    df: pd.DataFrame,
    text_col: str = "review_text",
    model_path: str = DEFAULT_MODEL_PATH,
    pipeline: Optional[Pipeline] = None,
) -> pd.DataFrame:
    """Predict sentiment for an entire pandas DataFrame.

    Adds 'predicted_sentiment' and 'confidence' columns to a copy of the DataFrame.

    Args:
        df: Input DataFrame containing review texts.
        text_col: Column name containing the text.
        model_path: Path to model artifact.
        pipeline: Optional pre-loaded pipeline.

    Returns:
        New DataFrame with predictions appended.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    if pipeline is None:
        pipeline = load_model(model_path)

    # Fill NaN or empty rows safely
    result_df = df.copy()
    raw_texts = result_df[text_col].fillna("").astype(str).str.strip().tolist()

    # For any genuinely blank rows, we mark them instead of failing the batch
    valid_indices = [i for i, t in enumerate(raw_texts) if len(t) > 0]
    valid_texts = [raw_texts[i] for i in valid_indices]

    sentiments = ["unclassifiable"] * len(raw_texts)
    confidences = [0.0] * len(raw_texts)

    if valid_texts:
        preds = pipeline.predict(valid_texts)
        probs = pipeline.predict_proba(valid_texts)
        classes = list(pipeline.classes_)

        for idx, pred_label, prob_vec in zip(valid_indices, preds, probs):
            sentiments[idx] = str(pred_label)
            cls_idx = classes.index(pred_label) if pred_label in classes else 0
            confidences[idx] = round(float(prob_vec[cls_idx]), 4)

    result_df["predicted_sentiment"] = sentiments
    result_df["confidence"] = confidences
    return result_df


if __name__ == "__main__":
    import sys

    # Quick interactive CLI test
    test_input = sys.argv[1] if len(sys.argv) > 1 else "This product exceeded all my expectations, absolutely wonderful!"
    print(f"Running inference for text: '{test_input}'")
    try:
        res = predict_sentiment(test_input)
        print("Prediction result:")
        print(f"  Sentiment:  {res['sentiment']}")
        print(f"  Confidence: {res['confidence'] * 100:.2f}%")
        print(f"  Class Probs: {res['probabilities']}")
    except Exception as e:
        print(f"Error during prediction: {e}")
