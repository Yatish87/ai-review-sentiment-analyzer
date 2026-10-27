"""Data loading, cleaning, and preprocessing utilities for sentiment analysis.

This module provides functions to:
1. Load a review dataset from CSV.
2. Validate and clean review text and sentiment labels.
3. Normalize labels to 'positive' and 'negative'.
4. Perform a stratified train/test split to avoid data leakage.
"""

from typing import Tuple, List, Optional
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Canonical label names used across the project
VALID_LABELS = {"positive", "negative"}


def normalize_label(label: str) -> str:
    """Normalize a raw sentiment label string to 'positive' or 'negative'.

    Handles variations like 'POS', 'Positive', 'pos', '1', 'neg', etc.

    Args:
        label: Raw label value from dataset.

    Returns:
        Normalized string 'positive' or 'negative'.

    Raises:
        ValueError: If label cannot be mapped to 'positive' or 'negative'.
    """
    if pd.isna(label):
        raise ValueError("Encountered null or missing label.")

    cleaned = str(label).strip().lower()

    if cleaned in {"positive", "pos", "1", "1.0", "true"}:
        return "positive"
    elif cleaned in {"negative", "neg", "0", "0.0", "false"}:
        return "negative"
    else:
        raise ValueError(
            f"Unrecognized sentiment label '{label}'. Expected positive or negative."
        )


def load_and_validate_data(
    filepath: str,
    text_col: str = "review_text",
    label_col: str = "sentiment",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load and validate a CSV dataset for sentiment classification.

    Performs the following safety checks and cleaning steps:
    - Verifies file exists on disk.
    - Checks required columns are present.
    - Drops rows with missing or blank text/labels.
    - Normalizes labels to 'positive' and 'negative'.
    - Checks that both 'positive' and 'negative' classes are present.
    - Drops exact duplicate review texts to prevent data leakage across splits.

    Args:
        filepath: Path to the CSV file.
        text_col: Name of column containing review text (default: 'review_text').
        label_col: Name of column containing sentiment label (default: 'sentiment').
        drop_duplicates: Whether to drop duplicate review texts (default: True).

    Returns:
        Cleaned pandas DataFrame with columns [text_col, label_col].

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is empty, missing required columns, has fewer than 2 classes,
                    or has no valid samples after cleaning.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at '{filepath}': {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset at '{filepath}' is completely empty.")

    # Validate required columns
    missing_cols = [col for col in (text_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only necessary columns and drop rows with null text or label
    df = df[[text_col, label_col]].copy()
    initial_count = len(df)
    df = df.dropna(subset=[text_col, label_col])

    # Convert text column to string and strip whitespace
    df[text_col] = df[text_col].astype(str).str.strip()

    # Filter out empty or whitespace-only reviews
    df = df[df[text_col].str.len() > 0]

    if df.empty:
        raise ValueError(
            f"No valid non-empty review rows found in '{filepath}' after dropping nulls/empty strings."
        )

    # Normalize labels
    try:
        df[label_col] = df[label_col].apply(normalize_label)
    except ValueError as exc:
        raise ValueError(f"Label normalization error in '{filepath}': {exc}") from exc

    # Optionally drop duplicates on review_text
    if drop_duplicates:
        df = df.drop_duplicates(subset=[text_col]).copy()

    # Verify both classes exist
    unique_classes = set(df[label_col].unique())
    if len(unique_classes) < 2:
        raise ValueError(
            f"Dataset must contain at least 2 classes ('positive' and 'negative') to train. "
            f"Found only: {unique_classes}"
        )

    return df.reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    text_col: str = "review_text",
    label_col: str = "sentiment",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train and test sets using stratified sampling.

    Stratified splitting ensures both the training and test sets have the
    exact same proportion of positive and negative reviews as the full dataset.

    Args:
        df: Cleaned pandas DataFrame.
        test_size: Fraction of samples to hold out for testing (e.g. 0.2 for 20%).
        random_state: Seed for reproducible random splitting.
        text_col: Name of review text column.
        label_col: Name of sentiment label column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df[text_col]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
