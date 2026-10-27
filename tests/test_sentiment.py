"""Unit and integration tests for the ML sentiment analysis pipeline.

Covers:
- Label normalization and data cleaning edge cases
- File and schema validation
- Stratified splitting
- Pipeline training, evaluation, and artifact serialization
- Single and batch inference
- Positive/negative prediction accuracy on clear benchmarks
- Error handling (empty inputs, missing files, corrupted schemas)
"""

import os
import tempfile
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_utils import normalize_label, load_and_validate_data, split_data
from src.train_model import build_pipeline, evaluate_model, train_and_save_model
from src.predict import load_model, predict_sentiment, predict_batch_dataframe


class TestDataUtils:
    """Tests for dataset loading, normalization, and validation."""

    def test_normalize_label_positive_variants(self):
        assert normalize_label("positive") == "positive"
        assert normalize_label("Positive") == "positive"
        assert normalize_label(" POS ") == "positive"
        assert normalize_label("1") == "positive"
        assert normalize_label(1) == "positive"

    def test_normalize_label_negative_variants(self):
        assert normalize_label("negative") == "negative"
        assert normalize_label("Negative") == "negative"
        assert normalize_label(" NEG ") == "negative"
        assert normalize_label("0") == "negative"
        assert normalize_label(0) == "negative"

    def test_normalize_label_invalid_raises_error(self):
        with pytest.raises(ValueError, match="Unrecognized sentiment label"):
            normalize_label("neutral")
        with pytest.raises(ValueError, match="Unrecognized sentiment label"):
            normalize_label("maybe")
        with pytest.raises(ValueError, match="null or missing label"):
            normalize_label(None)

    def test_load_sample_dataset_success(self):
        df = load_and_validate_data("data/sample_reviews.csv")
        assert not df.empty
        assert "review_text" in df.columns
        assert "sentiment" in df.columns
        assert set(df["sentiment"].unique()) == {"positive", "negative"}

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate_data("data/does_not_exist.csv")

    def test_missing_required_column_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("wrong_col,another_col\ntext1,positive\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="missing required column"):
                load_and_validate_data(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_single_class_only_raises_error(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("review_text,sentiment\nGreat product!,positive\nSuper loved it!,positive\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain at least 2 classes"):
                load_and_validate_data(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cleaning_drops_empty_and_duplicates(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "review_text,sentiment\n"
                "Good quality,positive\n"
                "Good quality,positive\n"  # duplicate
                "   ,positive\n"            # blank text
                ",negative\n"              # missing text
                "Broken,negative\n"
            )
            temp_path = f.name

        try:
            df = load_and_validate_data(temp_path, drop_duplicates=True)
            assert len(df) == 2
            assert set(df["sentiment"]) == {"positive", "negative"}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_stratified_split(self):
        df = load_and_validate_data("data/sample_reviews.csv")
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_train) + len(X_test) == len(df)

        # Both classes must be present in both train and test splits
        assert set(y_train) == {"positive", "negative"}
        assert set(y_test) == {"positive", "negative"}


@pytest.fixture(scope="module")
def trained_temp_model():
    """Train a temporary model to verify pipeline end-to-end."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        temp_model_path = f.name

    metrics = train_and_save_model(
        data_path="data/sample_reviews.csv",
        model_output_path=temp_model_path,
        test_size=0.2,
        random_state=42,
    )

    yield temp_model_path, metrics

    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)


class TestTrainingAndInference:
    """Tests for model training, artifact persistence, and predictions."""

    def test_model_artifact_created_and_loadable(self, trained_temp_model):
        model_path, metrics = trained_temp_model
        assert os.path.exists(model_path)
        assert metrics["accuracy"] >= 0.5  # better than random guess

        pipeline = load_model(model_path)
        assert isinstance(pipeline, Pipeline)
        assert hasattr(pipeline, "predict")

    def test_prediction_on_clear_positive(self, trained_temp_model):
        model_path, _ = trained_temp_model
        result = predict_sentiment(
            "The battery life is outstanding and lasts over two days. Love it!",
            model_path=model_path,
        )
        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.5
        assert "positive" in result["probabilities"]
        assert "negative" in result["probabilities"]
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3

    def test_prediction_on_clear_negative(self, trained_temp_model):
        model_path, _ = trained_temp_model
        result = predict_sentiment(
            "Terrible build quality. The hinge snapped within three days of normal use.",
            model_path=model_path,
        )
        assert result["sentiment"] == "negative"
        assert result["confidence"] > 0.5

    def test_prediction_on_list_of_reviews(self, trained_temp_model):
        model_path, _ = trained_temp_model
        reviews = [
            "Super fast shipping and excellent customer service!",
            "Customer support was rude, unhelpful, and refused replacement.",
        ]
        results = predict_sentiment(reviews, model_path=model_path)
        assert len(results) == 2
        assert results[0]["sentiment"] == "positive"
        assert results[1]["sentiment"] == "negative"

    def test_predict_validation_errors(self, trained_temp_model):
        model_path, _ = trained_temp_model
        # Empty string
        with pytest.raises(ValueError, match="empty or contains only whitespace"):
            predict_sentiment("   ", model_path=model_path)

        # Empty list
        with pytest.raises(ValueError, match="Cannot predict on an empty list"):
            predict_sentiment([], model_path=model_path)

        # Invalid type
        with pytest.raises(ValueError, match="Input texts must be a string or list"):
            predict_sentiment(12345, model_path=model_path)  # type: ignore

    def test_missing_model_file_raises_error(self):
        with pytest.raises(FileNotFoundError, match="Trained model not found"):
            load_model("models/nonexistent_model.joblib")

    def test_predict_batch_dataframe(self, trained_temp_model):
        model_path, _ = trained_temp_model
        df = pd.DataFrame(
            {
                "review_text": [
                    "High quality materials, exceeded expectations.",
                    "The package was completely damaged and the item inside was useless.",
                    "",  # blank edge case
                ],
                "product_id": [101, 102, 103],
            }
        )
        pred_df = predict_batch_dataframe(df, text_col="review_text", model_path=model_path)
        assert "predicted_sentiment" in pred_df.columns
        assert "confidence" in pred_df.columns
        assert pred_df.loc[0, "predicted_sentiment"] == "positive"
        assert pred_df.loc[1, "predicted_sentiment"] == "negative"
        assert pred_df.loc[2, "predicted_sentiment"] == "unclassifiable"

    def test_regression_target_phrases(self, trained_temp_model):
        """Verify the 5 benchmark target phrases evaluate to their expected sentiment."""
        model_path, _ = trained_temp_model

        test_cases = [
            ("good product", "positive"),
            ("This product is good", "positive"),
            ("excellent quality and fast delivery", "positive"),
            ("terrible product", "negative"),
            ("This product is bad and broke immediately", "negative"),
        ]

        for text, expected_sentiment in test_cases:
            res = predict_sentiment(text, model_path=model_path)
            assert (
                res["sentiment"] == expected_sentiment
            ), f"Failed for '{text}': got {res['sentiment']} (conf {res['confidence']}), expected {expected_sentiment}"

    def test_confidence_belongs_to_predicted_class(self, trained_temp_model):
        """Verify confidence strictly corresponds to the probability of the predicted class."""
        model_path, _ = trained_temp_model

        for phrase in ["good product", "terrible product"]:
            res = predict_sentiment(phrase, model_path=model_path)
            predicted_class = res["sentiment"]
            expected_conf = res["probabilities"][predicted_class]
            assert abs(res["confidence"] - expected_conf) < 1e-4

    def test_probabilities_map_to_classes_order(self, trained_temp_model):
        """Verify probabilities are dynamically mapped using classes_, not hard-coded indexes."""
        model_path, _ = trained_temp_model
        pipeline = load_model(model_path)

        res = predict_sentiment("good product", pipeline=pipeline)
        assert set(res["probabilities"].keys()) == set(pipeline.classes_)
        assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-3

    def test_saved_model_reloads_deterministically(self):
        """Verify the saved model file loads cleanly and yields deterministic predictions."""
        pipeline_a = load_model("models/sentiment_pipeline.joblib")
        pipeline_b = load_model("models/sentiment_pipeline.joblib")

        res_a = predict_sentiment("good product", pipeline=pipeline_a)
        res_b = predict_sentiment("good product", pipeline=pipeline_b)

        assert res_a["sentiment"] == res_b["sentiment"] == "positive"
        assert res_a["confidence"] == res_b["confidence"]

    def test_ml_predictions_work_without_llm_key(self, monkeypatch):
        """Verify that ML predictions run completely offline without an LLM API key."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        res = predict_sentiment("good product", model_path="models/sentiment_pipeline.joblib")
        assert res["sentiment"] == "positive"
        assert res["confidence"] > 0.5
