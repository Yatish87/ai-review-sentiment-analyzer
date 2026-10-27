"""Unit tests for the Generative AI insights and primary sentiment classification module.

All tests in this module use mock objects and do NOT make external network calls
or consume paid API credits.
"""

from unittest.mock import MagicMock
import json
import pytest

from src.genai_insights import (
    format_reviews_for_prompt,
    build_system_instruction,
    build_user_prompt,
    analyze_negative_feedback,
    get_api_key,
    create_genai_client,
    build_generation_config,
    classify_review_with_llm,
    classify_batch_with_llm,
    MAX_REVIEWS_TO_ANALYZE,
    DISCLAIMER_TEXT,
)


class TestGenAIPrimaryClassifier:
    """Test suite for LLM primary sentiment classification (single and batch)."""

    def test_classify_positive_review(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": "positive",
            "reason": "Customer praised the product performance and build quality."
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm("good product", client=mock_client)
        assert res["success"] is True
        assert res["sentiment"] == "positive"
        assert "praised" in res["reason"]
        assert res["confidence"] is None  # Confidence is explicitly None
        assert res["error"] is None

    def test_classify_negative_review(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": "negative",
            "reason": "Customer reported unit broke immediately."
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm("terrible product; broke immediately", client=mock_client)
        assert res["success"] is True
        assert res["sentiment"] == "negative"
        assert "broke" in res["reason"]
        assert res["confidence"] is None

    def test_classify_mixed_review(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": "mixed",
            "reason": "Expressed positive praise for battery but severe negative feedback for broken screen."
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm(
            "The battery is excellent, but the screen broke after one day",
            client=mock_client,
        )
        assert res["success"] is True
        assert res["sentiment"] == "mixed"
        assert "screen" in res["reason"]

    def test_classify_neutral_review(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": "neutral",
            "reason": "Factual statement regarding delivery day and packaging without sentiment polarity."
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm(
            "The package arrived on Tuesday in a brown cardboard box.",
            client=mock_client,
        )
        assert res["success"] is True
        assert res["sentiment"] == "neutral"

    def test_classify_empty_and_whitespace_input(self):
        res1 = classify_review_with_llm("")
        assert res1["success"] is False
        assert "empty or contains only whitespace" in res1["error"]

        res2 = classify_review_with_llm("    \t\n   ")
        assert res2["success"] is False
        assert "empty or contains only whitespace" in res2["error"]

        res3 = classify_review_with_llm(None)  # type: ignore
        assert res3["success"] is False
        assert "must be a non-empty string" in res3["error"]

    def test_classify_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        res = classify_review_with_llm("good product", client=None)
        assert res["success"] is False
        assert "Gemini API key not found" in res["error"]

    def test_classify_invalid_json_from_provider(self):
        mock_response = MagicMock()
        mock_response.text = "This is not valid JSON at all."
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm("good product", client=mock_client)
        assert res["success"] is False
        assert "Failed to parse structured JSON" in res["error"]

    def test_classify_unsupported_sentiment_label(self):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "sentiment": "super_fantastic",
            "reason": "Unsupported label"
        })
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        res = classify_review_with_llm("good product", client=mock_client)
        assert res["success"] is False
        assert "unsupported sentiment label" in res["error"]

    def test_classify_provider_exception_and_sanitizes_credentials(self):
        secret_key = "AIzaSySecretFakeKey123456"
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError(
            f"Rate limit exceeded contacting Gemini with key {secret_key}"
        )

        res = classify_review_with_llm("good product", api_key=secret_key, client=mock_client)
        assert res["success"] is False
        assert secret_key not in res["error"]
        assert "[REDACTED_API_KEY]" in res["error"]

    def test_classify_batch_preserves_order_and_row_mapping(self):
        reviews = [
            "good product",                                              # idx 0
            "terrible product; broke immediately",                       # idx 1
            "The battery is excellent, but the screen broke after one day",# idx 2
        ]

        # Mock batch response matching ids
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"id": 0, "sentiment": "positive", "reason": "Praise for product."},
            {"id": 1, "sentiment": "negative", "reason": "Broke immediately."},
            {"id": 2, "sentiment": "mixed", "reason": "Good battery, broken screen."},
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        batch_results = classify_batch_with_llm(reviews, client=mock_client)
        assert len(batch_results) == 3

        # Row 0
        assert batch_results[0]["review"] == reviews[0]
        assert batch_results[0]["sentiment"] == "positive"
        assert batch_results[0]["success"] is True

        # Row 1
        assert batch_results[1]["review"] == reviews[1]
        assert batch_results[1]["sentiment"] == "negative"
        assert batch_results[1]["success"] is True

        # Row 2
        assert batch_results[2]["review"] == reviews[2]
        assert batch_results[2]["sentiment"] == "mixed"
        assert batch_results[2]["success"] is True

    def test_classify_batch_handles_blank_rows_safely(self):
        reviews = ["good product", "", "terrible product"]
        mock_response = MagicMock()
        mock_response.text = json.dumps([
            {"id": 0, "sentiment": "positive", "reason": "Good."},
            {"id": 2, "sentiment": "negative", "reason": "Terrible."},
        ])
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        batch_results = classify_batch_with_llm(reviews, client=mock_client)
        assert len(batch_results) == 3
        assert batch_results[0]["sentiment"] == "positive"
        assert batch_results[1]["sentiment"] == "unclassified"
        assert "Empty" in batch_results[1]["reason"]
        assert batch_results[2]["sentiment"] == "negative"


class TestGenAIComplaintSummarizer:
    """Test suite for complaint theme summarization."""

    def test_format_reviews_bounds_and_truncates(self):
        reviews = [f"Complaint number {i} " + ("x" * 600) for i in range(50)]
        formatted = format_reviews_for_prompt(reviews, max_count=MAX_REVIEWS_TO_ANALYZE)
        assert f"Review {MAX_REVIEWS_TO_ANALYZE}:" in formatted
        assert f"Review {MAX_REVIEWS_TO_ANALYZE + 1}:" not in formatted
        assert "..." in formatted

    def test_build_system_instruction_contains_strict_rules(self):
        instruction = build_system_instruction()
        assert "STRICT GROUNDING RULES" in instruction
        assert "Do NOT invent" in instruction

    def test_build_user_prompt_structure(self):
        prompt = build_user_prompt("Review 1: broken", 1)
        assert "Executive Summary" in prompt
        assert "Top Recurring Complaint Themes" in prompt

    def test_get_api_key_resolution_order(self, monkeypatch):
        assert get_api_key("direct_key") == "direct_key"
        monkeypatch.setenv("GEMINI_API_KEY", "gemini_env_key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert get_api_key() == "gemini_env_key"

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "google_env_key")
        assert get_api_key() == "google_env_key"

    def test_placeholder_api_key_filtered_out(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert get_api_key() is None
        assert get_api_key("your_gemini_api_key_here") is None

    def test_analyze_empty_reviews_returns_clean_error(self):
        res = analyze_negative_feedback([])
        assert res["success"] is False
        assert "No negative reviews provided" in res["error"]

    def test_analyze_with_mocked_client_success(self):
        mock_response = MagicMock()
        mock_response.text = (
            "### 1. Executive Summary\n"
            "Hardware build failures.\n\n"
            "### 2. Top Recurring Complaint Themes\n"
            "- **Theme Title**: Hinge Fragility\n"
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        complaints = ["The hinge snapped within three days."]
        result = analyze_negative_feedback(complaints, client=mock_client)
        assert result["success"] is True
        assert "Hinge Fragility" in result["content"]
