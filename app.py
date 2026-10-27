"""Streamlit Web Application: AI Review Sentiment Analyzer.

Hybrid Intelligence Architecture:
1. Primary Classifier: Google Gemini LLM (`gemini-2.5-flash`) providing nuanced 4-class sentiment
   ('positive', 'negative', 'mixed', 'neutral') with grounded explanations.
2. Baseline Classifier: Local scikit-learn ML pipeline (`TF-IDF + Logistic Regression`)
   for fast, local, offline binary classification with calibrated probability confidence.
3. Complaint Theme Synthesizer: Google Gemini extracts recurring root-cause complaint themes
   from negative & mixed customer feedback.
"""

from typing import Optional, List, Dict, Any
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_model, predict_sentiment, predict_batch_dataframe, DEFAULT_MODEL_PATH
from src.genai_insights import (
    classify_review_with_llm,
    classify_batch_with_llm,
    analyze_negative_feedback,
    get_api_key,
    DEFAULT_MODEL_NAME,
    DISCLAIMER_TEXT,
    MAX_REVIEWS_TO_ANALYZE,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Review Sentiment Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading baseline ML model...")
def get_baseline_model(path: str = DEFAULT_MODEL_PATH):
    """Load and cache the baseline ML pipeline once across user sessions."""
    return load_model(path)


# -----------------------------------------------------------------------------
# Sidebar: Engine Configuration & Credentials
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Classification Engines")

    # 1. Primary Engine: Google Gemini LLM
    st.subheader("1. Primary Engine: Google Gemini LLM")
    st.caption("Classifies sentiment into positive, negative, mixed, or neutral with evidence-based reasons.")

    env_api_key = get_api_key()
    user_api_key = st.text_input(
        "Gemini API Key (optional override)",
        value="",
        type="password",
        help="Reads from GEMINI_API_KEY or GOOGLE_API_KEY in .env by default. Keys are kept in session memory and never committed.",
    )
    active_api_key = get_api_key(user_api_key.strip()) if user_api_key.strip() else env_api_key

    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash", "gemini-3.8-flash"],
        index=0,
        help="Fast and grounded models. 'gemini-flash-lite-latest' provides the fastest response time.",
    )

    if active_api_key:
        st.success("🔑 Gemini API Key configured")
    else:
        st.warning("⚠️ No API Key found. You can set GEMINI_API_KEY in .env or paste it above. The ML Baseline remains 100% functional offline.")

    st.markdown("---")

    # 2. Baseline Engine: Scikit-Learn Model
    st.subheader("2. Baseline: Scikit-Learn Model")
    st.caption("Local TF-IDF + Logistic Regression pipeline running 100% offline.")
    ml_available = False
    ml_pipeline = None
    try:
        ml_pipeline = get_baseline_model(DEFAULT_MODEL_PATH)
        ml_available = True
        st.success("✅ ML Baseline Loaded (`TF-IDF + Logistic Regression`)")
    except Exception as exc:
        st.error(f"⚠️ Baseline Model Unavailable: {exc}")
        st.caption("Run `python src/train_model.py` in your terminal to train the baseline model.")

    st.markdown("---")
    st.caption(
        "**Privacy & Data Policy:** When using the LLM classifier, review text is transmitted "
        "to Google Gemini for inference. Do not submit confidential or identifying customer data unless authorized. "
        "The ML baseline processes data entirely in local memory."
    )


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.title("🔍 AI Review Sentiment Analyzer")
st.markdown(
    "A hybrid sentiment intelligence platform. Uses **Google Gemini LLM** as the primary classifier "
    "for nuanced sentiment and reasoning, with a **Classical Scikit-Learn ML Model** available as an offline baseline."
)

st.warning(
    f"ℹ️ **Notice:** {DISCLAIMER_TEXT} Model classifications are probabilistic suggestions and should be verified.",
    icon="⚠️",
)

tabs = st.tabs(["💬 Single Review Analysis", "📊 Batch CSV Analysis & Themes", "📖 Architecture & Engine Comparison"])


# -----------------------------------------------------------------------------
# Tab 1: Single Review Prediction
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Classify a Single Customer Review")

    col_btn1, col_btn2, col_btn3, _ = st.columns([1, 1, 1, 2])
    if col_btn1.button("💡 Positive Example"):
        st.session_state["single_input"] = "The battery life is outstanding and lasts over two days. Really good product!"
    if col_btn2.button("💡 Negative Example"):
        st.session_state["single_input"] = "Terrible build quality. The hinge snapped within 3 days and support refused refund."
    if col_btn3.button("💡 Mixed Example"):
        st.session_state["single_input"] = "The audio quality is crystal clear, but the ear cushions are very uncomfortable after an hour."

    current_input = st.session_state.get("single_input", "")
    user_review = st.text_area(
        "Customer Review Text",
        value=current_input,
        placeholder="Enter a customer review here...",
        height=110,
    )

    # Engine Selection for Single Review
    engine_choice = st.radio(
        "Select Classification Engine:",
        options=[
            "🤖 Google Gemini LLM (Primary — Positive / Negative / Mixed / Neutral + Reason)",
            "⚙️ Classical ML Baseline (Offline — Positive / Negative Binary + Confidence)",
        ],
        index=0,
        horizontal=True,
    )

    if st.button("🔮 Analyze Sentiment", type="primary"):
        if not user_review.strip():
            st.warning("Please enter review text before submitting.")
        elif "Gemini LLM" in engine_choice:
            if not active_api_key:
                st.error(
                    "Gemini API key is required for LLM classification. "
                    "Please enter your key in the sidebar or add `GEMINI_API_KEY` to your `.env` file. "
                    "Alternatively, select the 'Classical ML Baseline' to run locally without an API key."
                )
            else:
                with st.spinner("Classifying with Google Gemini..."):
                    result = classify_review_with_llm(
                        user_review.strip(),
                        api_key=active_api_key,
                        model_name=selected_model,
                    )

                    if result["success"]:
                        st.markdown("### 🤖 LLM Classification Result")
                        sentiment = result["sentiment"].upper()
                        reason = result["reason"]

                        m1, m2 = st.columns([1, 3])
                        with m1:
                            if sentiment == "POSITIVE":
                                st.metric("Predicted Sentiment", "POSITIVE 👍")
                            elif sentiment == "NEGATIVE":
                                st.metric("Predicted Sentiment", "NEGATIVE 👎")
                            elif sentiment == "MIXED":
                                st.metric("Predicted Sentiment", "MIXED ⚖️")
                            else:
                                st.metric("Predicted Sentiment", "NEUTRAL ⚪")

                        with m2:
                            st.markdown(f"**Grounded Explanation:**\n> {reason}")

                        st.caption("ℹ️ *Confidence: LLM classifications provide evidence-based reasoning rather than synthetic probability scores.*")
                    else:
                        st.error(f"LLM Classification Failed: {result['error']}")
                        st.info("You can switch to the Classical ML Baseline option above to process offline.")

        else:
            # Classical ML Baseline
            if not ml_available or ml_pipeline is None:
                st.error("Baseline model is not loaded. Please train it with `python src/train_model.py`.")
            else:
                try:
                    result = predict_sentiment(user_review.strip(), pipeline=ml_pipeline)
                    st.markdown("### ⚙️ Classical ML Baseline Result")
                    sentiment = result["sentiment"].upper()
                    conf = result["confidence"]
                    probs = result["probabilities"]

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Sentiment (Binary)", sentiment)
                    with m2:
                        st.metric("Model Confidence", f"{conf * 100:.2f}%")
                    with m3:
                        st.caption("Class Probabilities:")
                        st.progress(probs.get("positive", 0.5), text=f"Positive: {probs.get('positive', 0.0) * 100:.1f}%")
                        st.progress(probs.get("negative", 0.5), text=f"Negative: {probs.get('negative', 0.0) * 100:.1f}%")

                except Exception as e:
                    st.error(f"Baseline prediction failed: {e}")


# -----------------------------------------------------------------------------
# Tab 2: Batch CSV Analysis & Complaint Themes
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Batch Review Classification & Complaint Synthesis")

    col_csv1, col_csv2 = st.columns([1, 1])
    with col_csv1:
        load_sample_btn = st.button("📂 Load Built-in Demo Dataset (100 Reviews)")
    with col_csv2:
        uploaded_file = st.file_uploader("Or Upload CSV File", type=["csv"], help="Must contain a text column of customer reviews.")

    df_to_analyze: Optional[pd.DataFrame] = None

    if load_sample_btn:
        sample_path = PROJECT_ROOT / "data" / "sample_reviews.csv"
        if sample_path.exists():
            df_to_analyze = pd.read_csv(sample_path)
            st.session_state["batch_raw_df"] = df_to_analyze
            st.success("Loaded demo dataset (`data/sample_reviews.csv`).")
        else:
            st.error("Demo dataset not found at `data/sample_reviews.csv`.")
    elif uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.session_state["batch_raw_df"] = df_to_analyze
        except Exception as exc:
            st.error(f"Error reading CSV file: {exc}")

    if "batch_raw_df" in st.session_state and df_to_analyze is None:
        df_to_analyze = st.session_state["batch_raw_df"]

    if df_to_analyze is not None and not df_to_analyze.empty:
        st.markdown(f"**Loaded {len(df_to_analyze)} rows.**")

        columns = list(df_to_analyze.columns)
        default_idx = columns.index("review_text") if "review_text" in columns else 0
        text_column = st.selectbox("Select review text column:", options=columns, index=default_idx)

        st.markdown("#### Choose Batch Classification Engine")
        batch_engine = st.radio(
            "Classification Engine:",
            options=[
                "🤖 Google Gemini LLM (Primary — Nuanced 4-Class + Explanations)",
                "⚙️ Classical ML Baseline (Fast, Offline — Binary Polarity)",
            ],
            index=0,
            horizontal=True,
        )

        # Cost and safety controls for LLM batch
        max_rows = len(df_to_analyze)
        if "Gemini LLM" in batch_engine:
            st.warning(
                "🔒 **Privacy & Cost Notice:** Review texts will be transmitted in bounded batches to the Google Gemini API. "
                "Ensure no confidential or personal identifiable information (PII) is included without authorization. "
                "You can limit the number of rows below to conserve API quota.",
                icon="⚠️",
            )
            max_rows = st.slider(
                "Maximum rows to process with LLM:",
                min_value=1,
                max_value=min(len(df_to_analyze), 50),
                value=min(len(df_to_analyze), 15),
                help="Limits the batch size to control token usage and prevent rate limits.",
            )

        if st.button("🚀 Run Batch Classification", type="primary"):
            target_df = df_to_analyze.head(max_rows).copy()

            if "Gemini LLM" in batch_engine:
                if not active_api_key:
                    st.error(
                        "Gemini API key is required. Please set GEMINI_API_KEY in .env "
                        "or enter it in the sidebar. You can also select the Classical ML Baseline to process offline."
                    )
                else:
                    reviews_list = target_df[text_column].astype(str).tolist()
                    progress_bar = st.progress(0, text="Sending reviews to Google Gemini in bounded batches...")

                    with st.spinner(f"Classifying {len(reviews_list)} reviews with Gemini..."):
                        batch_results = classify_batch_with_llm(
                            reviews_list,
                            api_key=active_api_key,
                            model_name=selected_model,
                            chunk_size=10,
                        )
                        progress_bar.progress(100, text="Classification complete!")

                    target_df["predicted_sentiment"] = [r["sentiment"] for r in batch_results]
                    target_df["reason"] = [r["reason"] for r in batch_results]
                    target_df["engine"] = "Google Gemini LLM"
                    st.session_state["batch_results_df"] = target_df
            else:
                # ML Baseline
                if not ml_available or ml_pipeline is None:
                    st.error("Baseline ML pipeline is not available.")
                else:
                    with st.spinner("Processing reviews with local ML pipeline..."):
                        pred_df = predict_batch_dataframe(target_df, text_col=text_column, pipeline=ml_pipeline)
                        pred_df["engine"] = "Classical ML Baseline"
                        st.session_state["batch_results_df"] = pred_df

        # Render Batch Results Table & Visualizations
        if "batch_results_df" in st.session_state:
            results_df = st.session_state["batch_results_df"]
            st.markdown("---")
            st.subheader(f"Batch Classification Results ({results_df['engine'].iloc[0]})")

            total_classified = len(results_df)
            counts = results_df["predicted_sentiment"].value_counts().to_dict()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Positive", counts.get("positive", 0))
            c2.metric("Negative", counts.get("negative", 0))
            c3.metric("Mixed", counts.get("mixed", 0))
            c4.metric("Neutral", counts.get("neutral", 0))

            # Distribution Chart
            chart_data = pd.DataFrame(
                {"Count": [counts.get(s, 0) for s in ["positive", "negative", "mixed", "neutral"]]},
                index=["positive", "negative", "mixed", "neutral"],
            )
            st.bar_chart(chart_data, color="#4F46E5")

            # Data Table
            st.dataframe(results_df, use_container_width=True)

            csv_download = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified CSV",
                data=csv_download,
                file_name="classified_reviews_output.csv",
                mime="text/csv",
            )

            # -----------------------------------------------------------------
            # GenAI Complaint Theme Extraction (Separate Feature)
            # -----------------------------------------------------------------
            st.markdown("---")
            st.subheader("🤖 Complaint Theme Synthesizer")
            st.markdown(
                "Synthesizes reviews predicted as **negative** or **mixed** into root-cause complaint themes, "
                "supporting quotes, and business investigation areas."
            )

            negative_or_mixed_reviews = results_df[
                results_df["predicted_sentiment"].isin(["negative", "mixed"])
            ][text_column].dropna().astype(str).tolist()

            st.write(f"Found **{len(negative_or_mixed_reviews)} feedback items** eligible for complaint synthesis.")

            if len(negative_or_mixed_reviews) == 0:
                st.info("No negative or mixed reviews detected in this batch. Nothing to synthesize!")
            else:
                if st.button("✨ Extract Complaint Themes with Gemini", type="primary"):
                    if not active_api_key:
                        st.error("Gemini API key is required. Please set it in the sidebar or in your `.env` file.")
                    else:
                        with st.spinner("Extracting recurring complaint themes..."):
                            insights = analyze_negative_feedback(
                                negative_reviews=negative_or_mixed_reviews,
                                api_key=active_api_key,
                                model_name=selected_model,
                            )

                            if insights["success"]:
                                st.success("✅ Complaint Theme Synthesis Complete!")
                                st.markdown("### Executive Insights & Themes")
                                st.markdown(insights["content"])
                                st.info(insights["disclaimer"])
                            else:
                                st.error(f"GenAI Synthesis Error: {insights['error']}")


# -----------------------------------------------------------------------------
# Tab 3: Architecture & Engine Comparison
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Hybrid Engine Comparison & Architecture")
    st.markdown("""
    | Feature | 🤖 Primary Classifier (Google Gemini LLM) | ⚙️ Baseline Classifier (Classical ML) |
    | :--- | :--- | :--- |
    | **Approach** | Pre-trained Large Language Model (`gemini-2.5-flash`) | TF-IDF Vectorizer + Logistic Regression |
    | **Sentiment Classes** | **4 Classes:** `positive`, `negative`, `mixed`, `neutral` | **Binary:** `positive`, `negative` |
    | **Nuance & Reasoning** | Understands context, sarcasm, negations, and provides a textual reason | Evaluates n-gram word frequencies |
    | **Inference Location** | Hosted API (Google Gemini) | 100% Local (CPU / in-memory) |
    | **Cost & Latency** | Network call; subject to token usage & rate limits | Zero API cost; processes thousands of rows/sec |
    | **Offline Capability** | Requires internet and API key | Works 100% offline without credentials |
    | **Confidence Measure** | Evidence-based grounded reason (no fake probability) | Calibrated probability distribution from logistic regression |
    
    ### Privacy & Security
    - **No Automatic API Calls:** The application never contacts external APIs upon startup. Requests are only triggered after an explicit button click.
    - **API Keys:** Read from local environment variables or password-masked sidebar input. Keys are never saved to disk, logged, or displayed in logs.
    - **Data Boundaries:** Customer reviews submitted to Gemini leave the local machine for API processing. Use the ML baseline for strictly offline or regulated data environments.
    """)
