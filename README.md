# AI Review Sentiment Analyzer

A portfolio-quality hybrid sentiment intelligence platform combining **Large Language Models (LLMs)** and **Classical Machine Learning**.

---

## 1. System Architecture

The application provides a dual-engine architecture:

1. **🤖 Primary Classifier: Google Gemini LLM (`gemini-2.5-flash`)**
   - **Nuanced 4-Class Sentiment:** Classifies reviews into `positive`, `negative`, `mixed`, or `neutral`.
   - **Evidence-Based Reasoning:** Extracts short, grounded explanations directly from the review text.
   - **Understands Complex Context:** Captures negation, contrastive conjunctions (*"great screen, but terrible battery"*), sarcasm, and mixed feedback that challenge bag-of-words models.
   - **No Hallucinated Probabilities:** Avoids synthetic confidence percentages; confidence is grounded in text justification.

2. **⚙️ Optional Baseline: Scikit-Learn ML Pipeline (`TF-IDF + Logistic Regression`)**
   - **100% Local & Offline:** Runs on-device with zero external API calls or latency.
   - **Fast Binary Classification:** Predicts binary `positive` / `negative` sentiment with calibrated probability scores.
   - **Reproducible Benchmark:** Fitted strictly on a stratified training split without data leakage.

3. **🔍 Complaint Theme Synthesizer (Separate GenAI Feature)**
   - Operates on customer reviews identified as `negative` or `mixed`.
   - Synthesizes recurring complaint themes, severity ratings, verbatim quotes, and actionable investigation recommendations.

```text
ai-review-sentiment-analyzer/
├── .env.example              # Template for API credentials (GEMINI_API_KEY / GOOGLE_API_KEY)
├── .gitignore                # Git ignore rules (.venv, .env, pycache, archives, etc.)
├── pytest.ini                # Pytest configuration specifying test paths and ignore rules
├── requirements.txt          # Direct runtime Python dependencies
├── README.md                 # Complete system documentation
├── app.py                    # Streamlit web app with LLM Primary & ML Baseline modes
├── data/
│   ├── .gitkeep
│   └── sample_reviews.csv    # Benchmark dataset with 100 balanced reviews
├── models/
│   ├── .gitkeep
│   └── sentiment_pipeline.joblib  # Trained scikit-learn pipeline artifact
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading, validation, label normalization, stratified split
│   ├── train_model.py        # TF-IDF + Logistic Regression training and evaluation
│   ├── predict.py            # Baseline single and batch inference
│   └── genai_insights.py     # LLM primary classification & complaint theme synthesis
└── tests/
    ├── __init__.py
    ├── test_sentiment.py     # 21 unit & regression tests for ML pipeline
    └── test_genai_insights.py# 17 unit tests for LLM classification & prompt grounding
```

---

## 2. Structured Classification Schema

Each review classified by the Primary LLM engine yields the following structured JSON output:

```json
{
  "success": true,
  "sentiment": "positive",
  "reason": "Customer expressed satisfaction with the battery longevity and fast delivery.",
  "confidence": null,
  "model": "gemini-2.5-flash",
  "error": null
}
```

- `sentiment`: Exactly one of `"positive"`, `"negative"`, `"mixed"`, or `"neutral"`.
- `reason`: Grounded explanation citing specific aspects mentioned in the review.
- `confidence`: Intentionally set to `null` because LLMs generate token probabilities, not calibrated frequentist confidence intervals.

---

## 3. Environment Setup & Configuration

### Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- macOS, Linux, or Windows

### 1. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials (For Primary LLM Engine)
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Set your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   # Or: GOOGLE_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```
> [!IMPORTANT]
> The `.env` file is excluded in `.gitignore` and is never committed. You can also paste your key directly into the Streamlit sidebar during your active session.

---

## 4. Cost, Privacy, and Rate-Limit Controls

1. **Explicit Invocations Only:**
   - The application **never** triggers LLM API calls on startup or tab switching. Invocations happen solely upon clicking **"Analyze Sentiment"**, **"Run Batch Classification"**, or **"Extract Complaint Themes"**.
2. **Batch Row Limits & Cost Safety:**
   - For batch CSV processing, the app provides a row-limit slider (default: 15, max: 50) and displays a clear cost and privacy notice before sending reviews to the external API.
3. **Data Privacy Notice:**
   - Reviews sent to the LLM are transmitted via Google Gemini's API. Do not upload confidential or personally identifiable customer data without appropriate authorization. For sensitive data, use the **Classical ML Baseline**, which runs 100% in local memory.
4. **Order Preservation:**
   - Batch classification assigns unique integer IDs to each review, chunking requests into groups of 10 to protect context windows and match each prediction strictly to its original row.

---

## 5. Launching the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### How to Use the App:
- **💬 Tab 1: Single Review Prediction**:
  - Choose between **"🤖 Google Gemini LLM (Primary)"** and **"⚙️ Classical ML Baseline"**.
  - Type or paste a review, or select one of the quick sample buttons (Positive, Negative, or Mixed).
  - Click **"🔮 Analyze Sentiment"** to view the sentiment badge and grounded reason.
- **📊 Tab 2: Batch CSV Analysis**:
  - Upload any CSV or click **"📂 Load Built-in Demo Dataset (100 Reviews)"**.
  - Select your classification engine (LLM Primary or ML Baseline).
  - Adjust the maximum rows slider if using the LLM.
  - Click **"🚀 Run Batch Classification"** to generate predictions, distribution charts, and a downloadable CSV.
  - Scroll down to the **Complaint Theme Synthesizer** to cluster negative and mixed reviews into actionable themes.
- **📖 Tab 3: Architecture & Comparison**:
  - Detailed feature-by-feature matrix comparing LLM vs. ML approaches.

---

## 6. Offline Testing

Run the full automated test suite (38 tests total):
```bash
python -m pytest -v
```

All 17 LLM unit tests use `unittest.mock` to simulate provider responses, verify JSON schema parsing, test credential redaction, and ensure row-order matching **without making live API calls or requiring internet access**.

---

## 7. Retraining the ML Baseline Model

The classical ML pipeline artifact is stored at `models/sentiment_pipeline.joblib`. To retrain and re-evaluate on the 100-sample benchmark:
```bash
python src/train_model.py
```

### Baseline Performance on Held-Out Test Set (20 Samples)
- **Accuracy:** **90.00%**
- **Positive Precision:** **90.00%**
- **Positive Recall:** **90.00%**
- **Positive F1-Score:** **90.00%**

---

## 8. Limitations & Responsible Use

1. **Probabilistic Outputs:** LLMs and ML models are statistical systems. Always verify synthesized complaint themes against operational business metrics before making product changes.
2. **Confidence vs. Reasoning:** Classical ML provides calibrated numeric probabilities ($0.0$ to $1.0$). The LLM classifier provides textual reasoning grounded in the review text, which is more interpretable for humans than arbitrary self-reported confidence numbers.
