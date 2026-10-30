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
├── APP_PY_GUIDE.md           # Beginner-friendly explanation of app.py
├── app.py                    # Streamlit web app with LLM Primary & ML Baseline modes
├── data/
│   ├── .gitkeep
│   └── sample_reviews.csv    # 2,186 balanced reviews (UCI Amazon + UCI Yelp + linguistic benchmarks)
├── models/
│   ├── .gitkeep
│   ├── sentiment_pipeline.md     # Readable summary of the saved baseline model
│   └── sentiment_pipeline.joblib  # Trained scikit-learn pipeline artifact
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading, validation, Potts/Pang&Lee negation tagging, stratified split
│   ├── train_model.py        # TF-IDF + Logistic Regression training with 5-fold CV tuning
│   ├── predict.py            # Baseline single and batch inference with calibrated probabilities
│   └── genai_insights.py     # LLM primary classification & complaint theme synthesis
└── tests/
    ├── __init__.py
    ├── test_sentiment.py     # 27 unit & regression tests for ML pipeline, negation, & generalization
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

For a step-by-step explanation of the interface code and how its features work,
see [`APP_PY_GUIDE.md`](./APP_PY_GUIDE.md).

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

Run the full automated test suite (44 tests total):
```bash
python -m pytest -v
```

- **27 ML Pipeline, Negation, & Generalization Tests:** Validates data normalization, Stratified splitting, `mark_negation` scope handling, TF-IDF n-gram tokenization, sentiment stop word retention, probability calibration, class-order alignment, and benchmark generalization phrases.
- **17 LLM Primary Classifier Tests:** Uses `unittest.mock` to simulate provider responses, verify JSON schema parsing, test credential redaction, and ensure row-order matching **without making live API calls or requiring internet access**.

---

## 7. Classical ML Baseline Model & Retraining

The classical ML pipeline artifact is stored at `models/sentiment_pipeline.joblib`.
For a readable description of its saved configuration and version compatibility, see [`models/sentiment_pipeline.md`](./models/sentiment_pipeline.md).

### Architecture & Preprocessing Enhancements
- **Pang & Lee / Potts Negation Marking (`mark_negation`):** Standard n-gram bag-of-words models suffer when negation tokens ("not", "never", "no", "didn't") appear with strong sentiment words. In naive n-grams, `"not bad"` inherits the strong negative weight of `"bad"` plus the negative weight of `"not"`, forcing it to be classified as overwhelmingly negative. The pipeline integrates `mark_negation`, transforming negated scopes into prefixed tokens (e.g., `not not_bad`). This allows the model to learn a positive weight for `not_bad` without fighting against `bad`.
- **Preserved Sentiment Vocabulary:** Retains critical sentiment-shifting tokens (`not`, `no`, `never`, `without`, `better`, `best`, `good`, `bad`, `very`, `much`) in the stop-word filtering stage.
- **N-Gram Range `(1, 2)` & Sublinear TF:** Extracts single words alongside word pairs and applies $1 + \log(\text{tf})$ scaling.
- **Stratified Hyperparameter Tuning:** Regularization parameter $C$ is tuned strictly on the training partition ($X_{\text{train}}, y_{\text{train}}$) via 5-fold Stratified Cross-Validation (`GridSearchCV`). The held-out test set is never touched during tuning.

To retrain and re-evaluate on the 2,186-sample benchmark:
```bash
python src/train_model.py
```

### Genuine Held-Out Evaluation Metrics (430 Real Test Samples)
Evaluated strictly on held-out test data (20% split, 100% real consumer reviews from the UCI Amazon & Yelp benchmarks, never seen during training or CV tuning):

| Metric | Baseline Before Fix (20 test samples) | Intermediate Model (33 test samples) | Generalized Model (430 Real Test Samples) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 90.00% | 84.85% | **86.98%** |
| **Positive Precision** | 90.00% | 86.67% | **86.30%** |
| **Positive Recall** | 90.00% | 81.25% | **87.91%** |
| **Positive F1-Score** | 90.00% | 83.87% | **87.10%** |
| **Brier Score Loss** (lower is better) | 0.2049 | 0.1502 | **0.1155** |
| **Log Loss** (lower is better) | 0.6022 | 0.4788 | **0.3788** |

Confusion Matrix on Held-Out Test Set (430 Real Consumer Reviews):
```text
                  Predicted Negative   Predicted Positive
Actual Negative          185                   30
Actual Positive           26                  189
```

### Benchmark Evaluation & Generalization Phrases (Real Model Output)
Verified real predictions without hard-coded phrases or artificial threshold shifts:

| Evaluation Phrase | Expected Sentiment | Baseline Before Fix | Generalized Model Output |
| :--- | :---: | :--- | :--- |
| **`"decent product"`** | **Positive** | `positive` (63.1%) | **`positive` (75.1%)** |
| **`"much better than before"`** | **Positive** | ❌ `negative` (53.0%) | **`positive` (58.0%)** |
| **`"not bad"`** | **Positive** | ❌ `negative` (96.2%) | **`positive` (63.9%)** |
| **`"average quality"`** | **Negative** | `positive` (50.3% coin flip) | **`negative` (79.1%)** |
| **`"excellent product"`** | **Positive** | `positive` (80.2%) | **`positive` (90.4%)** |
| **`"not as good as expected"`** | **Negative** | `negative` (66.2%) | **`negative` (81.2%)** |

### Target Sentiment Expressions Generalization

| Expression | Expected Sentiment | Baseline Before Fix | Generalized Model Output |
| :--- | :---: | :--- | :--- |
| **`"decent"`** | **Positive** | ❌ `negative` (50.1% coin flip) | **`positive` (84.3%)** |
| **`"better"`** | **Contextual** | `negative` (50.4% coin flip) | **`negative` (65.3%)** (comparative) |
| **`"excellent"`** | **Positive** | `positive` (79.4%) | **`positive` (97.0%)** |
| **`"satisfactory"`** | **Positive** | ❌ `negative` (50.1% coin flip) | **`positive` (80.1%)** |
| **`"amazing"`** | **Positive** | ❌ `negative` (50.1% coin flip) | **`positive` (91.0%)** |
| **`"disappointing"`** | **Negative** | `negative` (55.1%) | **`negative` (91.6%)** |
| **`"mediocre"`** | **Negative** | ❌ `negative` (50.1% coin flip) | **`negative` (82.6%)** |
| **`"wonderful"`** | **Positive** | `positive` (64.6%) | **`positive` (83.6%)** |
| **`"poor"`** | **Negative** | `negative` (81.9%) | **`negative` (96.8%)** |
| **`"not bad"`** | **Positive** | ❌ `negative` (96.2%) | **`positive` (63.9%)** |
| **`"could be better"`** | **Negative** | ❌ `negative` (50.4% coin flip) | **`negative` (65.3%)** |

### Dataset Source & Provenance
- **UCI Sentiment Labelled Sentences Dataset (Kotzias et al., 2015, KDD):** Contains 1,000 real customer reviews from Amazon Product Reviews and 1,000 real customer reviews from Yelp, manually labeled for binary sentiment. Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences).
- **Supplementary Linguistic Patterns:** 38 balanced training samples (~1.7% of total dataset) incorporating short reviews, comparative expressions, and mixed sentiments. All supplementary examples are tagged with `source: supplementary_linguistic` and are strictly excluded from the held-out test partition.

---

## 8. Limitations & Responsible Use

1. **Probabilistic Outputs:** LLMs and ML models are statistical systems. Always verify synthesized complaint themes against operational business metrics before making product changes.
2. **Confidence vs. Reasoning:** Classical ML provides calibrated numeric probabilities ($0.0$ to $1.0$). The LLM classifier provides textual reasoning grounded in the review text, which is more interpretable for humans than arbitrary self-reported confidence numbers.
