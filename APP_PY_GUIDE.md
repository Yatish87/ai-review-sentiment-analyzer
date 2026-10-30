# `app.py` — A Beginner-Friendly Guide

This guide explains what [`app.py`](./app.py) does, how its code is organized,
and how a user action travels through the application. It is written for readers
who are new to Streamlit or this project.

## 1. What is `app.py`?

`app.py` is the user-interface layer of the AI Review Sentiment Analyzer. It
builds the web page with Streamlit, collects review text or a CSV file, calls
the appropriate analysis functions, and displays the results.

It connects three separate capabilities:

1. **Gemini review classifier** — analyzes one review or a batch and returns
   positive, negative, mixed, or neutral sentiment with a written reason.
2. **Local machine-learning baseline** — uses the saved scikit-learn pipeline
   to predict positive or negative sentiment without sending review text to an
   API.
3. **Complaint theme synthesizer** — sends negative or mixed batch feedback to
   Gemini and asks it to summarize recurring complaints and investigation
   ideas.

`app.py` coordinates these features. The actual Gemini and scikit-learn logic
lives in other modules:

- [`src/genai_insights.py`](./src/genai_insights.py) handles Gemini requests,
  response validation, and complaint synthesis.
- [`src/predict.py`](./src/predict.py) loads the local model and performs
  single-review or DataFrame predictions.
- [`src/train_model.py`](./src/train_model.py) trains and saves the local
  model.
- [`src/data_utils.py`](./src/data_utils.py) contains data cleaning and
  negation preprocessing utilities.

## 2. A simple picture of the application

```text
User opens the Streamlit app
          |
          v
app.py builds the sidebar, page header, and three tabs
          |
          +--> Single Review tab
          |       +--> Gemini classifier OR local ML predictor
          |
          +--> Batch CSV tab
          |       +--> Gemini batch classifier OR local ML predictor
          |       +--> Optional Gemini complaint-theme synthesis
          |
          +--> Architecture tab
                  +--> Shows a comparison of both classifiers
```

Gemini requests are made only when the user clicks an analysis button. The
application does try to load the local ML model while rendering the sidebar,
but that operation is local and does not contact Gemini.

## 3. How Streamlit runs this file

Streamlit executes `app.py` from top to bottom to create the page. When a user
changes a widget or clicks a button, Streamlit runs the script again with the
updated widget values.

This explains why the file is organized as ordinary Python code containing
Streamlit widgets: the widgets describe what should appear on the page, and the
button conditions describe what work should happen on a rerun.

## 4. Imports and project setup

At the top, the file imports:

- **Python utilities:** `os`, `sys`, and `Path` for environment values and
  file paths.
- **Pandas:** used to read, transform, display, and export CSV data.
- **Streamlit:** used to create the web interface.
- **Project functions:** prediction helpers from `src.predict` and Gemini
  helpers from `src.genai_insights`.

`PROJECT_ROOT` is calculated from the location of `app.py`. The code adds that
directory to Python's import search path if necessary. This helps the imports
work when the app is launched from the repository.

Several imports are used later as follows:

| Imported item | Purpose in the app |
| --- | --- |
| `load_model` | Load a saved model; available to this file but not directly called in the current UI flow. |
| `predict_sentiment` | Predict one review using the local model. |
| `predict_batch_dataframe` | Predict reviews in a DataFrame using the local model. |
| `classify_review_with_llm` | Classify one review with Gemini. |
| `classify_batch_with_llm` | Classify a list of reviews with Gemini. |
| `analyze_negative_feedback` | Generate complaint themes with Gemini. |
| `get_api_key` | Find a usable key from an override or environment configuration. |

`Optional`, `List`, `Dict`, and `Any` are imported for Python type annotations.

## 5. Page setup and cached model

### Page configuration

`st.set_page_config(...)` sets the browser-tab title and icon, uses a wide page
layout, and opens the sidebar by default. Streamlit expects this configuration
to happen near the beginning of the file.

### `get_baseline_model`

The function decorated with `@st.cache_resource` loads the scikit-learn model
through `load_model()`. Streamlit caches the loaded object so the app does not
need to deserialize the model again on every rerun.

The cache is for the model resource, not for individual predictions. The
default artifact path comes from `src.predict.DEFAULT_MODEL_PATH`.

## 6. Sidebar: settings, key, and local model status

The `with st.sidebar:` block places its controls in the page sidebar.

### Gemini configuration

1. `get_api_key()` looks for a key already configured in the environment (the
   project supports `GEMINI_API_KEY` and `GOOGLE_API_KEY`).
2. `st.text_input(..., type="password")` offers an optional key override. The
   password field masks the characters while they are being entered.
3. If the user entered an override, `get_api_key()` checks it; otherwise the
   environment key is used.
4. `st.selectbox()` lets the user choose a Gemini model name.
5. A success or warning message shows whether a usable key is available.

The selected key is passed to the Gemini functions when the user explicitly
starts a request. `app.py` does not write it to a file.

### Local ML model status

The sidebar calls `get_baseline_model()` inside a `try` block. On success, the
app sets `ml_available` to `True` and keeps the loaded pipeline in
`ml_pipeline`. If loading raises an exception, the app displays the error and
suggests training the model.

Those two values are later used to prevent a local prediction from running
when the pipeline could not be loaded.

### Privacy notice

The sidebar explains that Gemini analysis sends review text to Google's API,
while the local baseline processes text on the machine. Users should select
the local model when they need to keep review text offline.

## 7. Main page header and tabs

The app displays its title, a short description, and a disclaimer imported
from `src.genai_insights`. The disclaimer reminds users that AI output is a
suggestion that should be verified.

`st.tabs()` creates three sections:

1. **Single Review Analysis**
2. **Batch CSV Analysis & Themes**
3. **Architecture & Engine Comparison**

Each `with tabs[index]:` block defines the contents of the corresponding tab.

## 8. Tab 1 — Analyze one review

### Example buttons and text area

The positive, negative, and mixed example buttons put example text into
`st.session_state["single_input"]`. Session state lets that value remain
available across Streamlit reruns.

The text area uses that saved value as its initial content. The user can
replace it with their own review.

### Choosing an engine

The radio control offers two choices:

- **Gemini:** four labels and a written explanation; needs a valid API key and
  an internet connection.
- **Classical ML:** two labels and class probabilities; intended to run
  locally without Gemini credentials.

### Clicking “Analyze Sentiment”

The button handler follows this decision sequence:

1. If the review is blank, show a warning and stop.
2. If Gemini was selected:
   - If there is no key, show an error and explain the local alternative.
   - Otherwise call `classify_review_with_llm()` with the trimmed review, key,
     and selected model.
   - If the result succeeds, display its sentiment and grounded reason.
   - If it fails, display the returned error and suggest the local baseline.
3. Otherwise, use the local model:
   - Check `ml_available` and `ml_pipeline`.
   - Call `predict_sentiment()` with the review and the already-loaded
     pipeline.
   - Show the binary sentiment, confidence, and positive/negative
     probabilities.
   - If an exception occurs during prediction, show its message in the UI.

The Gemini result intentionally does not display a numeric confidence value.
The local model shows probabilities returned by scikit-learn.

## 9. Tab 2 — Load and analyze a CSV

### Loading the data

The tab offers two ways to provide a DataFrame:

- **Built-in demo:** reads `data/sample_reviews.csv` using a path based on
  `PROJECT_ROOT`.
- **Upload:** accepts a CSV file and reads it with `pandas.read_csv()`.

The loaded DataFrame is stored in `st.session_state["batch_raw_df"]`. This
allows it to remain available when Streamlit reruns the script after widget
interactions. File-read errors are shown in the page.

The built-in button currently says “100 Reviews” in the UI; the actual current
CSV contains more than 100 rows. The label is descriptive UI text and does not
limit the number of rows loaded.

### Selecting the review column and engine

Once a non-empty DataFrame is loaded:

1. The app shows the number of rows.
2. It presents a dropdown containing the CSV's column names.
3. If `review_text` exists, that column is selected by default; otherwise the
   first column is used.
4. The user chooses Gemini or the local ML baseline.

### Gemini batch controls

When Gemini is selected, the app displays a privacy and cost warning and a
maximum-row slider. The slider defaults to at most 15 rows and cannot exceed
50 rows (or the number of rows in the DataFrame, if smaller).

On the run button:

1. `head(max_rows)` selects the rows to process.
2. The selected text column is converted to strings.
3. `classify_batch_with_llm()` is called with `chunk_size=10`.
4. Each result's sentiment and reason are added to the selected rows.
5. An `engine` column identifies the result as Gemini output.
6. The completed DataFrame is saved in
   `st.session_state["batch_results_df"]`.

The row limit controls the number of rows submitted; chunking controls how
those rows are grouped into requests inside the helper.

### Local ML batch controls

When the local model is selected, the app checks that the pipeline is
available, then calls `predict_batch_dataframe()`. That helper returns a copy
of the input DataFrame with prediction columns. The app adds an `engine`
column and saves the results in session state.

### Display, chart, and CSV download

If `batch_results_df` exists, the app:

- Counts each sentiment and shows metric cards.
- Draws a bar chart for positive, negative, mixed, and neutral counts.
- Displays the full results table.
- Creates a CSV download named `classified_reviews_output.csv`.

The chart includes all four sentiment names even when the selected local model
only produces positive and negative predictions. Missing categories display
with a count of zero.

## 10. Complaint Theme Synthesizer

This feature appears below the batch results. It is a separate Gemini request,
not part of either classifier's initial prediction.

The app filters the results to rows labeled `negative` or `mixed`, extracts
their review text, and shows how many reviews are eligible for synthesis.

- If there are no eligible reviews, the app displays an informational message.
- Otherwise, clicking **Extract Complaint Themes with Gemini** checks for an
  API key and calls `analyze_negative_feedback()`.
- A successful response is rendered as Markdown with its disclaimer.
- A failed response is shown as an error.

Because this step uses Gemini, selected review text is sent to Google's API.
The LLM helper applies its own limits and safety instructions; see
[`src/genai_insights.py`](./src/genai_insights.py) for those details.

## 11. Tab 3 — Compare the two approaches

The final tab displays a static Markdown table comparing the Gemini and local
ML options. It describes their approach, sentiment labels, reasoning,
execution location, expected cost/latency, offline use, and confidence style.

It also repeats important privacy points: external API calls are triggered by
user actions, and Gemini processing sends review text outside the local
machine. This section is explanatory UI; it does not run either model.

## 12. Important Streamlit state values

| State or variable | What it stores | Why it is used |
| --- | --- | --- |
| `st.session_state["single_input"]` | Example review text | Keeps a clicked example in the single-review text area. |
| `st.session_state["batch_raw_df"]` | Loaded input DataFrame | Keeps the selected CSV available across reruns. |
| `st.session_state["batch_results_df"]` | Classified output DataFrame | Keeps predictions visible after the run button reruns the app. |
| `active_api_key` | Key selected for the current run | Passed to Gemini helpers when invoked. |
| `ml_pipeline` | Cached scikit-learn pipeline | Reused for local predictions. |
| `ml_available` | Whether the model loaded successfully | Controls local model availability in the UI. |

## 13. Error handling in the UI

The app reports common user-facing problems directly:

- Empty single-review text produces a warning.
- Missing Gemini credentials produce an explanatory error.
- Gemini helper failures display the returned error.
- A missing or unreadable local model produces a sidebar error.
- Local prediction exceptions are shown in the single-review tab.
- CSV parsing errors are shown near the upload control.

This makes the app usable without requiring the user to inspect a terminal for
every routine input problem.

## 14. Current model compatibility note

The repository's saved `models/sentiment_pipeline.joblib` currently records
scikit-learn version `1.9.1`. The inspected environment uses scikit-learn
`1.7.2`; the artifact loads with a warning, but its `predict_proba()` call
fails there because the deserialized classifier lacks an attribute expected
by that version.

Since both local prediction flows use probabilities, the local baseline may
show as unavailable or fail in that environment. The sidebar reports load
errors, and single-review prediction reports prediction errors. See
[`models/sentiment_pipeline.md`](./models/sentiment_pipeline.md) for the model
details. Use a compatible scikit-learn version or retrain the artifact in the
target environment.

## 15. Running the application

From the repository root, install the dependencies listed in
[`requirements.txt`](./requirements.txt), configure a Gemini API key if you
want Gemini features, and start Streamlit:

```bash
streamlit run app.py
```

The local ML classifier does not require a Gemini key, but it does require a
compatible saved model artifact.

## 16. Quick code map

| Area in `app.py` | Responsibility |
| --- | --- |
| Imports and `PROJECT_ROOT` | Load dependencies and make project modules importable. |
| `st.set_page_config()` | Set the page title, icon, layout, and sidebar default. |
| `get_baseline_model()` | Load and cache the local scikit-learn pipeline. |
| `with st.sidebar:` | Collect Gemini settings and report model/key availability. |
| Main header and `st.tabs()` | Display the page introduction and organize the app. |
| `with tabs[0]:` | Collect and classify one review. |
| `with tabs[1]:` | Load, classify, display, and export batch CSV results; optionally synthesize complaint themes. |
| `with tabs[2]:` | Explain the differences between the two engines. |
