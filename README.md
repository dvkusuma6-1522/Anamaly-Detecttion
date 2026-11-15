# Social Media Trend Anomaly Detection

An unsupervised anomaly detection example for social media trend metrics. The repository contains a processed trend dataset, two Isolation Forest training entry points, a batch CLI, a Streamlit dashboard, a saved model, and an exploratory notebook.

> **Scope and data provenance:** The executable application uses `processed_trend_data.csv`. The notebook explores `raw_trend_data.csv.csv` and derives rolling features from `impressions`, but its paths and output artifacts do not match the root-level application files. No checked-in preprocessing script connects those datasets. Treat the notebook as exploratory material; do not assume it generated the current processed CSV or model.

## Contents

- [Pipeline and architecture](#pipeline-and-architecture)
- [Repository structure](#repository-structure)
- [Datasets and preprocessing](#datasets-and-preprocessing)
- [Model and labels](#model-and-labels)
- [Streamlit dashboard](#streamlit-dashboard)
- [Setup and operation](#setup-and-operation)
- [Troubleshooting and limitations](#troubleshooting-and-limitations)

## Pipeline and architecture

The app reads preprocessed trend records, fits an Isolation Forest, predicts on those same records, then either writes a batch CSV or presents selected records in the dashboard. The repository does not currently implement a streaming or online scoring service.

```text
raw_trend_data.csv.csv                 EDA_.ipynb
       |                                    |
       | (no repository preprocessing link) | timestamp cleaning,
       |                                    | rolling impression features
       |                                    v
       +----------------------------> exploratory notebook outputs

processed_trend_data.csv
       |
       +--> train_model.py ----------------> isolation_forest.pkl (5% contamination)
       |                                           |
       |                                           +--> dashboard.py
       |
       +--> app.py --> anomaly_detector.py --> model (1% contamination)
                                           |
                                           +--> anomaly_output.csv
```

The dashboard loads the existing root-level model; it does not train one on startup. The CLI calls the second trainer before scoring, so its model overwrites `isolation_forest.pkl` with a different contamination setting.

## Repository structure

| Path | Purpose |
|---|---|
| `EDA_.ipynb` | Exploratory notebook with data preview, cleaning, rolling/lag feature creation, plots, and a separate Isolation Forest experiment. Its input/output paths refer to files outside this repository layout. |
| `train_model.py` | Trains the dashboard-compatible model using four columns and saves `isolation_forest.pkl` beside the script. |
| `anomaly_detector.py` | Reusable CLI training and prediction functions. It uses the same four features, but its trainer uses 1% contamination and maps labels to `0`/`1`. |
| `app.py` | Batch entry point: reads the processed CSV, trains via `anomaly_detector.py`, predicts the same rows, and writes `anomaly_output.csv`. |
| `dashboard.py` | Authenticated Streamlit dashboard for interactive filtering, trend charts, anomaly summaries, and CSV export. |
| `requirements.txt` | Python package dependencies, including Plotly and bcrypt used by the dashboard. |
| `raw_trend_data.csv.csv` | Raw social post records (12,000 rows, 28 columns). |
| `processed_trend_data.csv` | Application input (10,698 rows, 11 columns). Its generating script is not present. |
| `anomaly_output.csv` | Existing CLI output (10,698 rows, 12 columns), with binary labels (`0` normal, `1` anomaly). Running `app.py` regenerates it. |
| `isolation_forest.pkl` | Serialized scikit-learn Isolation Forest loaded by the dashboard and CLI. It must be loaded with compatible Python and scikit-learn versions. |
| `README.md` | Project documentation. |
| `.gitignore` | Excludes local Python environments, bytecode, and common local artifacts. |
| `venv/` | Local virtual environment; create it locally and do not commit it. |

## Datasets and preprocessing

The figures below were measured from the CSV files in this repository. Column type labels are those inferred by pandas `read_csv`; CSV files do not store a formal type system.

### Raw source: `raw_trend_data.csv.csv`

- **Shape:** 12,000 rows × 28 columns.
- **Time range:** `timestamp` parses for all 12,000 rows, from `2024-05-01 01:04:04` through `2025-04-30 23:46:43`.
- **Missingness:** `mentions` has 3,941 missing values (32.84%); the remaining 27 columns have none. This is a valid optional field in the source and should not be blindly forward-filled.
- **Numeric ranges:** `sentiment_score` -0.9998 to 0.9999; `toxicity_score` 0 to 0.9999; `likes_count` 0–5,000; `shares_count` 0–2,000; `comments_count` 0–1,000; `impressions` 130–99,997; `engagement_rate` 0.00188–32.21171; `user_past_sentiment_avg` -0.9996–0.9994; `user_engagement_growth` -0.4999–0.4999; `buzz_change_rate` -99.9–99.9.

| Column(s) | pandas dtype | Description |
|---|---|---|
| `post_id`, `user_id` | string/object | Record and user identifiers |
| `timestamp` | string/object | Post timestamp; parse as datetime for time analysis |
| `day_of_week`, `platform`, `location`, `language` | string/object | Time and post context |
| `text_content`, `hashtags`, `mentions`, `keywords` | string/object | Text and social metadata; `mentions` is nullable |
| `topic_category`, `sentiment_label`, `emotion_type` | string/object | Categorized content attributes |
| `brand_name`, `product_name`, `campaign_name`, `campaign_phase` | string/object | Brand and campaign attributes |
| `sentiment_score`, `toxicity_score`, `engagement_rate`, `user_past_sentiment_avg`, `user_engagement_growth`, `buzz_change_rate` | float64 | Continuous measurements |
| `likes_count`, `shares_count`, `comments_count`, `impressions` | int64 | Count metrics |

### Application input: `processed_trend_data.csv`

- **Shape:** 10,698 × 11; `date` spans `2024-05-01` through `2025-04-30` (all parse as dates).
- **Missingness:** no missing values in any column.
- **Numeric ranges:** `post_count` 1–4; `likes` 0–16,235; `shares` 0–5,405; `comments` 0–2,338; `engagement_rate` 0.00232–32.21171; `rolling_mean` 0–2.3333; `rolling_std` 0–1.7321; `z_score` -1.1547–1.1547.

| Column | pandas dtype | Range / role |
|---|---|---|
| `date` | string/object | Calendar date; parsed and sorted by dashboard |
| `platform`, `hashtag` | string/object | Categories used by sidebar filters |
| `post_count` | int64 | 1–4 |
| `likes` | int64 | 0–16,235 |
| `shares` | int64 | 0–5,405 |
| `comments` | int64 | 0–2,338 |
| `engagement_rate` | float64 | 0.00232–32.21171 |
| `rolling_mean` | float64 | 0–2.3333 |
| `rolling_std` | float64 | 0–1.7321 |
| `z_score` | float64 | -1.1547–1.1547 |

The app assumes this schema already exists. There is no ingestion or preprocessing implementation for turning the 28-column raw source into this 11-column table. The processed table's `rolling_mean`, `rolling_std`, and `z_score` are present but **are not model inputs** in the current Python application.

### Existing output: `anomaly_output.csv`

- **Shape:** 10,698 × 12; same date range as the processed input.
- **Missingness:** none.
- **Schema:** the 11 processed columns above, plus `anomaly` (`int64`).
- **Observed label counts:** `0` normal = 10,591 (**98.999813%**); `1` anomaly = 107 (**1.000187%**).

These counts describe the artifact currently in the repository. They reflect `anomaly_detector.py`'s 1% contamination model and label remapping, not the standalone/dashboard trainer's configured 5% contamination.

### Exploratory notebook preprocessing and findings

The notebook's executed cells show a separate analysis path:

1. It reads `/content/raw_trend_data2.csv.csv` (not this repository's `raw_trend_data.csv.csv`). The saved notebook preview/output reports a 12,000 × 28 table and 3,941 missing `mentions` values.
2. It parses `timestamp` with invalid values coerced to missing, drops invalid timestamps, sorts chronologically, and forward-fills remaining missing values. The saved cell output does not report how many invalid timestamps were dropped. Forward-fill also fills the optional `mentions` gaps; this may propagate unrelated prior values and should be reviewed before using that notebook as a production preprocessing recipe.
3. It computes seven-row rolling mean and sample standard deviation of `impressions`, a one-row lag of `impressions`, and calendar `day`, `month`, and numeric weekday. It drops rows with missing values generated by the rolling and lag features.
4. It plots impressions over time, their distribution, rolling mean, and numerical correlations. The saved outputs show the plots but do not provide quantitative written conclusions.
5. It trains an Isolation Forest on `rolling_mean`, `rolling_std`, and `lag_1`, predicts `-1`/`1`, and plots anomaly timestamps. This differs from the app's four features and its processed dataset.
6. Its final cells write to `../data/processed_trend_data.csv` and `../models/isolation_forest.pkl`; they do not generate the root-level artifacts consumed by the application.

No scaling is applied by the application trainers. The notebook's rolling calculations are transformations, not feature standardization.

## Model and labels

### Method

Isolation Forest is an unsupervised method that isolates observations by repeatedly selecting a feature and a split point at random. A point separated after only a few splits has a shorter average path through the ensemble and is more likely to be unusual. In scikit-learn, `predict` returns `-1` for an outlier and `1` for an inlier. The `contamination` setting determines the expected outlier fraction used to set the decision threshold; it is not a measure of validated accuracy.

For the current application, each row is represented by:

```text
post_count, likes, shares, engagement_rate
```

These features are used as-is (no scaling). Isolation Forest random splits can be affected by differences in feature scales; evaluate preprocessing and thresholds on representative data before operational use.

### Configurations in the repository

| Entry point | `n_estimators` | `contamination` | `max_samples` | `max_features` | `bootstrap` | `random_state` | Label handling |
|---|---:|---:|---|---:|---|---:|---|
| `train_model.py` | 100 | 0.05 | `auto` (scikit-learn default) | 1.0 (default) | `False` (default) | 42 | Saves model; labels are assigned by dashboard as `-1` anomaly / `1` normal |
| `anomaly_detector.py:train_model` | 100 (scikit-learn default) | 0.01 | `auto` (default) | 1.0 (default) | `False` (default) | 42 | Saves model; detector remaps `-1` to `1` anomaly and `1` to `0` normal |
| Notebook experiment | 100 | 0.05 | `auto` (default) | 1.0 (default) | `False` (default) | 42 | Notebook adds readable `Anomaly` / `Normal` strings |

The standalone trainer and CLI trainer write to the same `isolation_forest.pkl` path with different contamination values. Run `python train_model.py` when the dashboard should use the 5% configuration; run `python app.py` when you want the CLI output, which retrains and replaces the model using 1%. The checked-in output has 107 anomalies, consistent with the CLI's 1% configuration.

Neither trainer nor the CLI saves `score_samples` or `decision_function` values. The output CSV stores only the remapped label. A larger `score_samples` value means more normal according to scikit-learn; `decision_function` is positive for inliers and negative for outliers. These scores can be computed from the loaded model if needed, but they are not currently part of the pipeline output.

Pickle/joblib artifacts can execute code when loaded. Only load `isolation_forest.pkl` from a trusted source. Rebuild the artifact in the target environment after material Python/scikit-learn version changes.

## Streamlit dashboard

Launch the app with `streamlit run dashboard.py`. It first displays a login form. The current code embeds username `admin` and password `1234`; it creates a bcrypt hash in memory on each process run and checks submitted passwords using `bcrypt.checkpw`.

> **Security:** These are development-only default credentials, hard-coded in source. Change them before exposing the dashboard to other users or networks. The current implementation has no account store, secret/environment-variable configuration, rate limiting, or production-grade access control.

After login:

- **Sidebar navigation:** `Overview`, `Anomalies`, or `Insights`.
- **Filters:** choose one platform and then a hashtag from that platform's rows. The selected records drive the KPIs, chart, and tables.
- **Overview:** data-point count, anomaly count, mean engagement rate, alert, trend line, and full filtered data table.
- **Anomalies:** detected anomalous rows, a heuristic explanation (`post_count`, `engagement_rate`, or `z_score` conditions), and a CSV download button.
- **Insights:** a simple message based on whether the filtered slice contains anomalies; it is a rule-based description, not generated or causal analysis.
- **Plot:** Plotly line chart of `post_count` over `date`, with anomaly markers.

The dashboard predicts directly using the serialized model's `-1`/`1` output. If a 1%-contamination model from `app.py` most recently replaced the artifact, dashboard anomaly counts will reflect that model.

## Setup and operation

Commands below are for Ubuntu/Linux from the repository root. Python 3.10+ is recommended; use the same environment for training and serving the serialized model.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt plotly bcrypt
```

`plotly` and `bcrypt` are already listed in `requirements.txt`; listing them on the install command makes the dashboard requirements explicit. The checked-in `.pkl` is loaded by default. To retrain it with the standalone 5% configuration:

```bash
python3 train_model.py
```

To run the batch pipeline (retrain at 1%, score the processed rows, and write/replace `anomaly_output.csv`):

```bash
python3 app.py
```

To run the dashboard (default local URL is usually `http://localhost:8501`):

```bash
streamlit run dashboard.py
```

Log in with `admin` / `1234` for the current development defaults. Keep the terminal running while using the app; stop it with `Ctrl+C`.

## Troubleshooting and limitations

- **Missing CSV/model:** scripts resolve their data and model files relative to their own directory. Ensure `processed_trend_data.csv` and `isolation_forest.pkl` are in the repository root.
- **Feature-count/schema error:** the model expects exactly `post_count`, `likes`, `shares`, and `engagement_rate` in the same order used for training. Retrain with `python train_model.py` after changing feature definitions.
- **Unexpected anomaly proportion:** `train_model.py` uses 5%, while `app.py` retrains with 1%. The latest trainer run determines the model in the shared pickle path.
- **Notebook file not found:** the notebook contains Colab-style absolute/parent-relative paths and expects different raw filenames. Update notebook paths and output directories before rerunning it locally.
- **Data lineage:** the repository does not include code that creates `processed_trend_data.csv` from the raw CSV. Reproducible operation requires documenting or implementing that preprocessing step.
- **Evaluation:** no labeled ground truth or validation metrics are included. Contamination is a threshold assumption, not evidence of model quality.
- **Batch behavior:** `app.py` scores the same rows it trains on. There is no held-out evaluation, persisted score output, API, scheduled job, or live stream consumer.
