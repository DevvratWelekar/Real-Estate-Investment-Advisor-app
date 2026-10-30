# Real Estate Investment Advisor

A Streamlit application for exploring Indian housing listings, estimating a
market-value-based investment proxy, and comparing five-year price scenarios.

## What the outputs mean

- **Investment classification:** a transparent proxy label. A listing is
  positive when its recalculated price per square foot is at or below the
  training median for its city and property type. The Logistic Regression
  model estimates this label; it does not predict realized investment returns.
  Classification metrics measure agreement with this constructed proxy only.
- **Five-year price:** the CSV has no historical future sale-price target.
  Training therefore uses the brief's explicit 8% annual appreciation
  assumption: `current price × (1 + rate)^5`. The regressor learns this
  scenario-derived target. The app can rescale it for an alternative annual
  appreciation scenario. It is not a market guarantee.
- The source data does not contain crime, RERA, rent, transaction costs,
  realized resale prices, or infrastructure scores. The app does not infer
  those unavailable facts.

All monetary values are in Indian rupees **lakhs** unless otherwise stated.
Price per square foot is recalculated from price and area, rather than trusting
the rounded source column.

## Run locally

Use Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-training.txt
python train.py
.\run.bat
```

The launcher starts the Streamlit app and opens it in Google Chrome at
`http://localhost:8501`. If the app is already running at that address, it opens
the existing app instead.

By default, training reads `data/raw/india_housing_prices.csv`, makes a reproducible
80/20 train/test split, saves the cleaned CSV under `data/processed/`, and
stores model and evaluation files under `models/`. It logs both model runs and
registers both models in the local MLflow SQLite registry (`mlflow.db`).
Set `MLFLOW_TRACKING_URI` to use a different tracking server.

To use another CSV for training:

```powershell
python train.py --data "C:\path\to\india_housing_prices.csv"
```

The dashboard's sidebar can also load a CSV for exploration. Train models
against that CSV before using its predictions.

## Included

- `app.py` — Streamlit investment form, filters, charts, cleaned-data download,
  evaluation metrics, and coefficient-based feature explanation.
- `train.py` — cleaning, reproducible model training/evaluation, local MLflow
  tracking/model registration, and deliverable exports.
- `src/pipeline.py` — shared data validation, engineered features, benchmark
  logic, and scikit-learn pipelines.
- `data/raw/india_housing_prices.csv` — supplied source dataset.

## Streamlit Community Cloud

Push this project to a GitHub repository, including `data/raw/` and the four
trained files in `models/` (`classifier.joblib`, `regressor.joblib`,
`benchmarks.json`, and `training_metadata.json`). In Streamlit Community Cloud,
create an app for that repository with `app.py` as the entry point. The Cloud
environment installs `requirements.txt`; MLflow is only needed for training
and is kept in `requirements-training.txt`. Model and dataset files must be
available in the deployed repository because the app does not train models at
startup.
