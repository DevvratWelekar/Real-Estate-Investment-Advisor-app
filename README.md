# Real Estate Investment Advisor

A Python Streamlit app for exploring Indian housing listings and estimating
property values.

## Run on Windows

Install Python 3.10 or newer. From this folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
.\run.bat
```

`run.bat` starts the app and opens it in Google Chrome at
`http://localhost:8501`. Google Chrome must be installed.

To retrain the models, install the training dependency and run:

```powershell
python -m pip install -r requirements-training.txt
python train.py
```

This is a Python project; it has no Node.js package manifest, so `npm install`
is not used. Install project dependencies with `pip` as shown above.

## Project files

- `app.py` — Streamlit dashboard.
- `train.py` — trains and evaluates the models.
- `src/pipeline.py` — data cleaning and model helpers.
- `data/raw/` — source housing data.
- `models/` — trained models and evaluation metadata.
- `tests/` — automated tests.

The investment label is a price-per-square-foot comparison, not a promise of
investment returns. Five-year estimates use an assumed annual growth rate.
