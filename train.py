"""Train, evaluate, and track investment and five-year price models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.pipeline import (
    CLASSIFIER_FEATURE_COLUMNS,
    DEFAULT_GROWTH_RATE,
    FORECAST_YEARS,
    REGRESSION_FEATURE_COLUMNS,
    build_benchmarks,
    future_price_target,
    investment_labels,
    load_and_clean_data,
    make_classifier,
    make_feature_frame,
    make_regressor,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"


def train(data_path: Path, export_cleaned: bool = True) -> dict[str, object]:
    frame, cleaning_report = load_and_clean_data(data_path)
    train_rows, test_rows = train_test_split(
        frame, test_size=0.2, random_state=42
    )
    benchmarks = build_benchmarks(train_rows)
    x_train = make_feature_frame(train_rows, benchmarks)
    x_test = make_feature_frame(test_rows, benchmarks)
    y_train_class = investment_labels(train_rows, benchmarks)
    y_test_class = investment_labels(test_rows, benchmarks)
    y_train_reg = future_price_target(train_rows)
    y_test_reg = future_price_target(test_rows)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if export_cleaned:
        frame.to_csv(PROCESSED_DIR / "india_housing_prices_cleaned.csv", index=False)
    with (MODEL_DIR / "benchmarks.json").open("w", encoding="utf-8") as file:
        json.dump(benchmarks, file, indent=2)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Real Estate Investment Advisor")

    classifier = make_classifier()
    with mlflow.start_run(run_name="investment-classifier") as run:
        classifier.fit(x_train[CLASSIFIER_FEATURE_COLUMNS], y_train_class)
        probabilities = classifier.predict_proba(
            x_test[CLASSIFIER_FEATURE_COLUMNS]
        )[:, 1]
        predictions = (probabilities >= 0.5).astype("int8")
        class_metrics = {
            "accuracy": float(accuracy_score(y_test_class, predictions)),
            "precision": float(
                precision_score(y_test_class, predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_test_class, predictions, zero_division=0)),
            "f1": float(f1_score(y_test_class, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test_class, probabilities)),
        }
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "label_rule": "price per sqft <= training city/property-type median",
                "features_exclude_price_fields": True,
                "test_size": 0.2,
                "random_state": 42,
                "training_rows": len(train_rows),
            }
        )
        mlflow.log_metrics(class_metrics)
        class_confusion_matrix = confusion_matrix(y_test_class, predictions)
        mlflow.log_dict(
            {"confusion_matrix": class_confusion_matrix.tolist()},
            "evaluation/classification.json",
        )
        mlflow.sklearn.log_model(  # pyright: ignore[reportPrivateImportUsage]
            classifier,
            artifact_path="classifier",
            registered_model_name="real-estate-investment-classifier",
            serialization_format="cloudpickle",
        )
        classifier_run_id = run.info.run_id

    regressor = make_regressor()
    with mlflow.start_run(run_name="five-year-price-regressor") as run:
        regressor.fit(x_train[REGRESSION_FEATURE_COLUMNS], y_train_reg)
        reg_predictions = regressor.predict(x_test[REGRESSION_FEATURE_COLUMNS])
        regression_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test_reg, reg_predictions))),
            "mae": float(mean_absolute_error(y_test_reg, reg_predictions)),
            "r2": float(r2_score(y_test_reg, reg_predictions)),
        }
        mlflow.log_params(
            {
                "model": "Ridge",
                "target": "Price_in_Lakhs compounded at fixed annual rate",
                "target_is_scenario_derived": True,
                "evaluation_note": "Metrics evaluate a formula-derived scenario, not realized future sale prices.",
                "annual_growth_rate": DEFAULT_GROWTH_RATE,
                "forecast_years": FORECAST_YEARS,
                "test_size": 0.2,
                "random_state": 42,
                "training_rows": len(train_rows),
            }
        )
        mlflow.log_metrics(regression_metrics)
        mlflow.sklearn.log_model(  # pyright: ignore[reportPrivateImportUsage]
            regressor,
            artifact_path="regressor",
            registered_model_name="real-estate-five-year-price-regressor",
            serialization_format="cloudpickle",
        )
        regressor_run_id = run.info.run_id

    joblib.dump(classifier, MODEL_DIR / "classifier.joblib", compress=3)
    joblib.dump(regressor, MODEL_DIR / "regressor.joblib", compress=3)
    metadata = {
        "classification_metrics": class_metrics,
        "classification_confusion_matrix": class_confusion_matrix.tolist(),
        "regression_metrics": regression_metrics,
        "cleaning_report": cleaning_report,
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
        "annual_growth_rate": DEFAULT_GROWTH_RATE,
        "forecast_years": FORECAST_YEARS,
        "classification_label_rule": (
            "Price per square foot at or below training city/property-type median"
        ),
        "regression_target_is_scenario_derived": True,
        "regression_evaluation_note": (
            "Metrics evaluate a formula-derived scenario, not realized future sale prices."
        ),
        "classifier_run_id": classifier_run_id,
        "regressor_run_id": regressor_run_id,
    }
    with (MODEL_DIR / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "india_housing_prices.csv",
        help="Path to the housing CSV (default: data/raw/india_housing_prices.csv).",
    )
    parser.add_argument(
        "--no-export-cleaned",
        action="store_true",
        help="Skip writing the cleaned CSV deliverable.",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset not found: {args.data}")
    results = train(args.data, export_cleaned=not args.no_export_cleaned)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()