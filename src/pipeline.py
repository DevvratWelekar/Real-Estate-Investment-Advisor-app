"""Data preparation, feature engineering, and model utilities."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = date.today().year
DEFAULT_GROWTH_RATE = 0.08
FORECAST_YEARS = 5

REQUIRED_COLUMNS = {
    "City",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
}

NUMERIC_FEATURES = [
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
    "Price_per_SqFt",
    "Year_Built",
    "Age_of_Property",
    "Floor_No",
    "Total_Floors",
    "Floor_Ratio",
    "Nearby_Schools",
    "Nearby_Hospitals",
    "Parking_Space_Count",
    "Amenities_Count",
    "Services_Count",
    "Relative_Price_Index",
]

CATEGORICAL_FEATURES = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "Furnished_Status",
    "Public_Transport_Accessibility",
    "Security",
    "Facing",
    "Owner_Type",
    "Availability_Status",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_clean_data(source: str | Path | Any) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load a CSV-like source, normalize expected columns, and remove unusable rows."""
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing)
        )

    original_rows = len(frame)
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().copy()

    defaults: dict[str, Any] = {
        "State": "Unknown",
        "Locality": "Unknown",
        "Year_Built": np.nan,
        "Age_of_Property": np.nan,
        "Floor_No": np.nan,
        "Total_Floors": np.nan,
        "Nearby_Schools": np.nan,
        "Nearby_Hospitals": np.nan,
        "Public_Transport_Accessibility": "Unknown",
        "Parking_Space": "No",
        "Security": "Unknown",
        "Amenities": "",
        "Furnished_Status": "Unknown",
        "Facing": "Unknown",
        "Owner_Type": "Unknown",
        "Availability_Status": "Unknown",
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    numeric_columns = [
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Year_Built",
        "Age_of_Property",
        "Floor_No",
        "Total_Floors",
        "Nearby_Schools",
        "Nearby_Hospitals",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        frame["Size_in_SqFt"].gt(0)
        & frame["Price_in_Lakhs"].gt(0)
        & frame["BHK"].gt(0)
    ].copy()
    invalid_rows = original_rows - duplicates - len(frame)
    if frame.empty:
        raise ValueError("No rows with valid positive price, area, and BHK remain.")

    frame["City"] = frame["City"].fillna("Unknown").astype(str).str.strip()
    frame["Property_Type"] = (
        frame["Property_Type"].fillna("Unknown").astype(str).str.strip()
    )
    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()

    frame["Price_per_SqFt"] = (
        frame["Price_in_Lakhs"] / frame["Size_in_SqFt"]
    )
    frame["Year_Built"] = frame["Year_Built"].where(
        frame["Year_Built"].between(1800, CURRENT_YEAR)
    )
    frame["Age_of_Property"] = (CURRENT_YEAR - frame["Year_Built"]).clip(lower=0)
    frame["Floor_No"] = frame["Floor_No"].clip(lower=0)
    frame["Total_Floors"] = frame["Total_Floors"].clip(lower=0)
    frame["Floor_Ratio"] = (
        frame["Floor_No"] / frame["Total_Floors"].replace(0, np.nan)
    )
    frame["Parking_Space_Count"] = frame["Parking_Space"].map(_parking_count)
    frame["Amenities_Count"] = frame["Amenities"].map(_amenities_count)
    frame["Services_Count"] = frame[["Nearby_Schools", "Nearby_Hospitals"]].sum(
        axis=1, min_count=1
    )

    report = {
        "rows_loaded": original_rows,
        "duplicates_removed": duplicates,
        "invalid_rows_removed": invalid_rows,
        "rows_cleaned": len(frame),
    }
    return frame.reset_index(drop=True), report


def _parking_count(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "true", "available"}:
        return 1.0
    if text in {"no", "false", "none", ""}:
        return 0.0
    try:
        return max(0.0, float(text))
    except ValueError:
        return np.nan


def _amenities_count(value: Any) -> float:
    if pd.isna(value) or not str(value).strip():
        return 0.0
    return float(len({item.strip().lower() for item in str(value).split(",") if item.strip()}))


def build_benchmarks(training_rows: pd.DataFrame) -> dict[str, Any]:
    """Calculate price-per-square-foot medians using training rows only."""
    keyed = (
        training_rows.groupby(["City", "Property_Type"], dropna=False)[
            "Price_per_SqFt"
        ]
        .median()
        .reset_index()
    )
    city_type: dict[str, dict[str, float]] = {}
    for city, property_type, price in zip(
        keyed["City"].astype(str).to_numpy(),
        keyed["Property_Type"].astype(str).to_numpy(),
        keyed["Price_per_SqFt"].to_numpy(dtype=float),
        strict=True,
    ):
        city_type.setdefault(city, {})[property_type] = price
    city_medians = training_rows.groupby("City")["Price_per_SqFt"].median()
    return {
        "city_property_type": city_type,
        "city": {str(key): float(value) for key, value in city_medians.items()},
        "overall": float(training_rows["Price_per_SqFt"].median()),
    }


def relative_price_index(frame: pd.DataFrame, benchmarks: dict[str, Any]) -> pd.Series:
    prices = pd.to_numeric(frame["Price_per_SqFt"], errors="coerce").to_numpy(
        dtype=float
    )
    cities = frame["City"].astype(str).to_numpy()
    property_types = frame["Property_Type"].astype(str).to_numpy()
    ratios = []
    for city, property_type, price in zip(
        cities, property_types, prices, strict=True
    ):
        median = (
            benchmarks.get("city_property_type", {}).get(city, {}).get(property_type)
            or benchmarks.get("city", {}).get(city)
            or benchmarks["overall"]
        )
        if not isinstance(median, (int, float)):
            raise ValueError(f"Invalid price benchmark for {city}, {property_type}.")
        ratios.append(price / max(median, 1e-9))

    return pd.Series(ratios, index=frame.index, dtype=float)


def make_feature_frame(
    frame: pd.DataFrame, benchmarks: dict[str, Any]
) -> pd.DataFrame:
    features = frame.copy()
    features["Relative_Price_Index"] = relative_price_index(features, benchmarks)
    return features[FEATURE_COLUMNS]


def investment_labels(frame: pd.DataFrame, benchmarks: dict[str, Any]) -> pd.Series:
    """Create the disclosed investment proxy: at/below the local-type median."""
    return relative_price_index(frame, benchmarks).le(1.0).astype("int8")


def future_price_target(
    frame: pd.DataFrame, annual_growth_rate: float = DEFAULT_GROWTH_RATE
) -> pd.Series:
    return frame["Price_in_Lakhs"] * (1.0 + annual_growth_rate) ** FORECAST_YEARS


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=300,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def make_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]
    )


def feature_importance(pipeline: Pipeline, limit: int = 12) -> pd.DataFrame:
    """Return logistic-regression coefficient magnitudes for readable drivers."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefficients = np.asarray(model.coef_).reshape(-1)
    result = pd.DataFrame(
        {"Feature": names, "Importance": np.abs(coefficients)}
    ).sort_values("Importance", ascending=False)
    return result.head(limit).reset_index(drop=True)


def forecast_for_growth_scenario(
    baseline_prediction: float, annual_growth_rate: float
) -> float:
    """Rescale the 8% model forecast to a user-selected five-year scenario."""
    base_factor = (1.0 + DEFAULT_GROWTH_RATE) ** FORECAST_YEARS
    scenario_factor = (1.0 + annual_growth_rate) ** FORECAST_YEARS
    return float(baseline_prediction * scenario_factor / base_factor)
