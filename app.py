"""Streamlit dashboard for Indian real-estate investment analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.pipeline import Pipeline

from src.pipeline import (
    DEFAULT_GROWTH_RATE,
    FEATURE_COLUMNS,
    forecast_for_growth_scenario,
    load_and_clean_data,
    make_feature_frame,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "raw" / "india_housing_prices.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data(
    path: str,
    modified: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    # `modified` intentionally participates in Streamlit's cache key.
    del modified
    return load_and_clean_data(path)


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[Pipeline, Pipeline, dict[str, Any], dict[str, Any]]:
    classifier = joblib.load(MODEL_DIR / "classifier.joblib")
    regressor = joblib.load(MODEL_DIR / "regressor.joblib")
    with (MODEL_DIR / "benchmarks.json").open(encoding="utf-8") as file:
        benchmarks = json.load(file)
    with (MODEL_DIR / "training_metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    return classifier, regressor, benchmarks, metadata


def money_lakhs(value: float) -> str:
    return f"₹{value:,.2f} lakh"


def show_analytics(data: pd.DataFrame) -> None:
    chart_data = data.sample(min(30_000, len(data)), random_state=42)
    st.subheader("Market overview")
    st.caption(
        "Summary metrics use every filtered listing; row-level charts use a "
        "reproducible sample of up to 30,000 listings."
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Properties", f"{len(data):,}")
    second.metric("Median asking price", money_lakhs(data["Price_in_Lakhs"].median()))
    second.caption("Price is expressed in lakhs (₹100,000 units).")
    third.metric("Median area", f"{data['Size_in_SqFt'].median():,.0f} sq ft")
    fourth.metric("Cities", f"{data['City'].nunique():,}")

    price_tab, location_tab, feature_tab, data_tab = st.tabs(
        ["Price & size", "Locations", "Features", "Cleaned data"]
    )
    with price_tab:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(
                    chart_data,
                    x="Price_in_Lakhs",
                    nbins=50,
                    title="Property price distribution",
                    labels={"Price_in_Lakhs": "Price (₹ lakh)"},
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.histogram(
                    chart_data,
                    x="Size_in_SqFt",
                    nbins=50,
                    title="Property size distribution",
                    labels={"Size_in_SqFt": "Area (sq ft)"},
                ),
                width="stretch",
            )
        st.plotly_chart(
            px.scatter(
                data.sample(min(10_000, len(data)), random_state=42),
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                color="Property_Type",
                opacity=0.55,
                title="Area vs asking price (sample of up to 10,000)",
                labels={
                    "Size_in_SqFt": "Area (sq ft)",
                    "Price_in_Lakhs": "Price (₹ lakh)",
                },
            ),
            width="stretch",
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.box(
                    chart_data,
                    y="Size_in_SqFt",
                    title="Area outliers",
                    labels={"Size_in_SqFt": "Area (sq ft)"},
                    points=False,
                ),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.box(
                    chart_data,
                    x="Property_Type",
                    y="Price_per_SqFt",
                    title="Price per sq ft by property type",
                    labels={
                        "Property_Type": "Property type",
                        "Price_per_SqFt": "₹ lakh per sq ft",
                    },
                    points=False,
                ),
                width="stretch",
            )
    with location_tab:
        left, right = st.columns(2)
        city_price = (
            data.groupby("City", as_index=False)
            .agg(Price_in_Lakhs=("Price_in_Lakhs", "mean"))
            .sort_values(by="Price_in_Lakhs", ascending=False)
            .head(20)
        )
        state_rate = (
            data.groupby("State", as_index=False)
            .agg(Price_per_SqFt=("Price_per_SqFt", "median"))
            .sort_values(by="Price_per_SqFt", ascending=False)
            .head(20)
        )
        with left:
            st.plotly_chart(
                px.bar(
                    city_price,
                    x="Price_in_Lakhs",
                    y="City",
                    orientation="h",
                    title="Average price by city (top 20)",
                    labels={"Price_in_Lakhs": "Average price (₹ lakh)"},
                ).update_layout(yaxis={"categoryorder": "total ascending"}),
                width="stretch",
            )
        with right:
            st.plotly_chart(
                px.bar(
                    state_rate,
                    x="Price_per_SqFt",
                    y="State",
                    orientation="h",
                    title="Median price per sq ft by state (top 20)",
                    labels={"Price_per_SqFt": "₹ lakh per sq ft"},
                ).update_layout(yaxis={"categoryorder": "total ascending"}),
                width="stretch",
            )
        locality = (
            data.groupby(["City", "Locality"], as_index=False)
            .agg(
                Median_Price=("Price_in_Lakhs", "median"),
                Median_Price_per_SqFt=("Price_per_SqFt", "median"),
                Median_Age=("Age_of_Property", "median"),
            )
            .sort_values(by="Median_Price", ascending=False)
            .head(10)
        )
        st.markdown("**Top 10 localities by median asking price**")
        st.dataframe(locality, width="stretch", hide_index=True)
        city_counts = data["City"].value_counts().head(15).index
        bhk_city = (
            data[data["City"].isin(city_counts)]
            .groupby(["City", "BHK"])
            .size()
            .rename("Listings")
            .reset_index()
        )
        st.plotly_chart(
            px.bar(
                bhk_city,
                x="City",
                y="Listings",
                color="BHK",
                title="BHK distribution across the 15 largest city samples",
                barmode="stack",
            ),
            width="stretch",
        )
        st.caption(
            "The dataset has no listing dates or historical transactions, so "
            "time-series price trends cannot be calculated; the locality table "
            "compares current listing prices instead."
        )
    with feature_tab:
        left, right = st.columns(2)
        with left:
            furnished = (
                data.groupby("Furnished_Status", as_index=False)
                .agg(Price_in_Lakhs=("Price_in_Lakhs", "median"))
                .sort_values("Price_in_Lakhs", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    furnished,
                    x="Furnished_Status",
                    y="Price_in_Lakhs",
                    title="Median price by furnishing",
                    labels={"Price_in_Lakhs": "Median price (₹ lakh)"},
                ),
                width="stretch",
            )
        with right:
            corr_columns = [
                "BHK",
                "Size_in_SqFt",
                "Price_in_Lakhs",
                "Price_per_SqFt",
                "Age_of_Property",
                "Nearby_Schools",
                "Nearby_Hospitals",
                "Parking_Space_Count",
            ]
            corr = data[corr_columns].corr(numeric_only=True)
            st.plotly_chart(
                px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Numeric feature correlations",
                ),
                width="stretch",
            )
        for column, title in [
            ("Nearby_Schools", "Nearby schools"),
            ("Nearby_Hospitals", "Nearby hospitals"),
        ]:
            access = (
                data.groupby(column, as_index=False)
                .agg(Price_per_SqFt=("Price_per_SqFt", "median"))
                .sort_values(by=column)
            )
            st.plotly_chart(
                px.line(
                    access,
                    x=column,
                    y="Price_per_SqFt",
                    markers=True,
                    title=f"Median price per sq ft vs {title.lower()}",
                    labels={
                        column: title,
                        "Price_per_SqFt": "₹ lakh per sq ft",
                    },
                ),
                width="stretch",
            )
        left, right = st.columns(2)
        with left:
            facing = (
                data.groupby("Facing", as_index=False)
                .agg(Price_per_SqFt=("Price_per_SqFt", "median"))
                .sort_values(by="Price_per_SqFt", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    facing,
                    x="Facing",
                    y="Price_per_SqFt",
                    title="Median price per sq ft by facing",
                    labels={"Price_per_SqFt": "₹ lakh per sq ft"},
                ),
                width="stretch",
            )
        with right:
            parking = (
                data.groupby("Parking_Space", as_index=False)
                .agg(Price_in_Lakhs=("Price_in_Lakhs", "median"))
                .sort_values(by="Price_in_Lakhs", ascending=False)
            )
            st.plotly_chart(
                px.bar(
                    parking,
                    x="Parking_Space",
                    y="Price_in_Lakhs",
                    title="Median asking price by parking availability",
                    labels={"Price_in_Lakhs": "Median price (₹ lakh)"},
                ),
                width="stretch",
            )
        left, right = st.columns(2)
        with left:
            owners = (
                data["Owner_Type"]
                .value_counts()
                .rename_axis("Owner_Type")
                .reset_index(name="Listings")
            )
            st.plotly_chart(
                px.bar(owners, x="Owner_Type", y="Listings", title="Listings by owner type"),
                width="stretch",
            )
            statuses = (
                data["Availability_Status"]
                .value_counts()
                .rename_axis("Availability_Status")
                .reset_index(name="Listings")
            )
            st.plotly_chart(
                px.bar(
                    statuses,
                    x="Availability_Status",
                    y="Listings",
                    title="Listings by availability",
                ),
                width="stretch",
            )
        with right:
            amenity_prices = (
                data.groupby("Amenities_Count", as_index=False)
                .agg(Price_per_SqFt=("Price_per_SqFt", "median"))
                .sort_values(by="Amenities_Count")
            )
            st.plotly_chart(
                px.bar(
                    amenity_prices,
                    x="Amenities_Count",
                    y="Price_per_SqFt",
                    title="Median price per sq ft by amenity count",
                    labels={
                        "Amenities_Count": "Distinct amenities",
                        "Price_per_SqFt": "₹ lakh per sq ft",
                    },
                ),
                width="stretch",
            )
        st.plotly_chart(
            px.box(
                chart_data,
                x="Public_Transport_Accessibility",
                y="Price_per_SqFt",
                title="Price per sq ft by public transport access",
                labels={
                    "Public_Transport_Accessibility": "Transport access",
                    "Price_per_SqFt": "₹ lakh per sq ft",
                },
                points=False,
            ),
            width="stretch",
        )
        st.caption(
            "Dataset does not include crime, RERA, resale transactions, or actual "
            "infrastructure scores; those effects cannot be measured here."
        )
    with data_tab:
        st.caption("Duplicate rows and rows with invalid price/area/BHK are removed.")
        st.dataframe(data.head(2_000), width="stretch", hide_index=True)
        st.download_button(
            "Download cleaned CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="india_housing_prices_cleaned.csv",
            mime="text/csv",
        )


def make_prediction_input(
    city: str,
    property_type: str,
    state: str,
    locality: str,
    bhk: int,
    size: int,
    price: float,
    year_built: int,
    furnished: str,
    floor: int,
    total_floors: int,
    schools: int,
    hospitals: int,
    transport: str,
    parking: str,
    security: str,
    amenities: str,
    facing: str,
    owner: str,
    availability: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "State": state,
                "City": city,
                "Locality": locality,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_in_Lakhs": price,
                "Year_Built": year_built,
                "Furnished_Status": furnished,
                "Floor_No": floor,
                "Total_Floors": total_floors,
                "Nearby_Schools": schools,
                "Nearby_Hospitals": hospitals,
                "Public_Transport_Accessibility": transport,
                "Parking_Space": parking,
                "Security": security,
                "Amenities": amenities,
                "Facing": facing,
                "Owner_Type": owner,
                "Availability_Status": availability,
            }
        ]
    )


def show_advisor(data: pd.DataFrame) -> None:
    if not (MODEL_DIR / "classifier.joblib").is_file():
        st.warning(
            "Models have not been trained yet. Run "
            "`python train.py` from the project folder, then refresh this page."
        )
        return

    classifier, regressor, benchmarks, metadata = load_models()
    st.subheader("Property investment assessment")
    st.caption(
        "The investment classification is a market-value proxy: a property is "
        "labeled attractive when its price per sq ft is at or below the training "
        "median for its city and property type. It is not a return guarantee."
    )
    cities = sorted(data["City"].dropna().astype(str).unique())
    states = sorted(data["State"].dropna().astype(str).unique())
    property_types = sorted(data["Property_Type"].dropna().astype(str).unique())
    if not cities or not states or not property_types:
        st.warning("The dataset has no valid city, state, or property type values.")
        return

    with st.form("investment-form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            city = st.selectbox("City", cities, index=0)
            state = st.selectbox("State", states, index=0)
            locality = st.text_input("Locality", "Unknown")
            property_type = st.selectbox("Property type", property_types, index=0)
            bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)
            size = st.number_input("Area (sq ft)", min_value=100, max_value=100_000, value=1_000)
        with col2:
            price = st.number_input("Asking price (₹ lakh)", min_value=1.0, value=100.0)
            year_built = st.number_input(
                "Year built",
                min_value=1800,
                max_value=pd.Timestamp.now().year,
                value=min(2015, pd.Timestamp.now().year),
            )
            furnished = st.selectbox(
                "Furnishing", ["Unfurnished", "Semi-furnished", "Furnished", "Unknown"]
            )
            floor = st.number_input("Floor number", min_value=0, max_value=200, value=1)
            total_floors = st.number_input(
                "Total floors", min_value=1, max_value=200, value=10
            )
            transport = st.selectbox(
                "Public transport", ["High", "Medium", "Low", "Unknown"]
            )
        with col3:
            schools = st.number_input("Nearby schools", min_value=0, value=2)
            hospitals = st.number_input("Nearby hospitals", min_value=0, value=1)
            parking = st.selectbox("Parking", ["No", "Yes"])
            security = st.selectbox("Security", ["Yes", "No", "Unknown"])
            amenities = st.text_input(
                "Amenities (comma separated)", "Gym, Garden"
            )
            facing = st.selectbox(
                "Facing", ["North", "South", "East", "West", "Unknown"]
            )
            owner = st.selectbox("Owner type", ["Owner", "Builder", "Broker", "Unknown"])
            availability = st.selectbox(
                "Availability", ["Ready_to_Move", "Under_Construction", "Unknown"]
            )
        annual_rate = st.slider(
            "Annual appreciation scenario (%)",
            min_value=0.0,
            max_value=20.0,
            value=DEFAULT_GROWTH_RATE * 100,
            step=0.5,
        )
        submitted = st.form_submit_button("Assess property", type="primary")

    if not submitted:
        return
    if floor > total_floors:
        st.error("Floor number cannot exceed total floors.")
        return
    if city is None or state is None or property_type is None:
        st.error("Select a valid city, state, and property type.")
        return
    raw = make_prediction_input(
        city,
        property_type,
        state,
        locality,
        int(bhk),
        int(size),
        float(price),
        int(year_built),
        furnished,
        int(floor),
        int(total_floors),
        int(schools),
        int(hospitals),
        transport,
        parking,
        security,
        amenities,
        facing,
        owner,
        availability,
    )
    clean, _ = load_and_clean_data(raw)
    x_input = make_feature_frame(clean, benchmarks)
    probability = float(classifier.predict_proba(x_input[FEATURE_COLUMNS])[0, 1])
    class_prediction = int(probability >= 0.5)
    baseline_forecast = float(regressor.predict(x_input[FEATURE_COLUMNS])[0])
    forecast = forecast_for_growth_scenario(baseline_forecast, annual_rate / 100)
    implied_rate = float(
        benchmarks["city_property_type"].get(city, {}).get(property_type)
        or benchmarks.get("city", {}).get(city)
        or benchmarks["overall"]
    )
    current_price_per_sqft = float(price) / int(size)

    left, middle, right = st.columns(3)
    left.metric(
        "Investment proxy",
        "Good Investment" if class_prediction else "Not a Good Investment",
        f"{probability:.0%} model probability",
    )
    middle.metric("Estimated price after 5 years", money_lakhs(forecast))
    right.metric("Current price per sq ft", f"₹{current_price_per_sqft:.4f} lakh")
    relative = current_price_per_sqft / max(implied_rate, 1e-9)
    st.caption(
        f"City/property-type median: ₹{implied_rate:.4f} lakh per sq ft. "
        f"This listing is {relative:.2f}× that benchmark. Forecast uses a "
        f"{annual_rate:.1f}% annual scenario; model training baseline is "
        f"{metadata['annual_growth_rate']:.0%}. Estimates exclude transaction "
        "costs, taxes, financing, rent, and market shocks."
    )


def main() -> None:
    st.title("Real Estate Investment Advisor")
    st.write(
        "Explore Indian housing listings, compare local value, and model a "
        "five-year appreciation scenario."
    )
    uploaded = st.sidebar.file_uploader("Use another housing CSV", type=["csv"])
    if uploaded is not None:
        try:
            data, cleaning = load_and_clean_data(uploaded)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not load uploaded CSV: {error}")
            st.stop()
    else:
        if not DATA_PATH.is_file():
            st.error(f"Dataset not found: {DATA_PATH}")
            st.stop()
        data, cleaning = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)

    with st.sidebar:
        st.subheader("Filters")
        cities = sorted(data["City"].dropna().astype(str).unique())
        selected_cities = st.multiselect("City", cities, default=cities)
        property_types = sorted(data["Property_Type"].dropna().astype(str).unique())
        selected_types = st.multiselect(
            "Property type", property_types, default=property_types
        )
        min_area, max_area = int(data["Size_in_SqFt"].min()), int(
            data["Size_in_SqFt"].max()
        )
        area_range = st.slider(
            "Area range (sq ft)",
            min_value=min_area,
            max_value=max_area,
            value=(min_area, max_area),
        )
        min_price, max_price = float(data["Price_in_Lakhs"].min()), float(
            data["Price_in_Lakhs"].max()
        )
        price_range = st.slider(
            "Price range (₹ lakh)",
            min_value=min_price,
            max_value=max_price,
            value=(min_price, max_price),
        )
        min_bhk, max_bhk = int(data["BHK"].min()), int(data["BHK"].max())
        bhk_range = st.slider(
            "BHK range",
            min_value=min_bhk,
            max_value=max_bhk,
            value=(min_bhk, max_bhk),
        )

    filtered = data[
        data["City"].isin(selected_cities)
        & data["Property_Type"].isin(selected_types)
        & data["Price_in_Lakhs"].between(*price_range)
        & data["Size_in_SqFt"].between(*area_range)
        & data["BHK"].between(*bhk_range)
    ]
    st.caption(
        f"Showing {len(filtered):,} of {len(data):,} properties. "
        f"Removed {cleaning['duplicates_removed']:,} duplicates and "
        f"{cleaning['invalid_rows_removed']:,} invalid rows."
    )
    analytics_tab, advisor_tab, model_tab = st.tabs(
        ["Market insights", "Investment advisor", "Model evaluation"]
    )
    with analytics_tab:
        if filtered.empty:
            st.info("No properties match those filters. Adjust the sidebar filters.")
        else:
            show_analytics(filtered)
    with advisor_tab:
        show_advisor(data)
    with model_tab:
        metadata_path = MODEL_DIR / "training_metadata.json"
        if not metadata_path.is_file():
            st.info("Train the models with `python train.py` to see evaluation metrics.")
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            st.subheader("Held-out evaluation")
            st.caption(
                f"Training rows: {metadata['rows_train']:,}; test rows: "
                f"{metadata['rows_test']:,}. Future-price targets are scenario-derived, "
                "not realized future sale prices. Classification metrics measure "
                "agreement with the disclosed market-median proxy, not actual returns."
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Investment classification**")
                st.json(metadata["classification_metrics"])
            with right:
                st.markdown("**Five-year price regression**")
                st.json(metadata["regression_metrics"])
            try:
                classifier, _, _, _ = load_models()
                from src.pipeline import feature_importance

                st.plotly_chart(
                    px.bar(
                        feature_importance(classifier),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Classification model: largest coefficient magnitudes",
                    ).update_layout(yaxis={"categoryorder": "total ascending"}),
                    width="stretch",
                )
            except (OSError, ValueError, KeyError) as error:
                st.warning(f"Could not load model explanation: {error}")


if __name__ == "__main__":
    main()