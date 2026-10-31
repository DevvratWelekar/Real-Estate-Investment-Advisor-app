import unittest

import pandas as pd

from src.pipeline import (
    CLASSIFIER_FEATURE_COLUMNS,
    build_benchmarks,
    forecast_for_growth_scenario,
    future_price_target,
    investment_labels,
    load_and_clean_data,
    make_feature_frame,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.source = pd.DataFrame(
            {
                "ID": [1, 2, 2, 4],
                "State": ["Karnataka", "Karnataka", "Karnataka", "Tamil Nadu"],
                "City": ["Mysuru", "Mysuru", "Mysuru", "Chennai"],
                "Locality": ["A", "B", "B", "C"],
                "Property_Type": ["Apartment"] * 4,
                "BHK": [2, 3, 3, 2],
                "Size_in_SqFt": [1000, 1000, 1000, 800],
                "Price_in_Lakhs": [100.0, 120.0, 120.0, 80.0],
                "Year_Built": [2015, 2010, 2010, 2020],
                "Nearby_Schools": [2, 3, 3, 1],
                "Nearby_Hospitals": [1, 2, 2, 1],
                "Parking_Space": ["Yes", "No", "No", "Yes"],
                "Amenities": ["Gym, Pool", "Garden", "Garden", ""],
            }
        )

    def test_cleaning_recalculates_price_per_sqft_and_removes_duplicate(self):
        cleaned, report = load_and_clean_data(self.source)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(report["duplicates_removed"], 1)
        self.assertAlmostEqual(cleaned.iloc[0]["Price_per_SqFt"], 0.1)
        self.assertEqual(cleaned.iloc[0]["Amenities_Count"], 2)

    def test_benchmarks_labels_and_features_are_consistent(self):
        cleaned, _ = load_and_clean_data(self.source)
        benchmarks = build_benchmarks(cleaned)
        labels = investment_labels(cleaned, benchmarks)
        features = make_feature_frame(cleaned, benchmarks)
        classifier_features = features[CLASSIFIER_FEATURE_COLUMNS]
        self.assertEqual(len(labels), len(cleaned))
        self.assertTrue(labels.isin([0, 1]).all())
        self.assertEqual(classifier_features.shape[1], 24)
        self.assertNotIn("Price_in_Lakhs", classifier_features.columns)
        self.assertNotIn("Price_per_SqFt", classifier_features.columns)
        self.assertNotIn("Relative_Price_Index", classifier_features.columns)
        self.assertAlmostEqual(features.iloc[0]["School_Density_Score"], 20.0)

    def test_classifier_features_do_not_change_with_listing_price(self):
        cleaned, _ = load_and_clean_data(self.source)
        benchmarks = build_benchmarks(cleaned)
        original_features = make_feature_frame(cleaned.iloc[:1], benchmarks)[
            CLASSIFIER_FEATURE_COLUMNS
        ]
        changed_price = cleaned.iloc[:1].copy()
        changed_price["Price_in_Lakhs"] *= 2
        changed_features = make_feature_frame(changed_price, benchmarks)[
            CLASSIFIER_FEATURE_COLUMNS
        ]
        pd.testing.assert_frame_equal(original_features, changed_features)

    def test_five_year_target_and_growth_scenario(self):
        cleaned, _ = load_and_clean_data(self.source)
        target = future_price_target(cleaned.iloc[:1])
        self.assertAlmostEqual(target.iloc[0], 100 * (1.08**5))
        self.assertAlmostEqual(
            forecast_for_growth_scenario(100 * (1.08**5), 0.08),
            100 * (1.08**5),
        )
        self.assertAlmostEqual(
            forecast_for_growth_scenario(100 * (1.08**5), 0.10),
            100 * (1.10**5),
        )

    def test_cleaning_report_counts_iqr_outliers(self):
        cleaned, report = load_and_clean_data(self.source)
        self.assertEqual(report["rows_cleaned"], len(cleaned))
        self.assertGreaterEqual(report["outliers_price_per_sqft"], 0)
        self.assertGreaterEqual(report["outliers_price_in_lakhs"], 0)
        self.assertGreaterEqual(report["outliers_size_in_sqft"], 0)


if __name__ == "__main__":
    unittest.main()
