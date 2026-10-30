import unittest

import pandas as pd

from src.pipeline import (
    build_benchmarks,
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
        self.assertEqual(len(labels), len(cleaned))
        self.assertTrue(labels.isin([0, 1]).all())
        self.assertEqual(features.shape[1], 25)
        self.assertTrue(features["Relative_Price_Index"].gt(0).all())

    def test_five_year_target_uses_compound_growth(self):
        cleaned, _ = load_and_clean_data(self.source)
        target = future_price_target(cleaned.iloc[:1])
        self.assertAlmostEqual(target.iloc[0], 100 * (1.08**5))


if __name__ == "__main__":
    unittest.main()
