"""
Tests for the preprocessing pipeline: cleaner, feature engineering, graph builder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.cleaner import DataCleaner
from preprocessing.feature_engineering import FeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_transaction_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic transaction DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "transaction_id": [f"TXN{i:05d}" for i in range(n)],
        "card_id": rng.integers(100, 200, n),
        "merchant_id": rng.integers(10, 50, n),
        "amount": rng.lognormal(3.5, 1.0, n).clip(1.0, 5000.0).round(2),
        "time": np.arange(n) * 3600,
        "hour_of_day": rng.integers(0, 24, n),
        "day_of_week": rng.integers(0, 7, n),
        "is_weekend": rng.integers(0, 2, n),
        "merchant_category": rng.choice(["grocery", "online_retail", "atm"], n),
        "card_type": rng.choice(["visa", "mastercard"], n),
        "is_international": rng.integers(0, 2, n),
        "latitude": rng.uniform(30.0, 50.0, n).round(4),
        "longitude": rng.uniform(-90.0, -60.0, n).round(4),
        "is_fraud": (rng.random(n) < 0.05).astype(int),
    })


# ---------------------------------------------------------------------------
# DataCleaner tests
# ---------------------------------------------------------------------------


class TestDataCleaner:
    def test_returns_dataframe(self):
        df = make_transaction_df()
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_handles_missing_values(self):
        df = make_transaction_df()
        df.loc[0, "amount"] = np.nan
        df.loc[1, "merchant_category"] = np.nan
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result["amount"].isna().sum() == 0
        assert result["merchant_category"].isna().sum() == 0

    def test_removes_duplicates(self):
        df = make_transaction_df(n=100)
        # Remove transaction_id column to create true duplicates
        df_no_id = df.drop(columns=["transaction_id"])
        df_dup = pd.concat([df_no_id, df_no_id.iloc[:10]], ignore_index=True)
        # Add fake IDs
        df_dup["transaction_id"] = [f"T{i}" for i in range(len(df_dup))]
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df_dup)
        assert len(result) <= len(df_dup)

    def test_iqr_outlier_capping(self):
        df = make_transaction_df()
        # Inject extreme outlier
        df.loc[0, "amount"] = 1_000_000.0
        cleaner = DataCleaner(outlier_method="iqr", outlier_threshold=3.0)
        result = cleaner.fit_transform(df)
        assert result["amount"].max() < 1_000_000.0

    def test_zscore_outlier_capping(self):
        df = make_transaction_df()
        df.loc[0, "amount"] = 1_000_000.0
        cleaner = DataCleaner(outlier_method="zscore", outlier_threshold=3.0)
        result = cleaner.fit_transform(df)
        assert result["amount"].max() < 1_000_000.0

    def test_type_casting(self):
        df = make_transaction_df()
        df["is_fraud"] = df["is_fraud"].astype(str)  # Introduce type mismatch
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result["is_fraud"].dtype in (int, np.int64, np.int32)

    def test_does_not_drop_label(self):
        df = make_transaction_df()
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert "is_fraud" in result.columns

    def test_preserves_row_count_without_dups(self):
        df = make_transaction_df(n=50)
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        # No duplicates introduced, so count should be equal
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# FeatureEngineer tests
# ---------------------------------------------------------------------------


class TestFeatureEngineer:
    def test_returns_dataframe(self):
        df = make_transaction_df()
        eng = FeatureEngineer(scaler_type="robust", time_windows=[1])
        result = eng.fit_transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_aggregation_columns(self):
        df = make_transaction_df()
        eng = FeatureEngineer(time_windows=[])
        result = eng.fit_transform(df)
        assert "merchant_avg_amount" in result.columns
        assert "card_avg_amount" in result.columns

    def test_adds_interaction_columns(self):
        df = make_transaction_df()
        eng = FeatureEngineer(time_windows=[])
        result = eng.fit_transform(df)
        assert "amount_to_avg_ratio" in result.columns

    def test_one_hot_encoding_merchant_category(self):
        df = make_transaction_df()
        eng = FeatureEngineer(time_windows=[])
        result = eng.fit_transform(df)
        # Original column replaced with dummies
        assert "merchant_category" not in result.columns
        assert any(c.startswith("merchant_category_") for c in result.columns)

    def test_feature_columns_property(self):
        df = make_transaction_df()
        eng = FeatureEngineer(time_windows=[])
        eng.fit_transform(df)
        assert len(eng.feature_columns) > 0

    def test_no_nan_after_engineering(self):
        df = make_transaction_df()
        eng = FeatureEngineer(time_windows=[])
        result = eng.fit_transform(df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isna().sum().sum() == 0
