"""
Feature engineering for credit card fraud detection.

Adds:
  - Time-based features (hour, day, is_weekend, time_since_last_txn)
  - Velocity features (txn_count_last_1h/24h, avg_amount_last_24h)
  - Aggregation features (merchant_avg_amount, card_avg_amount, merchant_fraud_rate)
  - Interaction features (amount_to_avg_ratio, is_high_amount)
  - Rolling statistics
  - Feature normalization (StandardScaler / RobustScaler)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from utils.logger import get_logger

logger = get_logger(__name__)

SCALER_MAP = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


class FeatureEngineer:
    """
    Compute and normalise features for downstream models.

    Args:
        scaler_type: One of "robust", "standard", "minmax".
        time_windows: List of hour-windows for velocity features.
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        time_windows: Optional[List[int]] = None,
    ) -> None:
        self.scaler_type = scaler_type
        self.time_windows = time_windows or [1, 6, 12, 24]
        self._scaler: Optional[object] = None
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features, fit scaler, and normalize."""
        logger.info("Starting feature engineering …")
        df = df.copy()
        df = self._add_time_features(df)
        df = self._add_velocity_features(df)
        df = self._add_aggregation_features(df)
        df = self._add_interaction_features(df)
        df = self._encode_categoricals(df)
        df = self._normalize(df, fit=True)
        logger.info(f"Feature engineering complete → {len(df.columns)} columns")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-fit transformations to new data."""
        df = df.copy()
        df = self._add_time_features(df)
        df = self._add_velocity_features(df)
        df = self._add_aggregation_features(df)
        df = self._add_interaction_features(df)
        df = self._encode_categoricals(df)
        df = self._normalize(df, fit=False)
        return df

    @property
    def feature_columns(self) -> List[str]:
        """Return the list of engineered feature column names."""
        return list(self._feature_cols)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)
            df["time_since_last_txn"] = df.groupby("card_id")["time"].diff().fillna(0)
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction count and average amount over configurable time windows."""
        if "time" not in df.columns or "card_id" not in df.columns:
            return df

        df = df.sort_values("time").reset_index(drop=True)

        for window_hours in self.time_windows:
            window_secs = window_hours * 3600
            count_col = f"txn_count_last_{window_hours}h"
            amt_col = f"avg_amount_last_{window_hours}h"

            counts = []
            avg_amounts = []
            for _, row in df.iterrows():
                mask = (
                    (df["card_id"] == row["card_id"])
                    & (df["time"] >= row["time"] - window_secs)
                    & (df["time"] < row["time"])
                )
                subset = df.loc[mask, "amount"]
                counts.append(len(subset))
                avg_amounts.append(subset.mean() if len(subset) > 0 else 0.0)

            df[count_col] = counts
            df[amt_col] = avg_amounts

        return df

    def _add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Card and merchant-level aggregations."""
        if "time" in df.columns:
            df = df.sort_values("time").reset_index(drop=True)

        if "merchant_id" in df.columns:
            if "time" in df.columns:
                grp_amt = df.groupby("merchant_id", sort=False)["amount"]
                prior_amt_sum = grp_amt.cumsum() - df["amount"]
                prior_amt_count = grp_amt.cumcount()
                df["merchant_avg_amount"] = (
                    prior_amt_sum / prior_amt_count.replace(0, np.nan)
                ).fillna(0.0)
            else:
                merchant_agg = df.groupby("merchant_id")["amount"].mean().rename("merchant_avg_amount")
                df = df.merge(merchant_agg, on="merchant_id", how="left")

            if "is_fraud" in df.columns:
                if "time" in df.columns:
                    grp = df.groupby("merchant_id", sort=False)["is_fraud"]
                    prior_fraud = grp.cumsum() - df["is_fraud"]
                    prior_count = grp.cumcount()
                    df["merchant_fraud_rate"] = (
                        prior_fraud / prior_count.replace(0, np.nan)
                    ).fillna(0.0)
                    # Uses only prior transactions per merchant to avoid label leakage.
                else:
                    merchant_fraud = df.groupby("merchant_id")["is_fraud"].mean().rename("merchant_fraud_rate")
                    df = df.merge(merchant_fraud, on="merchant_id", how="left")

        if "card_id" in df.columns:
            if "time" in df.columns:
                grp_amt = df.groupby("card_id", sort=False)["amount"]
                prior_amt_sum = grp_amt.cumsum() - df["amount"]
                prior_amt_count = grp_amt.cumcount()
                df["card_avg_amount"] = (
                    prior_amt_sum / prior_amt_count.replace(0, np.nan)
                ).fillna(0.0)
            else:
                card_agg = df.groupby("card_id")["amount"].mean().rename("card_avg_amount")
                df = df.merge(card_agg, on="card_id", how="left")

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ratio and binary interaction features."""
        if "amount" in df.columns and "card_avg_amount" in df.columns:
            df["amount_to_avg_ratio"] = (
                df["amount"] / (df["card_avg_amount"] + 1e-6)
            )
            df["is_high_amount"] = (df["amount_to_avg_ratio"] > 3.0).astype(int)

        if "hour_of_day" in df.columns:
            df["is_night"] = df["hour_of_day"].between(22, 5).astype(int)

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        cat_cols = ["merchant_category", "card_type"]
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=float)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def _normalize(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Normalize numeric feature columns (exclude IDs, labels)."""
        exclude = {
            "transaction_id", "card_id", "merchant_id", "is_fraud",
            "is_weekend", "is_international", "day_of_week",
        }
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

        if not numeric_cols:
            return df

        if fit:
            scaler_cls = SCALER_MAP.get(self.scaler_type, RobustScaler)
            self._scaler = scaler_cls()
            df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols].values)
            self._feature_cols = numeric_cols
        else:
            if self._scaler is None:
                logger.warning("Scaler not fitted — skipping normalization.")
                return df
            cols_to_scale = [c for c in self._feature_cols if c in df.columns]
            df[cols_to_scale] = self._scaler.transform(df[cols_to_scale].values)

        return df
