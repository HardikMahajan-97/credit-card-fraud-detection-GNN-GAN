"""
Data cleaning pipeline for credit card fraud detection.

Handles:
  - Missing value imputation
  - Duplicate removal
  - Outlier detection and capping (IQR and z-score methods)
  - Type validation and casting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Clean raw transaction DataFrames.

    Args:
        outlier_method: "iqr" or "zscore".
        outlier_threshold: Multiplier for IQR or number of std-devs for z-score.
        numeric_impute_strategy: Strategy for numeric columns ("median" or "mean").
        categorical_impute_strategy: Strategy for categorical columns ("mode" or constant).
    """

    def __init__(
        self,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        numeric_impute_strategy: str = "median",
        categorical_impute_strategy: str = "mode",
    ) -> None:
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit cleaning parameters on *df* and return the cleaned DataFrame.

        Cleans in place: missing values → duplicates → outliers → types.
        """
        logger.info(f"Starting data cleaning on {len(df):,} rows × {len(df.columns)} cols")
        df = df.copy()
        df = self._remove_duplicates(df)
        df = self._impute_missing(df)
        df = self._handle_outliers(df)
        df = self._cast_types(df)
        logger.info(f"Data cleaning complete → {len(df):,} rows")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously-fit cleaning logic to new data (same as fit_transform here)."""
        return self.fit_transform(df)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        # Avoid using transaction_id as duplicate key — it's unique by design
        subset = [c for c in df.columns if c != "transaction_id"]
        df = df.drop_duplicates(subset=subset, keep="first")
        dropped = before - len(df)
        if dropped:
            logger.info(f"  Removed {dropped:,} duplicate rows")
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if missing_cols.empty:
            return df

        logger.info(f"  Imputing missing values in {len(missing_cols)} column(s)")
        for col in missing_cols.index:
            if df[col].dtype in (np.float64, np.float32, np.int64, np.int32):
                if self.numeric_impute_strategy == "median":
                    fill_val = df[col].median()
                else:
                    fill_val = df[col].mean()
            else:
                if self.categorical_impute_strategy == "mode":
                    mode_vals = df[col].mode()
                    fill_val = mode_vals[0] if not mode_vals.empty else "unknown"
                else:
                    fill_val = self.categorical_impute_strategy

            df[col] = df[col].fillna(fill_val)
            logger.debug(f"    Imputed '{col}' with {fill_val} ({missing_counts[col]} missing)")

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers in numeric columns (excludes label and ID columns)."""
        skip_cols = {"is_fraud", "transaction_id", "card_id", "merchant_id"}
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in skip_cols
        ]

        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            if self.outlier_method == "iqr":
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
            else:  # zscore
                mean_ = series.mean()
                std_ = series.std()
                if std_ == 0:
                    continue
                lower = mean_ - self.outlier_threshold * std_
                upper = mean_ + self.outlier_threshold * std_

            clipped = df[col].clip(lower=lower, upper=upper)
            n_clipped = (df[col] != clipped).sum()
            if n_clipped > 0:
                logger.debug(f"    Clipped {n_clipped:,} values in '{col}' to [{lower:.2f}, {upper:.2f}]")
                df[col] = clipped

        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure expected column types."""
        int_cols = ["is_fraud", "is_weekend", "is_international", "hour_of_day", "day_of_week"]
        float_cols = ["amount", "latitude", "longitude"]

        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

        return df
