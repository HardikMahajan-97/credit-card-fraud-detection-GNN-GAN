"""
Synthetic credit card fraud data generator.

Generates realistic transaction data using statistical distributions that
mimic real-world fraud patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Merchant category codes (MCC)
MERCHANT_CATEGORIES = [
    "grocery", "gas_station", "restaurant", "online_retail", "travel",
    "entertainment", "pharmacy", "atm", "clothing", "electronics",
]

CARD_TYPES = ["visa", "mastercard", "amex", "discover"]


class SyntheticFraudGenerator:
    """
    Generate realistic synthetic credit card transaction data.

    Legitimate transactions follow typical spending patterns (log-normal
    amounts, business-hour heavy), while fraudulent transactions exhibit
    higher amounts, unusual hours, rapid succession, and more foreign activity.

    Args:
        n_samples: Total number of transactions to generate.
        fraud_ratio: Fraction of fraudulent transactions (default ~1.7%).
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 100_000,
        fraud_ratio: float = 0.017,
        random_seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_seed = random_seed
        np.random.seed(random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Generate the full transaction dataset.

        Returns:
            DataFrame with columns:
            transaction_id, card_id, merchant_id, amount, time,
            hour_of_day, day_of_week, is_weekend, merchant_category,
            card_type, is_international, latitude, longitude,
            is_fraud.
        """
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_legit = self.n_samples - n_fraud

        logger.info(
            f"Generating {self.n_samples:,} transactions "
            f"({n_fraud:,} fraud, {n_legit:,} legitimate)"
        )

        legit_df = self._generate_legitimate(n_legit)
        fraud_df = self._generate_fraudulent(n_fraud)

        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        df["transaction_id"] = [f"TXN{i:08d}" for i in range(len(df))]

        logger.info("Synthetic data generation complete.")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_legitimate(self, n: int) -> pd.DataFrame:
        """Generate n legitimate (non-fraud) transactions."""
        rng = np.random.default_rng(self.random_seed)

        # Amounts: log-normal (mean ~$50, std wide)
        amounts = rng.lognormal(mean=3.5, sigma=1.0, size=n).clip(1.0, 5000.0)

        # Time: seconds since epoch — spread over 6 months
        time_vals = rng.integers(0, 180 * 24 * 3600, size=n)

        # Hour: business-hours heavy (8 AM – 9 PM)
        hour_probs = np.array([
            0.005, 0.003, 0.002, 0.002, 0.003, 0.005,  # 0–5
            0.010, 0.030, 0.060, 0.070, 0.080, 0.085,  # 6–11
            0.090, 0.085, 0.080, 0.075, 0.070, 0.065,  # 12–17
            0.060, 0.055, 0.045, 0.030, 0.020, 0.010,  # 18–23
        ])
        hour_probs /= hour_probs.sum()
        hours = rng.choice(24, size=n, p=hour_probs)

        day_of_week = rng.integers(0, 7, size=n)
        is_weekend = (day_of_week >= 5).astype(int)

        merchant_categories = rng.choice(MERCHANT_CATEGORIES, size=n)
        card_types = rng.choice(CARD_TYPES, size=n)

        # Legitimate: ~5% international
        is_international = (rng.random(n) < 0.05).astype(int)

        # Location: concentrated around typical home area
        latitudes = rng.normal(40.7, 5.0, size=n).clip(-90, 90)
        longitudes = rng.normal(-74.0, 8.0, size=n).clip(-180, 180)

        # Card/merchant IDs
        card_ids = rng.integers(1000, 5000, size=n)
        merchant_ids = rng.integers(100, 2000, size=n)

        return pd.DataFrame({
            "card_id": card_ids,
            "merchant_id": merchant_ids,
            "amount": amounts.round(2),
            "time": time_vals,
            "hour_of_day": hours,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "merchant_category": merchant_categories,
            "card_type": card_types,
            "is_international": is_international,
            "latitude": latitudes.round(4),
            "longitude": longitudes.round(4),
            "is_fraud": 0,
        })

    def _generate_fraudulent(self, n: int) -> pd.DataFrame:
        """Generate n fraudulent transactions."""
        rng = np.random.default_rng(self.random_seed + 1)

        # Amounts: higher, often round numbers or large values
        amounts = rng.lognormal(mean=4.5, sigma=1.2, size=n).clip(5.0, 10000.0)

        # Time: fraudulent activity heavier late-night (2–5 AM)
        hour_probs = np.array([
            0.040, 0.045, 0.060, 0.070, 0.060, 0.040,  # 0–5
            0.025, 0.020, 0.030, 0.035, 0.040, 0.045,  # 6–11
            0.050, 0.045, 0.040, 0.040, 0.040, 0.040,  # 12–17
            0.045, 0.045, 0.050, 0.055, 0.055, 0.055,  # 18–23
        ])
        hour_probs /= hour_probs.sum()
        hours = rng.choice(24, size=n, p=hour_probs)

        time_vals = rng.integers(0, 180 * 24 * 3600, size=n)
        day_of_week = rng.integers(0, 7, size=n)
        is_weekend = (day_of_week >= 5).astype(int)

        # Fraud skewed toward online and atm
        fraud_cat_probs = {
            "online_retail": 0.30, "atm": 0.20, "electronics": 0.15,
            "travel": 0.10, "restaurant": 0.08, "gas_station": 0.07,
            "clothing": 0.05, "entertainment": 0.03, "pharmacy": 0.01,
            "grocery": 0.01,
        }
        categories = list(fraud_cat_probs.keys())
        cat_probs = np.array(list(fraud_cat_probs.values()))
        merchant_categories = rng.choice(categories, size=n, p=cat_probs)

        card_types = rng.choice(CARD_TYPES, size=n)

        # Fraudulent: ~40% international
        is_international = (rng.random(n) < 0.40).astype(int)

        # Location: more varied (farther from home)
        latitudes = rng.uniform(-90, 90, size=n)
        longitudes = rng.uniform(-180, 180, size=n)

        # Fraud often uses a small set of compromised cards
        card_ids = rng.integers(1000, 5000, size=n)
        merchant_ids = rng.integers(100, 2000, size=n)

        return pd.DataFrame({
            "card_id": card_ids,
            "merchant_id": merchant_ids,
            "amount": amounts.round(2),
            "time": time_vals,
            "hour_of_day": hours,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "merchant_category": merchant_categories,
            "card_type": card_types,
            "is_international": is_international,
            "latitude": latitudes.round(4),
            "longitude": longitudes.round(4),
            "is_fraud": 1,
        })
