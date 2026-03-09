"""
Multi-source data acquisition for credit card fraud detection.

Supports downloading/loading data from:
  - Kaggle (creditcardfraud dataset)
  - HuggingFace Hub
  - Synthetic generation (always available)
  - Mixed (combination of all available sources)
"""

from __future__ import annotations

import pandas as pd
from typing import Optional

from data.synthetic_generator import SyntheticFraudGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

# Canonical column names expected throughout the pipeline
CANONICAL_COLUMNS = [
    "transaction_id",
    "card_id",
    "merchant_id",
    "amount",
    "time",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "merchant_category",
    "card_type",
    "is_international",
    "latitude",
    "longitude",
    "is_fraud",
]


def get_data(
    source: str = "synthetic",
    n_samples: int = 100_000,
    fraud_ratio: float = 0.017,
    random_seed: int = 42,
    kaggle_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Unified data acquisition interface.

    Args:
        source: One of "kaggle", "huggingface", "synthetic", or "mixed".
        n_samples: Number of samples for synthetic generation.
        fraud_ratio: Fraud ratio for synthetic data.
        random_seed: Random seed.
        kaggle_path: Path to pre-downloaded Kaggle CSV file (optional).

    Returns:
        DataFrame with canonical columns.
    """
    source = source.lower()
    logger.info(f"Loading data from source: {source}")

    if source == "synthetic":
        return _load_synthetic(n_samples, fraud_ratio, random_seed)
    elif source == "kaggle":
        return _load_kaggle(kaggle_path, random_seed)
    elif source == "huggingface":
        return _load_huggingface(random_seed)
    elif source == "mixed":
        return _load_mixed(n_samples, fraud_ratio, random_seed, kaggle_path)
    else:
        raise ValueError(
            f"Unknown data source '{source}'. "
            "Choose from: kaggle, huggingface, synthetic, mixed."
        )


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------


def _load_synthetic(
    n_samples: int, fraud_ratio: float, random_seed: int
) -> pd.DataFrame:
    """Generate and return synthetic transaction data."""
    gen = SyntheticFraudGenerator(
        n_samples=n_samples, fraud_ratio=fraud_ratio, random_seed=random_seed
    )
    df = gen.generate()
    logger.info(f"Synthetic data loaded: {len(df):,} rows, fraud={df['is_fraud'].mean():.3%}")
    return df


def _load_kaggle(path: Optional[str], random_seed: int) -> pd.DataFrame:
    """
    Load the Kaggle Credit Card Fraud Detection dataset.

    The dataset is available at:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

    If *path* is provided, load directly from disk; otherwise attempt to
    download via kagglehub.

    Args:
        path: Local CSV path (creditcard.csv). May be None.
        random_seed: Used when generating fallback synthetic data.

    Returns:
        DataFrame with canonical columns.
    """
    import os

    csv_path = path

    if csv_path is None or not os.path.exists(str(csv_path)):
        # Try kagglehub download
        try:
            import kagglehub  # type: ignore

            logger.info("Attempting Kaggle download via kagglehub …")
            dl_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
            import glob as _glob

            csvs = _glob.glob(os.path.join(dl_path, "*.csv"))
            if csvs:
                csv_path = csvs[0]
                logger.info(f"Downloaded to {csv_path}")
            else:
                raise FileNotFoundError("No CSV found after kagglehub download.")
        except Exception as exc:
            logger.warning(
                f"Kaggle download failed ({exc}). "
                "Falling back to synthetic data."
            )
            return _load_synthetic(100_000, 0.017, random_seed)

    logger.info(f"Loading Kaggle dataset from {csv_path}")
    raw = pd.read_csv(csv_path)
    return _normalize_kaggle(raw)


def _normalize_kaggle(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the raw Kaggle creditcardfraud CSV to canonical columns.

    The Kaggle dataset has columns: Time, V1–V28, Amount, Class.
    We map these to the canonical schema, generating placeholder values for
    fields not present in the original data.
    """
    import numpy as np

    n = len(raw)
    rng = np.random.default_rng(0)

    df = pd.DataFrame()
    df["transaction_id"] = [f"KGL{i:08d}" for i in range(n)]
    df["card_id"] = rng.integers(1000, 5000, size=n)
    df["merchant_id"] = rng.integers(100, 2000, size=n)
    df["amount"] = raw["Amount"].values
    df["time"] = raw["Time"].values.astype(int)

    hours = (raw["Time"].values / 3600) % 24
    df["hour_of_day"] = hours.astype(int)
    df["day_of_week"] = ((raw["Time"].values / 86400) % 7).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["merchant_category"] = rng.choice(
        ["grocery", "online_retail", "atm", "restaurant", "gas_station"], size=n
    )
    df["card_type"] = rng.choice(["visa", "mastercard", "amex", "discover"], size=n)
    df["is_international"] = (rng.random(n) < 0.10).astype(int)
    df["latitude"] = rng.normal(40.7, 5.0, size=n).clip(-90, 90).round(4)
    df["longitude"] = rng.normal(-74.0, 8.0, size=n).clip(-180, 180).round(4)
    df["is_fraud"] = raw["Class"].values.astype(int)

    # Carry over PCA features for richer representation
    for col in raw.columns:
        if col.startswith("V"):
            df[col] = raw[col].values

    return df


def _load_huggingface(random_seed: int) -> pd.DataFrame:
    """
    Load a fraud dataset from HuggingFace Hub.

    Falls back to synthetic data if the Hub is unavailable.

    Returns:
        DataFrame with canonical columns.
    """
    try:
        from datasets import load_dataset  # type: ignore

        logger.info("Loading fraud dataset from HuggingFace Hub …")
        ds = load_dataset("iemilky/credit_card_transaction_fraud_detection", split="train")
        raw = ds.to_pandas()
        return _normalize_huggingface(raw, random_seed)
    except Exception as exc:
        logger.warning(
            f"HuggingFace load failed ({exc}). Falling back to synthetic data."
        )
        return _load_synthetic(100_000, 0.017, random_seed)


def _normalize_huggingface(raw: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """
    Normalise a raw HuggingFace fraud DataFrame to canonical columns.

    Column names may vary by dataset; this function attempts common mappings.
    """
    import numpy as np

    n = len(raw)
    rng = np.random.default_rng(random_seed)
    df = pd.DataFrame()

    df["transaction_id"] = [f"HF{i:08d}" for i in range(n)]

    # Flexible column mapping
    def _pick(candidates: list[str], default: object = None) -> object:
        for c in candidates:
            if c in raw.columns:
                return raw[c]
        if default is not None:
            return default
        return pd.Series(np.zeros(n))

    df["card_id"] = _pick(["cc_num", "card_id", "customer_id"], rng.integers(1000, 5000, n))
    df["merchant_id"] = _pick(["merchant", "merchant_id"], rng.integers(100, 2000, n))
    df["amount"] = _pick(["amt", "amount", "trans_amount"], rng.lognormal(3.5, 1.0, n))
    df["time"] = _pick(["unix_time", "time", "trans_date_trans_time"], rng.integers(0, 10**9, n))
    df["hour_of_day"] = _pick(["hour", "hour_of_day"], rng.integers(0, 24, n))
    df["day_of_week"] = _pick(["day_of_week"], rng.integers(0, 7, n))
    df["is_weekend"] = (pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0) >= 5).astype(int)
    df["merchant_category"] = _pick(["category", "merchant_category"], rng.choice(["grocery", "online_retail"], n))
    df["card_type"] = _pick(["card_type"], rng.choice(["visa", "mastercard"], n))
    df["is_international"] = _pick(["is_international"], (rng.random(n) < 0.10).astype(int))
    df["latitude"] = _pick(["lat", "latitude"], rng.normal(40.7, 5.0, n))
    df["longitude"] = _pick(["long", "longitude"], rng.normal(-74.0, 8.0, n))
    df["is_fraud"] = _pick(["is_fraud", "fraud", "label", "Class"], np.zeros(n)).astype(int)

    # Numeric conversion & cleanup
    for col in ["amount", "latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _load_mixed(
    n_samples: int, fraud_ratio: float, random_seed: int, kaggle_path: Optional[str]
) -> pd.DataFrame:
    """
    Combine all available data sources into a single DataFrame.

    Always includes synthetic data; Kaggle and HuggingFace are included if
    available. The combined dataset is deduplicated and shuffled.
    """
    frames = []

    # Synthetic — always available
    frames.append(_load_synthetic(n_samples // 2, fraud_ratio, random_seed))

    # Kaggle — best-effort
    try:
        kdf = _load_kaggle(kaggle_path, random_seed)
        frames.append(kdf)
        logger.info("Kaggle data included in mixed source.")
    except Exception as exc:
        logger.warning(f"Kaggle skipped in mixed mode: {exc}")

    # HuggingFace — best-effort
    try:
        hdf = _load_huggingface(random_seed)
        frames.append(hdf)
        logger.info("HuggingFace data included in mixed source.")
    except Exception as exc:
        logger.warning(f"HuggingFace skipped in mixed mode: {exc}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)
    combined = combined.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Mixed data: {len(combined):,} rows, fraud={combined['is_fraud'].mean():.3%}"
    )
    return combined
