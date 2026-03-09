"""
PyTorch Dataset and DataLoader classes for credit card fraud detection.

Provides:
  - FraudDataset: tabular dataset for GAN and MLP-based models.
  - FraudGraphDataset: graph-structured dataset for GNN models (PyG-compatible).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger(__name__)

# Try to import PyTorch Geometric; graph dataset degrades gracefully if missing.
try:
    from torch_geometric.data import Data  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("torch_geometric not available; FraudGraphDataset will be limited.")


class FraudDataset(Dataset):
    """
    PyTorch Dataset for tabular transaction features.

    Args:
        features: 2-D float array of shape (N, D).
        labels: 1-D integer array of shape (N,) with 0/1 labels.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def feature_dim(self) -> int:
        """Input feature dimensionality."""
        return self.features.shape[1]


class FraudGraphDataset(Dataset):
    """
    Dataset that wraps a list of PyTorch Geometric Data objects.

    Each element is a graph representing a window of transactions.

    Args:
        graph_list: List of PyG Data objects.
        labels: Graph-level integer labels (0 = no fraud, 1 = fraud present).
    """

    def __init__(self, graph_list: list, labels: Optional[np.ndarray] = None) -> None:
        if not _HAS_PYG:
            raise ImportError(
                "torch_geometric is required for FraudGraphDataset. "
                "Install it with: pip install torch-geometric"
            )
        self.graph_list = graph_list
        self.labels = (
            torch.tensor(labels, dtype=torch.long)
            if labels is not None
            else torch.zeros(len(graph_list), dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.graph_list)

    def __getitem__(self, idx: int):
        graph = self.graph_list[idx]
        graph.graph_label = self.labels[idx]
        return graph


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_fraud_datasets(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "is_fraud",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> Tuple[FraudDataset, FraudDataset, FraudDataset]:
    """
    Split a DataFrame into stratified train / val / test FraudDatasets.

    Args:
        df: Full dataset DataFrame.
        feature_cols: Columns to use as input features.
        label_col: Binary label column.
        test_size: Fraction for test split.
        val_size: Fraction for validation split (from the remaining train data).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    # val_size is relative to the full dataset; adjust to be relative to train_val
    adjusted_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val, stratify=y_train_val, random_state=random_seed
    )

    logger.info(
        f"Dataset split — train: {len(y_train):,}, val: {len(y_val):,}, test: {len(y_test):,}"
    )

    return FraudDataset(X_train, y_train), FraudDataset(X_val, y_val), FraudDataset(X_test, y_test)


def get_dataloader(
    dataset: FraudDataset,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Wrap a FraudDataset in a DataLoader.

    Args:
        dataset: FraudDataset instance.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of DataLoader worker processes.

    Returns:
        DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
