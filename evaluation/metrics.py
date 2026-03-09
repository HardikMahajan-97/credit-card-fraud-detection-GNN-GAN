"""
Evaluation metrics for credit card fraud detection.

Primary metric: AUPRC (Area Under Precision-Recall Curve) for imbalanced data.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth binary labels (N,).
        y_pred: Predicted binary labels (N,) at *threshold*.
        y_prob: Predicted fraud probabilities (N,). Required for AUC metrics.
        threshold: Decision threshold (default 0.5).

    Returns:
        Dictionary mapping metric names to float values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics: Dict[str, float] = {}

    # --- Basic classification ---
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    metrics["tn"] = float(cm[0, 0])
    metrics["fp"] = float(cm[0, 1])
    metrics["fn"] = float(cm[1, 0])
    metrics["tp"] = float(cm[1, 1])

    # --- Probability-based ---
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["auprc"] = float("nan")

        # Detection rate at various FPR thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        for target_fpr in [0.001, 0.01, 0.05, 0.1]:
            idx = np.searchsorted(fpr, target_fpr, side="right") - 1
            idx = max(0, min(idx, len(tpr) - 1))
            metrics[f"tpr_at_fpr_{int(target_fpr * 1000)}"] = float(tpr[idx])

    logger.info(
        f"Metrics — F1(w): {metrics.get('f1_weighted', 0):.4f} | "
        f"AUPRC: {metrics.get('auprc', float('nan')):.4f} | "
        f"ROC-AUC: {metrics.get('roc_auc', float('nan')):.4f}"
    )
    return metrics
