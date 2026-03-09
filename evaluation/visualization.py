"""
Visualization utilities for fraud detection model analysis.

Generates:
  - ROC curve
  - Precision-Recall curve
  - Confusion matrix heatmap
  - Training loss curves (GAN and GNN)
  - t-SNE embedding visualization
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Matplotlib imports are lazy to avoid failures in headless environments
_plt = None
_sns = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_sns():
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns


class Visualizer:
    """
    Create and save diagnostic plots.

    Args:
        output_dir: Directory to save plots.
        dpi: Image resolution.
    """

    def __init__(self, output_dir: str = "plots", dpi: int = 150) -> None:
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "roc_curve.png",
    ) -> None:
        """Plot and save the ROC curve."""
        from sklearn.metrics import roc_curve, roc_auc_score

        plt = _get_plt()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic")
        ax.legend(loc="lower right")
        self._save(fig, filename)

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        filename: str = "pr_curve.png",
    ) -> None:
        """Plot and save the Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        plt = _get_plt()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.step(recall, precision, where="post", lw=2, label=f"AUPRC = {ap:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        self._save(fig, filename)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        filename: str = "confusion_matrix.png",
    ) -> None:
        """Plot and save the confusion matrix heatmap."""
        from sklearn.metrics import confusion_matrix

        plt = _get_plt()
        sns = _get_sns()
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"],
            ax=ax,
        )
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")
        self._save(fig, filename)

    def plot_gan_loss_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "gan_loss.png",
    ) -> None:
        """Plot GAN training loss curves (G_loss, D_loss, GP)."""
        plt = _get_plt()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for ax, key, title in zip(
            axes,
            ["g_loss", "d_loss", "gradient_penalty"],
            ["Generator Loss", "Discriminator Loss", "Gradient Penalty"],
        ):
            if key in history:
                ax.plot(history[key], lw=2)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")

        plt.tight_layout()
        self._save(fig, filename)

    def plot_gnn_loss_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "gnn_loss.png",
    ) -> None:
        """Plot GNN training loss curves (CE, EWC, replay, val)."""
        plt = _get_plt()
        fig, ax = plt.subplots(figsize=(9, 5))

        for key, label in [
            ("train_loss", "Train (CE+EWC)"),
            ("val_loss", "Validation"),
            ("ewc_penalty", "EWC Penalty"),
            ("replay_loss", "Replay Loss"),
        ]:
            if key in history and history[key]:
                ax.plot(history[key], label=label, lw=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("GNN Training Curves")
        ax.legend()
        self._save(fig, filename)

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        filename: str = "tsne.png",
        perplexity: int = 30,
    ) -> None:
        """
        Plot t-SNE of transaction embeddings coloured by label.

        Args:
            embeddings: Feature matrix (N, D).
            labels: Integer labels (N,).
            filename: Output filename.
            perplexity: t-SNE perplexity parameter.
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("sklearn not available; skipping t-SNE plot.")
            return

        plt = _get_plt()

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="coolwarm", alpha=0.6, s=10
        )
        plt.colorbar(scatter, ax=ax, label="Fraud Label")
        ax.set_title("t-SNE of Transaction Embeddings")
        self._save(fig, filename)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save(self, fig, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        _get_plt().close(fig)
        logger.info(f"Plot saved to {path}")
