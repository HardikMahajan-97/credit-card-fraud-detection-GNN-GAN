"""
Ensemble fusion layer combining GAN anomaly scores and GNN fraud probabilities.

Fusion strategies:
  - weighted: fixed linear combination
  - learned: small MLP trained on validation data
  - stacking: logistic regression meta-learner

Includes Platt scaling calibration for probability outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.logger import get_logger

logger = get_logger(__name__)


class _LearnedFusion(nn.Module):
    """Two-layer MLP fusion network."""

    def __init__(self, input_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EnsembleModel:
    """
    Fuse GAN anomaly scores and GNN fraud probabilities.

    Args:
        fusion_method: One of "weighted", "learned", "stacking".
        gan_weight: Weight for GAN anomaly score (weighted fusion only).
        gnn_weight: Weight for GNN probability (weighted fusion only).
        calibration: Whether to apply Platt scaling calibration.
        device: Compute device.
    """

    def __init__(
        self,
        fusion_method: str = "learned",
        gan_weight: float = 0.3,
        gnn_weight: float = 0.7,
        calibration: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.fusion_method = fusion_method
        self.gan_weight = gan_weight
        self.gnn_weight = gnn_weight
        self.calibration = calibration
        self.device = device or torch.device("cpu")

        self._fusion_net: Optional[_LearnedFusion] = None
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._stacking_model = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        gan_scores: np.ndarray,
        gnn_probs: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
    ) -> None:
        """
        Fit the fusion layer on validation data.

        Args:
            gan_scores: GAN anomaly scores (N,) — higher = more anomalous.
            gnn_probs: GNN fraud probabilities (N,) — in [0, 1].
            labels: Binary ground-truth labels (N,).
            epochs: Training epochs for learned fusion.
        """
        logger.info(f"Fitting ensemble ({self.fusion_method}) on {len(labels)} samples …")
        features = self._prepare_features(gan_scores, gnn_probs)

        if self.fusion_method == "learned":
            self._fit_learned(features, labels, epochs)
        elif self.fusion_method == "stacking":
            self._fit_stacking(features, labels)
        # weighted: no fitting required

        if self.calibration:
            raw = self._predict_raw(gan_scores, gnn_probs)
            self._fit_platt(raw, labels)

        self._fitted = True
        logger.info("Ensemble fitting complete.")

    def predict(
        self, gan_score: float, gnn_prob: float
    ) -> Tuple[float, float]:
        """
        Predict fraud probability for a single transaction.

        Args:
            gan_score: GAN anomaly score.
            gnn_prob: GNN fraud probability.

        Returns:
            (fraud_probability, raw_score)
        """
        raw = self._predict_raw(
            np.array([gan_score]), np.array([gnn_prob])
        )[0]
        prob = self._calibrate(raw) if self.calibration and self._fitted else raw
        return float(prob), float(raw)

    def predict_batch(
        self, gan_scores: np.ndarray, gnn_probs: np.ndarray
    ) -> np.ndarray:
        """
        Predict fraud probabilities for a batch.

        Args:
            gan_scores: Array of GAN anomaly scores (N,).
            gnn_probs: Array of GNN fraud probabilities (N,).

        Returns:
            Array of fraud probabilities (N,).
        """
        raw = self._predict_raw(gan_scores, gnn_probs)
        if self.calibration and self._fitted:
            return np.array([self._calibrate(s) for s in raw])
        return raw

    def calibrate(
        self, gan_scores: np.ndarray, gnn_probs: np.ndarray, labels: np.ndarray
    ) -> None:
        """Fit Platt scaling on held-out validation data."""
        raw = self._predict_raw(gan_scores, gnn_probs)
        self._fit_platt(raw, labels)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_features(
        self, gan_scores: np.ndarray, gnn_probs: np.ndarray
    ) -> np.ndarray:
        gan_norm = (gan_scores - gan_scores.min()) / (gan_scores.ptp() + 1e-8)
        return np.stack([gan_norm, gnn_probs], axis=1).astype(np.float32)

    def _predict_raw(
        self, gan_scores: np.ndarray, gnn_probs: np.ndarray
    ) -> np.ndarray:
        gan_norm = (gan_scores - gan_scores.min()) / (gan_scores.ptp() + 1e-8)

        if self.fusion_method == "weighted":
            total = self.gan_weight + self.gnn_weight
            return (self.gan_weight * gan_norm + self.gnn_weight * gnn_probs) / total

        elif self.fusion_method == "learned" and self._fusion_net is not None:
            feats = torch.tensor(
                np.stack([gan_norm, gnn_probs], axis=1), dtype=torch.float32
            ).to(self.device)
            self._fusion_net.eval()
            with torch.no_grad():
                out = self._fusion_net(feats).cpu().numpy()
            return out

        elif self.fusion_method == "stacking" and self._stacking_model is not None:
            feats = np.stack([gan_norm, gnn_probs], axis=1)
            return self._stacking_model.predict_proba(feats)[:, 1]

        # Default: weighted average
        total = self.gan_weight + self.gnn_weight
        return (self.gan_weight * gan_norm + self.gnn_weight * gnn_probs) / total

    def _fit_learned(
        self, features: np.ndarray, labels: np.ndarray, epochs: int
    ) -> None:
        net = _LearnedFusion(input_dim=features.shape[1]).to(self.device)
        opt = optim.Adam(net.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        net.train()
        for _ in range(epochs):
            opt.zero_grad()
            preds = net(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()

        self._fusion_net = net

    def _fit_stacking(
        self, features: np.ndarray, labels: np.ndarray
    ) -> None:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(features, labels)
        self._stacking_model = lr

    def _fit_platt(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit Platt scaling (logistic) on raw scores."""
        from scipy.optimize import minimize  # type: ignore

        def neg_log_likelihood(params: np.ndarray) -> float:
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

        result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="L-BFGS-B")
        self._platt_a, self._platt_b = result.x
        logger.debug(f"Platt scaling: a={self._platt_a:.4f}, b={self._platt_b:.4f}")

    def _calibrate(self, score: float) -> float:
        return float(1.0 / (1.0 + np.exp(-(self._platt_a * score + self._platt_b))))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        d: Dict = {
            "fusion_method": self.fusion_method,
            "gan_weight": self.gan_weight,
            "gnn_weight": self.gnn_weight,
            "calibration": self.calibration,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "fitted": self._fitted,
        }
        if self._fusion_net is not None:
            d["fusion_net_state"] = self._fusion_net.state_dict()
        return d

    def load_state_dict(self, state: Dict) -> None:
        self.fusion_method = state["fusion_method"]
        self.gan_weight = state["gan_weight"]
        self.gnn_weight = state["gnn_weight"]
        self.calibration = state["calibration"]
        self._platt_a = state.get("platt_a", 1.0)
        self._platt_b = state.get("platt_b", 0.0)
        self._fitted = state.get("fitted", False)
        if "fusion_net_state" in state:
            self._fusion_net = _LearnedFusion()
            self._fusion_net.load_state_dict(state["fusion_net_state"])
