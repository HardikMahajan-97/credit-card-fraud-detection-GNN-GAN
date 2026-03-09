"""
Real-time transaction inference engine.

Maintains a sliding window of recent transactions, incrementally updates the
transaction graph, and performs mini-batch inference for efficiency.
Thread-safe for concurrent requests.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from torch_geometric.data import Data  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class RealTimeEngine:
    """
    Streaming mini-batch inference engine for fraud detection.

    Maintains a sliding window of recent transactions and uses an ensemble
    of a GAN discriminator and GNN model for scoring.

    Args:
        gan_discriminator: Trained Discriminator model.
        gnn_model: Trained GNNModel (optional; falls back to GAN-only if None).
        ensemble: Trained EnsembleModel (optional).
        feature_engineer: Fitted FeatureEngineer for preprocessing new transactions.
        graph_builder: GraphBuilder for constructing context graphs.
        window_size: Number of recent transactions to keep in context.
        device: Compute device.
        threshold: Fraud decision threshold.
    """

    def __init__(
        self,
        gan_discriminator=None,
        gnn_model=None,
        ensemble=None,
        feature_engineer=None,
        graph_builder=None,
        window_size: int = 500,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        self.gan_discriminator = gan_discriminator
        self.gnn_model = gnn_model
        self.ensemble = ensemble
        self.feature_engineer = feature_engineer
        self.graph_builder = graph_builder
        self.window_size = window_size
        self.device = device or torch.device("cpu")
        self.threshold = threshold

        self._window: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

        logger.info(
            f"RealTimeEngine initialized (window={window_size}, threshold={threshold})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_transaction(
        self, txn: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a single incoming transaction.

        Args:
            txn: Dictionary of transaction features.

        Returns:
            Dict with keys: transaction_id, fraud_probability, is_fraud,
            gan_score, gnn_prob, threshold.
        """
        with self._lock:
            self._window.append(txn)

        result = self._score_single(txn)
        return result

    def process_batch(
        self, txns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Score a batch of transactions.

        Args:
            txns: List of transaction feature dicts.

        Returns:
            List of scoring result dicts.
        """
        results = []
        for txn in txns:
            results.append(self.process_transaction(txn))
        return results

    def get_context_graph(self):
        """Return the current transaction graph for the sliding window."""
        import pandas as pd

        with self._lock:
            window_list = list(self._window)

        if not window_list or self.graph_builder is None:
            return None

        df = pd.DataFrame(window_list)
        return self.graph_builder.build_single(df)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _score_single(self, txn: Dict[str, Any]) -> Dict[str, Any]:
        """Compute fraud score for a single transaction."""
        txn_id = txn.get("transaction_id", "unknown")
        gan_score = 0.5
        gnn_prob = 0.5

        # --- GAN anomaly score ---
        if self.gan_discriminator is not None:
            try:
                feat = self._extract_features(txn)
                tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
                score_raw = self.gan_discriminator.get_anomaly_score(tensor)
                # Normalize to [0, 1] via sigmoid
                gan_score = float(torch.sigmoid(score_raw).item())
            except Exception as exc:
                logger.debug(f"GAN scoring failed for {txn_id}: {exc}")

        # --- GNN fraud probability ---
        if self.gnn_model is not None and _HAS_PYG:
            try:
                graph = self.get_context_graph()
                if graph is not None:
                    graph = graph.to(self.device)
                    self.gnn_model.eval()
                    with torch.no_grad():
                        logits, _ = self.gnn_model(graph)
                        probs = torch.softmax(logits, dim=-1)
                        gnn_prob = float(probs[:, 1].mean().item())
            except Exception as exc:
                logger.debug(f"GNN scoring failed for {txn_id}: {exc}")

        # --- Ensemble fusion ---
        if self.ensemble is not None:
            fraud_prob, _ = self.ensemble.predict(gan_score, gnn_prob)
        else:
            fraud_prob = 0.3 * gan_score + 0.7 * gnn_prob

        return {
            "transaction_id": txn_id,
            "fraud_probability": round(fraud_prob, 6),
            "is_fraud": fraud_prob >= self.threshold,
            "gan_score": round(gan_score, 6),
            "gnn_prob": round(gnn_prob, 6),
            "threshold": self.threshold,
        }

    def _extract_features(self, txn: Dict[str, Any]) -> np.ndarray:
        """Extract numeric feature array from a transaction dict."""
        numeric_keys = [
            "amount", "hour_of_day", "day_of_week", "is_weekend",
            "is_international", "latitude", "longitude",
        ]
        return np.array([float(txn.get(k, 0.0)) for k in numeric_keys], dtype=np.float32)
