"""
Real-time transaction inference engine.

Maintains a sliding window of recent transactions, incrementally updates the
transaction graph, and performs mini-batch inference for efficiency.
Thread-safe for concurrent requests.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
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
        self._card_seen_merchants: Dict[str, set] = {}
        self._card_timestamps: Dict[str, datetime] = {}

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
            window_snapshot = list(self._window)
            seen_snapshot = {k: set(v) for k, v in self._card_seen_merchants.items()}
            ts_snapshot = dict(self._card_timestamps)
        result = self._score_single(
            txn,
            window_snapshot=window_snapshot,
            seen_merchants=seen_snapshot,
            last_timestamps=ts_snapshot,
        )
        with self._lock:
            card_id = str(txn.get("card_id", ""))
            merchant_id = str(txn.get("merchant_id", ""))
            if card_id:
                self._card_seen_merchants.setdefault(card_id, set()).add(merchant_id)
                ts = self._parse_ts(txn)
                if ts is not None:
                    self._card_timestamps[card_id] = ts
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

    def _score_single(
        self,
        txn: Dict[str, Any],
        window_snapshot: Optional[List[Dict[str, Any]]] = None,
        seen_merchants: Optional[Dict[str, set]] = None,
        last_timestamps: Optional[Dict[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Compute fraud score for a single transaction."""
        txn_id = txn.get("transaction_id", "unknown")
        gan_score = 0.5
        gnn_prob = 0.5

        # --- GAN anomaly score ---
        if self.gan_discriminator is not None:
            try:
                feat = self._extract_features(
                    txn,
                    window_snapshot=window_snapshot,
                    seen_merchants=seen_merchants or {},
                    last_timestamps=last_timestamps or {},
                )
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

    def _extract_features(
        self,
        txn: Dict[str, Any],
        window_snapshot: Optional[List[Dict[str, Any]]] = None,
        seen_merchants: Optional[Dict[str, set]] = None,
        last_timestamps: Optional[Dict[str, datetime]] = None,
    ) -> np.ndarray:
        """Extract numeric feature array from a transaction dict."""
        ts = self._parse_ts(txn)
        amount = float(txn.get("amount", 0.0))
        card_id = str(txn.get("card_id", ""))
        merchant_id = str(txn.get("merchant_id", ""))
        window_data = window_snapshot if window_snapshot is not None else list(self._window)
        seen_map = seen_merchants if seen_merchants is not None else self._card_seen_merchants
        ts_map = last_timestamps if last_timestamps is not None else self._card_timestamps

        # rolling mean over last 5 transaction amounts for the same card
        recent_amounts = []
        if card_id:
            for t in window_data[-5:]:
                if str(t.get("card_id", "")) == card_id:
                    recent_amounts.append(float(t.get("amount", 0.0)))
        rolling_mean = float(np.mean(recent_amounts)) if recent_amounts else amount
        amount_delta = float(np.log1p(abs(amount - rolling_mean)))

        # seconds since previous transaction for card, clipped to 7 days
        prev_ts = ts_map.get(card_id) if card_id else None
        if ts is not None and prev_ts is not None:
            secs = min(max((ts - prev_ts).total_seconds(), 0.0), 604800.0)
        else:
            secs = 604800.0
        secs_feat = float(np.log1p(secs))

        # merchant novelty per card
        known_merchants = seen_map.get(card_id, set()) if card_id else set()
        is_new_merchant = 1.0 if merchant_id and merchant_id not in known_merchants else 0.0

        # burst count in last 30 minutes normalized by 10 and clipped to 1
        burst_count = 0
        if ts is not None and card_id:
            for t in window_data:
                if str(t.get("card_id", "")) != card_id:
                    continue
                t_ts = self._parse_ts(t)
                if t_ts is None:
                    continue
                if 0.0 <= (ts - t_ts).total_seconds() <= 1800.0:
                    burst_count += 1
        burst_norm = min(float(burst_count) / 10.0, 1.0)

        hour = ts.hour if ts is not None else int(txn.get("hour_of_day", 0))
        dow = ts.weekday() if ts is not None else int(txn.get("day_of_week", 0))
        return np.array(
            [
                np.log1p(amount),
                float(txn.get("is_international", 0)),
                np.sin(2 * np.pi * hour / 24.0),
                np.cos(2 * np.pi * hour / 24.0),
                np.sin(2 * np.pi * dow / 7.0),
                np.cos(2 * np.pi * dow / 7.0),
                amount_delta,
                secs_feat,
                is_new_merchant,
                burst_norm,
            ],
            dtype=np.float32,
        )

    def _parse_ts(self, txn: Dict[str, Any]) -> Optional[datetime]:
        ts = txn.get("timestamp")
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        try:
            return datetime.fromisoformat(str(ts))
        except Exception:
            return None
