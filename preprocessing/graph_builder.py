"""
Transaction graph builder for the GNN module.

Constructs PyTorch Geometric graphs from transaction DataFrames.
Each graph is a bipartite structure:
  - Card/account nodes
  - Merchant nodes
  - Transaction edges with temporal and monetary features
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import torch
    from torch_geometric.data import Data, HeteroData  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("torch_geometric not available; GraphBuilder will return None graphs.")


class GraphBuilder:
    """
    Build transaction graphs from a DataFrame of transactions.

    Args:
        window_size: Number of transactions per graph window.
        max_neighbors: Maximum edges per node (for sparse graphs).
        temporal_encoding_dim: Dimension of sinusoidal temporal encoding.
    """

    def __init__(
        self,
        window_size: int = 1000,
        max_neighbors: int = 50,
        temporal_encoding_dim: int = 16,
    ) -> None:
        self.window_size = window_size
        self.max_neighbors = max_neighbors
        self.temporal_encoding_dim = temporal_encoding_dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_graphs(self, df: pd.DataFrame) -> List:
        """
        Partition *df* into windows and build one PyG Data object per window.

        Args:
            df: Preprocessed transaction DataFrame.

        Returns:
            List of PyG Data objects (or dicts if PyG unavailable).
        """
        if not _HAS_PYG:
            logger.error("torch_geometric required for graph construction.")
            return []

        df = df.sort_values("time").reset_index(drop=True) if "time" in df.columns else df
        graphs = []
        n_windows = max(1, len(df) // self.window_size)

        for i in range(n_windows):
            start = i * self.window_size
            end = min(start + self.window_size, len(df))
            window_df = df.iloc[start:end].reset_index(drop=True)
            graph = self._build_single_graph(window_df)
            if graph is not None:
                graphs.append(graph)

        logger.info(f"Built {len(graphs)} graphs from {len(df):,} transactions")
        return graphs

    def build_single(self, df: pd.DataFrame) -> Optional[object]:
        """Build a single graph from the entire DataFrame."""
        return self._build_single_graph(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_single_graph(self, df: pd.DataFrame) -> Optional[object]:
        """Convert a window DataFrame into a PyG Data object."""
        if not _HAS_PYG or len(df) == 0:
            return None

        import torch

        # ------ Node construction -------
        # Card nodes: unique card_ids
        card_ids = df["card_id"].unique() if "card_id" in df.columns else np.array([0])
        merchant_ids = df["merchant_id"].unique() if "merchant_id" in df.columns else np.array([0])

        n_cards = len(card_ids)
        n_merchants = len(merchant_ids)

        card_idx_map = {cid: i for i, cid in enumerate(card_ids)}
        merchant_idx_map = {mid: i + n_cards for i, mid in enumerate(merchant_ids)}

        # Card node features: [avg_amount, txn_count]
        card_feats = []
        for cid in card_ids:
            subset = df[df["card_id"] == cid] if "card_id" in df.columns else df
            card_feats.append([
                float(subset["amount"].mean()) if "amount" in subset.columns else 0.0,
                float(len(subset)),
            ])

        # Merchant node features: [avg_amount, txn_count]
        merchant_feats = []
        for mid in merchant_ids:
            subset = df[df["merchant_id"] == mid] if "merchant_id" in df.columns else df
            merchant_feats.append([
                float(subset["amount"].mean()) if "amount" in subset.columns else 0.0,
                float(len(subset)),
            ])

        # Pad to same width
        node_feats = card_feats + merchant_feats
        x = torch.tensor(node_feats, dtype=torch.float)

        # ------ Edge construction -------
        src_list, dst_list = [], []
        edge_feats = []
        label_list = []

        for _, row in df.iterrows():
            cid = row.get("card_id", 0)
            mid = row.get("merchant_id", 0)
            src = card_idx_map.get(cid, 0)
            dst = merchant_idx_map.get(mid, n_cards)

            src_list.append(src)
            dst_list.append(dst)
            # Also add reverse edge for undirected graph
            src_list.append(dst)
            dst_list.append(src)

            amount = float(row.get("amount", 0.0))
            time_enc = self._temporal_encode(float(row.get("time", 0)))
            is_intl = float(row.get("is_international", 0))

            edge_feat = np.concatenate([[amount, is_intl], time_enc])
            edge_feats.append(edge_feat)
            edge_feats.append(edge_feat)  # reverse edge same features

            label_list.append(int(row.get("is_fraud", 0)))
            label_list.append(int(row.get("is_fraud", 0)))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float)
        y = torch.tensor(
            df["is_fraud"].values if "is_fraud" in df.columns else np.zeros(len(df)),
            dtype=torch.long,
        )

        # Graph label: 1 if any fraud in window
        graph_label = int(y.max().item()) if len(y) > 0 else 0

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            graph_label=torch.tensor(graph_label, dtype=torch.long),
            num_nodes=n_cards + n_merchants,
        )
        return data

    def _temporal_encode(self, t: float) -> np.ndarray:
        """Sinusoidal positional encoding for time values."""
        d = self.temporal_encoding_dim
        enc = np.zeros(d)
        for i in range(d // 2):
            div_term = 10000 ** (2 * i / d)
            enc[2 * i] = np.sin(t / div_term)
            enc[2 * i + 1] = np.cos(t / div_term)
        return enc
