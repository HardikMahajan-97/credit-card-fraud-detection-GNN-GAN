"""
Full GNN model for transaction fraud detection.

Architecture:
  Input → TemporalEncoding → GraphSAGE (×2) → GAT (×2, multi-head) → MLP classifier

Includes skip connections between layers and global graph pooling.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gnn.layers import GATConv, GraphSAGEConv, TemporalEncoding

try:
    from torch_geometric.nn import global_mean_pool  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

    def global_mean_pool(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Fallback: mean over all nodes."""
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, x.size(-1), device=x.device)
        count = torch.zeros(n_graphs, 1, device=x.device)
        for i in range(x.size(0)):
            out[batch[i]] += x[i]
            count[batch[i]] += 1
        return out / count.clamp(min=1)


class GNNModel(nn.Module):
    """
    GraphSAGE + GAT hybrid model for node-level fraud detection.

    Args:
        input_dim: Node feature dimensionality.
        hidden_dim: Hidden representation size.
        num_sage_layers: Number of GraphSAGE layers.
        num_gat_layers: Number of GAT layers.
        gat_heads: Number of GAT attention heads.
        num_classes: Number of output classes (2 for fraud/not-fraud).
        dropout: Dropout probability.
        temporal_dim: Temporal encoding dimensionality (0 to disable).
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 128,
        num_sage_layers: int = 2,
        num_gat_layers: int = 2,
        gat_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        temporal_dim: int = 0,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.temporal_dim = temporal_dim
        effective_input = input_dim + temporal_dim

        # ------ Input projection ------
        self.input_proj = nn.Linear(effective_input, hidden_dim)

        # ------ GraphSAGE layers ------
        self.sage_layers = nn.ModuleList()
        for _ in range(num_sage_layers):
            self.sage_layers.append(GraphSAGEConv(hidden_dim, hidden_dim))

        # ------ GAT layers ------
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            in_c = hidden_dim
            # All but last GAT layer: concat heads
            if i < num_gat_layers - 1:
                self.gat_layers.append(
                    GATConv(in_c, hidden_dim // gat_heads, heads=gat_heads, concat=True)
                )
            else:
                self.gat_layers.append(
                    GATConv(in_c, hidden_dim, heads=gat_heads, concat=False)
                )

        # ------ Skip connections ------
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_sage_layers + num_gat_layers)
        ])

        # ------ MLP classifier (node-level) ------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # ------ Graph-level pooling + MLP ------
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        if temporal_dim > 0:
            self.temporal_enc = TemporalEncoding(temporal_dim)
        else:
            self.temporal_enc = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        data,  # PyG Data object or a dict with x, edge_index, batch keys
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data object with attributes:
                  x (N, input_dim), edge_index (2, E), batch (N,) [optional].

        Returns:
            (node_logits, graph_embedding):
              node_logits: (N, num_classes) fraud logits per node.
              graph_embedding: (B, hidden_dim) per-graph embedding.
        """
        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)

        # Optional temporal encoding
        if self.temporal_enc is not None and hasattr(data, "t"):
            t_enc = self.temporal_enc(data.t)
            x = torch.cat([x, t_enc], dim=-1)

        # Input projection
        h = F.relu(self.input_proj(x))
        skip_idx = 0

        # GraphSAGE layers
        for sage_layer in self.sage_layers:
            h_new = sage_layer(h, edge_index)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new + self.skip_projs[skip_idx](h)  # skip connection
            h = F.relu(h)
            skip_idx += 1

        # GAT layers
        for gat_layer in self.gat_layers:
            h_new = gat_layer(h, edge_index)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            if h_new.shape == h.shape:
                h = h_new + self.skip_projs[skip_idx](h)
            else:
                h = h_new
            h = F.relu(h)
            skip_idx += 1

        # Node-level predictions
        node_logits = self.classifier(h)

        # Graph-level embedding via global pooling
        graph_emb = global_mean_pool(h, batch)

        return node_logits, graph_emb

    def predict_proba(self, data) -> torch.Tensor:
        """Return node-level fraud probabilities."""
        logits, _ = self.forward(data)
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
