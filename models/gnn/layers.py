"""
Custom GNN layers: GraphSAGE, GAT, and TemporalEncoding.

These layers are used by the GNNModel for node-level feature aggregation
in transaction graphs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing  # type: ignore
    from torch_geometric.utils import add_self_loops, softmax  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    MessagePassing = object  # type: ignore


# ---------------------------------------------------------------------------
# Temporal Encoding
# ---------------------------------------------------------------------------


class TemporalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transaction timestamps.

    Encodes the time dimension into a fixed-size embedding that can be
    concatenated with node/edge features.

    Args:
        encoding_dim: Output embedding dimensionality (must be even).
    """

    def __init__(self, encoding_dim: int = 16) -> None:
        super().__init__()
        self.encoding_dim = encoding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of timestamps.

        Args:
            t: Timestamp tensor of shape (N,) or (N, 1).

        Returns:
            Encoded tensor of shape (N, encoding_dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.float()

        d = self.encoding_dim
        div_term = torch.exp(
            torch.arange(0, d, 2, dtype=torch.float, device=t.device)
            * -(math.log(10000.0) / d)
        )
        enc = torch.zeros(t.size(0), d, device=t.device)
        enc[:, 0::2] = torch.sin(t * div_term)
        enc[:, 1::2] = torch.cos(t * div_term)
        return enc


# ---------------------------------------------------------------------------
# GraphSAGE Layer
# ---------------------------------------------------------------------------


class GraphSAGEConv(nn.Module if not _HAS_PYG else MessagePassing):  # type: ignore[misc]
    """
    GraphSAGE convolution with mean aggregation.

    Concatenates self-representation with neighbour mean, then applies a
    linear projection.

    Args:
        in_channels: Input feature dimensionality.
        out_channels: Output feature dimensionality.
        normalize: Whether to L2-normalise the output.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = True,
    ) -> None:
        if _HAS_PYG:
            super().__init__(aggr="mean")
        else:
            super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_channels).
            edge_index: Edge indices (2, E).

        Returns:
            Updated node features (N, out_channels).
        """
        if _HAS_PYG:
            return self._pyg_forward(x, edge_index)
        return self._fallback_forward(x, edge_index)

    def _pyg_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        agg = self.propagate(edge_index, x=x)
        out = self.lin_self(x) + self.lin_neigh(agg)
        if self.bias is not None:
            out = out + self.bias
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out

    def message(self, x_j: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x_j

    def _fallback_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simple mean aggregation fallback (no PyG required)."""
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        count = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst.unsqueeze(1), torch.ones(len(dst), 1, device=x.device))
        count = count.clamp(min=1)
        agg = agg / count

        out = self.lin_self(x) + self.lin_neigh(agg)
        if self.bias is not None:
            out = out + self.bias
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


# ---------------------------------------------------------------------------
# GAT Layer
# ---------------------------------------------------------------------------


class GATConv(nn.Module if not _HAS_PYG else MessagePassing):  # type: ignore[misc]
    """
    Graph Attention Network convolution layer.

    Multi-head attention aggregation of neighbour features.

    Args:
        in_channels: Input feature dimensionality.
        out_channels: Output dimensionality per attention head.
        heads: Number of attention heads.
        dropout: Attention weight dropout.
        concat: If True, concatenate heads; otherwise average.
        negative_slope: LeakyReLU slope for attention.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
        negative_slope: float = 0.2,
        bias: bool = True,
    ) -> None:
        if _HAS_PYG:
            super().__init__(aggr="add", node_dim=0)
        else:
            super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        out_dim = heads * out_channels if concat else out_channels
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_channels).
            edge_index: Edge indices (2, E).

        Returns:
            Updated node features (N, heads * out_channels) if concat else (N, out_channels).
        """
        if _HAS_PYG:
            return self._pyg_forward(x, edge_index)
        return self._fallback_forward(x, edge_index)

    def _pyg_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_channels)

        alpha_src = (h * self.att_src).sum(dim=-1)
        alpha_dst = (h * self.att_dst).sum(dim=-1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=n)

        out = self.propagate(
            edge_index,
            x=(h, h),
            alpha=(alpha_src, alpha_dst),
        )

        if self.concat:
            out = out.view(n, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(  # type: ignore[override]
        self,
        x_j: torch.Tensor,
        alpha_i: torch.Tensor,
        alpha_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        alpha = alpha_i + alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def _fallback_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simple single-head attention fallback."""
        n = x.size(0)
        h = self.lin(x).view(n, self.heads, self.out_channels)
        src, dst = edge_index[0], edge_index[1]

        alpha_src = (h[src] * self.att_src).sum(-1)
        alpha_dst = (h[dst] * self.att_dst).sum(-1)
        alpha = F.leaky_relu(alpha_src + alpha_dst, self.negative_slope)

        # Softmax per destination node
        alpha_exp = torch.exp(alpha)
        alpha_sum = torch.zeros(n, self.heads, device=x.device)
        alpha_sum.index_add_(0, dst, alpha_exp)
        alpha_norm = alpha_exp / (alpha_sum[dst] + 1e-16)

        agg = torch.zeros(n, self.heads, self.out_channels, device=x.device)
        agg.index_add_(0, dst, h[src] * alpha_norm.unsqueeze(-1))

        if self.concat:
            out = agg.view(n, self.heads * self.out_channels)
        else:
            out = agg.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias
        return out
