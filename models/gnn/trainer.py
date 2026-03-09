"""
GNN training with Elastic Weight Consolidation (EWC) and Experience Replay.

Implements continual learning to prevent catastrophic forgetting when
the model encounters new fraud types over time.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.gnn.model import GNNModel
from models.gnn.memory_buffer import ExperienceReplayBuffer
from utils.logger import get_logger
from utils.helpers import save_checkpoint

logger = get_logger(__name__)

try:
    from torch_geometric.data import Batch, Data  # type: ignore

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class GNNTrainer:
    """
    GNN trainer with EWC regularization and experience replay.

    Args:
        model: GNNModel instance.
        device: Compute device.
        lr: Learning rate.
        ewc_lambda: EWC regularization strength.
        replay_buffer: Optional ExperienceReplayBuffer.
        replay_ratio: Fraction of each batch drawn from the replay buffer.
        class_weights: Tensor of per-class weights for cross-entropy.
        checkpoint_dir: Directory for checkpoints.
        grad_clip: Max gradient norm (0 to disable).
        patience: Early stopping patience.
    """

    def __init__(
        self,
        model: GNNModel,
        device: torch.device,
        lr: float = 1e-3,
        ewc_lambda: float = 5000.0,
        replay_buffer: Optional[ExperienceReplayBuffer] = None,
        replay_ratio: float = 0.3,
        class_weights: Optional[torch.Tensor] = None,
        checkpoint_dir: str = "checkpoints/gnn",
        grad_clip: float = 1.0,
        patience: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.replay_buffer = replay_buffer
        self.replay_ratio = replay_ratio
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        self.patience = patience

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # EWC state
        self._fisher: Dict[str, torch.Tensor] = {}
        self._theta_star: Dict[str, torch.Tensor] = {}

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "ewc_penalty": [], "replay_loss": []
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        task_id: int = 0,
        tb_writer=None,
    ) -> Dict[str, List[float]]:
        """
        Train the GNN model.

        Args:
            train_loader: DataLoader for the current task's training data.
            val_loader: Optional validation DataLoader.
            epochs: Number of epochs.
            task_id: Current task identifier for continual learning.
            tb_writer: Optional TensorBoard SummaryWriter.

        Returns:
            Training history dictionary.
        """
        best_val_loss = float("inf")
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader, task_id)
            self.history["train_loss"].append(train_metrics["total"])
            self.history["ewc_penalty"].append(train_metrics["ewc"])
            self.history["replay_loss"].append(train_metrics["replay"])

            val_loss = None
            if val_loader is not None:
                val_loss = self._val_epoch(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    self._save(epoch, tag="best")
                else:
                    no_improve += 1

                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} — "
                    f"Train Loss: {train_metrics['total']:.4f} | "
                    f"EWC: {train_metrics['ewc']:.4f} | "
                    f"Replay: {train_metrics['replay']:.4f}"
                    + (f" | Val Loss: {val_loss:.4f}" if val_loss is not None else "")
                )

            if tb_writer is not None:
                tb_writer.add_scalar("GNN/train_loss", train_metrics["total"], epoch)
                if val_loss is not None:
                    tb_writer.add_scalar("GNN/val_loss", val_loss, epoch)

        # After training: compute Fisher for next task
        self._compute_fisher(train_loader)
        logger.info("GNN training complete.")
        return self.history

    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute the EWC penalty term."""
        if not self._fisher:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, param in self.model.named_parameters():
            if name in self._fisher:
                penalty += (
                    self._fisher[name] * (param - self._theta_star[name]) ** 2
                ).sum()
        return self.ewc_lambda * penalty

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, task_id: int
    ) -> Dict[str, float]:
        self.model.train()
        ce_losses, ewc_losses, replay_losses = [], [], []

        for batch in loader:
            if _HAS_PYG:
                batch = batch.to(self.device)
                node_logits, _ = self.model(batch)
                labels = batch.y
            else:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                # Create minimal fake data object
                class _FakeData:
                    def __init__(self, x, ei):
                        self.x = x
                        self.edge_index = ei
                        self.batch = None
                n = x.size(0)
                ei = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                fake_data = _FakeData(x, ei)
                node_logits, _ = self.model(fake_data)
                labels = labels

            ce_loss = self.criterion(node_logits, labels.to(self.device))
            ewc_loss = self.compute_ewc_loss()

            # Replay loss
            replay_loss = torch.tensor(0.0, device=self.device)
            if self.replay_buffer is not None and len(self.replay_buffer) > 0:
                n_replay = max(1, int(len(labels) * self.replay_ratio))
                entries = self.replay_buffer.get_class_balanced_sample(n_replay)
                if entries:
                    replay_loss = self._compute_replay_loss(entries)

            total_loss = ce_loss + ewc_loss + replay_loss
            self.optimizer.zero_grad()
            total_loss.backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            ce_losses.append(float(ce_loss.item()))
            ewc_losses.append(float(ewc_loss.item()))
            replay_losses.append(float(replay_loss.item()))

            # Store in replay buffer
            if self.replay_buffer is not None and _HAS_PYG:
                self._store_replay(batch, task_id)

        return {
            "total": float(np.mean(ce_losses)) + float(np.mean(ewc_losses)),
            "ce": float(np.mean(ce_losses)),
            "ewc": float(np.mean(ewc_losses)),
            "replay": float(np.mean(replay_losses)),
        }

    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                if _HAS_PYG:
                    batch = batch.to(self.device)
                    logits, _ = self.model(batch)
                    labels = batch.y.to(self.device)
                else:
                    x, labels = batch
                    x = x.to(self.device)
                    labels = labels.to(self.device)
                    class _FD:
                        def __init__(self, x, ei):
                            self.x = x
                            self.edge_index = ei
                            self.batch = None
                    ei = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                    logits, _ = self.model(_FD(x, ei))
                losses.append(float(self.criterion(logits, labels).item()))
        return float(np.mean(losses))

    def _compute_replay_loss(self, entries: list) -> torch.Tensor:
        """Compute cross-entropy loss on replay samples."""
        if not _HAS_PYG or not entries:
            return torch.tensor(0.0, device=self.device)

        graphs = [e.graph_data for e in entries]
        labels = torch.tensor([e.label for e in entries], dtype=torch.long, device=self.device)

        try:
            batch = Batch.from_data_list(graphs).to(self.device)
            logits, _ = self.model(batch)
            # Use graph-level labels for replay (one per graph)
            if logits.size(0) != labels.size(0):
                # Node-level logits: use mean per graph
                graph_logits = []
                for i in range(len(graphs)):
                    mask = batch.batch == i
                    graph_logits.append(logits[mask].mean(0, keepdim=True))
                logits = torch.cat(graph_logits, dim=0)
            return self.criterion(logits, labels)
        except Exception as exc:
            logger.debug(f"Replay loss failed: {exc}")
            return torch.tensor(0.0, device=self.device)

    def _compute_fisher(self, loader: DataLoader, n_samples: int = 1000) -> None:
        """Compute the diagonal Fisher Information Matrix."""
        logger.info("Computing Fisher Information Matrix …")
        self.model.eval()

        fisher: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        n_seen = 0
        for batch in loader:
            if n_seen >= n_samples:
                break

            if _HAS_PYG:
                batch = batch.to(self.device)
                logits, _ = self.model(batch)
                labels = batch.y.to(self.device)
            else:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                class _FD:
                    def __init__(self, x, ei):
                        self.x = x
                        self.edge_index = ei
                        self.batch = None
                ei = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                logits, _ = self.model(_FD(x, ei))

            loss = self.criterion(logits, labels)
            self.model.zero_grad()
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            n_seen += len(labels)

        n_batches = max(1, n_seen)
        self._fisher = {n: f / n_batches for n, f in fisher.items()}
        self._theta_star = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
        }
        logger.info("Fisher Information Matrix computed.")

    def _store_replay(self, batch, task_id: int) -> None:
        """Store current batch in replay buffer."""
        if not _HAS_PYG or self.replay_buffer is None:
            return
        data_list = batch.to_data_list()
        for data in data_list:
            label = int(data.graph_label.item()) if hasattr(data, "graph_label") else 0
            self.replay_buffer.add(data, label, task_id=task_id)

    def _save(self, epoch: int, tag: str = "") -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history,
            },
            checkpoint_dir=self.checkpoint_dir,
            filename=f"gnn_{tag}_epoch_{epoch}.pt",
        )
