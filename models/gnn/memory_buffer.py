"""
Experience Replay Buffer for continual learning in GNN fraud detection.

Uses reservoir sampling to maintain a representative distribution of past
examples. Supports priority-based sampling for harder examples.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReplayEntry:
    """A single entry in the replay buffer."""

    graph_data: object  # PyG Data or dict
    label: int
    embedding: Optional[torch.Tensor] = None
    task_id: int = 0
    priority: float = 1.0


class ExperienceReplayBuffer:
    """
    Fixed-size experience replay buffer with reservoir sampling.

    Uses reservoir sampling (Algorithm R) to maintain a uniform distribution
    over the stream of incoming examples. Supports class-balanced and
    priority-based sampling.

    Args:
        capacity: Maximum number of entries in the buffer.
        priority_alpha: Exponent for priority-based sampling (0 = uniform).
    """

    def __init__(self, capacity: int = 10_000, priority_alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self._buffer: List[ReplayEntry] = []
        self._n_seen = 0  # Total examples seen (for reservoir sampling)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        graph_data: object,
        label: int,
        embedding: Optional[torch.Tensor] = None,
        task_id: int = 0,
        priority: float = 1.0,
    ) -> None:
        """
        Add a new example to the buffer using reservoir sampling.

        Args:
            graph_data: PyG Data object or feature dict.
            label: Integer class label.
            embedding: Optional pre-computed node embedding.
            task_id: Task identifier (for multi-task continual learning).
            priority: Sampling priority score.
        """
        entry = ReplayEntry(
            graph_data=graph_data,
            label=label,
            embedding=embedding,
            task_id=task_id,
            priority=priority,
        )

        if len(self._buffer) < self.capacity:
            self._buffer.append(entry)
        else:
            # Reservoir sampling: replace with decreasing probability
            j = random.randint(0, self._n_seen)
            if j < self.capacity:
                self._buffer[j] = entry

        self._n_seen += 1

    def add_batch(
        self,
        graph_data_list: List,
        labels: List[int],
        embeddings: Optional[List[torch.Tensor]] = None,
        task_id: int = 0,
    ) -> None:
        """Add multiple examples at once."""
        for i, (gd, lbl) in enumerate(zip(graph_data_list, labels)):
            emb = embeddings[i] if embeddings is not None else None
            self.add(gd, lbl, emb, task_id)

    def sample(self, batch_size: int) -> List[ReplayEntry]:
        """
        Sample *batch_size* entries using priority-weighted sampling.

        Args:
            batch_size: Number of entries to sample.

        Returns:
            List of ReplayEntry objects.
        """
        if len(self._buffer) == 0:
            return []

        n = min(batch_size, len(self._buffer))

        if self.priority_alpha == 0:
            return random.sample(self._buffer, n)

        priorities = np.array([e.priority ** self.priority_alpha for e in self._buffer])
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self._buffer), size=n, replace=False, p=probs)
        return [self._buffer[i] for i in indices]

    def get_class_balanced_sample(self, batch_size: int) -> List[ReplayEntry]:
        """
        Return a class-balanced sample from the buffer.

        Attempts equal representation of each class. Falls back to random
        sampling if any class has insufficient examples.

        Args:
            batch_size: Desired total batch size.

        Returns:
            List of ReplayEntry objects.
        """
        if len(self._buffer) == 0:
            return []

        # Group by label
        by_class: dict = {}
        for entry in self._buffer:
            by_class.setdefault(entry.label, []).append(entry)

        n_classes = len(by_class)
        per_class = max(1, batch_size // n_classes)

        result = []
        for entries in by_class.values():
            k = min(per_class, len(entries))
            result.extend(random.sample(entries, k))

        # Shuffle and trim
        random.shuffle(result)
        return result[:batch_size]

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update the priority of specific buffer entries."""
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self._buffer):
                self._buffer[idx].priority = prio

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ExperienceReplayBuffer("
            f"size={len(self._buffer)}/{self.capacity}, "
            f"n_seen={self._n_seen})"
        )
