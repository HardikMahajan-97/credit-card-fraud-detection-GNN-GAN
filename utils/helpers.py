"""Helper utilities: seed setting, device selection, checkpointing."""

import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(device: str = "auto") -> torch.device:
    """
    Select the appropriate compute device.

    Args:
        device: One of "auto", "cpu", or "cuda". "auto" selects CUDA if available.

    Returns:
        torch.device instance.
    """
    if device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = device
    dev = torch.device(selected)
    logger.info(f"Using device: {dev}")
    return dev


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> str:
    """
    Save a model checkpoint to disk.

    Args:
        state: Dictionary containing model state dict and any extra info.
        checkpoint_dir: Directory to save the checkpoint.
        filename: Checkpoint filename.

    Returns:
        Full path to the saved checkpoint.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a model checkpoint from disk.

    Args:
        filepath: Path to the checkpoint file.
        device: Device to map the tensors to. Defaults to CPU.

    Returns:
        Dictionary containing checkpoint data.
    """
    if device is None:
        device = torch.device("cpu")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    state = torch.load(filepath, map_location=device)
    logger.info(f"Checkpoint loaded from {filepath}")
    return state
