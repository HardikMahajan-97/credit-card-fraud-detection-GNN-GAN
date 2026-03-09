"""
WGAN-GP training loop.

Trains a Generator and Discriminator using Wasserstein loss with gradient penalty.
Supports checkpointing, TensorBoard logging, and anomaly scoring.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.gan.generator import Generator
from models.gan.discriminator import Discriminator
from utils.logger import get_logger
from utils.helpers import save_checkpoint

logger = get_logger(__name__)


class GANTrainer:
    """
    WGAN-GP trainer for fraud transaction generation.

    Args:
        generator: Generator model.
        discriminator: Discriminator / Critic model.
        device: Compute device.
        lr: Learning rate (Adam).
        beta1: Adam beta1.
        beta2: Adam beta2.
        n_critic: Discriminator steps per generator step.
        gp_lambda: Gradient penalty coefficient.
        checkpoint_dir: Directory to save checkpoints.
    """

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        device: torch.device,
        lr: float = 1e-4,
        beta1: float = 0.0,
        beta2: float = 0.9,
        n_critic: int = 5,
        gp_lambda: float = 10.0,
        checkpoint_dir: str = "checkpoints/gan",
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.n_critic = n_critic
        self.gp_lambda = gp_lambda
        self.checkpoint_dir = checkpoint_dir

        self.opt_g = optim.Adam(
            generator.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.opt_d = optim.Adam(
            discriminator.parameters(), lr=lr, betas=(beta1, beta2)
        )

        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, T_max=200
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, T_max=200
        )

        self.history: Dict[str, List[float]] = {
            "g_loss": [], "d_loss": [], "gradient_penalty": []
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 200,
        log_interval: int = 50,
        tb_writer=None,
    ) -> Dict[str, List[float]]:
        """
        Run WGAN-GP training.

        Args:
            dataloader: DataLoader yielding (features, labels) batches.
            epochs: Number of training epochs.
            log_interval: Log every N batches.
            tb_writer: Optional TensorBoard SummaryWriter.

        Returns:
            Training history dict.
        """
        self.generator.train()
        self.discriminator.train()
        global_step = 0

        for epoch in range(1, epochs + 1):
            g_losses, d_losses, gps = [], [], []

            for batch_idx, (real_data, _) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)

                # ---- Train Discriminator ----
                d_loss, gp = self._train_discriminator_step(real_data, batch_size)
                d_losses.append(d_loss)
                gps.append(gp)

                # ---- Train Generator (every n_critic D steps) ----
                if batch_idx % self.n_critic == 0:
                    g_loss = self._train_generator_step(batch_size)
                    g_losses.append(g_loss)

                global_step += 1

                if global_step % log_interval == 0:
                    logger.debug(
                        f"Epoch {epoch}/{epochs} | Step {global_step} | "
                        f"D_loss: {np.mean(d_losses):.4f} | "
                        f"G_loss: {np.mean(g_losses) if g_losses else 0.0:.4f} | "
                        f"GP: {np.mean(gps):.4f}"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("GAN/D_loss", np.mean(d_losses), global_step)
                        tb_writer.add_scalar("GAN/G_loss", np.mean(g_losses) if g_losses else 0.0, global_step)
                        tb_writer.add_scalar("GAN/GP", np.mean(gps), global_step)

            epoch_g = float(np.mean(g_losses)) if g_losses else 0.0
            epoch_d = float(np.mean(d_losses))
            epoch_gp = float(np.mean(gps))

            self.history["g_loss"].append(epoch_g)
            self.history["d_loss"].append(epoch_d)
            self.history["gradient_penalty"].append(epoch_gp)

            self.scheduler_g.step()
            self.scheduler_d.step()

            if epoch % 50 == 0:
                logger.info(
                    f"Epoch {epoch}/{epochs} — G: {epoch_g:.4f}, D: {epoch_d:.4f}, GP: {epoch_gp:.4f}"
                )
                self._save(epoch)

        logger.info("GAN training complete.")
        return self.history

    def generate_samples(self, n: int) -> torch.Tensor:
        """
        Generate n synthetic transaction samples.

        Args:
            n: Number of samples to generate.

        Returns:
            Tensor of shape (n, output_dim).
        """
        self.generator.eval()
        with torch.no_grad():
            z = self.generator.sample_noise(n, self.device)
            samples = self.generator(z)
        self.generator.train()
        return samples.cpu()

    def get_anomaly_scores(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute anomaly scores for a batch of transactions.

        Args:
            data: Feature tensor of shape (N, input_dim).

        Returns:
            Anomaly scores of shape (N,).
        """
        self.discriminator.eval()
        data = data.to(self.device)
        scores = self.discriminator.get_anomaly_score(data)
        self.discriminator.train()
        return scores.cpu()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_discriminator_step(
        self, real_data: torch.Tensor, batch_size: int
    ) -> Tuple[float, float]:
        self.opt_d.zero_grad()

        z = self.generator.sample_noise(batch_size, self.device)
        fake_data = self.generator(z).detach()

        real_scores = self.discriminator(real_data)
        fake_scores = self.discriminator(fake_data)

        gp = self._gradient_penalty(real_data, fake_data)
        d_loss = fake_scores.mean() - real_scores.mean() + self.gp_lambda * gp

        d_loss.backward()
        self.opt_d.step()

        return float(d_loss.item()), float(gp.item())

    def _train_generator_step(self, batch_size: int) -> float:
        self.opt_g.zero_grad()

        z = self.generator.sample_noise(batch_size, self.device)
        fake_data = self.generator(z)
        fake_scores = self.discriminator(fake_data)

        g_loss = -fake_scores.mean()
        g_loss.backward()
        self.opt_g.step()

        return float(g_loss.item())

    def _gradient_penalty(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute WGAN-GP gradient penalty."""
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real)

        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def _save(self, epoch: int) -> None:
        save_checkpoint(
            {
                "epoch": epoch,
                "generator_state": self.generator.state_dict(),
                "discriminator_state": self.discriminator.state_dict(),
                "opt_g_state": self.opt_g.state_dict(),
                "opt_d_state": self.opt_d.state_dict(),
                "history": self.history,
            },
            checkpoint_dir=self.checkpoint_dir,
            filename=f"gan_epoch_{epoch}.pt",
        )
