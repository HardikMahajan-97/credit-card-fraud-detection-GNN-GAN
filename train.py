"""
Main training entrypoint for credit card fraud detection.

Phases:
  1. Load config and data
  2. Preprocess data
  3. Train GAN (learn data distribution, generate synthetic fraud)
  4. Augment training data with GAN-generated samples
  5. Train GNN with continual learning
  6. Train and calibrate ensemble
  7. Evaluate on test set
  8. Save results

Usage:
    python train.py
    python train.py --config config/config.yaml --data-source synthetic
    python train.py --gan-epochs 10 --gnn-epochs 5 --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml

from utils.helpers import get_device, set_seed, save_checkpoint
from utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/train.log")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg


def merge_args(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Override config values with CLI arguments."""
    if args.data_source:
        cfg.setdefault("data", {})["source"] = args.data_source
    if args.device:
        cfg.setdefault("training", {})["device"] = args.device
    if args.gan_epochs is not None:
        cfg.setdefault("gan", {})["epochs"] = args.gan_epochs
    if args.gnn_epochs is not None:
        cfg.setdefault("gnn", {})["epochs"] = args.gnn_epochs
    return cfg


# ---------------------------------------------------------------------------
# Phase 1: Data loading and preprocessing
# ---------------------------------------------------------------------------


def phase_load_data(cfg: Dict) -> tuple:
    """Load raw transaction data."""
    from data.download_data import get_data

    data_cfg = cfg.get("data", {})
    df = get_data(
        source=data_cfg.get("source", "synthetic"),
        n_samples=int(data_cfg.get("synthetic_n_samples", 100_000)),
        fraud_ratio=float(data_cfg.get("fraud_ratio", 0.017)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )
    logger.info(
        f"Data loaded: {len(df):,} rows | "
        f"fraud rate: {df['is_fraud'].mean():.3%}"
    )
    return df


def phase_preprocess(cfg: Dict, df) -> tuple:
    """Run the preprocessing pipeline."""
    from preprocessing.pipeline import PreprocessingPipeline
    from data.dataset import build_fraud_datasets

    pipe = PreprocessingPipeline(cfg=cfg)
    processed_df, _ = pipe.run(df, build_graphs=False)

    # Determine feature columns (numeric, non-label)
    exclude = {"transaction_id", "card_id", "merchant_id", "is_fraud"}
    feature_cols = [
        c for c in processed_df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    data_cfg = cfg.get("data", {})
    train_ds, val_ds, test_ds = build_fraud_datasets(
        processed_df,
        feature_cols=feature_cols,
        test_size=float(data_cfg.get("test_size", 0.2)),
        val_size=float(data_cfg.get("val_size", 0.1)),
        random_seed=int(data_cfg.get("random_seed", 42)),
    )

    logger.info(
        f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}"
    )
    return train_ds, val_ds, test_ds, feature_cols


# ---------------------------------------------------------------------------
# Phase 2: GAN training
# ---------------------------------------------------------------------------


def phase_train_gan(cfg: Dict, train_ds, device: torch.device):
    """Train the WGAN-GP model."""
    from models.gan.generator import Generator
    from models.gan.discriminator import Discriminator
    from models.gan.trainer import GANTrainer
    from data.dataset import get_dataloader

    gan_cfg = cfg.get("gan", {})
    feature_dim = train_ds.feature_dim

    generator = Generator(
        noise_dim=int(gan_cfg.get("noise_dim", 128)),
        output_dim=feature_dim,
        hidden_dims=list(gan_cfg.get("generator_hidden", [256, 512, 256])),
    )
    discriminator = Discriminator(
        input_dim=feature_dim,
        hidden_dims=list(gan_cfg.get("discriminator_hidden", [256, 512, 256])),
    )

    train_loader = get_dataloader(
        train_ds,
        batch_size=int(gan_cfg.get("batch_size", 256)),
        shuffle=True,
    )

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr=float(gan_cfg.get("learning_rate", 1e-4)),
        beta1=float(gan_cfg.get("beta1", 0.0)),
        beta2=float(gan_cfg.get("beta2", 0.9)),
        n_critic=int(gan_cfg.get("n_critic", 5)),
        gp_lambda=float(gan_cfg.get("gradient_penalty_lambda", 10.0)),
        checkpoint_dir=cfg.get("training", {}).get("checkpoint_dir", "checkpoints") + "/gan",
    )

    logger.info(f"Training GAN for {gan_cfg.get('epochs', 200)} epochs …")
    history = trainer.train(
        dataloader=train_loader,
        epochs=int(gan_cfg.get("epochs", 200)),
    )

    return trainer, generator, discriminator, history


def phase_augment_data(cfg: Dict, train_ds, generator, device: torch.device):
    """Generate synthetic fraud samples and augment training data."""
    import torch
    from data.dataset import FraudDataset

    gan_cfg = cfg.get("gan", {})
    multiplier = int(gan_cfg.get("synthetic_multiplier", 3))

    n_fraud = int((train_ds.labels == 1).sum().item())
    n_synthetic = n_fraud * multiplier

    logger.info(f"Generating {n_synthetic:,} synthetic fraud samples …")
    generator.eval()
    with torch.no_grad():
        z = generator.sample_noise(n_synthetic, device)
        synthetic_feats = generator(z).cpu().numpy()
    generator.train()

    synthetic_labels = np.ones(n_synthetic, dtype=np.int64)

    # Combine with original training data
    aug_feats = np.vstack([train_ds.features.numpy(), synthetic_feats])
    aug_labels = np.concatenate([train_ds.labels.numpy(), synthetic_labels])

    aug_ds = FraudDataset(aug_feats, aug_labels)
    logger.info(
        f"Augmented dataset: {len(aug_ds):,} samples | "
        f"fraud rate: {aug_labels.mean():.3%}"
    )
    return aug_ds


# ---------------------------------------------------------------------------
# Phase 3: GNN training (tabular fallback)
# ---------------------------------------------------------------------------


def phase_train_gnn(cfg: Dict, train_ds, val_ds, device: torch.device):
    """Train the GNN model (tabular fallback if PyG unavailable)."""
    from models.gnn.model import GNNModel
    from models.gnn.memory_buffer import ExperienceReplayBuffer
    from models.gnn.trainer import GNNTrainer
    from data.dataset import get_dataloader

    gnn_cfg = cfg.get("gnn", {})
    cl_cfg = cfg.get("continual_learning", {})

    feature_dim = train_ds.feature_dim
    model = GNNModel(
        input_dim=feature_dim,
        hidden_dim=int(gnn_cfg.get("hidden_dim", 128)),
        num_sage_layers=int(gnn_cfg.get("num_sage_layers", 2)),
        num_gat_layers=int(gnn_cfg.get("num_gat_layers", 2)),
        gat_heads=int(gnn_cfg.get("gat_heads", 4)),
        dropout=float(gnn_cfg.get("dropout", 0.3)),
        max_neighbors=int(gnn_cfg.get("max_neighbors", 30)),
    )

    replay_buffer = ExperienceReplayBuffer(
        capacity=int(cl_cfg.get("replay_buffer_size", 10_000))
    )

    # Compute class weights for imbalanced data
    focal_alpha = float(gnn_cfg.get("focal_alpha", 0.75))
    pos_weight = float(gnn_cfg.get("pos_weight", 1.0))
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32)

    trainer = GNNTrainer(
        model=model,
        device=device,
        lr=float(gnn_cfg.get("learning_rate", 5e-4)),
        lr_patience=int(gnn_cfg.get("lr_patience", 5)),
        ewc_lambda=float(cl_cfg.get("ewc_lambda", 5000.0)),
        replay_buffer=replay_buffer,
        replay_ratio=float(cl_cfg.get("replay_ratio", 0.3)),
        class_weights=class_weights,
        focal_alpha=focal_alpha,
        focal_gamma=float(gnn_cfg.get("focal_gamma", 2.0)),
        pos_weight=pos_weight,
        checkpoint_dir=cfg.get("training", {}).get("checkpoint_dir", "checkpoints") + "/gnn",
        patience=int(cfg.get("training", {}).get("early_stopping_patience", 10)),
    )

    train_loader = get_dataloader(
        train_ds,
        batch_size=int(gnn_cfg.get("batch_size", 64)),
        shuffle=True,
    )
    val_loader = get_dataloader(val_ds, batch_size=256, shuffle=False)

    logger.info(f"Training GNN for {gnn_cfg.get('epochs', 100)} epochs …")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(gnn_cfg.get("epochs", 100)),
        task_id=0,
    )

    return trainer, model, history


# ---------------------------------------------------------------------------
# Phase 4: Ensemble training and evaluation
# ---------------------------------------------------------------------------


def phase_train_ensemble(
    cfg: Dict,
    discriminator,
    gnn_model,
    val_ds,
    device: torch.device,
):
    """Train and calibrate the ensemble model."""
    from models.ensemble import EnsembleModel
    from data.dataset import get_dataloader
    import torch.nn.functional as F

    ens_cfg = cfg.get("ensemble", {})
    ensemble = EnsembleModel(
        fusion_method=ens_cfg.get("fusion_method", "weighted"),
        gan_weight=float(ens_cfg.get("gan_weight", 0.3)),
        gnn_weight=float(ens_cfg.get("gnn_weight", 0.7)),
        calibration=bool(ens_cfg.get("calibration", True)),
        device=device,
    )

    val_loader = get_dataloader(val_ds, batch_size=256, shuffle=False)

    all_gan_scores, all_gnn_probs, all_labels = [], [], []

    discriminator.eval()
    gnn_model.eval()

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)

            # GAN anomaly scores
            gan_s = discriminator.get_anomaly_score(features).cpu().numpy()
            all_gan_scores.append(gan_s)

            # GNN fraud probabilities (tabular fallback)
            class _FD:
                def __init__(self, x, ei):
                    self.x = x
                    self.edge_index = ei
                    self.batch = None
            ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            logits, _ = gnn_model(_FD(features, ei))
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_gnn_probs.append(probs)
            all_labels.append(labels.numpy())

    gan_scores = np.concatenate(all_gan_scores)
    gnn_probs = np.concatenate(all_gnn_probs)
    labels_np = np.concatenate(all_labels)

    ensemble.fit(gan_scores, gnn_probs, labels_np, epochs=50)
    logger.info("Ensemble training complete.")
    return ensemble


def phase_evaluate(
    cfg: Dict,
    discriminator,
    gnn_model,
    ensemble,
    test_ds,
    device: torch.device,
    output_dir: str = "results",
):
    """Evaluate the full model on the test set."""
    import torch.nn.functional as F
    from data.dataset import get_dataloader
    from evaluation.metrics import evaluate_model
    from evaluation.visualization import Visualizer

    os.makedirs(output_dir, exist_ok=True)
    test_loader = get_dataloader(test_ds, batch_size=256, shuffle=False)

    all_gan_scores, all_gnn_probs, all_labels = [], [], []

    discriminator.eval()
    gnn_model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            gan_s = discriminator.get_anomaly_score(features).cpu().numpy()
            all_gan_scores.append(gan_s)

            class _FD:
                def __init__(self, x, ei):
                    self.x = x
                    self.edge_index = ei
                    self.batch = None
            ei = torch.zeros(2, 0, dtype=torch.long, device=device)
            logits, _ = gnn_model(_FD(features, ei))
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_gnn_probs.append(probs)
            all_labels.append(labels.numpy())

    gan_scores = np.concatenate(all_gan_scores)
    gnn_probs = np.concatenate(all_gnn_probs)
    y_true = np.concatenate(all_labels)

    y_prob = ensemble.predict_batch(gan_scores, gnn_probs)
    y_pred = (y_prob >= ensemble.threshold if hasattr(ensemble, "threshold") else y_prob >= 0.5).astype(int)

    metrics = evaluate_model(y_true, y_pred, y_prob)

    # Visualize
    viz = Visualizer(output_dir=output_dir)
    if len(np.unique(y_true)) > 1:
        viz.plot_roc_curve(y_true, y_prob)
        viz.plot_precision_recall_curve(y_true, y_prob)
    viz.plot_confusion_matrix(y_true, y_pred)

    # Save metrics
    import json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({k: v for k, v in metrics.items() if not np.isnan(v)}, f, indent=2)

    logger.info(f"Test metrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GAN + GNN fraud detection model")
    parser.add_argument("--config", default="config/config.yaml", help="Config file path")
    parser.add_argument("--data-source", default=None, choices=["kaggle", "huggingface", "synthetic", "mixed"])
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--gan-epochs", type=int, default=None)
    parser.add_argument("--gnn-epochs", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)

    # Setup
    seed = cfg.get("data", {}).get("random_seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("training", {}).get("device", "auto"))

    # Phase 1: Load data
    df = phase_load_data(cfg)

    # Phase 2: Preprocess
    train_ds, val_ds, test_ds, feature_cols = phase_preprocess(cfg, df)

    # Phase 3: Train GAN
    gan_trainer, generator, discriminator, gan_history = phase_train_gan(cfg, train_ds, device)

    # Phase 4: Augment with synthetic fraud
    aug_train_ds = phase_augment_data(cfg, train_ds, generator, device)

    # Phase 5: Train GNN
    gnn_trainer, gnn_model, gnn_history = phase_train_gnn(cfg, aug_train_ds, val_ds, device)

    # Phase 6: Train Ensemble
    ensemble = phase_train_ensemble(cfg, discriminator, gnn_model, val_ds, device)

    # Phase 7: Evaluate
    metrics = phase_evaluate(
        cfg, discriminator, gnn_model, ensemble, test_ds, device,
        output_dir=args.output_dir,
    )

    # Phase 8: Save final models
    checkpoint_dir = cfg.get("training", {}).get("checkpoint_dir", "checkpoints")
    save_checkpoint(
        {
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "gnn_state": gnn_model.state_dict(),
            "ensemble_state": ensemble.state_dict(),
            "metrics": metrics,
            "feature_cols": feature_cols,
        },
        checkpoint_dir=checkpoint_dir,
        filename="final_model.pt",
    )

    logger.info("Training complete! Models saved.")
    logger.info(f"Final test metrics: AUPRC={metrics.get('auprc', 'N/A'):.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
