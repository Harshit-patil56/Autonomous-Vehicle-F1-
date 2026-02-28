"""Training script for the waypoint-prediction model.

Uses PyTorch Lightning for the training loop with mixed-precision,
cosine-annealing scheduler, model checkpointing, and optional W&B
logging.

The training loss is computed on **evaluated Bezier curve points**
(not raw control points) following the DeepRacing approach:
    predicted_curve_points = M @ predicted_control_points
    gt_curve_points        = M @ gt_fitted_control_points
    loss = smooth_l1(predicted_curve_points, gt_waypoints)

Usage::

    python -m model.train \\
        --train-parquet datasets/train.parquet \\
        --val-parquet   datasets/val.parquet \\
        --max-epochs 50

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.network import WaypointNet
from model.dataset import WaypointDataset


# ---------------------------------------------------------------
# Bezier utility used during training
# ---------------------------------------------------------------

def bernstein_basis(n: int, t: torch.Tensor) -> torch.Tensor:
    """Compute the Bernstein polynomial basis matrix.

    Equivalent to DeepRacing's ``bezierM(s, order)``.

    Args:
        n: Bezier order (= num_control_points - 1).
        t: (S,) parameter values in [0, 1].

    Returns:
        (S, n+1) matrix M such that ``M @ control_points`` evaluates
        the Bezier curve at the given parameter values.
    """
    s = len(t)
    k = torch.arange(n + 1, dtype=t.dtype, device=t.device)

    # Binomial coefficients: C(n, k).
    from math import comb
    binom = torch.tensor(
        [comb(n, int(ki)) for ki in k], dtype=t.dtype, device=t.device
    )

    # (S, n+1) = C(n,k) * t^k * (1-t)^(n-k)
    t_exp = t.unsqueeze(1)  # (S, 1)
    basis = binom * (t_exp ** k) * ((1.0 - t_exp) ** (n - k))
    return basis


# ---------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------

class WaypointLitModule(pl.LightningModule):
    """PyTorch Lightning wrapper for WaypointNet.

    Attributes:
        net: The WaypointNet model.
        num_eval_points: Number of evenly-spaced parameter values used
            to evaluate the Bezier curve for the loss computation.
        lr: Learning rate.
        momentum: SGD momentum.
        weight_decay: L2 regularisation.
    """

    def __init__(
        self,
        num_control_points: int = 9,
        num_eval_points: int = 20,
        use_speed: bool = True,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        t_max_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.net = WaypointNet(
            num_control_points=num_control_points,
            use_speed=use_speed,
        )
        self.num_control_points = num_control_points
        self.num_eval_points = num_eval_points
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.t_max_epochs = t_max_epochs

        # Pre-compute the Bernstein basis matrix (moved to device in
        # on_fit_start).
        order = num_control_points - 1
        t = torch.linspace(0.0, 1.0, num_eval_points)
        self.register_buffer("bezier_M", bernstein_basis(order, t))

    def forward(
        self,
        image: torch.Tensor,
        speed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.net(image, speed)

    def _compute_loss(
        self,
        pred_control_points: torch.Tensor,
        gt_waypoints: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate predicted Bezier and compute Smooth-L1 loss.

        Args:
            pred_control_points: (batch, CP, 2) from the network.
            gt_waypoints: (batch, num_eval_points, 2) ground-truth
                local waypoints.

        Returns:
            Scalar loss.
        """
        # (batch, num_eval_points, 2) = M @ control_points
        pred_points = torch.matmul(self.bezier_M, pred_control_points)
        return F.smooth_l1_loss(pred_points, gt_waypoints)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, waypoints, speeds = batch
        pred_cp = self.net(images, speeds)
        loss = self._compute_loss(pred_cp, waypoints)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, waypoints, speeds = batch
        pred_cp = self.net(images, speeds)
        loss = self._compute_loss(pred_cp, waypoints)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=True,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max_epochs
        )
        return [optimizer], [scheduler]


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main() -> None:
    """Entry point for model training."""
    parser = argparse.ArgumentParser(description="Train WaypointNet.")
    parser.add_argument("--train-parquet", type=str, required=True)
    parser.add_argument("--val-parquet", type=str, required=True)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-control-points", type=int, default=9)
    parser.add_argument("--num-waypoints", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--wandb-project", type=str, default="f1-waypoint")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="model/checkpoints",
    )
    args = parser.parse_args()

    # --- Reproducibility ---
    pl.seed_everything(args.seed, workers=True)

    # --- Datasets ---
    train_ds = WaypointDataset(
        args.train_parquet,
        num_waypoints=args.num_waypoints,
        augment=True,
    )
    val_ds = WaypointDataset(
        args.val_parquet,
        num_waypoints=args.num_waypoints,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # --- Model ---
    model = WaypointLitModule(
        num_control_points=args.num_control_points,
        num_eval_points=args.num_waypoints,
        lr=args.lr,
        t_max_epochs=args.max_epochs,
    )

    # --- Logger ---
    logger = None
    if not args.no_wandb:
        try:
            logger = WandbLogger(project=args.wandb_project)
        except Exception as exc:
            print(f"W&B init failed ({exc}). Training without W&B.")
            logger = None

    # --- Callbacks ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_cb],
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_loss:   {checkpoint_cb.best_model_score:.6f}")


if __name__ == "__main__":
    main()
