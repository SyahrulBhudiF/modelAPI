"""
Lightning TabR Model

Main TabR model implementation using PyTorch Lightning.
Implements retrieval-augmented tabular learning with FAISS-based nearest neighbor search.
"""

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.models.tabr.encoder import TabREncoder
from app.models.tabr.value_correction import ValueCorrection


class LitTabR(pl.LightningModule):
    """
    Lightning module for TabR (Tabular Retrieval-Augmented) model.

    TabR enhances predictions by retrieving similar examples from a context set
    and using their labels to inform the prediction.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        d: int = 128,
        m: int = 96,
        encoder_blocks: int = 0,
        dropout: float = 0.1,
        lr: float = 1e-3,
        freeze_context_epoch: int = 5,
    ):
        """
        Initialize LitTabR model.

        Args:
            in_dim: Input feature dimension
            num_classes: Number of output classes
            d: Embedding dimension
            m: Number of neighbors to retrieve
            encoder_blocks: Number of encoder residual blocks
            dropout: Dropout probability
            lr: Learning rate
            freeze_context_epoch: Epoch at which to freeze context embeddings
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = TabREncoder(in_dim, d, encoder_blocks, dropout)
        self.WK = nn.Linear(d, d, bias=False)
        self.WY = nn.Embedding(num_classes, d)
        self.T = ValueCorrection(d, dropout)

        self.predictor = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.head = nn.Linear(d, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # Context state
        self.ctx_frozen = False
        self.ctx_k: torch.Tensor | None = None
        self.ctx_y: torch.Tensor | None = None
        self.faiss_index: faiss.IndexFlatL2 | None = None
        self.ctx_val_ready = False

        # Inference helpers
        self.imputer = None
        self.scaler = None
        self.feature_cols: list[str] | None = None
        self.default_context: tuple[torch.Tensor, torch.Tensor] | None = None

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    # ---- FAISS ----
    def build_faiss(self) -> None:
        """Build FAISS index from context keys."""
        k = self.ctx_k.detach().cpu().numpy().astype("float32")
        index = faiss.IndexFlatL2(k.shape[1])
        index.add(k)
        self.faiss_index = index

    def retrieve_topk(self, k_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k nearest neighbors using FAISS.

        Args:
            k_query: Query key tensor

        Returns:
            Tuple of (distances, indices) tensors
        """
        kq = k_query.detach().cpu().numpy().astype("float32")
        dist, idx = self.faiss_index.search(kq, self.hparams.m)
        return (
            torch.tensor(dist, device=k_query.device),
            torch.tensor(idx, device=k_query.device),
        )

    # ---- Forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_dim)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        z = self.encoder(x)
        k = self.WK(z)

        dist, idx = self.retrieve_topk(k)
        sim = -dist
        weights = F.softmax(sim, dim=1)
        weights = F.dropout(weights, p=0.1, training=self.training)

        k_i = self.ctx_k[idx]
        y_i = self.WY(self.ctx_y[idx])

        delta = k.unsqueeze(1) - k_i
        V = y_i + self.T(delta)
        R = torch.sum(weights.unsqueeze(-1) * V, dim=1)

        z_hat = z + R
        z_hat = z_hat + self.predictor(z_hat)
        return self.head(z_hat)

    # ---- Training Steps ----
    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc = (logits.argmax(1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

    # ---- Context Freeze ----
    def on_train_epoch_start(self):
        """Handle context freezing at specified epoch."""
        if (
            not self.ctx_frozen
            and self.current_epoch >= self.hparams.freeze_context_epoch
        ):
            self.freeze_context(
                self.trainer.datamodule.ctx_train_x,
                self.trainer.datamodule.ctx_train_y,
            )

    def freeze_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor) -> None:
        """
        Freeze context embeddings.

        Args:
            ctx_x: Context features tensor
            ctx_y: Context labels tensor
        """
        self.ctx_frozen = True
        self.encoder.eval()

        ctx_x = ctx_x.to(self.get_device())
        ctx_y = ctx_y.to(self.get_device())

        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()
        print("Context frozen & FAISS built")

    def on_validation_epoch_start(self):
        """Setup validation context."""
        if self.ctx_val_ready:
            return

        device = self.get_device()
        ctx_x = self.trainer.datamodule.ctx_val_x.to(device)
        ctx_y = self.trainer.datamodule.ctx_val_y.to(device)

        self.encoder.eval()
        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()
        self.ctx_val_ready = True
        print("Validation context frozen & FAISS built")

    def set_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor) -> None:
        """
        Set context for inference.

        Args:
            ctx_x: Context features tensor
            ctx_y: Context labels tensor
        """
        device = self.get_device()
        ctx_x = ctx_x.to(device)
        ctx_y = ctx_y.to(device)

        self.encoder.eval()
        with torch.no_grad():
            z_ctx = self.encoder(ctx_x)
            self.ctx_k = self.WK(z_ctx)
            self.ctx_y = ctx_y

        self.build_faiss()

    def set_preprocessor(self, imputer, scaler, feature_cols: list[str]) -> None:
        """
        Set preprocessing artifacts.

        Args:
            imputer: Fitted imputer for handling missing values
            scaler: Fitted scaler for normalization
            feature_cols: List of feature column names
        """
        self.imputer = imputer
        self.scaler = scaler
        self.feature_cols = feature_cols

    def set_default_context(self, ctx_x: torch.Tensor, ctx_y: torch.Tensor) -> None:
        """
        Set default context for inference.

        Args:
            ctx_x: Context features tensor
            ctx_y: Context labels tensor
        """
        self.default_context = (ctx_x, ctx_y)
        self.set_context(ctx_x, ctx_y)

    def get_device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.parameters()).device

    def predict_from_df(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions from a DataFrame.

        Args:
            df: Input DataFrame with features
            threshold: Classification threshold

        Returns:
            Tuple of (predictions, probabilities)
        """
        assert self.imputer is not None, "imputer not set"
        assert self.scaler is not None, "scaler not set"
        assert self.feature_cols is not None, "feature_cols not set"

        self.eval()

        # Preprocess
        X_np = self.scaler.transform(self.imputer.transform(df[self.feature_cols]))

        x = torch.tensor(X_np, dtype=torch.float32, device=self.get_device())

        # Ensure context
        if self.default_context is not None:
            ctx_x, ctx_y = self.default_context
            self.set_context(ctx_x, ctx_y)

        with torch.no_grad():
            logits = self(x)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = (prob > threshold).long()

        return pred.cpu().numpy(), prob.cpu().numpy()

    def predict_from_csv(
        self, csv_path: str, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions from a CSV file.

        Args:
            csv_path: Path to CSV file
            **kwargs: Additional arguments passed to predict_from_df

        Returns:
            Tuple of (predictions, probabilities)
        """
        df = pd.read_csv(csv_path)
        return self.predict_from_df(df, **kwargs)
