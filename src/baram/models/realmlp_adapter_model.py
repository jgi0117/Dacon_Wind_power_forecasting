'''PyTorch shared-trunk MLP with one task-specific adapter per wind group.'''

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import numpy as np
import pandas as pd

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS, ficr_aware_loss_torch, target_score
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')
SHARED_SIZES = (256, 128)
ADAPTER_SIZE = 64
DROPOUT = 0.15
WEIGHT_DECAY = 2e-2


class _AdapterNetwork:
    @staticmethod
    def build(n_features: int, n_targets: int) -> Any:
        import torch
        from torch import nn

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                shared_layers: list[nn.Module] = []
                in_features = n_features
                for out_features in SHARED_SIZES:
                    shared_layers.extend([
                        nn.Linear(in_features, out_features),
                        nn.LayerNorm(out_features),
                        nn.Mish(),
                        nn.Dropout(DROPOUT),
                    ])
                    in_features = out_features
                self.shared_trunk = nn.Sequential(*shared_layers)
                self.adapters = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(SHARED_SIZES[-1], ADAPTER_SIZE),
                        nn.Mish(),
                        nn.Dropout(DROPOUT),
                        nn.Linear(ADAPTER_SIZE, 1),
                    )
                    for _ in range(n_targets)
                ])
                self.apply(self._initialize)

            @staticmethod
            def _initialize(module: nn.Module) -> None:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(
                        module.weight, nonlinearity='relu'
                    )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

            def forward(self, features: Any) -> Any:
                shared = self.shared_trunk(features)
                return torch.cat(
                    [adapter(shared) for adapter in self.adapters],
                    dim=-1,
                )

        return Network()


class RealMLPAdapterModel(RegressionModel):
    def __init__(self, config: PipelineConfig, epochs: int | None = None) -> None:
        self.config = config
        self.epochs = int(epochs or config.max_epochs)
        self.best_iteration = self.epochs
        self.model: Any | None = None
        self.feature_median: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.feature_columns: list[str] = []
        self.target_names = list(TARGET_COLS)
        self.training_history: list[dict[str, Any]] = []
        self.elapsed_seconds = 0.0
        self.device = 'cpu'
        self.n_threads = 1

    def _fit_scaler(self, X: pd.DataFrame) -> None:
        values = X.to_numpy(dtype=np.float32, copy=True)
        self.feature_columns = [str(column) for column in X.columns]
        self.feature_median = np.nanmedian(values, axis=0).astype(np.float32)
        q25, q75 = np.nanpercentile(values, [25.0, 75.0], axis=0)
        scale = (q75 - q25).astype(np.float32)
        self.feature_scale = np.where(scale > 1e-6, scale, 1.0)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_median is None or self.feature_scale is None:
            raise RuntimeError('Feature scaler must be fitted before transform.')
        if [str(column) for column in X.columns] != self.feature_columns:
            raise ValueError('AdapterMLP feature columns differ from fit schema.')
        values = X.to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(values)
        if invalid.any():
            values[invalid] = np.broadcast_to(
                self.feature_median, values.shape
            )[invalid]
        values = (values - self.feature_median) / self.feature_scale
        return np.clip(values, -10.0, 10.0).astype(np.float32, copy=False)

    @staticmethod
    def _masked_targets(y: pd.DataFrame | pd.Series) -> np.ndarray:
        values = np.array(y.to_numpy(dtype=np.float32), copy=True)
        if values.ndim == 1:
            values = values[:, None]
        values[~np.isfinite(values)] = 0.0
        return values

    @staticmethod
    def _coslog4_factor(progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        phase = math.log2(1.0 + 15.0 * progress)
        return 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))

    def _loss(self, actual: Any, prediction: Any) -> Any:
        return ficr_aware_loss_torch(
            actual,
            prediction,
            ficr_weight=self.config.ficr_weight,
            temperature=self.config.ficr_temperature,
        ).mean()

    def _history_row(
        self,
        epoch: int,
        train_losses: list[float],
        group_train_losses: dict[str, list[float]],
        actual: Any,
        prediction: Any,
    ) -> dict[str, Any]:
        actual_np = actual.detach().cpu().numpy()
        prediction_np = prediction.detach().cpu().numpy()
        row: dict[str, Any] = {
            'step': epoch,
            'train_loss': float(np.mean(train_losses)),
            'validation_loss': float(
                self._loss(actual, prediction).detach().cpu()
            ),
        }
        scores: list[float] = []
        for index, target in enumerate(self.target_names):
            group_loss = float(
                self._loss(
                    actual[:, index:index + 1],
                    prediction[:, index:index + 1],
                ).detach().cpu()
            )
            score = target_score(
                actual_np[:, index], prediction_np[:, index]
            )
            scores.append(score)
            row[f'{target}__train_loss'] = float(
                np.mean(group_train_losses[target])
            )
            row[f'{target}__validation_loss'] = group_loss
            row[f'{target}__validation_score'] = score
        row['validation_score'] = float(np.mean(scores))
        return row

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.DataFrame | pd.Series | None = None,
    ) -> 'RealMLPAdapterModel':
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        started = time.perf_counter()
        self.device = self.config.device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is not available.')
        self.n_threads = (
            os.cpu_count() or 1
            if self.config.n_jobs < 1
            else self.config.n_jobs
        )
        if self.device == 'cpu':
            torch.set_num_threads(max(1, self.n_threads))
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        if isinstance(y, pd.DataFrame):
            self.target_names = [str(column) for column in y.columns]
        else:
            self.target_names = [str(y.name or TARGET_COLS[0])]
        self._fit_scaler(X)
        train_X = torch.from_numpy(self._transform(X))
        train_y = torch.from_numpy(self._masked_targets(y))
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=self.device.startswith('cuda'),
        )
        self.model = _AdapterNetwork.build(
            train_X.shape[1], len(self.target_names)
        ).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=WEIGHT_DECAY,
        )
        total_steps = max(1, self.epochs * len(loader))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: self._coslog4_factor(
                step / max(1, total_steps - 1)
            ),
        )
        has_validation = X_valid is not None and y_valid is not None
        if has_validation:
            valid_X = torch.from_numpy(self._transform(X_valid)).to(self.device)
            valid_y = torch.from_numpy(
                self._masked_targets(y_valid)
            ).to(self.device)
        best_loss = float('inf')
        best_state: dict[str, Any] | None = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses: list[float] = []
            group_train_losses = {
                target: [] for target in self.target_names
            }
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(batch_X)
                loss = self._loss(batch_y, prediction)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=5.0
                )
                optimizer.step()
                scheduler.step()
                train_losses.append(float(loss.detach().cpu()))
                for index, target in enumerate(self.target_names):
                    group_loss = self._loss(
                        batch_y[:, index:index + 1],
                        prediction[:, index:index + 1],
                    )
                    group_train_losses[target].append(
                        float(group_loss.detach().cpu())
                    )
            if has_validation:
                self.model.eval()
                with torch.no_grad():
                    valid_prediction = self.model(valid_X)
                    row = self._history_row(
                        epoch,
                        train_losses,
                        group_train_losses,
                        valid_y,
                        valid_prediction,
                    )
                self.training_history.append(row)
                validation_loss = float(row['validation_loss'])
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    self.best_iteration = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                LOGGER.info(
                    'AdapterMLP epoch=%d train=%.6f val=%.6f score=%.6f',
                    epoch,
                    row['train_loss'],
                    validation_loss,
                    row['validation_score'],
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.elapsed_seconds = time.perf_counter() - started
        LOGGER.info('AdapterMLP best epoch=%d', self.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError('AdapterMLP must be fitted before predict().')
        self.model.eval()
        features = torch.from_numpy(self._transform(X))
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), 4096):
                prediction = self.model(
                    features[start:start + 4096].to(self.device)
                )
                batches.append(prediction.detach().cpu().numpy())
        return np.concatenate(batches, axis=0).reshape(
            len(X), len(self.target_names)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'RealMLP-Adapter',
            'training': 'supervised-gradient',
            'architecture': 'shared-trunk-task-adapters',
            'shared_hidden_sizes': list(SHARED_SIZES),
            'adapter_hidden_size': ADAPTER_SIZE,
            'targets': self.target_names,
            'max_epochs': self.epochs,
            'best_iteration': self.best_iteration,
            'validation_metric': 'ficr-aware-loss',
            'loss_name': 'ficr-aware',
            'selection_metric': 'ficr-aware-loss',
            'ficr_weight': self.config.ficr_weight,
            'ficr_temperature': self.config.ficr_temperature,
            'learning_rate': self.config.learning_rate,
            'lr_schedule': 'coslog4',
            'dropout': DROPOUT,
            'dropout_schedule': 'constant',
            'weight_decay': WEIGHT_DECAY,
            'weight_decay_schedule': 'constant',
            'optimizer': 'adam',
            'squared_momentum': 0.95,
            'gradient_clip_norm': 5.0,
            'preprocessing': 'median-iqr-clip10',
            'target_normalization': False,
            'target_masking': 'finite-target-and-capacity-factor>=0.10',
            'early_stopping': False,
            'training_history': self.training_history,
            'device': self.device,
            'n_threads': self.n_threads,
            'elapsed_seconds': self.elapsed_seconds,
        }
