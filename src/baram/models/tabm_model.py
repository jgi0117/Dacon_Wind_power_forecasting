'''Supervised TabM regressor with competition-score early stopping.'''

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from baram.config import PipelineConfig
from baram.metrics import target_score
from .base import RegressionModel


LOGGER = logging.getLogger('baram.pipeline')


class TabMModel(RegressionModel):
    def __init__(self, config: PipelineConfig, epochs: int | None = None) -> None:
        self.config = config
        self.epochs = int(epochs or config.max_epochs)
        self.best_iteration = self.epochs
        self.best_score: float | None = None
        self.model: Any | None = None
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.device = 'cpu'
        self.elapsed_seconds = 0.0
        self.training_history: list[dict[str, float | int | None]] = []

    def _transform(self, X: pd.DataFrame, *, fit: bool = False) -> np.ndarray:
        values = X.to_numpy(dtype=np.float32, copy=True)
        if fit:
            values = self.imputer.fit_transform(values)
            values = self.scaler.fit_transform(values)
        else:
            values = self.scaler.transform(self.imputer.transform(values))
        return np.asarray(values, dtype=np.float32)

    def fit(
        self, X: pd.DataFrame, y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> 'TabMModel':
        import torch
        from tabm import TabM

        started = time.perf_counter()
        torch.manual_seed(self.config.seed)
        self.device = self.config.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(self.device)
        X_train = self._transform(X, fit=True)
        y_train = y.to_numpy(dtype=np.float32, copy=True)
        X_val = None if X_valid is None else self._transform(X_valid)
        y_val = None if y_valid is None else y_valid.to_numpy(dtype=np.float32, copy=True)

        self.model = TabM.make(
            n_num_features=X_train.shape[1], d_out=1, k=32,
            n_blocks=3, d_block=512, dropout=0.1, arch_type='tabm',
        ).to(device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=1e-4,
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            generator=generator,
        )
        best_state = None
        wait = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_rows = 0
            for features, target in loader:
                features, target = features.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(features).squeeze(-1)
                loss = ((prediction - target[:, None]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += float(loss.detach().cpu()) * len(features)
                train_rows += len(features)
            if X_val is None or y_val is None:
                continue
            prediction = self._predict_array(X_val)
            validation_loss = float(np.mean((prediction - y_val) ** 2))
            score = target_score(y_val, prediction)
            self.training_history.append({
                'step': epoch,
                'train_loss': train_loss_sum / max(train_rows, 1),
                'validation_loss': validation_loss,
                'validation_score': score,
            })
            LOGGER.info('TabM epoch=%d score=%.7f', epoch, score)
            if self.best_score is None or score > self.best_score + self.config.early_stopping_min_delta:
                self.best_score, self.best_iteration, wait = score, epoch, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
                if wait >= self.config.early_stopping_patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.elapsed_seconds = time.perf_counter() - started
        return self

    def _predict_array(self, values: np.ndarray) -> np.ndarray:
        import torch
        if self.model is None:
            raise RuntimeError('TabM must be fitted before predict().')
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), self.config.batch_size * 4):
                batch = torch.from_numpy(values[start:start + self.config.batch_size * 4]).to(self.device)
                predictions.append(self.model(batch).squeeze(-1).mean(dim=1).cpu().numpy())
        return np.concatenate(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_array(self._transform(X))

    def metadata(self) -> dict[str, Any]:
        return {
            'model': 'TabM', 'training': 'supervised-gradient',
            'max_epochs': self.epochs, 'best_iteration': self.best_iteration,
            'best_validation_score': self.best_score, 'ensemble_size': 32,
            'loss_name': 'mse', 'selection_metric': 'competition-score',
            'training_history': self.training_history,
            'device': self.device, 'elapsed_seconds': self.elapsed_seconds,
        }
