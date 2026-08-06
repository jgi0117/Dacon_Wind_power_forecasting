from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from baram.config import PipelineConfig
from baram.metrics import TARGET_COLS
from baram.models.realmlp_adapter_model import (
    ADAPTER_SIZE,
    SHARED_SIZES,
    RealMLPAdapterModel,
    _AdapterNetwork,
)


class AdapterArchitectureTest(unittest.TestCase):
    def test_each_target_has_a_separate_adapter(self) -> None:
        model = _AdapterNetwork.build(n_features=8, n_targets=3)
        prediction = model(torch.zeros(4, 8))

        self.assertEqual(tuple(prediction.shape), (4, 3))
        self.assertEqual(len(model.adapters), 3)
        self.assertEqual(model.adapters[0][0].in_features, SHARED_SIZES[-1])
        self.assertEqual(model.adapters[0][0].out_features, ADAPTER_SIZE)
        self.assertIsNot(
            model.adapters[0][0].weight,
            model.adapters[1][0].weight,
        )

    def test_small_masked_fit_records_group_history(self) -> None:
        rng = np.random.default_rng(42)
        columns = [f'x_{index}' for index in range(8)]
        X = pd.DataFrame(
            rng.normal(size=(64, 8)).astype(np.float32),
            columns=columns,
        )
        y = pd.DataFrame(
            rng.uniform(0.11, 0.8, size=(64, 3)).astype(np.float32),
            columns=TARGET_COLS,
        )
        y.loc[:24, 'kpx_group_3'] = np.nan
        X_valid = pd.DataFrame(
            rng.normal(size=(16, 8)).astype(np.float32),
            columns=columns,
        )
        y_valid = pd.DataFrame(
            rng.uniform(0.11, 0.8, size=(16, 3)).astype(np.float32),
            columns=TARGET_COLS,
        )
        config = PipelineConfig(
            max_epochs=2,
            batch_size=16,
            learning_rate=0.02,
            n_jobs=1,
            device='cpu',
        )

        model = RealMLPAdapterModel(config).fit(
            X, y, X_valid, y_valid
        )
        prediction = model.predict(X_valid)

        self.assertEqual(prediction.shape, (16, 3))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertEqual(len(model.training_history), 2)
        self.assertIn(
            'kpx_group_3__validation_loss',
            model.training_history[0],
        )
        self.assertIn(model.best_iteration, (1, 2))


if __name__ == '__main__':
    unittest.main()
