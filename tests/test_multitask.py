from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from baram.metrics import CAPACITY_KWH, TARGET_COLS, ficr_aware_loss_torch
from baram.workflows.training import (
    _all_history_masked_training_data,
    _capacity_factor_frame,
    _restore_prediction_frame,
)


class MultiTaskLossTest(unittest.TestCase):
    def test_missing_target_has_zero_gradient(self) -> None:
        actual = torch.tensor([
            [0.20, 0.30, float('nan')],
            [0.40, 0.20, 0.50],
        ])
        prediction = torch.full((1, 2, 3), 0.25, requires_grad=True)

        loss = ficr_aware_loss_torch(actual, prediction)
        loss.mean().backward()

        self.assertEqual(loss.shape, (1,))
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(float(prediction.grad[0, 0, 2]), 0.0)


class MultiTaskScalingTest(unittest.TestCase):
    def test_all_history_keeps_rows_with_partial_targets(self) -> None:
        index = pd.date_range('2022-12-31 23:00', periods=4, freq='h')
        features = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0]}, index=index)
        targets = pd.DataFrame(
            [
                [10_800.0, 5_400.0, np.nan],
                [12_000.0, 6_000.0, np.nan],
                [13_000.0, 7_000.0, 10_500.0],
                [np.nan, np.nan, np.nan],
            ],
            index=index,
            columns=TARGET_COLS,
        )

        selected_x, selected_y = _all_history_masked_training_data(
            features, targets
        )

        self.assertEqual(list(selected_x.index), list(index[:3]))
        self.assertTrue(
            pd.isna(selected_y.loc[index[0], 'kpx_group_3'])
        )
        self.assertAlmostEqual(
            selected_y.loc[index[0], 'kpx_group_1'], 0.5
        )

    def test_capacity_factor_round_trip_and_shape(self) -> None:
        index = pd.date_range('2024-01-01', periods=2, freq='h')
        generation = pd.DataFrame(
            [[10_800.0, 5_400.0, 5_250.0], [21_600.0, 10_800.0, 10_500.0]],
            index=index,
            columns=TARGET_COLS,
        )

        factors = _capacity_factor_frame(generation)
        restored = _restore_prediction_frame(factors.to_numpy(), index)

        for target in TARGET_COLS:
            np.testing.assert_allclose(restored[target], generation[target])
        self.assertEqual(CAPACITY_KWH['kpx_group_3'], 21_000.0)

    def test_prediction_shape_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            _restore_prediction_frame(np.zeros((2, 2)), pd.RangeIndex(2))


if __name__ == '__main__':
    unittest.main()
