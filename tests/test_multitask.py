from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from baram.metrics import (
    CAPACITY_KWH, TARGET_COLS, activity_loss_torch, ficr_aware_loss_torch,
    ficr_boundary_consistency_loss_torch,
    relu_ficr_aware_loss_torch,
    temporal_group_dro_ficr_loss_torch,
)
from baram.models.realmlp_model import (
    _pack_activity_block_metadata,
    _pack_reliability_metadata,
    _quarter_block_ids,
    _unpack_activity_block_metadata,
    _unpack_reliability_metadata,
)
from baram.models.stacked_realmlp_model import (
    STACK_FEATURE_COLUMNS,
    _append_stack_features,
)
from baram.models.temporal_correction_model import (
    N_PREDICTION_LAGS,
    _append_prediction_context,
)
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

    def test_sigmoid_loss_respects_continuous_sample_weight(self) -> None:
        actual = torch.full((2, 1), 0.50)
        prediction = torch.tensor([[0.50], [0.80]])
        unweighted = ficr_aware_loss_torch(
            actual, prediction, ficr_weight=0.0
        )
        weighted = ficr_aware_loss_torch(
            actual,
            prediction,
            ficr_weight=0.0,
            sample_weight=torch.tensor([[1.0], [0.2]]),
        )
        self.assertLess(float(weighted), float(unweighted))
        self.assertAlmostEqual(float(weighted), 0.05, places=3)

    def test_activity_loss_uses_low_output_and_masks_missing(self) -> None:
        actual = torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]])
        logits = torch.zeros((1, 2, 3), requires_grad=True)
        loss = activity_loss_torch(actual, logits)
        loss.mean().backward()
        self.assertEqual(loss.shape, (1,))
        self.assertGreater(abs(float(logits.grad[0, 0, 0])), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 2]), 0.0)

    def test_temporal_group_dro_emphasizes_hard_block(self) -> None:
        actual = torch.full((4, 1), 0.50)
        prediction = torch.tensor(
            [[[0.50], [0.50], [0.70], [0.70]]], requires_grad=True
        )
        block_ids = torch.tensor([0.0, 0.0, 1.0, 1.0])
        balanced, losses = temporal_group_dro_ficr_loss_torch(
            actual, prediction, block_ids, {0: 0.5, 1: 0.5},
            ficr_weight=1.0,
        )
        hard_weighted, _ = temporal_group_dro_ficr_loss_torch(
            actual, prediction, block_ids, {0: 0.1, 1: 0.9},
            ficr_weight=1.0,
        )
        self.assertEqual(set(losses), {0, 1})
        self.assertGreater(
            float(losses[1].detach()), float(losses[0].detach())
        )
        self.assertGreater(
            float(hard_weighted.detach()), float(balanced.detach())
        )
        hard_weighted.mean().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_boundary_consistency_penalizes_ficr_decision_disagreement(self) -> None:
        actual = torch.full((4, 1), 0.50)
        consistent = torch.full((2, 4, 1), 0.565, requires_grad=True)
        disagreeing = torch.stack([
            torch.full((4, 1), 0.55),
            torch.full((4, 1), 0.59),
        ]).requires_grad_()
        consistent_loss = ficr_boundary_consistency_loss_torch(
            actual, consistent
        )
        disagreement_loss = ficr_boundary_consistency_loss_torch(
            actual, disagreeing
        )
        self.assertEqual(tuple(disagreement_loss.shape), (2,))
        self.assertAlmostEqual(float(consistent_loss.mean().detach()), 0.0)
        self.assertGreater(
            float(disagreement_loss.mean().detach()),
            float(consistent_loss.mean().detach()),
        )
        disagreement_loss.mean().backward()
        self.assertTrue(torch.isfinite(disagreeing.grad).all())
        self.assertGreater(float(disagreeing.grad.abs().sum()), 0.0)

    def test_relu_ficr_loss_has_three_gradient_regions(self) -> None:
        actual = torch.full((3, 1), 0.50)
        prediction = torch.tensor(
            [[0.53], [0.56], [0.59]], requires_grad=True
        )
        loss = relu_ficr_aware_loss_torch(
            actual, prediction, ficr_weight=0.75, margin=0.005
        )
        loss.mean().backward()
        per_sample_gradient = prediction.grad[:, 0] * len(actual)
        expected = torch.tensor([0.25, 0.4375, 1.0])
        torch.testing.assert_close(
            per_sample_gradient, expected, atol=2e-3, rtol=2e-3
        )

    def test_quarter_blocks_are_chronological_and_compact(self) -> None:
        index = pd.to_datetime([
            '2022-12-31 23:00', '2023-01-01 00:00', '2023-04-01 00:00'
        ])
        ids, labels = _quarter_block_ids(index)
        np.testing.assert_array_equal(ids, [0.0, 1.0, 2.0])
        self.assertEqual(labels, {0: '2022Q4', 1: '2023Q1', 2: '2023Q2'})

    def test_temporal_metadata_round_trip_preserves_activity(self) -> None:
        activity = np.asarray([
            [0.0, 1.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]
        ], dtype=np.float32)
        packed = _pack_activity_block_metadata(
            activity, np.asarray([0.0, 3.0, 7.0], dtype=np.float32)
        )
        decoded, block_ids = _unpack_activity_block_metadata(
            torch.from_numpy(packed)
        )
        np.testing.assert_array_equal(decoded.numpy(), activity)
        np.testing.assert_array_equal(block_ids.numpy(), [0.0, 3.0, 7.0])

    def test_reliability_metadata_round_trip_with_temporal_codes(self) -> None:
        activity = np.asarray([
            [0.0, 1.0, -1.0], [1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]
        ], dtype=np.float32)
        reliability = np.asarray([
            [1.0, 1.0, 0.2], [1.0, 1.0, 0.65], [1.0, 1.0, 1.0]
        ], dtype=np.float32)
        packed_temporal = _pack_activity_block_metadata(
            activity, np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
        )
        packed = _pack_reliability_metadata(
            packed_temporal, reliability
        )
        temporal_codes, restored_reliability = (
            _unpack_reliability_metadata(torch.tensor(packed))
        )
        restored_activity, block_ids = _unpack_activity_block_metadata(
            temporal_codes
        )
        restored_activity = torch.where(
            restored_activity == 2.0,
            torch.full_like(restored_activity, -1.0),
            restored_activity,
        )
        np.testing.assert_array_equal(restored_activity.numpy(), activity)
        np.testing.assert_array_equal(block_ids.numpy(), [0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            restored_reliability.numpy(), reliability, atol=2e-6
        )

    def test_group3_stack_features_use_only_stage1_predictions(self) -> None:
        index = pd.date_range('2023-01-01', periods=2, freq='h')
        features = pd.DataFrame({'weather': [3.0, 4.0]}, index=index)
        prediction = np.asarray([[0.2, 0.4], [0.8, 0.5]])
        stacked = _append_stack_features(features, prediction)
        self.assertEqual(list(stacked.columns[-4:]), list(STACK_FEATURE_COLUMNS))
        np.testing.assert_allclose(
            stacked['stack__proxy_mean_cf'], [0.3, 0.65]
        )
        np.testing.assert_allclose(
            stacked['stack__proxy_difference_cf'], [-0.2, 0.3]
        )
        np.testing.assert_array_equal(stacked['weather'], features['weather'])


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

    def test_read_only_eligible_mask_is_supported(self) -> None:
        index = pd.date_range('2024-01-01', periods=2, freq='h')
        features = pd.DataFrame({'x': [1.0, 2.0]}, index=index)
        targets = pd.DataFrame(
            [[10_800.0, 10_800.0, 10_500.0]] * 2,
            index=index,
            columns=TARGET_COLS,
        )
        eligible = np.array([True, False])
        eligible.flags.writeable = False

        selected_x, _ = _all_history_masked_training_data(
            features, targets, eligible
        )

        self.assertEqual(list(selected_x.index), [index[0]])


class TemporalPredictionContextTest(unittest.TestCase):
    def test_context_uses_current_and_previous_base_predictions(self) -> None:
        index = pd.date_range('2024-01-01 01:00', periods=15, freq='h')
        prediction = pd.DataFrame(
            np.repeat(np.arange(15, dtype=np.float32)[:, None], 3, axis=1),
            index=index,
            columns=TARGET_COLS,
        )
        features = pd.DataFrame({'weather': [1.0, 2.0, 3.0]}, index=index[12:])

        result = _append_prediction_context(features, prediction)

        self.assertEqual(N_PREDICTION_LAGS, 12)
        self.assertEqual(
            result.loc[index[12], 'base_prediction__kpx_group_1__lag_00'],
            12.0,
        )
        self.assertEqual(
            result.loc[index[12], 'base_prediction__kpx_group_1__lag_12'],
            0.0,
        )
        self.assertFalse(any('target' in column for column in result.columns))

    def test_context_rejects_incomplete_prediction_history(self) -> None:
        index = pd.date_range('2024-01-01 01:00', periods=12, freq='h')
        prediction = pd.DataFrame(
            np.zeros((12, 3)), index=index, columns=TARGET_COLS
        )
        features = pd.DataFrame({'weather': [1.0]}, index=[index[-1]])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            _append_prediction_context(features, prediction)

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
