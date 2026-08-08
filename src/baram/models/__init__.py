"""Lazy model registry for independent optional-dependency execution."""

from __future__ import annotations

from baram.config import PipelineConfig
from baram.constants import SUPPORTED_MODEL_NAMES

from .base import RegressionModel


MODEL_NAMES = SUPPORTED_MODEL_NAMES


def build_model(
    name: str,
    config: PipelineConfig,
    *,
    iterations: dict[str, int] | int | None = None,
) -> RegressionModel:
    if name == "realmlp":
        if config.group3_stacking:
            from .stacked_realmlp_model import StackedGroup3RealMLPModel

            return StackedGroup3RealMLPModel(
                config,
                iterations=iterations,
            )

        if config.teacher_student_distillation:
            from .distilled_realmlp_model import (
                DistilledTemporalRealMLPModel,
            )

            return DistilledTemporalRealMLPModel(
                config,
                iterations=iterations,
            )

        if config.temporal_prediction_correction:
            from .temporal_correction_model import (
                TemporalCorrectionRealMLPModel,
            )

            return TemporalCorrectionRealMLPModel(
                config,
                iterations=iterations,
            )

        # Direct baseline after this change:
        # weather/time + group/turbine metadata -> shared single-target RealMLP.
        from .group_conditioned_realmlp_model import (
            GroupConditionedRealMLPModel,
        )

        epochs = (
            iterations.get("student", iterations.get("multitask"))
            if isinstance(iterations, dict)
            else iterations
        )
        return GroupConditionedRealMLPModel(
            config,
            epochs=epochs,
        )

    raise ValueError(f"Unsupported model: {name}")


__all__ = ["MODEL_NAMES", "RegressionModel", "build_model"]