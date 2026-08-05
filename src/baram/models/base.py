"""모델 구현이 따르는 최소 학습/추론 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class RegressionModel(ABC):
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> "RegressionModel":
        """모델을 학습한다."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """capacity factor를 추론한다."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """재현과 전체 재학습에 필요한 정보를 반환한다."""
