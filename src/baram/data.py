"""전처리 산출물 로딩과 제출 파일 생성."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import TARGET_COLS


def load_artifacts(
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "X_train": artifacts_dir / "X_train.pkl",
        "y_train": artifacts_dir / "y_train.pkl",
        "X_test": artifacts_dir / "X_test.pkl",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"전처리 산출물이 없습니다: {missing}")

    X_train = pd.read_pickle(paths["X_train"])
    y_train = pd.read_pickle(paths["y_train"])
    X_test = pd.read_pickle(paths["X_test"])
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train과 y_train의 시간 인덱스가 다릅니다.")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("학습/평가 특성 스키마가 다릅니다.")
    if not set(TARGET_COLS).issubset(y_train.columns):
        raise ValueError(f"정답 열이 없습니다: {TARGET_COLS}")
    return X_train, y_train[TARGET_COLS], X_test


def write_submission(
    sample_path: Path,
    prediction: pd.DataFrame,
    destination: Path,
) -> None:
    sample = pd.read_csv(sample_path)
    if len(sample) != len(prediction):
        raise ValueError("sample_submission과 평가 예측의 행 수가 다릅니다.")
    for target in TARGET_COLS:
        sample[target] = prediction[target].to_numpy(dtype=float)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(destination, index=False, encoding="utf-8")
