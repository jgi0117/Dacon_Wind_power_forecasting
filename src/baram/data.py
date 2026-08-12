"""전처리 산출물 로딩과 제출 파일 생성."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import TARGET_COLS

from baram.constants import TARGET_COLS


def load_artifacts(
    artifacts_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts_dir = Path(artifacts_dir)

    parquet_paths = {
        "X_train": artifacts_dir / "X_train.parquet",
        "y_train": artifacts_dir / "y_train.parquet",
        "X_test": artifacts_dir / "X_test.parquet",
    }

    pickle_paths = {
        "X_train": artifacts_dir / "X_train.pkl",
        "y_train": artifacts_dir / "y_train.pkl",
        "X_test": artifacts_dir / "X_test.pkl",
    }

    # parquet 우선
    if all(path.exists() for path in parquet_paths.values()):
        print(f"[data] Loading parquet artifacts from: {artifacts_dir}")

        X_train = pd.read_parquet(parquet_paths["X_train"])
        y_train = pd.read_parquet(parquet_paths["y_train"])
        X_test = pd.read_parquet(parquet_paths["X_test"])

    # 기존 pickle fallback
    elif all(path.exists() for path in pickle_paths.values()):
        print(f"[data] Loading pickle artifacts from: {artifacts_dir}")

        X_train = pd.read_pickle(pickle_paths["X_train"])
        y_train = pd.read_pickle(pickle_paths["y_train"])
        X_test = pd.read_pickle(pickle_paths["X_test"])

    else:
        missing = [
            str(path)
            for path in parquet_paths.values()
            if not path.exists()
        ]

        raise FileNotFoundError(
            "전처리 산출물을 찾을 수 없습니다. "
            f"Parquet missing: {missing}"
        )

    if not X_train.index.equals(y_train.index):
        raise ValueError(
            "X_train과 y_train의 시간 인덱스가 다릅니다."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "X_train과 X_test의 특성 스키마가 다릅니다."
        )

    missing_targets = [
        col
        for col in TARGET_COLS
        if col not in y_train.columns
    ]

    if missing_targets:
        raise ValueError(
            f"정답 열이 없습니다: {missing_targets}"
        )

    return (
        X_train,
        y_train[TARGET_COLS],
        X_test,
    )


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
