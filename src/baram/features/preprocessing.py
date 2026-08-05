"""BARAM 2026 누수 방지형 기상 전처리 파이프라인.

실행 예시
---------
python preprocessing.py --data-dir data --output-dir artifacts --mode hybrid

생성물
------
- X_train.pkl / X_test.pkl: 시간 인덱스를 가진 float32 특성
- y_train.pkl: 원 단위(kWh)의 그룹별 라벨
- feature_manifest.json: 설정, 크기, 결측 및 컬럼 목록
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..metrics import CAPACITY_KWH, TARGET_COLS


TIME_COL = "forecast_kst_dtm"
AVAILABLE_COL = "data_available_kst_dtm"
GRID_COL = "grid_id"
WEATHER_KEY_COLS = [TIME_COL, AVAILABLE_COL, GRID_COL, "latitude", "longitude"]

LDAPS_STATIC_COLS = ["surface_0_lsm", "surface_0_h"]

LDAPS_GRID_FEATURES = [
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
    "ws10",
    "ws50_mid",
    "ws50_component_range",
]

GFS_GRID_FEATURES = [
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "surface_0_gust",
    "ws10",
    "ws80",
    "ws100",
    "wind_shear_100_10",
    "gust_excess_10",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """전처리 동작 설정."""

    mode: Literal["aggregate", "wind_grid", "hybrid"] = "hybrid"
    aggregate_stats: tuple[str, ...] = ("mean", "std", "min", "max")
    dtype: str = "float32"
    drop_constant: bool = True
    add_missing_indicators: bool = True


@dataclass
class FeatureBundle:
    """학습·평가 특성과 라벨 묶음."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    train_available_time: pd.Series
    test_available_time: pd.Series
    diagnostics: dict[str, object]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator.abs() > 1e-6)


def _prediction_cutoff(forecast_time: pd.Series) -> pd.Series:
    """각 예측 행에 적용되는 전일 14:00 KST를 계산한다."""

    delivery_day = (forecast_time - pd.Timedelta(hours=1)).dt.normalize()
    return delivery_day - pd.Timedelta(days=1) + pd.Timedelta(hours=14)


def validate_weather_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    expected_grid_count: int,
) -> dict[str, object]:
    """예보 키, 공간 완전성, 공개시각을 검사한다."""

    missing_keys = set(WEATHER_KEY_COLS) - set(frame.columns)
    if missing_keys:
        raise ValueError(f"{source}: 키 컬럼 누락 {sorted(missing_keys)}")
    if frame[WEATHER_KEY_COLS[:3]].isna().any().any():
        raise ValueError(f"{source}: 시간 또는 grid 키에 결측이 있습니다.")
    if frame.duplicated([TIME_COL, GRID_COL]).any():
        raise ValueError(f"{source}: (시간, grid_id) 중복이 있습니다.")

    counts = frame.groupby(TIME_COL, sort=False)[GRID_COL].nunique()
    if not counts.eq(expected_grid_count).all():
        bad = counts[~counts.eq(expected_grid_count)]
        raise ValueError(
            f"{source}: 시간당 격자 수가 {expected_grid_count}가 아닌 시각이 "
            f"{len(bad)}개 있습니다."
        )

    available_counts = frame.groupby(TIME_COL, sort=False)[AVAILABLE_COL].nunique()
    if not available_counts.eq(1).all():
        raise ValueError(f"{source}: 한 대상시각에 여러 공개시각이 있습니다.")

    unique_time = frame[[TIME_COL, AVAILABLE_COL]].drop_duplicates(TIME_COL)
    cutoff = _prediction_cutoff(unique_time[TIME_COL])
    violations = unique_time[AVAILABLE_COL] > cutoff
    if violations.any():
        raise ValueError(
            f"{source}: 예측기준시점 이후 공개된 예보가 {int(violations.sum())}개 있습니다."
        )

    numeric = frame.select_dtypes(include=np.number)
    return {
        "rows": int(len(frame)),
        "times": int(frame[TIME_COL].nunique()),
        "grids": int(frame[GRID_COL].nunique()),
        "missing_cells": int(frame.isna().sum().sum()),
        "infinite_cells": int(np.isinf(numeric.to_numpy()).sum()),
        "availability_violations": int(violations.sum()),
    }


def load_weather(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    frame[AVAILABLE_COL] = pd.to_datetime(frame[AVAILABLE_COL], errors="raise")
    return frame.sort_values([AVAILABLE_COL, GRID_COL, TIME_COL]).reset_index(drop=True)


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"], errors="raise")
    if labels["kst_dtm"].duplicated().any():
        raise ValueError("라벨 시간에 중복이 있습니다.")
    return labels.set_index("kst_dtm").sort_index()[TARGET_COLS]


class WeatherFeaturePipeline:
    """Train에 적합하고 Train/Test에 동일하게 적용하는 전처리기."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self.grid_fallbacks_: dict[str, pd.DataFrame] = {}
        self.feature_columns_: list[str] | None = None
        self.constant_columns_: list[str] = []
        self._is_fitted = False

    @staticmethod
    def _fit_grid_fallback(frame: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            col
            for col in frame.select_dtypes(include=np.number).columns
            if col != GRID_COL
        ]
        return frame.groupby(GRID_COL, sort=True)[numeric_cols].median()

    def fit(self, ldaps_train: pd.DataFrame, gfs_train: pd.DataFrame) -> "WeatherFeaturePipeline":
        validate_weather_frame(ldaps_train, source="ldaps_train", expected_grid_count=16)
        validate_weather_frame(gfs_train, source="gfs_train", expected_grid_count=9)
        self.grid_fallbacks_["ldaps"] = self._fit_grid_fallback(ldaps_train)
        self.grid_fallbacks_["gfs"] = self._fit_grid_fallback(gfs_train)

        features, _ = self._build_features(ldaps_train, gfs_train, fit_stage=True)
        if self.config.drop_constant:
            self.constant_columns_ = [
                col
                for col in features.columns
                if features[col].nunique(dropna=False) <= 1
                and 'row_had_missing' not in col
            ]
            features = features.drop(columns=self.constant_columns_)
        self.feature_columns_ = features.columns.tolist()
        self._is_fitted = True
        return self

    def transform(
        self, ldaps: pd.DataFrame, gfs: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not self._is_fitted or self.feature_columns_ is None:
            raise RuntimeError("fit()을 먼저 호출해야 합니다.")
        features, available = self._build_features(ldaps, gfs, fit_stage=False)
        features = features.drop(columns=self.constant_columns_, errors="ignore")

        missing = set(self.feature_columns_) - set(features.columns)
        extra = set(features.columns) - set(self.feature_columns_)
        if missing or extra:
            raise ValueError(
                "Train/Test 특성 스키마가 다릅니다. "
                f"누락={sorted(missing)[:10]}, 추가={sorted(extra)[:10]}"
            )
        features = features[self.feature_columns_]
        if features.isna().any().any():
            bad = features.columns[features.isna().any()].tolist()
            raise ValueError(f"최종 특성에 결측이 남았습니다: {bad[:10]}")
        if np.isinf(features.to_numpy()).any():
            raise ValueError("최종 특성에 inf가 남았습니다.")
        return features.astype(self.config.dtype), available

    def fit_transform(
        self, ldaps_train: pd.DataFrame, gfs_train: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.fit(ldaps_train, gfs_train)
        return self.transform(ldaps_train, gfs_train)

    def _impute_from_train(self, frame: pd.DataFrame, source: str) -> pd.DataFrame:
        result = frame.copy()
        numeric_cols = [
            col
            for col in result.select_dtypes(include=np.number).columns
            if col != GRID_COL
        ]
        original_missing = result[numeric_cols].isna()

        # 동일 공개시각의 한 예보 배치 안에서만 grid별 시간 보간한다.
        result[numeric_cols] = result.groupby(
            [AVAILABLE_COL, GRID_COL], sort=False
        )[numeric_cols].transform(
            lambda series: series.interpolate(method="linear", limit_direction="both")
        )

        # 배치 경계 또는 전체 구간 결측은 Train의 grid별 중앙값으로만 대체한다.
        fallback = self.grid_fallbacks_[source]
        for col in numeric_cols:
            if result[col].isna().any():
                result[col] = result[col].fillna(result[GRID_COL].map(fallback[col]))

        if result[numeric_cols].isna().any().any():
            bad = result.columns[result.isna().any()].tolist()
            raise ValueError(f"{source}: Train 기반 대체 후 결측이 남았습니다: {bad}")

        if self.config.add_missing_indicators:
            result["row_had_missing"] = original_missing.any(axis=1).astype(np.int8)
        return result

    @staticmethod
    def _derive_ldaps(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        u10 = out["heightAboveGround_10_10u"]
        v10 = out["heightAboveGround_10_10v"]
        out["ws10"] = np.hypot(u10, v10)
        out["dir10_u"] = _safe_ratio(u10, out["ws10"]).fillna(0.0)
        out["dir10_v"] = _safe_ratio(v10, out["ws10"]).fillna(0.0)

        u50 = (
            out["heightAboveGround_50_50MUmax"]
            + out["heightAboveGround_50_50MUmin"]
        ) / 2.0
        v50 = (
            out["heightAboveGround_50_50MVmax"]
            + out["heightAboveGround_50_50MVmin"]
        ) / 2.0
        out["ws50_mid"] = np.hypot(u50, v50)
        out["dir50_u"] = _safe_ratio(u50, out["ws50_mid"]).fillna(0.0)
        out["dir50_v"] = _safe_ratio(v50, out["ws50_mid"]).fillna(0.0)
        out["ws50_component_range"] = np.hypot(
            out["heightAboveGround_50_50MUmax"]
            - out["heightAboveGround_50_50MUmin"],
            out["heightAboveGround_50_50MVmax"]
            - out["heightAboveGround_50_50MVmin"],
        )
        out["ws10_sq"] = out["ws10"].pow(2)
        out["ws10_cube"] = out["ws10"].pow(3)
        out["ws50_sq"] = out["ws50_mid"].pow(2)
        out["ws50_cube"] = out["ws50_mid"].pow(3)
        out["wind_shear_50_10"] = out["ws50_mid"] - out["ws10"]
        out["wind_ratio_50_10"] = _safe_ratio(out["ws50_mid"], out["ws10"]).clip(0, 10)
        return out

    @staticmethod
    def _derive_gfs(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        pairs = {
            "10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
            "80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
            "100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
        }
        for height, (u_col, v_col) in pairs.items():
            speed = f"ws{height}"
            out[speed] = np.hypot(out[u_col], out[v_col])
            out[f"dir{height}_u"] = _safe_ratio(out[u_col], out[speed]).fillna(0.0)
            out[f"dir{height}_v"] = _safe_ratio(out[v_col], out[speed]).fillna(0.0)
            out[f"{speed}_sq"] = out[speed].pow(2)
            out[f"{speed}_cube"] = out[speed].pow(3)

        out["wind_shear_100_10"] = out["ws100"] - out["ws10"]
        out["wind_shear_100_80"] = out["ws100"] - out["ws80"]
        out["wind_ratio_100_10"] = _safe_ratio(out["ws100"], out["ws10"]).clip(0, 10)
        out["wind_ratio_100_80"] = _safe_ratio(out["ws100"], out["ws80"]).clip(0, 10)
        out["gust_excess_10"] = out["surface_0_gust"] - out["ws10"]
        return out

    def _aggregate(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        excluded = set(WEATHER_KEY_COLS) | set(LDAPS_STATIC_COLS)
        value_cols = [
            col
            for col in frame.select_dtypes(include=np.number).columns
            if col not in excluded
        ]
        aggregated = frame.groupby(TIME_COL, sort=True)[value_cols].agg(
            list(self.config.aggregate_stats)
        )
        aggregated.columns = [
            f"{prefix}__{column}__{stat}" for column, stat in aggregated.columns
        ]
        return aggregated

    @staticmethod
    def _pivot_selected(
        frame: pd.DataFrame, prefix: str, selected: list[str]
    ) -> pd.DataFrame:
        selected = [col for col in selected if col in frame.columns]
        pivoted = frame.pivot(index=TIME_COL, columns=GRID_COL, values=selected)
        pivoted.columns = [
            f"{prefix}__{column}__grid_{int(grid):02d}"
            for column, grid in pivoted.columns
        ]
        return pivoted.sort_index(axis=1)

    @staticmethod
    def _time_features(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        left = ldaps[[TIME_COL, AVAILABLE_COL]].drop_duplicates(TIME_COL).set_index(TIME_COL)
        right = gfs[[TIME_COL, AVAILABLE_COL]].drop_duplicates(TIME_COL).set_index(TIME_COL)
        if not left.index.equals(right.index):
            raise ValueError("LDAPS와 GFS의 대상 시간 집합이 다릅니다.")
        if not left[AVAILABLE_COL].equals(right[AVAILABLE_COL]):
            raise ValueError("LDAPS와 GFS의 공개시각이 다릅니다.")

        forecast = left.index.to_series()
        available = left[AVAILABLE_COL].copy()
        center = forecast - pd.Timedelta(minutes=30)
        delivery_start = forecast - pd.Timedelta(hours=1)
        features = pd.DataFrame(index=left.index)
        features.index.name = TIME_COL
        features["time__lead_hour"] = (
            (forecast - available).dt.total_seconds() / 3600.0
        ).to_numpy()
        features["time__end_hour"] = forecast.dt.hour.to_numpy()
        features["time__interval_hour"] = delivery_start.dt.hour.to_numpy()
        features["time__delivery_month"] = delivery_start.dt.month.to_numpy()
        features["time__delivery_doy"] = delivery_start.dt.dayofyear.to_numpy()

        center_hour = center.dt.hour + center.dt.minute / 60.0
        features["time__hour_sin"] = np.sin(2 * np.pi * center_hour / 24.0).to_numpy()
        features["time__hour_cos"] = np.cos(2 * np.pi * center_hour / 24.0).to_numpy()
        features["time__doy_sin"] = np.sin(
            2 * np.pi * center.dt.dayofyear / 365.25
        ).to_numpy()
        features["time__doy_cos"] = np.cos(
            2 * np.pi * center.dt.dayofyear / 365.25
        ).to_numpy()
        features["time__month_sin"] = np.sin(
            2 * np.pi * (center.dt.month - 1) / 12.0
        ).to_numpy()
        features["time__month_cos"] = np.cos(
            2 * np.pi * (center.dt.month - 1) / 12.0
        ).to_numpy()
        return features, available

    def _build_features(
        self,
        ldaps_raw: pd.DataFrame,
        gfs_raw: pd.DataFrame,
        *,
        fit_stage: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        validate_weather_frame(ldaps_raw, source="ldaps", expected_grid_count=16)
        validate_weather_frame(gfs_raw, source="gfs", expected_grid_count=9)

        if fit_stage:
            ldaps = ldaps_raw.copy()
            gfs = gfs_raw.copy()
            if self.config.add_missing_indicators:
                ldaps["row_had_missing"] = ldaps.isna().any(axis=1).astype(np.int8)
                gfs["row_had_missing"] = gfs.isna().any(axis=1).astype(np.int8)
        else:
            ldaps = self._impute_from_train(ldaps_raw, "ldaps")
            gfs = self._impute_from_train(gfs_raw, "gfs")

        ldaps = self._derive_ldaps(ldaps)
        gfs = self._derive_gfs(gfs)
        time_features, available = self._time_features(ldaps, gfs)

        blocks = [time_features]
        if self.config.mode in {"aggregate", "hybrid"}:
            blocks.extend(
                [self._aggregate(ldaps, "ldaps"), self._aggregate(gfs, "gfs")]
            )
        if self.config.mode in {"wind_grid", "hybrid"}:
            blocks.extend(
                [
                    self._pivot_selected(ldaps, "ldaps", LDAPS_GRID_FEATURES),
                    self._pivot_selected(gfs, "gfs", GFS_GRID_FEATURES),
                ]
            )

        features = pd.concat(blocks, axis=1).sort_index()
        if features.columns.duplicated().any():
            duplicates = features.columns[features.columns.duplicated()].tolist()
            raise ValueError(f"중복 특성명이 있습니다: {duplicates[:10]}")
        return features, available.reindex(features.index)


def make_target_data(
    labels: pd.DataFrame,
    target: str,
    *,
    strategy: Literal["all", "hard", "soft", "settlement"] = "soft",
    low_generation_weight: float = 0.2,
    settlement_alpha: float = 1.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """그룹별 학습 mask, CF 타깃, sample weight를 만든다."""

    if target not in TARGET_COLS:
        raise ValueError(f"알 수 없는 타깃: {target}")
    y = labels[target]
    valid_label = y.notna()
    evaluation_zone = y >= CAPACITY_KWH[target] * 0.10

    if strategy == "hard":
        train_mask = valid_label & evaluation_zone
        weight = pd.Series(1.0, index=y.index)
    elif strategy == "soft":
        train_mask = valid_label
        weight = pd.Series(
            np.where(evaluation_zone, 1.0, low_generation_weight), index=y.index
        )
    elif strategy == "settlement":
        train_mask = valid_label & evaluation_zone
        weight = 1.0 + settlement_alpha * (y / CAPACITY_KWH[target]).clip(0, 1.1)
    elif strategy == "all":
        train_mask = valid_label
        weight = pd.Series(1.0, index=y.index)
    else:
        raise ValueError(f"지원하지 않는 strategy: {strategy}")

    target_cf = y / CAPACITY_KWH[target]
    return train_mask, target_cf, weight.astype(float)


def delivery_year(index: pd.DatetimeIndex) -> np.ndarray:
    """00:00을 전일 마지막 구간으로 처리한 발전 구간 연도."""

    return (index - pd.Timedelta(hours=1)).year.to_numpy()


def make_2024_holdout(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    years = delivery_year(index)
    return years <= 2023, years == 2024


def build_feature_bundle(
    data_dir: str | Path = "data",
    config: PreprocessingConfig | None = None,
) -> FeatureBundle:
    data_dir = Path(data_dir)
    paths = {
        "ldaps_train": data_dir / "train" / "ldaps_train.csv",
        "gfs_train": data_dir / "train" / "gfs_train.csv",
        "labels": data_dir / "train" / "train_labels.csv",
        "ldaps_test": data_dir / "test" / "ldaps_test.csv",
        "gfs_test": data_dir / "test" / "gfs_test.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"필수 파일이 없습니다: {missing_files}")

    ldaps_train = load_weather(paths["ldaps_train"])
    gfs_train = load_weather(paths["gfs_train"])
    ldaps_test = load_weather(paths["ldaps_test"])
    gfs_test = load_weather(paths["gfs_test"])
    labels = load_labels(paths["labels"])

    diagnostics = {
        "ldaps_train": validate_weather_frame(
            ldaps_train, source="ldaps_train", expected_grid_count=16
        ),
        "gfs_train": validate_weather_frame(
            gfs_train, source="gfs_train", expected_grid_count=9
        ),
        "ldaps_test": validate_weather_frame(
            ldaps_test, source="ldaps_test", expected_grid_count=16
        ),
        "gfs_test": validate_weather_frame(
            gfs_test, source="gfs_test", expected_grid_count=9
        ),
    }

    pipeline = WeatherFeaturePipeline(config)
    X_train, train_available = pipeline.fit_transform(ldaps_train, gfs_train)
    X_test, test_available = pipeline.transform(ldaps_test, gfs_test)
    labels = labels.reindex(X_train.index)

    if not labels.index.equals(X_train.index):
        raise ValueError("학습 특성과 라벨 시간 인덱스가 다릅니다.")
    if len(X_train) != 26_304 or len(X_test) != 8_760:
        raise ValueError(
            f"예상하지 못한 시간 수: train={len(X_train)}, test={len(X_test)}"
        )

    diagnostics.update(
        {
            "feature_mode": pipeline.config.mode,
            "n_features": int(X_train.shape[1]),
            "constant_columns_dropped": pipeline.constant_columns_,
            "train_feature_missing": int(X_train.isna().sum().sum()),
            "test_feature_missing": int(X_test.isna().sum().sum()),
        }
    )
    return FeatureBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=labels,
        train_available_time=train_available,
        test_available_time=test_available,
        diagnostics=diagnostics,
    )


def save_bundle(
    bundle: FeatureBundle,
    output_dir: str | Path,
    config: PreprocessingConfig,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.X_train.to_pickle(output_dir / "X_train.pkl")
    bundle.X_test.to_pickle(output_dir / "X_test.pkl")
    bundle.y_train.to_pickle(output_dir / "y_train.pkl")

    manifest = {
        "config": asdict(config),
        "train_shape": list(bundle.X_train.shape),
        "test_shape": list(bundle.X_test.shape),
        "train_start": str(bundle.X_train.index.min()),
        "train_end": str(bundle.X_train.index.max()),
        "test_start": str(bundle.X_test.index.min()),
        "test_end": str(bundle.X_test.index.max()),
        "features": bundle.X_train.columns.tolist(),
        "diagnostics": bundle.diagnostics,
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--mode",
        choices=["aggregate", "wind_grid", "hybrid"],
        default="hybrid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessingConfig(mode=args.mode)
    bundle = build_feature_bundle(args.data_dir, config)
    save_bundle(bundle, args.output_dir, config)
    print(
        f"완료: X_train={bundle.X_train.shape}, "
        f"X_test={bundle.X_test.shape}, output={Path(args.output_dir).resolve()}"
    )
    print(json.dumps(bundle.diagnostics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
