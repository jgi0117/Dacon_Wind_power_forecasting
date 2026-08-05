"""비용이 큰 fold별 반복 수 탐색 결과의 검증 가능한 캐시."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PipelineConfig
from ..splitting import IterationFold


CACHE_SCHEMA_VERSION = 1
MODEL_CONFIG_VERSION = "lgbm-catboost-2026-08-04-v1"
CACHE_FILENAME = "iteration_selection_cache.json"


def cache_signature(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    folds: list[IterationFold],
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_config_version": MODEL_CONFIG_VERSION,
        "seed": config.seed,
        "train_shape": X_train.shape,
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "features": list(X_train.columns),
        "folds": [asdict(fold) for fold in folds],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_iteration_cache(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    folds: list[IterationFold],
    *,
    required_models: tuple[str, ...] = (),
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]] | None:
    if config.refresh_iterations:
        return None
    path = config.iteration_cache_dir / CACHE_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != cache_signature(config, X_train, folds):
        return None
    schedule = payload.get("iteration_schedule", {})
    if any(
        model_name not in schedule
        or any(
            target not in schedule[model_name]
            for target in ("kpx_group_1", "kpx_group_2", "kpx_group_3")
        )
        for model_name in required_models
    ):
        return None
    return payload["iteration_schedule"], payload["iteration_selection"]


def save_iteration_cache(
    config: PipelineConfig,
    X_train: pd.DataFrame,
    folds: list[IterationFold],
    schedule: dict[str, dict[str, int]],
    audit: dict[str, dict[str, Any]],
) -> Path:
    config.iteration_cache_dir.mkdir(parents=True, exist_ok=True)
    path = config.iteration_cache_dir / CACHE_FILENAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("signature") == cache_signature(config, X_train, folds):
            existing_schedule = existing.get("iteration_schedule", {})
            existing_audit = existing.get("iteration_selection", {})
            schedule = existing_schedule | schedule
            audit = existing_audit | audit
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "signature": cache_signature(config, X_train, folds),
        "iteration_schedule": schedule,
        "iteration_selection": audit,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path
