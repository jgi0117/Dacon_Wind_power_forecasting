"""KPX group/turbine metadata loading utilities.

This module reads ``data/info.xlsx`` and converts the competition metadata into
one metadata row per KPX target group.

The loader is intentionally tolerant to Korean/English column names because
the exact display labels in info.xlsx may differ. Required information is
resolved by aliases and helpful errors are raised when the group column cannot
be identified.

Categorical metadata:
- group id
- manufacturer
- turbine model

Numeric metadata:
- official group capacity (from CAPACITY_KWH)
- turbine rated power (when present in info.xlsx)
- rotor diameter (when present)
- hub height (when present)
- latitude / longitude (when present)

Missing optional fields are retained as NaN / "UNKNOWN" instead of fabricating
values.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from baram.metrics import CAPACITY_KWH, TARGET_COLS


_CATEGORICAL_OUTPUTS = (
    "meta__group_id",
    "meta__manufacturer",
    "meta__model",
)

_NUMERIC_OUTPUTS = (
    "meta__group_capacity_kw",
    "meta__turbine_rated_power_kw",
    "meta__rotor_diameter_m",
    "meta__hub_height_m",
    "meta__latitude",
    "meta__longitude",
)


def _normalise_name(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[\s\-()/\[\].]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


_ALIASES = {
    "group": (
        "kpx_group",
        "group",
        "group_id",
        "kpx그룹",
        "kpx_그룹",
        "그룹",
        "발전그룹",
    ),
    "manufacturer": (
        "manufacturer",
        "maker",
        "제조사",
        "제조업체",
        "터빈제조사",
        "turbine_manufacturer",
    ),
    "model": (
        "model",
        "model_name",
        "turbine_model",
        "터빈모델",
        "모델",
        "기종",
        "터빈기종",
    ),
    "rated_power": (
        "rated_power",
        "rated_capacity",
        "turbine_capacity",
        "unit_capacity",
        "정격출력",
        "정격용량",
        "터빈용량",
        "터빈정격용량",
    ),
    "rotor": (
        "rotor",
        "rotor_diameter",
        "rotor_diameter_m",
        "로터",
        "로터직경",
        "로터_직경",
    ),
    "hub": (
        "hub",
        "hub_height",
        "hub_height_m",
        "허브",
        "허브높이",
        "허브_높이",
    ),
    "latitude": (
        "latitude",
        "lat",
        "위도",
    ),
    "longitude": (
        "longitude",
        "lon",
        "lng",
        "경도",
    ),
}


def _find_column(columns: Iterable[object], aliases: Iterable[str]) -> str | None:
    normalised = {_normalise_name(col): str(col) for col in columns}
    alias_norm = [_normalise_name(alias) for alias in aliases]

    for alias in alias_norm:
        if alias in normalised:
            return normalised[alias]

    # Fuzzy fallback: useful for labels such as "터빈 모델명", "KPX Group 번호".
    for norm, original in normalised.items():
        if any(alias in norm or norm in alias for alias in alias_norm):
            return original
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    extracted = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce")


def _canonical_target(value: object) -> str | None:
    text = _normalise_name(value)
    if text in TARGET_COLS:
        return text

    # Examples: Group 1, KPX Group 1, kpx_group_1, 그룹1.
    match = re.search(r"(?:group|그룹|kpx)[^0-9]*([123])", text, flags=re.I)
    if match:
        return f"kpx_group_{match.group(1)}"

    # Conservative final fallback: accept a bare 1/2/3.
    if text in {"1", "2", "3"}:
        return f"kpx_group_{text}"
    return None


def _mode_or_unknown(series: pd.Series | None) -> str:
    if series is None:
        return "UNKNOWN"
    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )
    values = values[~values.str.lower().isin({"", "nan", "none"})]
    if values.empty:
        return "UNKNOWN"
    mode = values.mode()
    return str(mode.iloc[0] if not mode.empty else values.iloc[0])


def _median_or_nan(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    numeric = _to_numeric(series).dropna()
    return float(numeric.median()) if not numeric.empty else float("nan")


def _sheet_frames(path: Path) -> list[pd.DataFrame]:
    workbook = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []

    for sheet_name, frame in workbook.items():
        if frame is None or frame.empty:
            continue
        local = frame.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()
        if local.empty:
            continue
        local["__sheet_name__"] = sheet_name
        frames.append(local)

    if not frames:
        raise ValueError(f"No readable metadata table was found in {path}.")
    return frames


def _extract_group_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    extracted: list[pd.DataFrame] = []

    for frame in frames:
        group_col = _find_column(frame.columns, _ALIASES["group"])

        # Some workbooks place one group per sheet and omit a group column.
        if group_col is None:
            sheet_target = _canonical_target(frame["__sheet_name__"].iloc[0])
            if sheet_target is None:
                continue
            local = frame.copy()
            local["__target__"] = sheet_target
        else:
            local = frame.copy()
            local["__target__"] = local[group_col].map(_canonical_target)
            local = local[local["__target__"].notna()]

        if not local.empty:
            extracted.append(local)

    if not extracted:
        available = sorted(
            {
                str(column)
                for frame in frames
                for column in frame.columns
                if column != "__sheet_name__"
            }
        )
        raise ValueError(
            "Could not identify KPX group rows in info.xlsx. "
            "Expected a group-like column or group-specific sheet names. "
            f"Available columns={available}"
        )

    # Different sheets can have different schemas. concat handles missing columns.
    return pd.concat(extracted, axis=0, ignore_index=True, sort=False)


def load_group_metadata(path: str | Path) -> pd.DataFrame:
    """Return one metadata row for each target in TARGET_COLS.

    The returned DataFrame is indexed by target name.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Competition metadata file not found: {path}. "
            "Place info.xlsx under the configured data directory."
        )

    frames = _sheet_frames(path)
    rows = _extract_group_rows(frames)

    manufacturer_col = _find_column(rows.columns, _ALIASES["manufacturer"])
    model_col = _find_column(rows.columns, _ALIASES["model"])
    rated_power_col = _find_column(rows.columns, _ALIASES["rated_power"])
    rotor_col = _find_column(rows.columns, _ALIASES["rotor"])
    hub_col = _find_column(rows.columns, _ALIASES["hub"])
    latitude_col = _find_column(rows.columns, _ALIASES["latitude"])
    longitude_col = _find_column(rows.columns, _ALIASES["longitude"])

    metadata_rows: list[dict[str, object]] = []
    for target in TARGET_COLS:
        group = rows.loc[rows["__target__"] == target]
        if group.empty:
            raise ValueError(f"info.xlsx contains no metadata rows for {target}.")

        metadata_rows.append(
            {
                "target": target,
                "meta__group_id": target,
                "meta__manufacturer": _mode_or_unknown(
                    None if manufacturer_col is None else group[manufacturer_col]
                ),
                "meta__model": _mode_or_unknown(
                    None if model_col is None else group[model_col]
                ),
                # CAPACITY_KWH is the official total group capacity used by scoring.
                "meta__group_capacity_kw": float(CAPACITY_KWH[target]),
                "meta__turbine_rated_power_kw": _median_or_nan(
                    None if rated_power_col is None else group[rated_power_col]
                ),
                "meta__rotor_diameter_m": _median_or_nan(
                    None if rotor_col is None else group[rotor_col]
                ),
                "meta__hub_height_m": _median_or_nan(
                    None if hub_col is None else group[hub_col]
                ),
                "meta__latitude": _median_or_nan(
                    None if latitude_col is None else group[latitude_col]
                ),
                "meta__longitude": _median_or_nan(
                    None if longitude_col is None else group[longitude_col]
                ),
            }
        )

    metadata = pd.DataFrame(metadata_rows).set_index("target").loc[TARGET_COLS]

    for column in _CATEGORICAL_OUTPUTS:
        metadata[column] = metadata[column].astype("string")
    for column in _NUMERIC_OUTPUTS:
        metadata[column] = pd.to_numeric(metadata[column], errors="coerce").astype(float)

    return metadata


def categorical_metadata_columns() -> tuple[str, ...]:
    return _CATEGORICAL_OUTPUTS


def numeric_metadata_columns() -> tuple[str, ...]:
    return _NUMERIC_OUTPUTS
