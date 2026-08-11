"""KPX group/turbine metadata loading utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

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
        "발전_그룹",
    ),

    "manufacturer": (
        "manufacturer",
        "maker",
        "제작사",
        "제조사",
        "제조업체",
        "터빈제조사",
        "터빈_제조사",
        "turbine_manufacturer",
    ),

    "model": (
        "model",
        "model_name",
        "turbine_model",
        "모델",
        "모델명",
        "터빈모델",
        "터빈_모델",
        "기종",
        "터빈기종",
    ),

    "rated_power": (
        "rated_power",
        "rated_capacity",
        "turbine_capacity",
        "unit_capacity",
        "설비용량",
        "설비용량_mw",
        "설비용량(MW)",
        "정격출력",
        "정격_출력",
        "정격용량",
        "정격_용량",
        "터빈용량",
        "터빈정격용량",
    ),

    "rotor": (
        "rotor",
        "rotor_diameter",
        "rotor_diameter_m",
        "Rotor Diameter(m)",
        "로터",
        "로터직경",
        "로터_직경",
    ),

    "hub": (
        "hub",
        "hub_height",
        "hub_height_m",
        "Hub Height(m)",
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


def _find_column(
    columns: Iterable[object],
    aliases: Iterable[str],
) -> str | None:
    normalised = {
        _normalise_name(col): str(col)
        for col in columns
    }

    alias_norm = [
        _normalise_name(alias)
        for alias in aliases
    ]

    for alias in alias_norm:
        if alias in normalised:
            return normalised[alias]

    for norm, original in normalised.items():
        if any(
            alias in norm or norm in alias
            for alias in alias_norm
        ):
            return original

    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    extracted = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(
            r"([-+]?\d+(?:\.\d+)?)",
            expand=False,
        )
    )

    return pd.to_numeric(
        extracted,
        errors="coerce",
    )


def _canonical_target(value: object) -> str | None:
    """Convert KPX group representation to target column name.

    Supports examples such as:
    1
    1.0
    "1"
    "1.0"
    "Group 1"
    "KPX Group 1"
    "KPX 1그룹"
    "1그룹"
    "kpx_group_1"
    """

    if pd.isna(value):
        return None

    # Excel frequently reads group IDs as floats: 1.0, 2.0, 3.0.
    if isinstance(value, (int, float)):
        numeric = float(value)

        if numeric.is_integer():
            group_number = int(numeric)

            if group_number in (1, 2, 3):
                return f"kpx_group_{group_number}"

    raw = str(value).strip()

    # "1.0", "2.0", "3.0"
    try:
        numeric = float(raw)

        if numeric.is_integer():
            group_number = int(numeric)

            if group_number in (1, 2, 3):
                return f"kpx_group_{group_number}"
    except ValueError:
        pass

    text = _normalise_name(raw)

    if text in TARGET_COLS:
        return text

    # kpx_group_1 / group_1 / kpx 1 / 그룹1 etc.
    match = re.search(
        r"(?:kpx|group|그룹)[^0-9]*([123])",
        text,
        flags=re.I,
    )

    if match:
        return f"kpx_group_{match.group(1)}"

    # 1그룹 / 2그룹 / 3그룹
    match = re.search(
        r"([123])[^0-9]*(?:group|그룹)",
        text,
        flags=re.I,
    )

    if match:
        return f"kpx_group_{match.group(1)}"

    # Last-resort: a standalone group number
    match = re.fullmatch(
        r"([123])(?:_0+)?",
        text,
    )

    if match:
        return f"kpx_group_{match.group(1)}"

    return None


def _mode_or_unknown(
    series: pd.Series | None,
) -> str:
    if series is None:
        return "UNKNOWN"

    values = (
        series.dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        ~values.str.lower().isin(
            {
                "",
                "nan",
                "none",
            }
        )
    ]

    if values.empty:
        return "UNKNOWN"

    mode = values.mode()

    return str(
        mode.iloc[0]
        if not mode.empty
        else values.iloc[0]
    )


def _median_or_nan(
    series: pd.Series | None,
) -> float:
    if series is None:
        return float("nan")

    numeric = _to_numeric(series).dropna()

    if numeric.empty:
        return float("nan")

    return float(numeric.median())


def _header_score(values: Iterable[object]) -> int:
    """Score a raw Excel row as a possible table header."""
    names = [
        _normalise_name(value)
        for value in values
        if pd.notna(value)
    ]

    if not names:
        return 0

    score = 0

    all_aliases = {
        key: tuple(
            _normalise_name(alias)
            for alias in aliases
        )
        for key, aliases in _ALIASES.items()
    }

    for aliases in all_aliases.values():
        matched = any(
            alias == name
            or alias in name
            or name in alias
            for name in names
            for alias in aliases
            if name
        )

        if matched:
            score += 1

    return score


def _detect_header_row(
    path: Path,
    sheet_name: str,
    max_scan_rows: int = 30,
) -> int | None:
    """Find the most likely metadata header row.

    info.xlsx can contain title / description rows above
    the real table header.  We therefore read the first
    rows with header=None and score them against known
    metadata column aliases.
    """
    preview = pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=max_scan_rows,
        engine="openpyxl",
    )

    if preview.empty:
        return None

    best_row: int | None = None
    best_score = 0

    for row_index in range(len(preview)):
        score = _header_score(
            preview.iloc[row_index].tolist()
        )

        if score > best_score:
            best_score = score
            best_row = row_index

    # At minimum require two recognisable metadata columns.
    if best_score >= 2:
        return best_row

    return None


def _read_sheet_with_detected_header(
    path: Path,
    sheet_name: str,
) -> pd.DataFrame:
    header_row = _detect_header_row(
        path,
        sheet_name,
    )

    if header_row is not None:
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=header_row,
            engine="openpyxl",
        )
    else:
        # Fallback for ordinary first-row-header sheets.
        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            header=0,
            engine="openpyxl",
        )

    frame = (
        frame
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .copy()
    )

    # Remove duplicate columns caused by merged / decorated Excel cells.
    frame = frame.loc[
        :,
        ~frame.columns.astype(str).str.match(
            r"^Unnamed:"
        ),
    ]

    return frame


def _sheet_frames(
    path: Path,
) -> list[pd.DataFrame]:
    excel = pd.ExcelFile(
        path,
        engine="openpyxl",
    )

    frames: list[pd.DataFrame] = []

    for sheet_name in excel.sheet_names:
        frame = _read_sheet_with_detected_header(
            path,
            sheet_name,
        )

        if frame.empty:
            continue

        frame["__sheet_name__"] = sheet_name
        frames.append(frame)

    if not frames:
        raise ValueError(
            f"No readable metadata table was found in {path}."
        )

    return frames


def _extract_group_rows(
    frames: list[pd.DataFrame],
) -> pd.DataFrame:
    extracted: list[pd.DataFrame] = []

    for frame in frames:
        group_col = _find_column(
            frame.columns,
            _ALIASES["group"],
        )

        if group_col is None:
            sheet_target = _canonical_target(
                frame["__sheet_name__"].iloc[0]
            )

            if sheet_target is None:
                continue

            local = frame.copy()
            local["__target__"] = sheet_target

        else:
            local = frame.copy()

            local["__target__"] = (
                local[group_col]
                .map(_canonical_target)
            )

            local = local[
                local["__target__"].notna()
            ]

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
            "Could not identify KPX group rows in "
            "info.xlsx after automatic header detection. "
            "Expected a group-like column or "
            "group-specific sheet names. "
            f"Available columns={available}"
        )

    return pd.concat(
        extracted,
        axis=0,
        ignore_index=True,
        sort=False,
    )


def load_group_metadata(
    path: str | Path,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Competition metadata file not found: {path}. "
            "Place info.xlsx under the configured "
            "data directory."
        )

    frames = _sheet_frames(path)
    rows = _extract_group_rows(frames)

    manufacturer_col = _find_column(
        rows.columns,
        _ALIASES["manufacturer"],
    )
    model_col = _find_column(
        rows.columns,
        _ALIASES["model"],
    )
    rated_power_col = _find_column(
        rows.columns,
        _ALIASES["rated_power"],
    )
    rotor_col = _find_column(
        rows.columns,
        _ALIASES["rotor"],
    )
    hub_col = _find_column(
        rows.columns,
        _ALIASES["hub"],
    )
    latitude_col = _find_column(
        rows.columns,
        _ALIASES["latitude"],
    )
    longitude_col = _find_column(
        rows.columns,
        _ALIASES["longitude"],
    )

    metadata_rows: list[
        dict[str, object]
    ] = []

    for target in TARGET_COLS:
        group = rows.loc[
            rows["__target__"] == target
        ]

        if group.empty:
            raise ValueError(
                f"info.xlsx contains no metadata rows "
                f"for {target}."
            )

        metadata_rows.append(
            {
                "target": target,
                "meta__group_id": target,
                "meta__manufacturer": (
                    _mode_or_unknown(
                        None
                        if manufacturer_col is None
                        else group[manufacturer_col]
                    )
                ),
                "meta__model": (
                    _mode_or_unknown(
                        None
                        if model_col is None
                        else group[model_col]
                    )
                ),
                "meta__group_capacity_kw": float(
                    CAPACITY_KWH[target]
                ),
                "meta__turbine_rated_power_kw": (
                    _median_or_nan(
                        None
                        if rated_power_col is None
                        else group[rated_power_col]
                    )
                ),
                "meta__rotor_diameter_m": (
                    _median_or_nan(
                        None
                        if rotor_col is None
                        else group[rotor_col]
                    )
                ),
                "meta__hub_height_m": (
                    _median_or_nan(
                        None
                        if hub_col is None
                        else group[hub_col]
                    )
                ),
                "meta__latitude": (
                    _median_or_nan(
                        None
                        if latitude_col is None
                        else group[latitude_col]
                    )
                ),
                "meta__longitude": (
                    _median_or_nan(
                        None
                        if longitude_col is None
                        else group[longitude_col]
                    )
                ),
            }
        )

    metadata = (
        pd.DataFrame(metadata_rows)
        .set_index("target")
        .loc[TARGET_COLS]
    )

    for column in _CATEGORICAL_OUTPUTS:
        metadata[column] = (
            metadata[column]
            .astype("string")
        )

    for column in _NUMERIC_OUTPUTS:
        metadata[column] = (
            pd.to_numeric(
                metadata[column],
                errors="coerce",
            )
            .astype(float)
        )

    return metadata


def categorical_metadata_columns() -> tuple[str, ...]:
    return _CATEGORICAL_OUTPUTS


def numeric_metadata_columns() -> tuple[str, ...]:
    return _NUMERIC_OUTPUTS