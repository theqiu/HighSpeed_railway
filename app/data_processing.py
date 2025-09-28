"""Utilities for loading and cleaning the inspection dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import re

import numpy as np
import pandas as pd


@dataclass
class FilterParams:
    """Container with commonly used filter parameters."""

    lot_numbers: Optional[Iterable[str]] = None
    qc_items: Optional[Iterable[str]] = None
    creators: Optional[Iterable[str]] = None
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None


def read_dataset(source: Union[str, "os.PathLike[str]", bytes]) -> pd.DataFrame:
    """Load the dataset from a CSV file or bytes.

    Parameters
    ----------
    source:
        Path to the CSV file or the raw bytes.

    Returns
    -------
    pandas.DataFrame
        The raw dataset with column names stripped from surrounding
        whitespace.
    """

    if isinstance(source, bytes):
        df = pd.read_csv(pd.io.common.BytesIO(source))
    else:
        df = pd.read_csv(source)

    df.columns = [str(col).strip() for col in df.columns]
    return df


def _parse_datetime(value: object) -> Optional[pd.Timestamp]:
    """Parse the ``qcdate`` column into a ``Timestamp``.

    The source data mixes integer timestamps written in scientific
    notation with string based timestamps in ``YYYYMMDDHHMMSS`` format.
    Any value that cannot be parsed is converted to ``NaT``.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    # Remove decimal point representation (e.g. ``2.02401E+13``)
    try:
        as_int = int(float(text))
        text = f"{as_int:014d}"
    except (TypeError, ValueError):
        pass

    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return pd.to_datetime(text, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue

    # Fall back to pandas' parser which handles ISO formats gracefully.
    try:
        return pd.to_datetime(text, errors="coerce")
    except (ValueError, TypeError):  # pragma: no cover - defensive guard.
        return None


_KEY_VALUE_PATTERN = re.compile(r"([^:：；;]+)[:：]\s*([-+]?\d+(?:\.\d+)?)")


def _extract_markers(text: str) -> dict[str, float]:
    """Extract numeric flags from key-value style status strings."""

    normalized = (
        text.replace("；", ";")
        .replace("：", ":")
        .replace("，", ",")
        .replace(" ", "")
    )
    matches = _KEY_VALUE_PATTERN.findall(normalized)
    markers: dict[str, float] = {}
    for key, value in matches:
        key_clean = key.strip()
        if not key_clean:
            continue
        try:
            markers[key_clean] = float(value)
        except ValueError:
            continue
    return markers


def _lookup_marker(markers: dict[str, float], keyword: str) -> Optional[float]:
    """Return the numeric flag for keys containing ``keyword``."""

    for key, value in markers.items():
        if keyword in key:
            return value
    return None


def _parse_status_from_text(text: str) -> Optional[str]:
    """Infer the inspection status from textual markers."""

    cleaned = text.strip()
    if not cleaned:
        return None

    lower = cleaned.lower()
    if "不适用" in cleaned or "n/a" in lower or lower == "na":
        return "not_applicable"

    markers = _extract_markers(cleaned)
    not_ok = _lookup_marker(markers, "不合格")
    ok = _lookup_marker(markers, "合格")

    if not_ok is not None:
        if not_ok > 0:
            return "unqualified"
        # 明确标记为 0 且存在合格标识时按合格处理
        if ok is not None and ok > 0:
            return "qualified"
        if ok is None:
            return "unqualified"

    if ok is not None and ok > 0:
        return "qualified"

    if "不合格" in cleaned:
        return "unqualified"
    if "合格" in cleaned:
        return "qualified"

    # 处理类似“101适用”这类字段，统一归入 not_applicable/unknown
    if re.search(r"\d+适用", cleaned):
        return "not_applicable"

    return None


def _parse_status(row: pd.Series) -> str:
    """Derive a normalised status for each record."""

    value = row.get("inputvalue")
    order = row.get("qcorder")

    for candidate in (value, order):
        if pd.isna(candidate):
            continue
        text = str(candidate).strip()
        if not text:
            continue
        try:
            numeric = float(text)
        except ValueError:
            status = _parse_status_from_text(text)
            if status:
                return status
            continue

        if np.isfinite(numeric):
            if numeric > 0:
                return "qualified"
            if numeric == 0:
                return "unqualified"

    return "unknown"


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw dataset.

    The function adds the following columns:

    ``qc_timestamp``
        Parsed timestamp representation of the ``qcdate`` column.
    ``status``
        Normalised inspection status (``qualified``/``unqualified``/``not_applicable``/``unknown``).
    ``year`` ``month`` ``day``
        Calendar fields extracted from the timestamp (where available).
    """

    df = df.copy()
    df["qc_timestamp"] = df.get("qcdate").apply(_parse_datetime)
    df["status"] = df.apply(_parse_status, axis=1)

    df["year"] = df["qc_timestamp"].dt.year
    df["month"] = df["qc_timestamp"].dt.month
    df["day"] = df["qc_timestamp"].dt.day

    return df


def apply_filters(df: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    """Filter the dataframe according to the provided parameters."""

    mask = pd.Series(True, index=df.index)

    if params.lot_numbers:
        mask &= df["lotno"].isin(list(params.lot_numbers))
    if params.qc_items:
        mask &= df["qcitem"].isin(list(params.qc_items))
    if params.creators:
        mask &= df["creator"].isin(list(params.creators))
    if params.date_range and any(params.date_range):
        start, end = params.date_range
        if start is not None:
            mask &= df["qc_timestamp"] >= start
        if end is not None:
            mask &= df["qc_timestamp"] <= end

    return df.loc[mask].copy()
