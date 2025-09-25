"""Statistical helpers for inspection analytics."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

STATUS_ORDER = ["qualified", "unqualified", "not_applicable", "unknown"]


def compute_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return high level metrics for the filtered dataset."""

    total = int(len(df))
    counts = df["status"].value_counts().to_dict()
    return {
        "total_records": total,
        "qualified": int(counts.get("qualified", 0)),
        "unqualified": int(counts.get("unqualified", 0)),
        "not_applicable": int(counts.get("not_applicable", 0)),
        "unknown": int(counts.get("unknown", 0)),
    }


def status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy dataframe with the status distribution."""

    counts = df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    result = counts.reset_index()
    result.columns = ["status", "count"]
    result["percentage"] = result["count"] / max(counts.sum(), 1) * 100
    return result


def aggregate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Aggregate inspection counts and status ratios by a dimension."""

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not present in dataframe")

    grouped = df.groupby(column)
    summary = grouped.size().to_frame("count")

    status_counts = (
        df.pivot_table(index=column, columns="status", values="#", aggfunc="count")
        if "#" in df.columns
        else grouped["status"].value_counts().unstack(fill_value=0)
    )

    summary = summary.join(status_counts, how="left").fillna(0)
    summary = summary.sort_values("count", ascending=False)
    return summary.reset_index()


def time_series(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Compute the number of inspections per time period."""

    if "qc_timestamp" not in df:
        raise KeyError("Dataset must contain the 'qc_timestamp' column")

    valid = df.dropna(subset=["qc_timestamp"])
    if valid.empty:
        return pd.DataFrame(columns=["period", "count", "qualified", "unqualified"])

    resampled = (
        valid.set_index("qc_timestamp")["status"].groupby(pd.Grouper(freq=freq)).value_counts()
    )
    tidy = resampled.unstack(fill_value=0)
    tidy["count"] = tidy.sum(axis=1)
    tidy = tidy.reset_index().rename(columns={"qc_timestamp": "period"})
    columns = ["period", "count"] + [col for col in STATUS_ORDER if col in tidy]
    return tidy.loc[:, columns]


def top_entities(
    df: pd.DataFrame,
    column: str,
    limit: int = 10,
    status: Optional[str] = None,
) -> pd.DataFrame:
    """Return the top ``limit`` entities sorted by inspection counts.

    Parameters
    ----------
    column:
        Dimension to aggregate by (e.g. ``"creator"``).
    status:
        Optional status filter. When provided only rows with this
        ``status`` will be considered for the ranking.
    """

    if status and status not in STATUS_ORDER:
        raise ValueError(f"Unknown status '{status}'")

    if status:
        subset = df[df["status"] == status]
    else:
        subset = df

    result = aggregate_by(subset, column)
    return result.head(limit)


_NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _coerce_numeric(value: object) -> Optional[float]:
    """Convert ``inputvalue`` style fields into numeric values when possible."""

    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace(",", "")
    if not _NUMERIC_PATTERN.match(normalized):
        return None

    try:
        return float(normalized)
    except ValueError:  # pragma: no cover - defensive guard
        return None


def attach_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with an additional ``numeric_value`` column."""

    result = df.copy()
    if "inputvalue" not in result:
        result["numeric_value"] = pd.NA
        return result

    result["numeric_value"] = result["inputvalue"].apply(_coerce_numeric)
    return result


def stage_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a matrix showing which ``opno`` steps exist for each ``xb005``."""

    required = {"opno", "xb005"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise KeyError(f"Dataset缺少必要的列: {missing}")

    subset = df.dropna(subset=["opno", "xb005"])
    if subset.empty:
        return pd.DataFrame()

    matrix = (
        subset.assign(has_record=1)
        .drop_duplicates(subset=["opno", "xb005"])
        .pivot(index="opno", columns="xb005", values="has_record")
        .fillna(0)
        .astype(int)
    )

    matrix = matrix.sort_index().sort_index(axis=1)
    matrix.columns = matrix.columns.astype(str)
    return matrix


def qcitem_catalog(
    df: pd.DataFrame, wheel_numbers: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Return a tidy table mapping wheels to processes and QC items."""

    subset = df.copy()
    if wheel_numbers:
        wheel_set = list(wheel_numbers)
        subset = subset[subset["xb005"].isin(wheel_set)]

    if subset.empty:
        return pd.DataFrame(columns=["xb005", "opno", "qcitem", "qcorder", "inputvalue"])

    sort_keys = [key for key in ["xb005", "opno", "qcorder"] if key in subset.columns]
    if sort_keys:
        subset = subset.sort_values(sort_keys, na_position="last")

    columns: List[str] = [
        "xb005",
        "opno",
        "qcitem",
        "qcorder",
        "inputvalue",
        "status" if "status" in subset.columns else None,
        "creator" if "creator" in subset.columns else None,
        "qc_timestamp" if "qc_timestamp" in subset.columns else None,
    ]
    columns = [col for col in columns if col is not None]
    return subset.loc[:, columns]


def numeric_qcitem_candidates(df: pd.DataFrame) -> List[str]:
    """Return QC 项列表，这些项目的 ``inputvalue`` 可以转换为数值。"""

    working = df if "numeric_value" in df.columns else attach_numeric_values(df)
    mask = working["numeric_value"].notna()
    return sorted(working.loc[mask, "qcitem"].dropna().unique().tolist())


def qcitem_numeric_trend(
    df: pd.DataFrame, qcitem: str, strategy: str = "latest"
) -> pd.DataFrame:
    """Aggregate numeric ``qcitem`` readings across wheels.

    Parameters
    ----------
    qcitem:
        需要分析的检测项目名称。
    strategy:
        ``"latest"`` 表示针对每个 ``xb005`` 取最近一次记录；
        ``"mean"`` 则对所有记录求平均。
    """

    working = df if "numeric_value" in df.columns else attach_numeric_values(df)

    subset = working[(working["qcitem"] == qcitem) & working["numeric_value"].notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=["xb005", "numeric_value", "qc_timestamp", "timestamp_label"])

    if strategy == "mean":
        aggregated = subset.groupby("xb005", as_index=False)["numeric_value"].mean()
        aggregated["qc_timestamp"] = pd.NaT
        result = aggregated
    else:
        if "qc_timestamp" in subset.columns:
            subset = subset.sort_values(["xb005", "qc_timestamp"], na_position="last")
            latest = subset.groupby("xb005", as_index=False).tail(1)
            result = latest.loc[:, ["xb005", "numeric_value", "qc_timestamp"]]
        else:
            result = subset.loc[:, ["xb005", "numeric_value"]]
            result["qc_timestamp"] = pd.NaT

    if "qc_timestamp" in result.columns and result["qc_timestamp"].notna().any():
        result = result.sort_values(["qc_timestamp", "xb005"], na_position="last")
    else:
        result = result.sort_values("xb005")
    result = result.reset_index(drop=True)
    if "qc_timestamp" in result.columns:
        result["timestamp_label"] = result["qc_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        result["timestamp_label"] = None

    return result