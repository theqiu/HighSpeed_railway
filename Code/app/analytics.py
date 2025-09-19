"""Statistical helpers for inspection analytics."""
from __future__ import annotations
from typing import Dict, Optional
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