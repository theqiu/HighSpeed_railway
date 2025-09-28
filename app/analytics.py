"""Statistical helpers for inspection analytics."""

from __future__ import annotations

import re
import string
from typing import Dict, Iterable, List, Optional, Tuple

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
    tidy = tidy.reindex(columns=STATUS_ORDER, fill_value=0)
    tidy["count"] = tidy.sum(axis=1)
    tidy = tidy.reset_index().rename(columns={"qc_timestamp": "period"})
    columns = ["period", "count"] + [col for col in STATUS_ORDER if col in tidy]
    return tidy.loc[:, columns]


def top_entities(
    df: pd.DataFrame,
    column: str,
    limit: Optional[int] = 10,
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
    if limit is not None and limit > 0:
        return result.head(limit)
    return result


_NUMERIC_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _extract_order_token(value: object) -> Optional[float]:
    """Parse an order indicator (e.g. ``qcorder``) into a numeric rank."""

    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


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


_SECTION_SUFFIX_PATTERN = re.compile(
    r"(平均值?|平均)(?:\s*[-_/]?\s*\d+)?\s*$",
    re.IGNORECASE,
)
_TRAILING_NOTE_PATTERN = re.compile(r"[（(][^）)]*[)）]\s*$")
_SECTION_PATTERNS = [
    re.compile(
        r"(?P<prefix>.+?)(?:\s*[：:\-—]?\s*)?截面\s*(?P<section>[A-Za-z0-9]+)(?:\s*平均值?|\s*平均)?(?:\s*[-_/]?\s*\d+)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<prefix>.+?)(?:\s*[：:\-—]?\s*)?(?P<section>[A-Za-z0-9]+)\s*截面(?:\s*平均值?|\s*平均)?(?:\s*[-_/]?\s*\d+)?\s*$",
        re.IGNORECASE,
    ),
]


def _normalise_panel_prefix(prefix: Optional[str]) -> Optional[str]:
    """Standardise the extracted panel prefix."""

    if prefix is None:
        return None

    cleaned = str(prefix).strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"[：:\-—]+$", "", cleaned).strip()
    return cleaned or None


_SECTION_ORDER = {letter: idx for idx, letter in enumerate(string.ascii_uppercase, start=1)}


def _infer_panel_side(panel_name: object) -> Optional[str]:
    """Return a side label when a panel refers to the left or right wheel."""

    if pd.isna(panel_name):
        return None

    text = str(panel_name)
    if not text:
        return None

    if "左" in text:
        return "左侧"
    if "右" in text:
        return "右侧"
    return None


def _parse_wheelseat_average(label: object) -> Tuple[Optional[str], Optional[str]]:
    """Extract the panel name and section token from an average ``qcitem`` label."""

    if pd.isna(label):
        return None, None

    text = str(label).strip()
    if not text:
        return None, None

    text = _TRAILING_NOTE_PATTERN.sub("", text).strip()

    for pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            prefix = _normalise_panel_prefix(match.group("prefix"))
            section = match.group("section")
            if prefix and section:
                return prefix, section.upper()

    stripped = _SECTION_SUFFIX_PATTERN.sub("", text).strip()
    if stripped:
        for pattern in _SECTION_PATTERNS:
            match = pattern.search(stripped)
            if match:
                prefix = _normalise_panel_prefix(match.group("prefix"))
                section = match.group("section")
                if prefix and section:
                    return prefix, section.upper()

    stripped = stripped.rstrip("截面").rstrip(":：-—").strip()
    if stripped:
        tail_match = re.search(r"([A-Za-z0-9]+)$", stripped)
        if tail_match:
            section = tail_match.group(1).upper()
            prefix = _normalise_panel_prefix(stripped[: tail_match.start()])
            if prefix and section:
                return prefix, section

    return None, None


def _section_rank(section: Optional[str]) -> float:
    """Return a sortable numeric rank for a wheel seat section token."""

    if section is None:
        return 10_000.0

    text = str(section).strip()
    if not text:
        return 10_000.0

    upper = text.upper()
    if upper in _SECTION_ORDER:
        return float(_SECTION_ORDER[upper])

    if text.isdigit():
        return float(text)

    try:
        return float(text)
    except ValueError:
        # Place unrecognised tokens towards the end while preserving grouping order.
        return 5_000.0


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


def wheelseat_monotonic_analysis(
    df: pd.DataFrame,
    keyword: str = "轮座",
    wheel_column: str = "xb005",
    type_column: Optional[str] = "lotno",
    type_value: Optional[str] = None,
    tolerance: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse whether wheel seat measurements decrease from inside to outside.

    Parameters
    ----------
    keyword:
        Substring used to select relevant ``qcitem`` entries.
    wheel_column:
        Column representing the wheel identifier (default ``xb005``).
    type_column:
        Optional column describing the wheel type (e.g. ``lotno``).
    type_value:
        Optional filter value for ``type_column``. When provided only
        records with this value will be analysed.
    tolerance:
        Allowed positive increase between adjacent measurements. Values
        larger than ``tolerance`` are treated as violations.
    """

    working = attach_numeric_values(df)
    if "numeric_value" not in working:
        return pd.DataFrame(), pd.DataFrame()

    mask = working["qcitem"].astype(str).str.contains(keyword, na=False)
    subset = working[mask & working["numeric_value"].notna()].copy()

    if type_column and type_column in subset.columns and type_value:
        subset = subset[subset[type_column] == type_value]

    if subset.empty or wheel_column not in subset.columns:
        return pd.DataFrame(), pd.DataFrame()

    group_keys = [wheel_column]
    if type_column and type_column in subset.columns:
        group_keys.insert(0, type_column)

    parsed = subset["qcitem"].apply(_parse_wheelseat_average)
    subset["panel_name"] = parsed.map(lambda item: item[0] if item else None)
    subset["section"] = parsed.map(lambda item: item[1] if item else None)

    if "qcorder" in subset.columns:
        subset["order_rank"] = subset["qcorder"].apply(_extract_order_token)
    else:
        subset["order_rank"] = pd.NA

    subset["fallback_rank"] = subset.groupby(group_keys).cumcount().astype(float)
    subset["order_rank"] = subset["order_rank"].astype(float)
    subset["order_rank"] = subset["order_rank"].fillna(subset["fallback_rank"])
    subset = subset.drop(columns="fallback_rank")
    subset = subset.sort_values(group_keys + ["order_rank", "qc_timestamp"], na_position="last")

    panel_subset = subset[subset["panel_name"].notna()].copy()
    if not panel_subset.empty:
        analysis_df = panel_subset
    else:
        analysis_df = subset

    if analysis_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if analysis_df["panel_name"].notna().any():
        panel_order_map: dict[tuple, dict[str, int]] = {}

        def _panel_order(group: pd.Series) -> pd.Series:
            key = tuple(group.name) if isinstance(group.name, tuple) else (group.name,)
            if key not in panel_order_map:
                unique_panels = pd.Index(group.dropna().unique())
                panel_order_map[key] = {panel: idx for idx, panel in enumerate(unique_panels, start=1)}
            mapping = panel_order_map[key]
            return group.map(mapping)

        analysis_df["panel_order"] = (
            analysis_df.groupby(group_keys)["panel_name"].transform(_panel_order)
        )
    else:
        analysis_df["panel_order"] = pd.NA

    analysis_df["section_rank"] = analysis_df["section"].map(_section_rank)
    analysis_df = analysis_df.sort_values(
        group_keys
        + ["panel_order", "section_rank", "order_rank", "qc_timestamp"],
        na_position="last",
    )

    sequences_cols = (
        group_keys
        + [
            "panel_name",
            "section",
            "section_rank",
            "order_rank",
            "qcitem",
            "numeric_value",
            "qc_timestamp",
        ]
    )
    sequences = analysis_df.loc[:, sequences_cols].copy()
    sequences["point_index"] = sequences.groupby(group_keys).cumcount()

    panel_group_keys = group_keys.copy()
    if analysis_df["panel_name"].notna().any():
        panel_group_keys.append("panel_name")

    panel_rows: List[dict[str, object]] = []
    for keys, group in analysis_df.groupby(panel_group_keys):
        ordered = group.copy()
        if ordered["section"].notna().any():
            ordered = ordered[ordered["section"].notna()].copy()
            ordered = ordered.sort_values(["section_rank", "qc_timestamp"], na_position="last")
            ordered = ordered.drop_duplicates(subset="section", keep="last")
            ordered = ordered.sort_values("section_rank")
            values = ordered["numeric_value"].tolist()
            section_sequence = ordered["section"].tolist()
        else:
            ordered = ordered.sort_values(["order_rank", "qc_timestamp"], na_position="last")
            values = ordered["numeric_value"].tolist()
            section_sequence = [str(idx + 1) for idx in range(len(values))]

        if len(values) < 2:
            max_positive = 0.0
            violation_count = 0
            monotonic = True
            violating_steps: List[str] = []
        else:
            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            epsilon = 1e-9
            positive_diffs = [diff for diff in diffs if diff > (tolerance + epsilon)]
            max_positive = max(positive_diffs) if positive_diffs else 0.0
            violation_count = len(positive_diffs)
            monotonic = violation_count == 0
            violating_steps = []
            if positive_diffs:
                for idx, diff in enumerate(diffs):
                    if diff > (tolerance + epsilon):
                        start_section = section_sequence[idx] if idx < len(section_sequence) else str(idx + 1)
                        end_section = (
                            section_sequence[idx + 1]
                            if idx + 1 < len(section_sequence)
                            else str(idx + 2)
                        )
                        violating_steps.append(f"{start_section}->{end_section}")
            else:
                violating_steps = []

        entry: dict[str, object] = {
            "point_count": len(values),
            "monotonic_descending": monotonic,
            "violation_count": violation_count,
            "max_positive_step": round(float(max_positive), 5),
            "first_value": values[0] if values else None,
            "last_value": values[-1] if values else None,
            "violating_sections": ",".join(violating_steps) if violating_steps else pd.NA,
        }

        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        for name, key_value in zip(panel_group_keys, key_tuple):
            entry[name] = key_value

        if "panel_name" not in entry:
            entry["panel_name"] = group.get("panel_name", pd.Series([pd.NA])).iloc[0]

        panel_rows.append(entry)

    panel_summary = pd.DataFrame(panel_rows)
    if panel_summary.empty:
        return pd.DataFrame(), sequences

    panel_summary["panel_side"] = panel_summary["panel_name"].map(_infer_panel_side)
    panel_summary["panel_group"] = panel_summary["panel_name"].combine_first(panel_summary["panel_side"])
    panel_summary["panel_group"] = panel_summary["panel_group"].fillna("未命名测点")

    group_columns = group_keys + ["panel_group"]

    def _merge_sections(values: pd.Series) -> object:
        tokens: List[str] = []
        for value in values.dropna():
            parts = [token for token in str(value).split(",") if token]
            for token in parts:
                if token not in tokens:
                    tokens.append(token)
        return ",".join(tokens) if tokens else pd.NA

    aggregation = panel_summary.groupby(group_columns, as_index=False).agg(
        point_count=("point_count", "sum"),
        violation_count=("violation_count", "sum"),
        max_positive_step=("max_positive_step", "max"),
        first_value=("first_value", "first"),
        last_value=("last_value", "last"),
        panel_side=("panel_side", "first"),
        panel_name=("panel_name", "first"),
        violating_sections=("violating_sections", _merge_sections),
    )
    aggregation["monotonic_descending"] = aggregation["violation_count"] == 0

    order_columns = [
        col for col in [type_column, wheel_column] if col and col in aggregation.columns
    ]
    remaining_cols = [col for col in aggregation.columns if col not in order_columns]
    summary = aggregation[order_columns + remaining_cols]
    sort_columns = order_columns + ["panel_group", "monotonic_descending"]
    ascending = [True] * len(order_columns) + [True, False]
    summary = summary.sort_values(sort_columns, ascending=ascending)

    return summary, sequences


def wheelseat_average_panels(sequence: pd.DataFrame) -> pd.DataFrame:
    """Return average-value panels for a single wheel sequence."""

    if sequence.empty:
        return pd.DataFrame()

    required_columns = {"qcitem", "numeric_value"}
    if not required_columns.issubset(sequence.columns):
        return pd.DataFrame()

    working = sequence.copy()
    working["qcitem"] = working["qcitem"].astype(str)
    mask = working["qcitem"].str.contains("平均", na=False)
    working = working[mask & working["numeric_value"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    parsed = working["qcitem"].apply(_parse_wheelseat_average)
    working["panel_name"] = parsed.map(lambda item: item[0] if item else None)
    working["section"] = parsed.map(lambda item: item[1] if item else None)
    working = working.dropna(subset=["panel_name"])
    if working.empty:
        return pd.DataFrame()

    panel_order_map = {
        panel: idx for idx, panel in enumerate(working["panel_name"].drop_duplicates(), start=1)
    }
    working["panel_order"] = working["panel_name"].map(panel_order_map)
    working["section_rank"] = working["section"].apply(_section_rank)

    sort_keys = ["panel_order"]
    if "order_rank" in working.columns:
        sort_keys.append("order_rank")
    if "point_index" in working.columns:
        sort_keys.append("point_index")
    if "qc_timestamp" in working.columns:
        sort_keys.append("qc_timestamp")
    sort_keys.append("section_rank")

    working = working.sort_values(sort_keys, na_position="last")
    deduped = working.drop_duplicates(subset=["panel_name", "section"], keep="last")
    deduped = deduped.sort_values(["panel_order", "section_rank", "section"])

    if "qc_timestamp" not in deduped.columns:
        deduped["qc_timestamp"] = pd.NaT

    deduped["section"] = deduped["section"].fillna("平均值")

    columns = ["panel_name", "section", "numeric_value", "qc_timestamp", "section_rank"]
    result = deduped.loc[:, columns].reset_index(drop=True)
    return result


def measurement_gap_analysis(
    df: pd.DataFrame,
    machine_item: str,
    manual_item: str,
) -> pd.DataFrame:
    """Compare equipment output with manual re-measurements."""

    working = attach_numeric_values(df)
    subset = working[working["qcitem"].isin({machine_item, manual_item})].copy()
    subset = subset.dropna(subset=["numeric_value"])

    if subset.empty:
        return pd.DataFrame()

    index_cols = ["xb005"]
    if "lotno" in subset.columns:
        index_cols.append("lotno")
    if "qc_timestamp" in subset.columns:
        index_cols.append("qc_timestamp")
    if "rn" in subset.columns:
        index_cols.append("rn")

    pivot = subset.pivot_table(
        index=index_cols,
        columns="qcitem",
        values="numeric_value",
        aggfunc="last",
    )

    if pivot.empty:
        return pd.DataFrame()

    pivot = pivot.dropna(subset=[machine_item, manual_item]).reset_index()
    pivot["diff"] = pivot[manual_item] - pivot[machine_item]
    pivot["abs_diff"] = pivot["diff"].abs()

    if "qc_timestamp" in pivot.columns:
        pivot["month"] = pivot["qc_timestamp"].dt.to_period("M").dt.to_timestamp()
    else:
        pivot["month"] = pd.NaT

    return pivot


def monthly_breach_summary(gap_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarise months where the measurement gap exceeded the threshold."""

    if gap_df.empty or "month" not in gap_df:
        return pd.DataFrame(columns=["month", "breach_count"])

    breaches = gap_df[gap_df["abs_diff"] > threshold]
    if breaches.empty:
        return pd.DataFrame(columns=["month", "breach_count"])

    summary = breaches.groupby("month").size().reset_index(name="breach_count")
    return summary.sort_values("month")


def combined_indicator_alerts(
    df: pd.DataFrame,
    diameter_item: str,
    flange_item: str,
    diameter_limit: Optional[float],
    flange_threshold: Optional[float],
) -> pd.DataFrame:
    """Identify potential risks based on combined indicator thresholds."""

    working = attach_numeric_values(df)
    subset = working[working["qcitem"].isin({diameter_item, flange_item})].copy()
    subset = subset.dropna(subset=["numeric_value"])

    if subset.empty:
        return pd.DataFrame()

    index_cols = ["xb005"]
    if "lotno" in subset.columns:
        index_cols.append("lotno")
    if "qc_timestamp" in subset.columns:
        index_cols.append("qc_timestamp")
    if "rn" in subset.columns:
        index_cols.append("rn")

    pivot = subset.pivot_table(
        index=index_cols,
        columns="qcitem",
        values="numeric_value",
        aggfunc="last",
    )

    if pivot.empty:
        return pd.DataFrame()

    pivot = pivot.dropna(subset=[diameter_item, flange_item]).reset_index()

    condition = pd.Series(True, index=pivot.index)
    if diameter_limit is not None:
        condition &= pivot[diameter_item] <= diameter_limit
    if flange_threshold is not None:
        condition &= pivot[flange_item] < flange_threshold

    alerts = pivot[condition].copy()
    if "qc_timestamp" in alerts.columns:
        alerts["month"] = alerts["qc_timestamp"].dt.to_period("M").dt.to_timestamp()

    return alerts
