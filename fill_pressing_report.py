"""Utility for populating the wheel pressing analysis workbook.

This script reads raw inspection records from ``data/2024-2025.csv`` and
injects the matching measurements into ``data/轮对压装统计分析表-2025.10.30.xlsx``.

The Excel template uses a two-row header: the first row contains the
presentation labels while the second row stores the detailed measurement
names that are recorded inside the MES export (CSV).  The script treats the
second header row as the lookup key when searching for a wheel's inspection
records.  Columns whose presentation label already denotes an averaged or
derived metric (``*均``/``*平均``) are skipped to avoid overwriting workbook
formulas.

Usage::

    python fill_pressing_report.py [--csv 自定义CSV路径] [--workbook 模板路径]

The script appends new rows for every wheel number found in the CSV.  If the
target workbook already contains data, the numbering in the first column is
continued from the last populated row.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

try:  # pragma: no cover - optional import guard
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover
    load_workbook = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openpyxl.cell import Cell
    from openpyxl.worksheet.worksheet import Worksheet
else:  # pragma: no cover
    Cell = Any
    Worksheet = Any


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "2024-2025.csv"
WORKBOOK_PATH = BASE_DIR / "data" / "轮对压装统计分析表-2025.10.30.xlsx"

# Measurements often contain decimal places.  Longer numeric strings (e.g.
# 20240117110418) should remain textual, so the matcher only converts values
# that either include a decimal point or are short integer tokens.
NUMERIC_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")

# Columns whose labels include these tokens are assumed to rely on workbook
# formulas (averages or summary statistics) and are therefore skipped.
SKIP_TOKENS = ("均", "平均", "过盈量")


def normalise_key(value: str) -> str:
    """Return a normalised token for matching template labels to qcitem rows."""

    if value is None:
        return ""
    # Remove whitespace, punctuation commonly used in headers, and harmonise case.
    cleaned = re.sub(r"[\s:：;；\-_/\\]", "", str(value))
    return cleaned.upper()


@dataclass
class ColumnDescriptor:
    """Metadata for a target Excel column."""

    index: int
    display_name: Optional[str]
    detail_name: Optional[str]

    @property
    def should_skip(self) -> bool:
        display = self.display_name or ""
        detail = self.detail_name or ""
        return any(token in display for token in SKIP_TOKENS) or any(
            token in detail for token in SKIP_TOKENS
        )


@dataclass
class Record:
    """A simplified row extracted from the MES CSV."""

    inputvalue: str
    creator: str
    qcitem: str
    opno: str


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", force=True)
    if verbose:
        LOGGER.debug("Verbose logging enabled")


def load_records(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV source not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    # Normalise whitespace for downstream lookups.
    for column in ("xb005", "opno", "qcitem", "qcorder", "inputvalue", "creator"):
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    if "xb005" not in df.columns:
        raise ValueError("CSV缺少 'xb005' 列，无法按轮对号分组。")

    return df


def iterate_wheels(df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
    grouped = df.groupby("xb005", sort=True)
    for wheel_id, group in grouped:
        yield wheel_id, group.sort_values("qcdate") if "qcdate" in group.columns else group


def build_lookup(group: pd.DataFrame) -> Tuple[Dict[str, List[Record]], Dict[str, List[Record]]]:
    by_qcitem: Dict[str, List[Record]] = defaultdict(list)
    by_opno: Dict[str, List[Record]] = defaultdict(list)

    for _, row in group.iterrows():
        record = Record(
            inputvalue=str(row.get("inputvalue", "")).strip(),
            creator=str(row.get("creator", "")).strip(),
            qcitem=str(row.get("qcitem", "")).strip(),
            opno=str(row.get("opno", "")).strip(),
        )
        if record.qcitem:
            by_qcitem[record.qcitem].append(record)
            norm_qcitem = normalise_key(record.qcitem)
            if norm_qcitem and norm_qcitem != record.qcitem:
                by_qcitem[norm_qcitem].append(record)
        if record.opno:
            by_opno[record.opno].append(record)
            norm_opno = normalise_key(record.opno)
            if norm_opno and norm_opno != record.opno:
                by_opno[norm_opno].append(record)

    return by_qcitem, by_opno


def normalise_value(raw: str) -> Optional[object]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None

    if NUMERIC_PATTERN.match(value) and not (value.isdigit() and len(value) >= 8 and "." not in value):
        if "." in value:
            return float(value)
        try:
            return int(value)
        except ValueError:
            # Fall back to string if it doesn't fit into Python's int range.
            return value

    return value


def resolve_value(
    detail_name: str,
    by_qcitem: Dict[str, List[Record]],
    by_opno: Dict[str, List[Record]],
    *,
    wheel_id: Optional[str] = None,
) -> Optional[object]:
    if not detail_name:
        return None

    detail_name = detail_name.strip()
    candidates = by_qcitem.get(detail_name)
    if not candidates:
        normalised = normalise_key(detail_name)
        if normalised and normalised != detail_name:
            candidates = by_qcitem.get(normalised)
        if not candidates:
            candidates = by_opno.get(detail_name)
        if not candidates and normalised and normalised != detail_name:
            candidates = by_opno.get(normalised)
        if not candidates:
            LOGGER.debug(
                "[%s] 未找到匹配的记录: %s",
                wheel_id or "?",
                detail_name,
            )
            return None

    record = candidates[0]
    if any(token in detail_name for token in ("作业人员", "测量人")):
        value = record.creator or record.inputvalue
    elif detail_name in {"轮对号", "轮对编号"}:
        # `轮对号` columns simply echo the wheel identifier; this branch allows
        # templates that redundantly store the name in the detail row.
        value = record.qcitem or record.inputvalue
    else:
        value = record.inputvalue or record.creator

    normalised = normalise_value(value)
    LOGGER.debug(
        "[%s] 命中 %s -> %s",
        wheel_id or "?",
        detail_name,
        normalised,
    )
    return normalised


def load_workbook_columns(sheet: Worksheet) -> List[ColumnDescriptor]:
    columns: List[ColumnDescriptor] = []
    for idx in range(1, sheet.max_column + 1):
        display = sheet.cell(row=1, column=idx).value
        detail = sheet.cell(row=2, column=idx).value
        if display is None and detail is None:
            continue
        columns.append(ColumnDescriptor(index=idx, display_name=display, detail_name=detail))
    return columns


def find_insert_row(sheet: Worksheet) -> Tuple[int, int]:
    row = 3
    last_seq = 0
    while True:
        seq_cell = sheet.cell(row=row, column=1)
        value = seq_cell.value
        if value in (None, ""):
            break
        if isinstance(value, (int, float)):
            last_seq = max(last_seq, int(value))
        else:
            try:
                last_seq = max(last_seq, int(str(value)))
            except (TypeError, ValueError):
                pass
        row += 1
    return row, last_seq


def write_rows(
    sheet: Worksheet,
    columns: Iterable[ColumnDescriptor],
    wheel_data: Iterator[Tuple[str, pd.DataFrame]],
) -> int:
    start_row, last_seq = find_insert_row(sheet)
    written = 0

    for offset, (wheel_id, group) in enumerate(wheel_data, start=1):
        row_idx = start_row + offset - 1
        seq_value = last_seq + offset
        by_qcitem, by_opno = build_lookup(group)
        LOGGER.info("处理轮对 %s，记录数 %s", wheel_id, len(group))

        for column in columns:
            if column.index == 1:
                sheet.cell(row=row_idx, column=column.index, value=seq_value)
                continue

            if column.should_skip:
                continue

            detail = column.detail_name or ""
            value: Optional[object]

            if detail.strip() in {"轮对号", "轮对编号"}:
                value = wheel_id
            else:
                value = resolve_value(detail, by_qcitem, by_opno, wheel_id=wheel_id)

            if value is None and column.display_name and "轮对号" in column.display_name:
                value = wheel_id

            if value is None:
                LOGGER.debug(
                    "[%s] 列 %s/%s 未找到匹配值",
                    wheel_id,
                    column.display_name,
                    detail,
                )
                continue

            target_cell: Cell = sheet.cell(row=row_idx, column=column.index)
            # Skip if the template already provides a formula for this column.
            if isinstance(target_cell.value, str) and target_cell.value.startswith("="):
                continue
            target_cell.value = value

        written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill the wheel pressing workbook")
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_PATH,
        help="Path to the MES CSV export (default: data/2024-2025.csv)",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=WORKBOOK_PATH,
        help="Path to the Excel workbook to populate",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=None,
        help="Optional worksheet name; defaults to the active sheet",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging column matches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if load_workbook is None:  # pragma: no cover - handled during runtime
        raise RuntimeError("openpyxl 未安装，无法读写工作簿。")

    df = load_records(args.csv)
    wb = load_workbook(args.workbook)
    sheet = wb[args.sheet] if args.sheet else wb.active
    columns = load_workbook_columns(sheet)

    if not columns:
        raise ValueError("目标工作簿缺少表头，无法写入数据。")

    total_written = write_rows(sheet, columns, iterate_wheels(df))
    if total_written == 0:
        logging.warning("未写入任何轮对数据，请检查模板列名与CSV是否匹配。")
    else:
        logging.info("成功写入 %s 条轮对记录。", total_written)

    wb.save(args.workbook)
    logging.info("结果已保存至 %s", args.workbook)


if __name__ == "__main__":
    main()

