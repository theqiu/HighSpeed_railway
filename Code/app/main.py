"""Streamlit application for inspection analytics."""
from __future__ import annotations
from io import BytesIO
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st
from app.analytics import compute_summary, status_breakdown, time_series, top_entities
from app.data_processing import FilterParams, apply_filters, preprocess_dataset, read_dataset
from app.visualization import bar_ranking, heatmap, pie_status, stacked_status_trend, timeline_chart
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA = BASE_DIR / "data" / "sample_inspections.csv"
@st.cache_data(show_spinner=False)
def load_data(file: Optional[BytesIO]) -> pd.DataFrame:
    """Load and preprocess the dataset from the provided file."""
    if file is None:
        raw = read_dataset(SAMPLE_DATA)
    else:
        raw = read_dataset(file.getvalue())
    return preprocess_dataset(raw)
def sidebar_filters(df: pd.DataFrame) -> FilterParams:
    """Render sidebar widgets and return the selected filters."""
    st.sidebar.header("筛选条件")
    lot_options = sorted(df["lotno"].dropna().unique().tolist())
    creator_options = sorted(df["creator"].dropna().unique().tolist())
    qc_options = sorted(df["qcitem"].dropna().unique().tolist())
    selected_lots = st.sidebar.multiselect("批次号", options=lot_options)
    selected_creators = st.sidebar.multiselect("检修人员", options=creator_options)
    selected_qc = st.sidebar.multiselect("检测项目", options=qc_options)
    min_date = df["qc_timestamp"].min()
    max_date = df["qc_timestamp"].max()
    date_range = None
    if pd.notna(min_date) and pd.notna(max_date):
        start, end = st.sidebar.date_input(
            "检测日期范围",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if isinstance(start, tuple):  # streamlit returns tuple when single date is selected
            start, end = start
        date_range = (
            pd.to_datetime(start) if start else None,
            pd.to_datetime(end) if end else None,
        )
    return FilterParams(
        lot_numbers=selected_lots or None,
        creators=selected_creators or None,
        qc_items=selected_qc or None,
        date_range=date_range,
    )
def render_summary(metrics: dict[str, int]) -> None:
    """Display high-level summary cards."""
    st.subheader("总体概览")
    total = metrics.get("total_records", 0)
    cols = st.columns(4)
    cols[0].metric("记录总数", f"{total:,}")
    cols[1].metric("合格", f"{metrics.get('qualified', 0):,}")
    cols[2].metric("不合格", f"{metrics.get('unqualified', 0):,}")
    cols[3].metric("未知/不适用", f"{metrics.get('unknown', 0) + metrics.get('not_applicable', 0):,}")
def render_charts(filtered: pd.DataFrame) -> None:
    """Render all visualisations for the filtered dataset."""
    st.subheader("状态分布")
    breakdown = status_breakdown(filtered)
    pie_col, table_col = st.columns((2, 1))
    pie_col.plotly_chart(pie_status(breakdown), use_container_width=True)
    table_col.dataframe(breakdown, use_container_width=True)
    st.subheader("时间趋势")
    ts = time_series(filtered, freq="D")
    if ts.empty:
        st.info("所选条件下没有可用于绘制趋势图的时间数据。")
    else:
        trend_col, stacked_col = st.columns(2)
        trend_col.plotly_chart(timeline_chart(ts), use_container_width=True)
        stacked_col.plotly_chart(stacked_status_trend(ts), use_container_width=True)
    st.subheader("检修人员 Top 10")
    top_creators = top_entities(filtered, "creator", limit=10)
    st.plotly_chart(bar_ranking(top_creators, "creator"), use_container_width=True)
    st.subheader("检测项目 Top 10")
    top_qc = top_entities(filtered, "qcitem", limit=10)
    st.plotly_chart(bar_ranking(top_qc, "qcitem"), use_container_width=True)
    st.subheader("人员与项目关联热力图")
    pivot = (
        filtered.groupby(["creator", "qcitem"]).size().reset_index(name="count")
    )
    if pivot.empty:
        st.info("所选条件下没有数据用于绘制热力图。")
    else:
        st.plotly_chart(heatmap(pivot, x="creator", y="qcitem", value="count"), use_container_width=True)
def main() -> None:
    st.set_page_config(page_title="检修数据分析平台", layout="wide")
    st.title("检修数据分析与可视化平台")
    st.write(
        """
        上传检修记录 CSV 文件或使用示例数据，快速查看批次、人员与检测项目的统计情况，
        并通过多种图表了解整体趋势。左侧提供筛选器以聚焦特定批次或员工。
        """
    )
    uploaded_file = st.sidebar.file_uploader("上传 CSV 文件", type=["csv"])
    dataset = load_data(uploaded_file)
    st.sidebar.caption(
        "未上传文件时，系统将自动加载位于 `data/sample_inspections.csv` 的示例数据。"
    )
    filters = sidebar_filters(dataset)
    filtered = apply_filters(dataset, filters)
    if filtered.empty:
        st.warning("当前筛选条件下无数据，请调整筛选项。")
        return
    metrics = compute_summary(filtered)
    render_summary(metrics)
    render_charts(filtered)
    st.subheader("明细数据")
    st.dataframe(filtered, use_container_width=True)
    buffer = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("导出筛选结果 CSV", data=buffer, file_name="inspection_filtered.csv", mime="text/csv")
if __name__ == "__main__":
    main()