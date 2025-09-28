from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

if __package__ in {None, ""}:
    # Allow running via ``streamlit run app/main.py`` by ensuring the project root is importable.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.analytics import (
    attach_numeric_values,
    combined_indicator_alerts,
    compute_summary,
    measurement_gap_analysis,
    monthly_breach_summary,
    numeric_qcitem_candidates,
    qcitem_catalog,
    qcitem_numeric_trend,
    stage_presence_matrix,
    status_breakdown,
    time_series,
    top_entities,
    wheelseat_average_panels,
    wheelseat_monotonic_analysis,
)
from app.data_processing import FilterParams, apply_filters, preprocess_dataset, read_dataset
from app.visualization import (
    bar_ranking,
    heatmap,
    pie_status,
    qcitem_numeric_trend_chart,
    stage_presence_heatmap,
    stacked_status_trend,
    timeline_chart,
    wheelseat_average_panel_chart,
    wheel_profile_chart,
    measurement_gap_chart,
    monthly_breach_bar,
    combined_alert_scatter,
)
from app.exporting import (
    export_tables_to_excel,
    export_to_pdf,
    export_to_word,
    figure_to_image_bytes,
    prepare_chart_images,
)

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA = BASE_DIR / "data" / "sample_inspections.csv"


def _init_artifacts() -> dict:
    return {"summary": {}, "tables": {}, "figures": {}}


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


def render_summary(metrics: dict[str, int], artifacts: dict) -> None:
    """Display high-level summary cards."""

    st.subheader("总体概览")
    total = metrics.get("total_records", 0)
    cols = st.columns(4)
    cols[0].metric("记录总数", f"{total:,}")
    cols[1].metric("合格", f"{metrics.get('qualified', 0):,}")
    cols[2].metric("不合格", f"{metrics.get('unqualified', 0):,}")
    cols[3].metric("未知/不适用", f"{metrics.get('unknown', 0) + metrics.get('not_applicable', 0):,}")

    artifacts["summary"] = metrics


def render_charts(filtered: pd.DataFrame, artifacts: dict) -> None:
    """Render all visualisations for the filtered dataset."""

    st.subheader("状态分布")
    breakdown = status_breakdown(filtered)
    pie_col, table_col = st.columns((2, 1))
    pie_fig = pie_status(breakdown)
    pie_col.plotly_chart(pie_fig, use_container_width=True)
    table_col.dataframe(breakdown, use_container_width=True)
    artifacts["figures"]["状态分布饼图"] = pie_fig
    artifacts["tables"]["状态分布"] = breakdown.copy()

    st.subheader("时间趋势")
    ts = time_series(filtered, freq="D")
    if ts.empty:
        st.info("所选条件下没有可用于绘制趋势图的时间数据。")
    else:
        trend_col, stacked_col = st.columns(2)
        timeline_fig = timeline_chart(ts)
        stacked_fig = stacked_status_trend(ts)
        trend_col.plotly_chart(timeline_fig, use_container_width=True)
        stacked_col.plotly_chart(stacked_fig, use_container_width=True)
        artifacts["figures"]["检修数量时间趋势"] = timeline_fig
        artifacts["figures"]["状态堆叠趋势"] = stacked_fig
        artifacts["tables"]["时间序列明细"] = ts.copy()

    st.subheader("检修人员排行")
    creator_rankings = top_entities(filtered, "creator", limit=None)
    if creator_rankings.empty:
        st.info("当前筛选条件下没有检修人员数据。")
    else:
        max_creators = len(creator_rankings)
        min_window = 1 if max_creators == 1 else min(10, max_creators)
        window_size = st.slider(
            "显示检修人员数量",
            min_value=min_window,
            max_value=max_creators,
            value=min(15, max_creators),
            key="creator_window",
        )
        if max_creators > window_size:
            start_index = st.slider(
                "起始序号",
                min_value=0,
                max_value=max_creators - window_size,
                value=0,
                key="creator_start",
            )
        else:
            start_index = 0
        visible_creators = creator_rankings.iloc[start_index : start_index + window_size]
        figure_height = max(360, 40 * len(visible_creators))
        creator_fig = bar_ranking(visible_creators, "creator", height=figure_height)
        st.plotly_chart(creator_fig, use_container_width=True)
        artifacts["figures"]["检修人员排行"] = creator_fig
        artifacts["tables"]["检修人员排行"] = creator_rankings.copy()

    st.subheader("检测项目排行")
    exclusion_pattern = r"(备注|说明)"
    qc_filtered = filtered.copy()
    if "qcitem" in qc_filtered.columns:
        qc_filtered = qc_filtered[~qc_filtered["qcitem"].astype(str).str.contains(exclusion_pattern, na=False)]
    qc_rankings = top_entities(qc_filtered, "qcitem", limit=None)
    if qc_rankings.empty:
        st.info("当前筛选条件下没有检测项目数据。")
    else:
        max_items = len(qc_rankings)
        min_window = 1 if max_items == 1 else min(10, max_items)
        window_size = st.slider(
            "显示检测项目数量",
            min_value=min_window,
            max_value=max_items,
            value=min(15, max_items),
            key="qc_window",
        )
        if max_items > window_size:
            start_index = st.slider(
                "项目起始序号",
                min_value=0,
                max_value=max_items - window_size,
                value=0,
                key="qc_start",
            )
        else:
            start_index = 0
        visible_qc = qc_rankings.iloc[start_index : start_index + window_size]
        figure_height = max(360, 40 * len(visible_qc))
        qc_fig = bar_ranking(visible_qc, "qcitem", height=figure_height)
        st.plotly_chart(qc_fig, use_container_width=True)
        artifacts["figures"]["检测项目排行"] = qc_fig
        artifacts["tables"]["检测项目排行"] = qc_rankings.copy()

    st.subheader("人员与项目关联热力图")
    pivot = (
        filtered.groupby(["creator", "qcitem"]).size().reset_index(name="count")
    )
    if pivot.empty:
        st.info("所选条件下没有数据用于绘制热力图。")
    else:
        relation_fig = heatmap(pivot, x="creator", y="qcitem", value="count")
        st.plotly_chart(relation_fig, use_container_width=True)
        artifacts["figures"]["人员与项目热力图"] = relation_fig
        artifacts["tables"]["人员项目交叉频次"] = pivot.copy()


def render_process_matrix(df: pd.DataFrame, artifacts: dict) -> None:
    """Render the wheel vs process coverage matrix."""

    st.subheader("轮对流程覆盖矩阵")
    matrix = stage_presence_matrix(df)
    if matrix.empty:
        st.info("所选数据中没有可用于构建流程矩阵的记录。")
        return

    total_rows = len(matrix.index)
    total_cols = len(matrix.columns)

    row_window = total_rows
    row_start = 0
    if total_rows > 30:
        min_window = min(10, total_rows)
        row_window = st.slider(
            "每页显示流程数量",
            min_value=min_window,
            max_value=total_rows,
            value=min(30, total_rows),
            step=1,
        )
        max_row_start = max(total_rows - row_window, 0)
        row_start = st.slider(
            "流程起始位置",
            min_value=0,
            max_value=max_row_start,
            value=0,
            step=1,
        )

    visible = matrix.iloc[row_start : row_start + row_window]

    col_window = total_cols
    col_start = 0
    if total_cols > 15:
        min_col_window = min(10, total_cols)
        col_window = st.slider(
            "每页显示轮对数量",
            min_value=min_col_window,
            max_value=total_cols,
            value=min(15, total_cols),
            step=1,
        )
        max_col_start = max(total_cols - col_window, 0)
        col_start = st.slider(
            "轮对起始位置",
            min_value=0,
            max_value=max_col_start,
            value=0,
            step=1,
        )
    visible = visible.iloc[:, col_start : col_start + col_window]

    figure_height = max(400, row_window * 24)
    figure_width = max(700, col_window * 60)
    matrix_fig = stage_presence_heatmap(visible, height=figure_height, width=figure_width)
    st.plotly_chart(matrix_fig, use_container_width=True)

    artifacts["figures"]["轮对流程覆盖矩阵"] = matrix_fig
    matrix_table = matrix.reset_index().rename(columns={matrix.index.name or "index": "opno"})
    artifacts["tables"]["轮对流程覆盖矩阵"] = matrix_table

    with st.expander("查看完整矩阵数据"):
        st.dataframe(matrix_table, use_container_width=True)


def render_flow_details(df: pd.DataFrame, artifacts: dict) -> None:
    """Show detailed QC items grouped by wheel number."""

    st.subheader("轮对检测项目明细")
    wheel_options = sorted(df["xb005"].dropna().unique().tolist())
    if not wheel_options:
        st.info("当前筛选条件下没有轮对号。")
        return

    default_selection = wheel_options[: min(len(wheel_options), 5)]
    selected = st.multiselect(
        "选择需要查看的轮对号",
        options=wheel_options,
        default=default_selection,
    )

    detail_df = qcitem_catalog(df, selected or None)
    if detail_df.empty:
        st.info("未找到匹配的检测明细。")
        return

    st.dataframe(detail_df, use_container_width=True)
    artifacts["tables"]["轮对检测项目明细"] = detail_df.copy()
    csv = detail_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "导出明细 CSV",
        data=csv,
        file_name="qcitem_detail.csv",
        mime="text/csv",
    )


def render_numeric_trends(df: pd.DataFrame, artifacts: dict) -> None:
    """Plot numeric qcitem trends across wheels."""

    st.subheader("检测项目数值趋势")
    enriched = attach_numeric_values(df)
    candidates = numeric_qcitem_candidates(enriched)
    if not candidates:
        st.info("当前数据集中没有可转换为数值的检测项目。")
        return

    selected_qcitem = st.selectbox("选择检测项目", options=candidates)
    trend_df = qcitem_numeric_trend(enriched, selected_qcitem)
    if trend_df.empty:
        st.info("所选检测项目暂无有效的数值数据。")
        return

    trend_fig = qcitem_numeric_trend_chart(trend_df)
    st.plotly_chart(trend_fig, use_container_width=True)
    artifacts["figures"][f"{selected_qcitem}数值趋势"] = trend_fig
    with st.expander("查看趋势数据表"):
        st.dataframe(trend_df, use_container_width=True)
    artifacts["tables"][f"{selected_qcitem}趋势数据"] = trend_df.copy()


def render_wheelseat_analysis(df: pd.DataFrame, artifacts: dict) -> None:
    """Analyse wheel seat measurements for monotonic patterns."""

    st.subheader("轮座尺寸规律检测")
    keyword = st.text_input("检测项目关键字", value="轮座", help="输入用于匹配轮座相关测点的关键词。")

    type_columns = [col for col in ["lotno", "opno"] if col in df.columns]
    type_column = st.selectbox("轮型字段", options=type_columns or ["无可选字段"], index=0)
    if type_column == "无可选字段":
        type_column = None

    type_value = None
    if type_column:
        type_candidates = sorted(df[type_column].dropna().unique().tolist())
        if type_candidates:
            selection = st.selectbox(
                "选择轮型/批次",
                options=["全部"] + type_candidates,
                index=0,
            )
            if selection != "全部":
                type_value = selection

    tolerance = st.number_input("允许的正向波动 (mm)", value=0.0, step=0.01, min_value=0.0)

    summary, sequences = wheelseat_monotonic_analysis(
        df,
        keyword=keyword,
        wheel_column="xb005",
        type_column=type_column,
        type_value=type_value,
        tolerance=tolerance,
    )

    if summary.empty:
        st.info("未找到符合条件的轮座测点数据。")
        return

    st.dataframe(summary, use_container_width=True)
    artifacts["tables"]["轮座规律检测汇总"] = summary.copy()
    if not sequences.empty:
        artifacts["tables"]["轮座测量序列"] = sequences.copy()

        wheel_options = summary["xb005"].astype(str).tolist()
        selected_wheel = st.selectbox("选择轮对查看曲线", options=wheel_options)
        if selected_wheel:
            seq = sequences[sequences["xb005"].astype(str) == selected_wheel]
            title_prefix = f"{selected_wheel} 轮座测量曲线"
            if type_column and type_value:
                title = f"{title_prefix} ({type_value})"
            else:
                title = title_prefix
            panel_data = wheelseat_average_panels(seq)
            if not panel_data.empty:
                max_panels = len(panel_data["panel_name"].unique())
                default_cols = 2 if max_panels >= 2 else 1
                columns_per_row = st.slider(
                    "每行展示子图数量",
                    min_value=1,
                    max_value=min(3, max_panels) if max_panels else 1,
                    value=default_cols,
                    key=f"panel_columns_{selected_wheel}",
                )
                panel_fig = wheelseat_average_panel_chart(
                    panel_data, title, columns=columns_per_row
                )
                st.plotly_chart(panel_fig, use_container_width=True)
                artifacts["figures"][title] = panel_fig

                export_table = panel_data.drop(columns=["section_rank"]) if "section_rank" in panel_data else panel_data
                artifacts["tables"][f"{selected_wheel}轮座平均测量"] = export_table.copy()
            else:
                st.info("该轮对缺少平均值测点，显示原始趋势曲线。")
                profile_fig = wheel_profile_chart(seq, title)
                st.plotly_chart(profile_fig, use_container_width=True)
                artifacts["figures"][title] = profile_fig

            violation_info = summary[summary["xb005"].astype(str) == selected_wheel]
            if not violation_info.empty and violation_info["violation_count"].iloc[0] > 0:
                st.error("该轮对存在由内向外不递减的测点，请复核加工过程。")
            else:
                st.success("该轮对测点符合由内向外逐渐减小的规律。")


def render_measurement_gap_section(df: pd.DataFrame, artifacts: dict) -> None:
    """Analyse differences between machine output and manual measurements."""

    st.subheader("设备与人工偏差预警")
    enriched = attach_numeric_values(df)
    candidates = numeric_qcitem_candidates(enriched)
    if len(candidates) < 2:
        st.info("当前数据缺少足够的数值型检测项目用于偏差分析。")
        return

    machine_item = st.selectbox("设备输出项目", options=candidates)
    manual_candidates = [item for item in candidates if item != machine_item]
    if not manual_candidates:
        st.info("请选择不同的人工复测项目以进行对比分析。")
        return
    manual_item = st.selectbox("人工复测项目", options=manual_candidates)
    threshold = st.number_input("偏差阈值", value=1.0, step=0.1, min_value=0.0)
    frequency = st.number_input("预警次数 N", value=3, step=1, min_value=1)

    gap_df = measurement_gap_analysis(enriched, machine_item, manual_item)
    if gap_df.empty:
        st.info("未找到同时包含设备输出和人工复测的数据。")
        return

    st.dataframe(gap_df, use_container_width=True)
    artifacts["tables"][f"{machine_item}与{manual_item}对比"] = gap_df.copy()

    gap_fig = measurement_gap_chart(gap_df, machine_item, manual_item)
    st.plotly_chart(gap_fig, use_container_width=True)
    artifacts["figures"][f"{machine_item}与{manual_item}测量对比"] = gap_fig

    monthly = monthly_breach_summary(gap_df, threshold)
    artifacts["tables"]["偏差超阈值月份"] = monthly.copy()
    if monthly.empty:
        st.success("当前时间范围内未发现偏差超出阈值的情况。")
    else:
        st.dataframe(monthly, use_container_width=True)
        breach_fig = monthly_breach_bar(monthly)
        st.plotly_chart(breach_fig, use_container_width=True)
        artifacts["figures"]["偏差预警次数"] = breach_fig

        triggered = monthly[monthly["breach_count"] >= frequency]
        if not triggered.empty:
            st.error(
                f"共有 {len(triggered)} 个月偏差超过阈值且次数不少于 {int(frequency)} 次，请关注设备或量具状态。"
            )
        else:
            st.warning("存在偏差超阈值的月份，但未达到设定的预警次数 N。")


def render_combined_alerts_section(df: pd.DataFrame, artifacts: dict) -> None:
    """Detect combined indicator risks based on configurable thresholds."""

    st.subheader("组合指标预警")
    enriched = attach_numeric_values(df)
    candidates = numeric_qcitem_candidates(enriched)
    if len(candidates) < 2:
        st.info("当前数据缺少足够的数值型检测项目用于组合预警。")
        return

    diameter_default = next((item for item in candidates if "轮径" in str(item)), candidates[0])
    diameter_item = st.selectbox("轮径检测项目", options=candidates, index=candidates.index(diameter_default))
    flange_candidates = [item for item in candidates if item != diameter_item]
    if not flange_candidates:
        st.info("请选择另一项轮缘相关的检测项目进行组合预警。")
        return
    flange_default = next((item for item in flange_candidates if "轮缘" in str(item)), flange_candidates[0])
    flange_item = st.selectbox("轮缘厚度项目", options=flange_candidates, index=flange_candidates.index(flange_default))

    st.caption("当轮径仍在限值内但轮缘厚度偏小时，可提前提示换轮。阈值留空则不参与判断。")
    diameter_limit_text = st.text_input("轮径上限 (可选)", value="")
    flange_threshold_text = st.text_input("轮缘厚度下限 (可选)", value="")

    if not diameter_limit_text and not flange_threshold_text:
        st.info("请输入至少一个阈值以执行组合预警分析。")
        return

    def _parse_threshold(text: str) -> Optional[float]:
        try:
            return float(text)
        except ValueError:
            return None

    diameter_limit = _parse_threshold(diameter_limit_text) if diameter_limit_text else None
    flange_threshold = _parse_threshold(flange_threshold_text) if flange_threshold_text else None

    alerts = combined_indicator_alerts(enriched, diameter_item, flange_item, diameter_limit, flange_threshold)
    if alerts.empty:
        st.success("未检测到需要预警的组合指标记录。")
        return

    st.dataframe(alerts, use_container_width=True)
    artifacts["tables"]["组合指标预警"] = alerts.copy()

    alert_fig = combined_alert_scatter(alerts, diameter_item, flange_item)
    st.plotly_chart(alert_fig, use_container_width=True)
    artifacts["figures"]["组合指标预警散点图"] = alert_fig


def render_exports(artifacts: dict) -> None:
    """Provide export options for tables and charts."""

    st.subheader("导出与分享")
    tables = artifacts.get("tables", {})
    figures = artifacts.get("figures", {})
    summary = artifacts.get("summary", {})

    if not tables and not figures:
        st.info("暂无可导出的数据或图表。")
        return

    excel_bytes = export_tables_to_excel({name: table.copy() for name, table in tables.items()})
    st.download_button(
        "导出 Excel", data=excel_bytes, file_name="inspection_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    image_payloads = prepare_chart_images(figures)
    word_bytes = export_to_word(summary, tables, image_payloads)
    st.download_button(
        "导出 Word", data=word_bytes, file_name="inspection_analysis.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

    pdf_bytes = export_to_pdf(summary, tables, image_payloads)
    st.download_button("导出 PDF", data=pdf_bytes, file_name="inspection_analysis.pdf", mime="application/pdf")

    if figures:
        fig_names = list(figures.keys())
        selected_fig = st.selectbox("选择导出的图表 (JPG)", options=fig_names)
        if selected_fig:
            jpg_bytes = figure_to_image_bytes(figures[selected_fig], fmt="jpg")
            st.download_button(
                "下载选中图表 JPG",
                data=jpg_bytes,
                file_name=f"{selected_fig}.jpg",
                mime="image/jpeg",
            )
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

    artifacts = _init_artifacts()
    metrics = compute_summary(filtered)
    render_summary(metrics, artifacts)
    render_process_matrix(filtered, artifacts)
    render_wheelseat_analysis(filtered, artifacts)
    render_numeric_trends(filtered, artifacts)
    render_measurement_gap_section(filtered, artifacts)
    render_combined_alerts_section(filtered, artifacts)
    render_flow_details(filtered, artifacts)
    render_charts(filtered, artifacts)

    st.subheader("明细数据")
    st.dataframe(filtered, use_container_width=True)
    artifacts["tables"]["筛选明细数据"] = filtered.copy()

    buffer = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("导出筛选结果 CSV", data=buffer, file_name="inspection_filtered.csv", mime="text/csv")

    render_exports(artifacts)


if __name__ == "__main__":
    main()
