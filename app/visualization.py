"""Plotting utilities using Plotly."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from .analytics import STATUS_ORDER

COLOR_MAP = {
    "qualified": "#2ca02c",
    "unqualified": "#d62728",
    "not_applicable": "#ff7f0e",
    "unknown": "#7f7f7f",
}


def pie_status(df: pd.DataFrame) -> px.pie:
    """Build a pie chart with the inspection status distribution."""

    data = df.copy()
    if "percentage" not in data.columns:
        total = max(data["count"].sum(), 1)
        data["percentage"] = data["count"] / total * 100

    fig = px.pie(
        data,
        names="status",
        values="count",
        color="status",
        color_discrete_map=COLOR_MAP,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{value} (%{percent})")
    fig.update_layout(margin=dict(t=20, b=20, l=10, r=10))
    return fig


def bar_ranking(df: pd.DataFrame, dimension: str, metric: str = "count") -> px.bar:
    """Horizontal bar chart for ranking type data."""

    fig = px.bar(
        df.sort_values(metric),
        x=metric,
        y=dimension,
        orientation="h",
        color=metric,
        text=metric,
        color_continuous_scale="Blues",
    )
    fig.update_layout(margin=dict(l=80, r=20, t=30, b=20))
    fig.update_traces(texttemplate="%{x}", textposition="outside", cliponaxis=False)
    return fig


def timeline_chart(ts_df: pd.DataFrame) -> px.line:
    """Plot the inspection count trend over time."""

    fig = px.line(
        ts_df,
        x="period",
        y="count",
        markers=True,
        labels={"period": "日期", "count": "检修数量"},
    )
    fig.update_layout(margin=dict(t=30, b=40, l=40, r=20))
    return fig


def stacked_status_trend(ts_df: pd.DataFrame) -> px.area:
    """Create an area chart showing the status mix over time."""

    value_vars = [col for col in STATUS_ORDER if col in ts_df]
    if not value_vars:
        raise ValueError("Time series dataframe does not contain status columns")

    melted = ts_df.melt(
        id_vars=["period"],
        value_vars=value_vars,
        var_name="status",
        value_name="status_count",
    )
    fig = px.area(
        melted,
        x="period",
        y="status_count",
        color="status",
        color_discrete_map=COLOR_MAP,
        labels={"period": "日期", "status_count": "数量", "status": "状态"},
    )
    fig.update_layout(margin=dict(t=30, b=40, l=40, r=20))
    return fig


def heatmap(df: pd.DataFrame, x: str, y: str, value: str) -> px.density_heatmap:
    """Return a heatmap for the combination of two dimensions."""

    fig = px.density_heatmap(
        df,
        x=x,
        y=y,
        z=value,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(t=30, b=40, l=60, r=20))
    return fig


def stage_presence_heatmap(
    matrix: pd.DataFrame, *, height: Optional[int] = None, width: Optional[int] = None
) -> go.Figure:
    """Visualise the process coverage matrix as a binary heatmap."""

    if matrix.empty:
        raise ValueError("Matrix is empty and cannot be plotted")

    z = matrix.to_numpy()
    text = np.where(z >= 1, "✔", "")

    heatmap = go.Heatmap(
        z=z,
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale=[[0, "#f0f0f0"], [1, "#2ca02c"]],
        showscale=False,
        text=text,
        texttemplate="%{text}",
        hovertemplate="轮对 %{x}<br>流程 %{y}<extra></extra>",
    )

    fig = go.Figure(data=[heatmap])
    layout_kwargs = dict(margin=dict(t=30, b=30, l=80, r=20))
    if height is not None:
        layout_kwargs["height"] = height
    if width is not None:
        layout_kwargs["width"] = width
    fig.update_layout(**layout_kwargs)
    fig.update_yaxes(automargin=True)
    return fig


def qcitem_numeric_trend_chart(trend_df: pd.DataFrame) -> px.line:
    """Plot numeric qcitem values across wheels."""

    if trend_df.empty:
        raise ValueError("Trend dataframe is empty")

    plotting = trend_df.copy()
    if "timestamp_label" in plotting:
        plotting["timestamp_label"] = plotting["timestamp_label"].fillna("无时间戳")

    fig = px.line(
        plotting,
        x="xb005",
        y="numeric_value",
        markers=True,
        hover_data={
            "xb005": True,
            "numeric_value": ":.3f",
            "timestamp_label": True,
        },
        labels={"xb005": "轮对号", "numeric_value": "检测值", "timestamp_label": "最新检测时间"},
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="top center")
    fig.update_layout(margin=dict(t=30, b=40, l=40, r=20))
    return fig


def wheel_profile_chart(sequence: pd.DataFrame, title: str) -> px.line:
    """Plot the wheel seat measurements sequence for a single wheel."""

    plotting = sequence.copy()
    plotting = plotting.sort_values("point_index")
    plotting["label"] = plotting.get("qcitem", plotting.get("order_rank", plotting["point_index"]))

    fig = px.line(
        plotting,
        x="point_index",
        y="numeric_value",
        markers=True,
        hover_data={
            "point_index": True,
            "numeric_value": ":.3f",
            "qcitem": True,
            "qc_timestamp": True,
        },
        labels={"point_index": "测点顺序", "numeric_value": "测量值"},
        title=title,
    )
    fig.update_traces(texttemplate="%{y:.3f}", textposition="top center")
    fig.update_layout(margin=dict(t=50, b=40, l=40, r=20))
    fig.update_xaxes(tickmode="linear")
    return fig


def wheelseat_average_panel_chart(panels: pd.DataFrame, title: str) -> go.Figure:
    """Render wheel seat average values as per-panel subplots."""

    if panels.empty:
        raise ValueError("Panels dataframe is empty")

    data = panels.copy()
    data["panel_name"] = data["panel_name"].astype(str)
    data["section"] = data["section"].astype(str)

    sort_columns = ["panel_name"]
    if "section_rank" in data.columns:
        sort_columns.append("section_rank")
    if "section" not in sort_columns:
        sort_columns.append("section")

    data = data.sort_values(sort_columns).reset_index(drop=True)
    panels_order = data["panel_name"].unique().tolist()

    subplot_titles = [f"{name}" for name in panels_order]
    rows = len(subplot_titles)

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    for idx, panel_name in enumerate(panels_order, start=1):
        subset = data[data["panel_name"] == panel_name].copy()
        if "section_rank" in subset.columns:
            subset = subset.sort_values(["section_rank", "section"])  # ensure consistent ordering
        else:
            subset = subset.sort_values("section")

        text_values = subset["numeric_value"].map(lambda val: f"{val:.3f}")

        if "qc_timestamp" in subset.columns and subset["qc_timestamp"].notna().any():
            if pd.api.types.is_datetime64_any_dtype(subset["qc_timestamp"]):
                hover_times = subset["qc_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("无时间戳")
            else:
                hover_times = subset["qc_timestamp"].astype(str)
            customdata = np.array(hover_times, dtype=object).reshape(-1, 1)
            hovertemplate = "截面:%{x}<br>测量值:%{y:.3f}<br>检测时间:%{customdata[0]}<extra></extra>"
        else:
            customdata = None
            hovertemplate = "截面:%{x}<br>测量值:%{y:.3f}<extra></extra>"

        trace_kwargs = {
            "x": subset["section"],
            "y": subset["numeric_value"],
            "mode": "lines+markers",
            "name": panel_name,
            "text": text_values,
            "textposition": "top center",
            "hovertemplate": hovertemplate,
        }
        if customdata is not None:
            trace_kwargs["customdata"] = customdata

        fig.add_trace(go.Scatter(**trace_kwargs), row=idx, col=1)
        fig.update_yaxes(title_text="测量值", row=idx, col=1)
        fig.update_xaxes(title_text="截面", row=idx, col=1, type="category")

    fig.update_layout(
        title=title,
        height=max(360, 260 * rows),
        showlegend=False,
        margin=dict(t=80, b=40, l=60, r=30),
    )

    return fig


def measurement_gap_chart(
    gap_df: pd.DataFrame, machine_item: str, manual_item: str
) -> go.Figure:
    """Plot equipment output and manual measurements with their difference."""

    if gap_df.empty:
        raise ValueError("Gap dataframe is empty")

    fig = go.Figure()
    if "qc_timestamp" in gap_df.columns:
        x_axis = gap_df["qc_timestamp"]
    else:
        x_axis = gap_df.index

    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=gap_df[machine_item],
            name=f"设备输出({machine_item})",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=gap_df[manual_item],
            name=f"人工复测({manual_item})",
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x_axis,
            y=gap_df["diff"],
            name="差值 (人工-设备)",
            opacity=0.3,
            marker_color="#d62728",
            yaxis="y2",
        )
    )

    fig.update_layout(
        margin=dict(t=40, b=60, l=50, r=50),
        yaxis=dict(title="测量值"),
        yaxis2=dict(title="差值", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def monthly_breach_bar(summary_df: pd.DataFrame) -> px.bar:
    """Visualise the monthly count of measurement breaches."""

    plotting = summary_df.copy()
    if plotting.empty:
        raise ValueError("Summary dataframe is empty")

    fig = px.bar(
        plotting,
        x="month",
        y="breach_count",
        labels={"month": "月份", "breach_count": "偏差超阈值次数"},
    )
    fig.update_layout(margin=dict(t=30, b=60, l=50, r=20))
    return fig


def combined_alert_scatter(
    alerts: pd.DataFrame, diameter_item: str, flange_item: str
) -> px.scatter:
    """Scatter plot highlighting potential combined indicator risks."""

    if alerts.empty:
        raise ValueError("Alerts dataframe is empty")

    fig = px.scatter(
        alerts,
        x=diameter_item,
        y=flange_item,
        color="xb005" if "xb005" in alerts.columns else None,
        hover_data=[col for col in ["xb005", "lotno", "qc_timestamp"] if col in alerts.columns],
        labels={diameter_item: "轮径", flange_item: "轮缘厚度"},
    )
    fig.update_layout(margin=dict(t=30, b=40, l=50, r=20))
    return fig
