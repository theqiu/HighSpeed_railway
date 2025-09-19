"""Plotting utilities using Plotly."""
from __future__ import annotations
from typing import Optional
import pandas as pd
import plotly.express as px
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
    melted = ts_df.melt(id_vars=["period"], value_vars=value_vars, var_name="status", value_name="count")
    fig = px.area(
        melted,
        x="period",
        y="count",
        color="status",
        color_discrete_map=COLOR_MAP,
        labels={"period": "日期", "count": "数量", "status": "状态"},
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