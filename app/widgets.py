"""Reusable UI widgets for the Streamlit dashboard."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def vertical_slider(
    *,
    key: str,
    label: Optional[str],
    min_value: int,
    max_value: int,
    value: int,
    step: int = 1,
    height: int = 220,
) -> int:
    """Render a slider used for vertical scrolling controls."""

    if max_value <= min_value:
        return int(min_value)

    if key in st.session_state:
        value = int(st.session_state[key])

    value = max(int(min_value), min(int(max_value), int(value)))

    selected = st.slider(
        label or "滚动条",
        min_value=int(min_value),
        max_value=int(max_value),
        value=value,
        step=max(1, int(step)),
        key=key,
        label_visibility="collapsed" if not label else "visible",
    )

    return int(selected)

