"""Reusable UI widgets for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).resolve().parent / "components" / "vertical_slider"
_vertical_slider_component = components.declare_component(
    "vertical_slider",
    path=str(_COMPONENT_DIR),
)


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
    """Render a vertical slider component and return the selected value."""

    state_key = f"{key}__value"

    if max_value <= min_value:
        st.session_state[state_key] = min_value
        return min_value

    current_value = int(st.session_state.get(state_key, value))
    current_value = max(min_value, min(max_value, current_value))

    result = _vertical_slider_component(
        key=key,
        default=current_value,
        label=label,
        min=min_value,
        max=max_value,
        value=current_value,
        step=max(1, step),
        height=max(height, 120),
    )

    if result is None:
        return current_value

    selected = int(result)
    st.session_state[state_key] = selected
    return selected

