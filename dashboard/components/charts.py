"""
Shared dashboard components — chart helpers, metric cards, section headers, and layout utilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render_kpi_row(
    kpis: List[Tuple[str, str, str, str]],
    cols: int = 6,
) -> None:
    """
    Render a row of KPI metric cards.

    Parameters
    ----------
    kpis : List[Tuple[str, str, str, str]]
        List of (label, value, delta, color) tuples
    cols : int
        Number of columns in the row
    """
    columns = st.columns(cols)
    for i, (label, value, delta, color) in enumerate(kpis):
        if i < len(columns):
            with columns[i]:
                st.markdown(
                    f"""
                    <div style="background: #161b22; border: 1px solid #30363d;
                                border-radius: 8px; padding: 0.75rem 1rem;
                                border-left: 3px solid {'#58a6ff' if color == 'accent'
                                else '#3fb950' if color == 'green'
                                else '#d29922' if color == 'orange'
                                else '#8b949e'};
                                transition: border-color 0.2s ease;">
                        <div style="color: #8b949e; font-size: 0.7rem;
                                    text-transform: uppercase; letter-spacing: 0.05em;
                                    font-weight: 600;">
                            {label}
                        </div>
                        <div style="color: #f0f6fc; font-size: 1.4rem; font-weight: 700;
                                    letter-spacing: -0.02em; margin-top: 0.2rem;">
                            {value}
                        </div>
                        <div style="color: #8b949e; font-size: 0.75rem; margin-top: 0.1rem;">
                            {delta}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header with optional subtitle."""
    st.markdown("---")
    st.markdown(
        f"""
        <div style="margin-bottom: 0.75rem;">
            <h2 style="font-size: 1.25rem; font-weight: 600; color: #f0f6fc;
                       letter-spacing: -0.02em; margin-bottom: 0.15rem;">
                {title}
            </h2>
            <p style="color: #8b949e; font-size: 0.85rem; margin: 0;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_figure(fig: go.Figure, key: str) -> None:
    """
    Render a Plotly figure with consistent dark theme settings.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to render
    key : str
        Unique key for the Streamlit component
    """
    fig.update_layout(
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#21262d",
            font_size=12,
            font_color="#f0f6fc",
        ),
        font=dict(family="Inter, -apple-system, sans-serif"),
        coloraxis=dict(colorbar=dict(
            outlinecolor="#30363d",
            tickcolor="#8b949e",
            tickfont=dict(color="#8b949e", size=10),
        )),
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


