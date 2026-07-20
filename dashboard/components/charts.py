"""
Shared dashboard components — Glassmorphism chart helpers, KPI cards, section headers.
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
    Render a row of glassmorphism KPI metric cards.

    Parameters
    ----------
    kpis : List[Tuple[str, str, str, str]]
        List of (label, value, delta, color) tuples
        color: 'accent', 'green', 'amber', 'rose', 'violet'
    cols : int
        Number of columns in the row
    """
    columns = st.columns(cols)

    color_map = {
        "accent": ("#00d4ff", "#0a0e17"),
        "green": ("#10b981", "#0a0e17"),
        "amber": ("#f59e0b", "#0a0e17"),
        "rose": ("#f43f5e", "#0a0e17"),
        "violet": ("#7c3aed", "#0a0e17"),
        "cyan": ("#38bdf8", "#0a0e17"),
    }

    for i, (label, value, delta, color) in enumerate(kpis):
        if i < len(columns):
            border_color, _ = color_map.get(color, ("#38bdf8", "#0a0e17"))
            with columns[i]:
                st.markdown(
                    f"""
                    <div class="glass"
                         style="border-left: 2px solid {border_color}; padding: 0.9rem 1rem;">
                        <div style="color: #475569; font-size: 0.6rem; text-transform: uppercase;
                                    letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.2rem;">
                            {label}
                        </div>
                        <div style="color: #f1f5f9; font-size: 1.35rem; font-weight: 700;
                                    letter-spacing: -0.02em; font-family: 'JetBrains Mono', monospace;">
                            {value}
                        </div>
                        <div style="color: #64748b; font-size: 0.65rem; margin-top: 0.1rem;">
                            {delta}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_section_header(title: str, subtitle: str = "") -> None:
    """Render a section header with glassmorphism styling and gradient accent line."""
    st.markdown(
        f"""
        <div class="section-divider">
            <div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #f1f5f9;
                            letter-spacing: -0.01em;">
                    {title}
                </div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.1rem;">
                    {subtitle}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_figure(fig: go.Figure, key: str) -> None:
    """
    Render a Plotly figure with glassmorphism dark theme settings.

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
            bgcolor="rgba(17, 24, 39, 0.9)",
            bordercolor="rgba(56, 189, 248, 0.2)",
            font_size=11,
            font_color="#f1f5f9",
            font_family="Inter, sans-serif",
        ),
        font=dict(family="Inter, -apple-system, sans-serif"),
        coloraxis=dict(
            colorbar=dict(
                outlinecolor="rgba(56, 189, 248, 0.1)",
                tickcolor="#64748b",
                tickfont=dict(color="#64748b", size=9),
            )
        ),
        dragmode=False,
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


def glass_card(title: str, content: str, color: str = "accent") -> str:
    """
    Generate an HTML string for a glassmorphism card.

    Parameters
    ----------
    title : str
        Card title
    content : str
        Card body content (HTML allowed)
    color : str
        Accent color: 'accent', 'green', 'amber', 'rose', 'violet'

    Returns
    -------
    str
        HTML string for the card
    """
    border_map = {
        "accent": "#00d4ff",
        "green": "#10b981",
        "amber": "#f59e0b",
        "rose": "#f43f5e",
        "violet": "#7c3aed",
        "cyan": "#38bdf8",
    }
    border = border_map.get(color, "#38bdf8")

    return f"""
    <div class="glass" style="border-left: 2px solid {border}; padding: 1.25rem; height: 100%;">
        <div style="color: #475569; font-size: 0.65rem; text-transform: uppercase;
                    letter-spacing: 0.08em; font-weight: 600; margin-bottom: 0.5rem;">
            {title}
        </div>
        <div style="color: #f1f5f9; font-size: 0.85rem; line-height: 1.6;">
            {content}
        </div>
    </div>
    """


def render_glass_card(title: str, content: str, color: str = "accent") -> None:
    """Render a glassmorphism info card directly to Streamlit."""
    st.markdown(glass_card(title, content, color), unsafe_allow_html=True)


def gradient_metric(value: str, label: str, delta: str = "", gradient: str = "accent") -> str:
    """
    Generate HTML for a gradient metric display.

    Parameters
    ----------
    value : str
        The metric value
    label : str
        The metric label
    delta : str
        Optional delta/change text
    gradient : str
        'accent', 'green', 'amber', or 'rose'

    Returns
    -------
    str
        HTML string
    """
    grad_map = {
        "accent": "linear-gradient(135deg, #00d4ff, #7c3aed)",
        "green": "linear-gradient(135deg, #10b981, #34d399)",
        "amber": "linear-gradient(135deg, #f59e0b, #f97316)",
        "rose": "linear-gradient(135deg, #f43f5e, #e11d48)",
    }
    grad = grad_map.get(gradient, grad_map["accent"])

    return f"""
    <div style="background: rgba(17, 24, 39, 0.6); backdrop-filter: blur(12px);
                border: 1px solid rgba(56, 189, 248, 0.08); border-radius: 10px;
                padding: 1rem; text-align: center;">
        <div style="font-size: 1.75rem; font-weight: 800; background: {grad};
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; font-family: 'JetBrains Mono', monospace;
                    letter-spacing: -0.03em;">
            {value}
        </div>
        <div style="color: #64748b; font-size: 0.7rem; text-transform: uppercase;
                    letter-spacing: 0.08em; font-weight: 600; margin-top: 0.2rem;">
            {label}
        </div>
        {f'<div style="color: #475569; font-size: 0.65rem; margin-top: 0.1rem;">{delta}</div>' if delta else ''}
    </div>
    """
