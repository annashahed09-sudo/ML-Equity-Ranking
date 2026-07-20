"""
Quantitative Equity Research Platform — Institutional Dashboard

Premium Streamlit dashboard inspired by Bloomberg Terminal, Palantir, and
institutional quant research platforms. Dark matte-navy theme with
professional typography and glassmorphism accents.

Usage:
    streamlit run dashboard/app.py --server.port 8501
    python main.py dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────

st.set_page_config(
    page_title="QERP | Quantitative Equity Research Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Institutional Dark Theme ──────────────────────────────────

st.markdown(
    """
<style>
    /* ── Base Layer ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --navy: #0d1117;
        --navy-muted: #161b22;
        --navy-soft: #21262d;
        --navy-border: #30363d;
        --accent: #58a6ff;
        --accent-green: #3fb950;
        --accent-orange: #d29922;
        --accent-red: #f85149;
        --accent-purple: #bc8cff;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #484f58;
        --glass-bg: rgba(22, 27, 34, 0.85);
        --glass-border: rgba(48, 54, 61, 0.5);
    }

    /* ── Reset & Base ───────────────────────────────────────────── */
    .stApp {
        background: var(--navy);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--navy-muted);
        border-right: 1px solid var(--navy-border);
        width: 280px !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background: transparent;
        border: 1px solid var(--navy-border);
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        width: 100%;
        text-align: left;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: var(--navy-soft);
        border-color: var(--accent);
        color: var(--accent);
    }
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #fff;
    }

    /* ── Metric Cards ───────────────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: var(--navy-muted);
        border: 1px solid var(--navy-border);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: var(--accent);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.08);
    }

    /* ── Headers ────────────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    h1 {
        font-size: 1.75rem;
        border-bottom: 1px solid var(--navy-border);
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
    }
    h2 {
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    h3 {
        font-size: 1rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── DataFrames ─────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--navy-border);
        border-radius: 8px;
        overflow: hidden;
    }
    .stDataFrame thead tr th {
        background: var(--navy-soft) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.5rem 0.75rem !important;
    }
    .stDataFrame tbody tr {
        background: var(--navy-muted) !important;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background: var(--navy-soft) !important;
    }
    .stDataFrame tbody td {
        color: var(--text-primary) !important;
        font-size: 0.85rem;
        padding: 0.4rem 0.75rem !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: transparent;
        border-bottom: 1px solid var(--navy-border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        border-radius: 6px 6px 0 0;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--navy-soft);
        color: var(--text-primary);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: transparent;
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }

    /* ── Cards & Containers ─────────────────────────────────────── */
    .card {
        background: var(--navy-muted);
        border: 1px solid var(--navy-border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    .card:hover {
        border-color: var(--navy-border);
    }
    .card-accent {
        border-left: 3px solid var(--accent);
    }
    .card-green {
        border-left: 3px solid var(--accent-green);
    }
    .card-orange {
        border-left: 3px solid var(--accent-orange);
    }
    .card-red {
        border-left: 3px solid var(--accent-red);
    }

    /* ── Stat Labels ────────────────────────────────────────────── */
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .stat-value {
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .stat-value-positive { color: var(--accent-green); }
    .stat-value-negative { color: var(--accent-red); }
    .stat-value-neutral  { color: var(--accent-orange); }

    /* ── Divider ────────────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid var(--navy-border);
        margin: 1.5rem 0;
    }

    /* ── Expander ───────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--navy-muted);
        border: 1px solid var(--navy-border);
        border-radius: 8px;
        font-weight: 500;
        color: var(--text-primary);
    }

    /* ── Select / Input ─────────────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label, .stTextInput label {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--navy-muted) !important;
        border: 1px solid var(--navy-border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }

    /* ── Info / Warning / Error ─────────────────────────────────── */
    .stAlert {
        background: var(--navy-muted) !important;
        border: 1px solid var(--navy-border) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    .stAlert > div[data-baseweb="notification"] {
        background: transparent !important;
    }

    /* ── Plotly ─────────────────────────────────────────────────── */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }
    .js-plotly-plot .plotly .gridlayer .crisp {
        stroke: var(--navy-border) !important;
        opacity: 0.5;
    }

    /* ── Footer ─────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.75rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid var(--navy-border);
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State Initialization ──────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Executive Dashboard"


# ── Sidebar Navigation ───────────────────────────────────────────────────

st.sidebar.markdown(
    """
    <div style="padding: 1rem 0; text-align: center;">
        <span style="font-size: 1.5rem; font-weight: 700; letter-spacing: -0.03em;
                     background: linear-gradient(135deg, #58a6ff, #bc8cff);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            QERP
        </span>
        <br>
        <span style="font-size: 0.7rem; color: #484f58; text-transform: uppercase;
                     letter-spacing: 0.15em;">
            Quantitative Equity Research Platform
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Navigation")

pages = [
    "Executive Dashboard",
    "Research Workspace",
    "Factor Explorer",
    "Feature Importance",
    "Portfolio Construction",
    "Risk Analytics",
    "Backtest Explorer",
    "News Intelligence",
    "Market Heatmap",
    "Correlation Matrix",
    "Model Comparison",
    "Experiment Tracking",
]

for page in pages:
    if st.sidebar.button(page, key=f"nav_{page}"):
        st.session_state.page = page
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶ Run Pipeline", key="sidebar_run", type="primary"):
        st.session_state.page = "Research Workspace"
        st.session_state.run_pipeline = True
        st.rerun()
with col2:
    if st.button("⟳ Refresh", key="sidebar_refresh"):
        st.rerun()

st.sidebar.markdown(
    """
    <div style="margin-top: 2rem; padding: 1rem; background: rgba(33, 38, 45, 0.5);
                border-radius: 8px; border: 1px solid #30363d;">
        <div style="color: #484f58; font-size: 0.7rem; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 0.5rem;">System Status</div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="display: inline-block; width: 8px; height: 8px;
                         background: #3fb950; border-radius: 50%;"></span>
            <span style="color: #8b949e; font-size: 0.8rem;">All systems operational</span>
        </div>
        <div style="color: #484f58; font-size: 0.7rem; margin-top: 0.5rem;">
            v2.0.0 • Python 3.10+
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Route to Pages ────────────────────────────────────────────────────────

page = st.session_state.page
st.session_state.run_pipeline = st.session_state.get("run_pipeline", False)

if page == "Executive Dashboard":
    from dashboard.pages.executive import render
    render()
elif page == "Research Workspace":
    from dashboard.pages.research import render
    render()
elif page == "Factor Explorer":
    from dashboard.pages.factors import render
    render()
elif page == "Feature Importance":
    from dashboard.pages.feature_importance import render
    render()
elif page == "Portfolio Construction":
    from dashboard.pages.portfolio import render
    render()
elif page == "Risk Analytics":
    from dashboard.pages.risk import render
    render()
elif page == "Backtest Explorer":
    from dashboard.pages.backtest import render
    render()
elif page == "News Intelligence":
    from dashboard.pages.news import render
    render()
elif page == "Market Heatmap":
    from dashboard.pages.heatmap import render
    render()
elif page == "Correlation Matrix":
    from dashboard.pages.correlation import render
    render()
elif page == "Model Comparison":
    from dashboard.pages.models import render
    render()
elif page == "Experiment Tracking":
    from dashboard.pages.experiments import render
    render()

# ── Footer ────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">QERP v2.0.0 — Institutional Quantitative Research Platform — © 2026</div>',
    unsafe_allow_html=True,
)
