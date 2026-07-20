"""
Quantitative Equity Research Platform — Institutional Dashboard

Futuristic glassmorphism dark theme inspired by:
- Bloomberg Terminal
- Palantir Foundry
- Linear
- Stripe

Design System:
- Glassmorphism cards with backdrop blur
- Dark navy base (#0a0e17) with gradient accents
- Cyan (#00d4ff), violet (#7c3aed), emerald (#10b981)
- Inter font family with JetBrains Mono for data
- Smooth transitions and micro-interactions
- Responsive grid layout
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="QERP | Quantitative Equity Research Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Glassmorphism Dark Theme ──────────────────────────────────────

st.markdown(
    """
<style>
    /* ── Fonts ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Design Tokens ─────────────────────────────────────── */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.75);
        --bg-card-hover: rgba(30, 41, 59, 0.85);
        --glass-border: rgba(56, 189, 248, 0.08);
        --glass-border-hover: rgba(56, 189, 248, 0.25);
        --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        --glass-blur: blur(16px);

        --accent-cyan: #00d4ff;
        --accent-violet: #7c3aed;
        --accent-emerald: #10b981;
        --accent-amber: #f59e0b;
        --accent-rose: #f43f5e;
        --accent-sky: #38bdf8;

        --gradient-accent: linear-gradient(135deg, #00d4ff, #7c3aed);
        --gradient-green: linear-gradient(135deg, #10b981, #34d399);
        --gradient-warm: linear-gradient(135deg, #f59e0b, #f97316);
        --gradient-danger: linear-gradient(135deg, #f43f5e, #e11d48);

        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #475569;

        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    /* ── Base ──────────────────────────────────────────────── */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── Gradient background decoration ────────────────────── */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at 20% 50%, rgba(124, 58, 237, 0.04) 0%, transparent 60%),
                    radial-gradient(ellipse at 80% 20%, rgba(0, 212, 255, 0.03) 0%, transparent 50%),
                    radial-gradient(ellipse at 50% 80%, rgba(16, 185, 129, 0.02) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 15, 25, 0.98), rgba(15, 23, 42, 0.95));
        border-right: 1px solid rgba(56, 189, 248, 0.06);
        backdrop-filter: blur(20px);
        width: 280px !important;
    }

    section[data-testid="stSidebar"] .stButton button {
        background: transparent;
        border: 1px solid transparent;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.55rem 1rem;
        border-radius: var(--radius-sm);
        width: 100%;
        text-align: left;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    section[data-testid="stSidebar"] .stButton button::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 2px;
        background: var(--gradient-accent);
        opacity: 0;
        transition: opacity 0.2s ease;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(56, 189, 248, 0.06);
        color: var(--text-primary);
        border-color: rgba(56, 189, 248, 0.1);
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] .stButton button:hover::before {
        opacity: 1;
    }

    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: var(--gradient-accent);
        border-color: transparent;
        color: #fff;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
    }

    section[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.35);
        transform: translateY(-1px);
    }

    /* ── Glass Card ────────────────────────────────────────── */
    .glass {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--glass-shadow);
        position: relative;
        overflow: hidden;
    }

    .glass::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.2), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .glass:hover {
        border-color: var(--glass-border-hover);
        background: var(--bg-card-hover);
        transform: translateY(-1px);
    }

    .glass:hover::before {
        opacity: 1;
    }

    .glass-accent {
        border-left: 2px solid var(--accent-cyan);
    }
    .glass-green {
        border-left: 2px solid var(--accent-emerald);
    }
    .glass-amber {
        border-left: 2px solid var(--accent-amber);
    }
    .glass-rose {
        border-left: 2px solid var(--accent-rose);
    }
    .glass-violet {
        border-left: 2px solid var(--accent-violet);
    }

    /* ── Metric Cards ──────────────────────────────────────── */
    div[data-testid="metric-container"] {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--glass-shadow);
    }

    div[data-testid="metric-container"]:hover {
        border-color: var(--glass-border-hover);
        background: var(--bg-card-hover);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }

    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] div {
        color: var(--text-primary) !important;
    }

    /* ── Headers ───────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
    }

    h1 {
        font-size: 1.65rem;
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
        padding-bottom: 0;
        border-bottom: none;
    }

    h2 {
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }

    h3 {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── DataFrames ─────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        overflow: hidden;
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
    }

    .stDataFrame thead tr th {
        background: rgba(30, 41, 59, 0.6) !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.55rem 0.75rem !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.06);
    }

    .stDataFrame tbody tr {
        background: transparent !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.03);
    }

    .stDataFrame tbody tr:hover {
        background: rgba(56, 189, 248, 0.03) !important;
    }

    .stDataFrame tbody td {
        color: var(--text-primary) !important;
        font-size: 0.8rem;
        padding: 0.4rem 0.75rem !important;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Tabs ──────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-sm);
        padding: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        font-weight: 500;
        font-size: 0.8rem;
        padding: 0.4rem 1rem;
        border-radius: 4px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(56, 189, 248, 0.05);
        color: var(--text-secondary);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(56, 189, 248, 0.1);
        color: var(--accent-cyan);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.05);
    }

    /* ── Buttons ────────────────────────────────────────────── */
    .stButton button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        border-radius: var(--radius-sm);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        background: rgba(56, 189, 248, 0.04);
        border: 1px solid rgba(56, 189, 248, 0.1);
        color: var(--text-secondary);
    }

    .stButton button:hover {
        background: rgba(56, 189, 248, 0.08);
        border-color: rgba(56, 189, 248, 0.2);
        color: var(--text-primary);
        transform: translateY(-1px);
    }

    .stButton button[kind="primary"] {
        background: var(--gradient-accent);
        color: #fff;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.15);
    }

    .stButton button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.25);
        transform: translateY(-1px);
    }

    /* ── Select / Input ─────────────────────────────────────── */
    .stSelectbox label, .stMultiSelect label, .stTextInput label, .stSlider label {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s ease;
    }

    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover,
    .stTextInput > div > div:hover {
        border-color: rgba(56, 189, 248, 0.2) !important;
    }

    /* ── Info / Warning / Error ─────────────────────────────── */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-md) !important;
        backdrop-filter: var(--glass-blur);
    }

    .stAlert > div[data-baseweb="notification"] {
        background: transparent !important;
    }

    /* ── Plotly ─────────────────────────────────────────────── */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    .js-plotly-plot .plotly .gridlayer .crisp {
        stroke: rgba(56, 189, 248, 0.06) !important;
    }

    /* ── Expander ───────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500;
        color: var(--text-primary) !important;
        backdrop-filter: var(--glass-blur);
    }

    /* ── Divider ────────────────────────────────────────────── */
    hr {
        border: none;
        border-top: 1px solid rgba(56, 189, 248, 0.06);
        margin: 1.5rem 0;
    }

    /* ── Section Separator ──────────────────────────────────── */
    .section-divider {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 1.5rem 0 1rem;
    }

    .section-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.15), transparent);
    }

    /* ── Status Dot ─────────────────────────────────────────── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 0.4rem;
    }

    .status-dot.active { background: var(--accent-emerald); box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
    .status-dot.warning { background: var(--accent-amber); box-shadow: 0 0 8px rgba(245, 158, 11, 0.5); }
    .status-dot.error { background: var(--accent-rose); box-shadow: 0 0 8px rgba(244, 63, 94, 0.5); }
    .status-dot.idle { background: var(--text-muted); }

    /* ── Gradient Text ──────────────────────────────────────── */
    .gradient-text {
        background: var(--gradient-accent);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Hero Section ───────────────────────────────────────── */
    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 3rem 1rem 2rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.75rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        max-width: 600px;
        line-height: 1.6;
        margin-bottom: 2rem;
    }

    /* ── Feature Grid ───────────────────────────────────────── */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .feature-card {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--glass-border);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        box-shadow: var(--glass-shadow);
    }

    .feature-card:hover {
        border-color: var(--glass-border-hover);
        background: var(--bg-card-hover);
        transform: translateY(-4px);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }

    .feature-title {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
        color: var(--text-primary);
    }

    .feature-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* ── Footer ─────────────────────────────────────────────── */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.7rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid rgba(56, 189, 248, 0.06);
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session State ────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "run_pipeline" not in st.session_state:
    st.session_state.run_pipeline = False

# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.markdown(
    f"""
    <div style="padding: 1.25rem 0; text-align: center;">
        <span style="font-size: 1.75rem; font-weight: 800; letter-spacing: -0.03em;
                     background: linear-gradient(135deg, #00d4ff, #7c3aed);
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ◈ QERP
        </span>
        <br>
        <span style="font-size: 0.65rem; color: #475569; text-transform: uppercase;
                     letter-spacing: 0.18em; font-weight: 500;">
            Quantitative Equity Research
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
    <div style="padding: 0 0.5rem 0.5rem;">
        <span style="font-size: 0.65rem; color: #475569; text-transform: uppercase;
                     letter-spacing: 0.1em; font-weight: 600;">Platform</span>
    </div>
    """
)

pages = [
    ("🏠", "Home"),
    ("📊", "Executive Dashboard"),
    ("🔬", "Research Workspace"),
    ("📈", "Factor Explorer"),
    ("💡", "Feature Importance"),
    ("💼", "Portfolio Construction"),
    ("⚠️", "Risk Analytics"),
    ("📜", "Backtest Explorer"),
    ("📰", "News Intelligence"),
    ("🗺️", "Market Heatmap"),
    ("🔗", "Correlation Matrix"),
    ("⚙️", "Model Comparison"),
    ("📋", "Experiment Tracking"),
]

for icon, page_name in pages:
    active = st.session_state.page == page_name
    label = f"{icon} {page_name}"
    if st.sidebar.button(
        label,
        key=f"nav_{page_name}",
        type="primary" if active else "secondary",
        use_container_width=True,
    ):
        st.session_state.page = page_name
        st.rerun()

st.sidebar.markdown("---")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶ Run", key="sidebar_run", type="primary", use_container_width=True):
        if st.session_state.page == "Home":
            st.session_state.page = "Research Workspace"
        st.session_state.run_pipeline = True
        st.rerun()
with col2:
    if st.button("⟳ Refresh", key="sidebar_refresh", use_container_width=True):
        st.rerun()

st.sidebar.markdown("---")

# System status indicator
st.sidebar.markdown(
    f"""
    <div style="margin-top: 0.5rem; padding: 1rem;
                background: rgba(17, 24, 39, 0.6);
                border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.06);
                backdrop-filter: blur(8px);">
        <div style="color: #475569; font-size: 0.65rem; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.5rem; font-weight: 600;">
            System Status
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span class="status-dot active"></span>
            <span style="color: #94a3b8; font-size: 0.8rem;">All systems operational</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem;">
            <span class="status-dot active" style="width:6px;height:6px;"></span>
            <span style="color: #94a3b8; font-size: 0.8rem;">Model API: Connected</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem;">
            <span class="status-dot active" style="width:6px;height:6px;"></span>
            <span style="color: #94a3b8; font-size: 0.8rem;">Data Pipeline: Active</span>
        </div>
        <div style="color: #475569; font-size: 0.65rem; margin-top: 0.5rem;">
            v2.0.0 • Python 3.10+
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Page Router ──────────────────────────────────────────────────────

page = st.session_state.page
st.session_state.run_pipeline = st.session_state.get("run_pipeline", False)

# Import and render the selected page
page_modules = {
    "Home": "dashboard.pages.home",
    "Executive Dashboard": "dashboard.pages.executive",
    "Research Workspace": "dashboard.pages.research",
    "Factor Explorer": "dashboard.pages.factors",
    "Feature Importance": "dashboard.pages.feature_importance",
    "Portfolio Construction": "dashboard.pages.portfolio",
    "Risk Analytics": "dashboard.pages.risk",
    "Backtest Explorer": "dashboard.pages.backtest",
    "News Intelligence": "dashboard.pages.news",
    "Market Heatmap": "dashboard.pages.heatmap",
    "Correlation Matrix": "dashboard.pages.correlation",
    "Model Comparison": "dashboard.pages.models",
    "Experiment Tracking": "dashboard.pages.experiments",
}

if page in page_modules:
    mod = __import__(page_modules[page], fromlist=["render"])
    mod.render()
else:
    from dashboard.pages.home import render
    render()

# ── Footer ──────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">◈ QERP v2.0.0 — Institutional Quantitative Research Platform</div>',
    unsafe_allow_html=True,
)
