"""Visual theme, KPI computation, and chart builders for the dashboard.

Kept separate from ``dashboard.py`` so the data-shaping and figure-building
logic is importable and unit-testable without a running Streamlit server.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Palette (dark "IBM Plex" fintech aesthetic) ----------------------------
BG = "#0A0D14"
PANEL = "#12161F"
PANEL_2 = "#161B26"
BORDER = "#1E2530"
TEXT = "#E6E8EE"
MUTED = "#8A93A6"
GRID = "#1E2530"

ACCENT_BLUE = "#3B82F6"
ACCENT_CYAN = "#22D3EE"
ACCENT_PURPLE = "#A855F7"
ACCENT_MAGENTA = "#EC4899"
POS = "#34D399"
NEG = "#F87171"

FONT_FAMILY = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
MONO_FAMILY = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"


# CSS injected once into the app. Loads IBM Plex and restyles Streamlit chrome
# into a dark, card-based financial-analysis layout.
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #0A0D14; --panel: #12161F; --panel2: #161B26; --border: #1E2530;
  --text: #E6E8EE; --muted: #8A93A6; --blue: #3B82F6; --purple: #A855F7;
}

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div, label {
  font-family: 'IBM Plex Sans', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 2.2rem; max-width: 1320px; }

h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.01em; color: var(--text); }

/* Numbered section header */
.mlq-section { display: flex; align-items: baseline; gap: 0.9rem; margin: 2.2rem 0 1.0rem; }
.mlq-section .num {
  font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; color: var(--blue);
  border: 1px solid var(--border); border-radius: 8px; padding: 0.15rem 0.55rem; background: var(--panel);
}
.mlq-section .title { font-size: 1.35rem; font-weight: 600; }
.mlq-section .sub { font-size: 0.85rem; color: var(--muted); margin-left: auto; font-family: 'IBM Plex Mono', monospace; }

/* Hero */
.mlq-hero { border: 1px solid var(--border); border-radius: 18px; padding: 2.0rem 2.2rem;
  background: radial-gradient(1200px 300px at 85% -20%, rgba(59,130,246,0.16), transparent 60%),
              radial-gradient(900px 300px at 10% 120%, rgba(168,85,247,0.12), transparent 60%),
              var(--panel); margin-bottom: 0.6rem; }
.mlq-hero .eyebrow { font-family: 'IBM Plex Mono', monospace; color: var(--blue); letter-spacing: 0.18em;
  text-transform: uppercase; font-size: 0.72rem; }
.mlq-hero h1 { font-size: 2.1rem; margin: 0.4rem 0 0.3rem; }
.mlq-hero p { color: var(--muted); max-width: 640px; margin: 0; }

/* KPI cards */
.mlq-kpi { background: var(--panel); border: 1px solid var(--border); border-radius: 14px;
  padding: 1.0rem 1.1rem; height: 100%; }
.mlq-kpi .k-label { color: var(--muted); font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace; }
.mlq-kpi .k-value { font-size: 1.55rem; font-weight: 600; margin-top: 0.35rem; font-variant-numeric: tabular-nums; }
.mlq-kpi .k-sub { color: var(--muted); font-size: 0.75rem; margin-top: 0.2rem; }
.k-pos { color: #34D399; } .k-neg { color: #F87171; } .k-neutral { color: var(--text); }

/* Bucket chips */
.mlq-chip { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
  border:1px solid var(--border); border-radius:8px; padding:0.28rem 0.6rem; margin:0.2rem 0.3rem 0.2rem 0;
  background: var(--panel2); }
.mlq-chip.long { border-color: rgba(52,211,153,0.4); color:#7ff0c8; }
.mlq-chip.short { border-color: rgba(248,113,113,0.4); color:#ffb0b0; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }

/* Buttons */
.stButton > button { background: var(--blue); color: white; border: none; border-radius: 10px;
  font-weight: 600; padding: 0.5rem 1.1rem; }
.stButton > button:hover { background: #2f6fe0; color: white; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--text); border-bottom: 2px solid var(--blue); }

/* Native metric fallback */
div[data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--border);
  padding: 1rem; border-radius: 14px; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }
</style>
"""


def apply_plotly_theme(fig: go.Figure, height: int = 320) -> go.Figure:
    """Apply the dark fintech theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=TEXT, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.12, x=0),
        hoverlabel=dict(font_family=MONO_FAMILY, bgcolor=PANEL),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER)
    return fig


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def compute_portfolio_kpis(
    portfolio_returns: pd.DataFrame,
    portfolio_summary: Dict[str, float],
    fold_metrics: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Derive headline KPIs from the per-date returns and summary.

    Returns raw numeric values (not formatted) so callers can style them.
    """
    net = portfolio_returns["net_return"].astype(float)

    equity = (1.0 + net).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    hit_rate = float((net > 0).mean()) if len(net) else 0.0
    cost_drag = float(portfolio_summary["cumulative_gross"] - portfolio_summary["cumulative_net"])

    mean_ic = std_ic = ic_ir = float("nan")
    if fold_metrics is not None and len(fold_metrics):
        mean_ic = float(fold_metrics["mean_ic"].mean())
        std_ic = float(fold_metrics["mean_ic"].std(ddof=0))
        ic_ir = float(mean_ic / std_ic) if std_ic and not np.isnan(std_ic) else float("nan")

    return {
        "sharpe_net": float(portfolio_summary["sharpe_net"]),
        "sharpe_gross": float(portfolio_summary["sharpe_gross"]),
        "cumulative_net": float(portfolio_summary["cumulative_net"]),
        "cumulative_gross": float(portfolio_summary["cumulative_gross"]),
        "mean_turnover": float(portfolio_summary["mean_turnover"]),
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "cost_drag": cost_drag,
        "mean_ic": mean_ic,
        "ic_ir": ic_ir,
        "n_periods": int(len(portfolio_returns)),
    }


def long_short_buckets(ranking: pd.DataFrame, quantile: float = 0.2) -> tuple[List[str], List[str]]:
    """Return (long_tickers, short_tickers) from a scored ranking."""
    ranked = ranking.sort_values("model_score", ascending=False)
    n = len(ranked)
    k = max(1, int(n * quantile))
    longs = ranked.head(k)["ticker"].tolist()
    shorts = ranked.tail(k)["ticker"].tolist()
    return longs, shorts


def cumulative_pnl_figure(portfolio_returns: pd.DataFrame) -> go.Figure:
    """Cumulative gross vs net PnL over the backtest."""
    df = portfolio_returns.copy()
    df["cum_gross"] = (1.0 + df["gross_return"].astype(float)).cumprod() - 1.0
    df["cum_net"] = (1.0 + df["net_return"].astype(float)).cumprod() - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cum_gross"],
            name="Gross",
            line=dict(color=ACCENT_PURPLE, width=2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cum_net"],
            name="Net of costs",
            line=dict(color=ACCENT_BLUE, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.10)",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return apply_plotly_theme(fig)


def drawdown_figure(portfolio_returns: pd.DataFrame) -> go.Figure:
    """Underwater (drawdown) curve for the net equity line."""
    net = portfolio_returns["net_return"].astype(float)
    equity = (1.0 + net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_returns["date"],
            y=drawdown,
            name="Drawdown",
            line=dict(color=NEG, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(248,113,113,0.15)",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return apply_plotly_theme(fig, height=240)


def turnover_figure(portfolio_returns: pd.DataFrame) -> go.Figure:
    """Per-rebalance turnover bars."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=portfolio_returns["date"],
            y=portfolio_returns["turnover"].astype(float),
            name="Turnover",
            marker_color=ACCENT_CYAN,
            marker_line_width=0,
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return apply_plotly_theme(fig, height=240)


def ic_by_fold_figure(fold_metrics: pd.DataFrame) -> go.Figure:
    """Mean IC per walk-forward fold with std error bars."""
    labels = [f"Fold {int(f)}" for f in fold_metrics["fold_id"]]
    colors = [POS if v >= 0 else NEG for v in fold_metrics["mean_ic"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=fold_metrics["mean_ic"],
            marker_color=colors,
            name="Mean IC",
            error_y=dict(type="data", array=fold_metrics["std_ic"], color=MUTED, thickness=1),
        )
    )
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    return apply_plotly_theme(fig, height=260)


def score_figure(report: pd.DataFrame) -> go.Figure:
    """Model score by ticker, colored by expected direction."""
    colors = [POS if str(d).lower() == "up" else NEG for d in report["expected_direction"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=report["ticker"],
            y=report["model_score"],
            marker_color=colors,
            text=report["rank"],
            textposition="outside",
            name="Model score",
        )
    )
    return apply_plotly_theme(fig, height=320)


def sentiment_figure(report: pd.DataFrame) -> go.Figure:
    """Review-sentiment composition donut."""
    counts = report["review_sentiment"].value_counts()
    palette = {
        "positive": POS,
        "neutral": ACCENT_BLUE,
        "negative": NEG,
    }
    colors = [palette.get(str(s).lower(), MUTED) for s in counts.index]
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            marker=dict(colors=colors, line=dict(color=BG, width=2)),
        )
    )
    return apply_plotly_theme(fig, height=320)
