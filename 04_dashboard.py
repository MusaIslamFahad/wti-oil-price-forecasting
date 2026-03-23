"""
04_dashboard.py
Crude Oil Price Analysis & Forecasting — Streamlit Dashboard

Run locally:
    pip install streamlit plotly pandas numpy
    streamlit run 04_dashboard.py

Deploy to Streamlit Cloud:
    1. Push this file + outputs/ + data/ folders to a GitHub repo
    2. Go to share.streamlit.io  →  New app  →  connect repo
    3. Set Main file path: 04_dashboard.py
    4. Deploy

Folder structure expected alongside this file:
    data/
        oil_clean.csv
    outputs/
        forecasts/
            forward_forecast_12m.csv
            test_predictions.csv
            model_evaluation_metrics.csv
            event_impact_metrics.csv
        figures/          (optional – only used if local images are present)
    structural_breaks.json
    forecast_meta.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crude Oil Price Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME & COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BLUE   = "#3266AD"
C_RED    = "#E8593C"
C_AMBER  = "#D4930A"
C_TEAL   = "#1D9E75"
C_PURPLE = "#7F77DD"
C_GRAY   = "#888780"
C_GREEN  = "#639922"
C_PINK   = "#D4537E"

MODEL_COLORS = {
    "ARIMA":   C_BLUE,
    "Prophet": C_TEAL,
    "LSTM":    C_PURPLE,
}

def ts(x):
    """Convert a pandas Timestamp to milliseconds since Unix epoch (int).
    Plotly 6.x add_vline / add_vrect / add_annotation internally calls
    float(sum([x, x])) to find the annotation midpoint — which works for
    numbers but crashes on ISO strings.  Returning ms-since-epoch (the
    native numeric format Plotly uses for datetime axes) fixes the crash
    while keeping date positioning correct."""
    if hasattr(x, "timestamp"):          # pandas Timestamp / datetime
        return int(x.timestamp() * 1000)
    return x

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Typography ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2 { font-family: 'DM Serif Display', serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1e2a3a;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] hr { border-color: #1e2a3a; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #f8f9fb;
        border: 1px solid #e8eaed;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    [data-testid="stMetricLabel"] { font-size: 12px !important; color: #666 !important; }
    [data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 600 !important; }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e8eaed;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 500;
        color: #555;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        color: #3266AD !important;
        border-bottom: 2px solid #3266AD !important;
        background: transparent !important;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        color: #1a1a2e;
        margin: 1.5rem 0 0.5rem;
        padding-bottom: 6px;
        border-bottom: 2px solid #3266AD;
        display: inline-block;
    }

    /* ── Event badges ── */
    .event-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-supply   { background: #fde8e3; color: #c0392b; }
    .badge-demand   { background: #e3f0fd; color: #2471a3; }
    .badge-financial{ background: #f0e8fd; color: #6c3483; }

    /* ── Info box ── */
    .insight-box {
        background: linear-gradient(135deg, #f0f4ff 0%, #f8f0ff 100%);
        border-left: 4px solid #3266AD;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        line-height: 1.6;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 12px;
        padding: 2rem 0 1rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }

    /* ── Hide default Streamlit branding ── */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached so reloads don't re-read Drive)
# ─────────────────────────────────────────────────────────────────────────────

def find_data_root():
    """
    Detect the data root whether running locally next to the data folders
    or inside Google Colab/Drive at the project root.
    """
    candidates = [
        ".",                                                       # local / Streamlit Cloud
        "/content/drive/MyDrive/crude_oil_project",               # Colab Drive
        os.path.dirname(os.path.abspath(__file__)),               # same dir as this file
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, "data", "oil_clean.csv")):
            return path
    return "."   # fallback – will raise a friendly error below


@st.cache_data(ttl=3600)
def load_data():
    root = find_data_root()

    # ── Main price series ──────────────────────────────────────────────────
    df = pd.read_csv(
        os.path.join(root, "data", "oil_clean.csv"),
        index_col="Date", parse_dates=True,
    )
    df.index = pd.DatetimeIndex(df.index, freq="MS")

    # ── Forward forecasts ──────────────────────────────────────────────────
    fwd_path = os.path.join(root, "outputs", "forecasts", "forward_forecast_12m.csv")
    fwd_df   = pd.read_csv(fwd_path, index_col=0, parse_dates=True) \
               if os.path.exists(fwd_path) else pd.DataFrame()

    # ── Test-period predictions ────────────────────────────────────────────
    test_path = os.path.join(root, "outputs", "forecasts", "test_predictions.csv")
    test_df   = pd.read_csv(test_path, index_col=0, parse_dates=True) \
                if os.path.exists(test_path) else pd.DataFrame()

    # ── Evaluation metrics ─────────────────────────────────────────────────
    metrics_path = os.path.join(root, "outputs", "forecasts", "model_evaluation_metrics.csv")
    metrics_df   = pd.read_csv(metrics_path, index_col=0) \
                   if os.path.exists(metrics_path) else pd.DataFrame()

    # ── Event impact metrics ───────────────────────────────────────────────
    events_path = os.path.join(root, "outputs", "forecasts", "event_impact_metrics.csv")
    events_df   = pd.read_csv(events_path) \
                  if os.path.exists(events_path) else pd.DataFrame()

    # ── Structural break metadata ──────────────────────────────────────────
    breaks_path = os.path.join(root, "structural_breaks.json")
    breaks      = json.load(open(breaks_path)) if os.path.exists(breaks_path) else {}

    # ── Forecast metadata ──────────────────────────────────────────────────
    meta_path = os.path.join(root, "forecast_meta.json")
    meta      = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    return df, fwd_df, test_df, metrics_df, events_df, breaks, meta


# Load — show friendly error if data not found
try:
    df, fwd_df, test_df, metrics_df, events_df, breaks, meta = load_data()
    DATA_LOADED = True
except Exception as e:
    DATA_LOADED = False
    LOAD_ERROR  = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# GEOPOLITICAL EVENTS CATALOGUE  (same as Notebook 02)
# ─────────────────────────────────────────────────────────────────────────────
GEOPOLITICAL_EVENTS = [
    {"date": "1973-10-01", "label": "Arab Oil Embargo",       "type": "Supply shock",    "color": C_RED},
    {"date": "1979-01-01", "label": "Iranian Revolution",     "type": "Supply shock",    "color": C_AMBER},
    {"date": "1986-01-01", "label": "OPEC Price Collapse",    "type": "Supply shock",    "color": C_BLUE},
    {"date": "1990-08-01", "label": "Gulf War",               "type": "Supply shock",    "color": C_TEAL},
    {"date": "1998-01-01", "label": "Asian Financial Crisis", "type": "Demand shock",    "color": C_PURPLE},
    {"date": "2008-07-01", "label": "Pre-GFC Peak",           "type": "Demand shock",    "color": C_PINK},
    {"date": "2008-09-01", "label": "Global Financial Crisis","type": "Financial crisis","color": C_RED},
    {"date": "2014-11-01", "label": "OPEC Price War",         "type": "Supply shock",    "color": C_GREEN},
    {"date": "2020-03-01", "label": "COVID-19 Crash",         "type": "Demand shock",    "color": C_AMBER},
    {"date": "2022-02-01", "label": "Russia-Ukraine War",     "type": "Supply shock",    "color": C_PURPLE},
]


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME  (applied to every figure)
# ─────────────────────────────────────────────────────────────────────────────
_PLOTLY_BASE = dict(
    font=dict(family="DM Sans, sans-serif", size=12, color="#333"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e8eaed",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=1, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", gridwidth=1, zeroline=False),
    hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=12),
)

def PLOTLY_LAYOUT(**overrides):
    """Return a merged layout dict — caller's kwargs win over base defaults."""
    merged = dict(_PLOTLY_BASE)
    merged.update(overrides)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛢️ Oil Price Dashboard")
    st.markdown("---")

    if DATA_LOADED:
        st.markdown("### Filters")

        # Date range
        min_year = int(df.index.year.min())
        max_year = int(df.index.year.max())
        year_range = st.slider(
            "Year range",
            min_value=min_year,
            max_value=max_year,
            value=(1970, max_year),
            step=1,
        )

        st.markdown("---")
        st.markdown("### Event overlays")
        show_events    = st.toggle("Show geopolitical events",   value=True)
        show_moving_avg= st.toggle("Show 12-month moving avg",   value=True)
        show_5yr_avg   = st.toggle("Show 5-year moving avg",     value=False)
        show_regimes   = st.toggle("Show regime shading",        value=False)

        st.markdown("---")
        st.markdown("### Forecasting")
        selected_models = st.multiselect(
            "Models to display",
            options=["ARIMA", "Prophet", "LSTM"],
            default=["ARIMA", "Prophet", "LSTM"],
        )

        st.markdown("---")
        st.markdown("### Dataset info")
        st.caption(f"**Records:** {len(df):,} months")
        st.caption(f"**From:** {df.index.min().date()}")
        st.caption(f"**To:** {df.index.max().date()}")
        st.caption(f"**All-time high:** ${df['price'].max():.2f}  ({df['price'].idxmax().date()})")
        st.caption(f"**All-time low:** ${df['price'].min():.2f}  ({df['price'].idxmin().date()})")
    else:
        st.error("Data not found. Run notebooks 00–03 first.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px; color:#555; line-height:1.6'>"
        "Built with Python · Streamlit · Plotly<br>"
        "Data: WTI Crude Oil 1970–2026<br>"
        "Models: ARIMA · Prophet · LSTM"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT — only if data loaded
# ─────────────────────────────────────────────────────────────────────────────

if not DATA_LOADED:
    st.error(f"Could not load data: {LOAD_ERROR}")
    st.info(
        "Make sure you have run notebooks 00–03 and that the `data/` and "
        "`outputs/` folders are in the same directory as this file."
    )
    st.stop()

# Apply year-range filter
df_filtered = df[
    (df.index.year >= year_range[0]) &
    (df.index.year <= year_range[1])
].copy()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family: DM Serif Display, serif; font-size:2.2rem; "
    "margin-bottom:0; color:#FFF8E7;'>Crude Oil Price Analysis & Forecasting</h1>",
    #"<h1 style='font-family: DM Serif Display, serif; font-size:2.2rem; "
    #"margin-bottom:0; color:#1a1a2e;'>Crude Oil Price Analysis & Forecasting</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666; font-size:14px; margin-top:4px;'>"
    "1970 – 2026 · WTI Crude Oil · Monthly · USD per barrel</p>",
    unsafe_allow_html=True,
)

# ── KPI metric row ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

latest_price    = df["price"].iloc[-1]
prev_price      = df["price"].iloc[-2]
price_delta     = latest_price - prev_price
ytd_start       = df[df.index.year == df.index.year.max()].iloc[0]["price"]
ytd_change      = (latest_price - ytd_start) / ytd_start * 100
decade_mean     = df[df.index.year >= (df.index.year.max() - 9)]["price"].mean()
all_time_high   = df["price"].max()
vol_12m         = df["rolling_std_12"].iloc[-1]

k1.metric("Latest price",     f"${latest_price:.2f}",   f"{price_delta:+.2f} prev month")
k2.metric("YTD change",       f"{ytd_change:+.1f}%",    f"from ${ytd_start:.2f}")
k3.metric("10-year avg",      f"${decade_mean:.2f}",    "USD / barrel")
k4.metric("All-time high",    f"${all_time_high:.2f}",  f"{df['price'].idxmax().year}")
k5.metric("12M volatility",   f"${vol_12m:.2f}",        "rolling std")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_price, tab_events, tab_forecast, tab_compare, tab_data = st.tabs([
    "📈  Price History",
    "🌍  Event Analysis",
    "🔮  Forecasts",
    "🏆  Model Comparison",
    "📋  Data Explorer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_price:

    st.markdown('<div class="section-header">Full Price Series</div>', unsafe_allow_html=True)
    st.caption(
        "Monthly WTI crude oil price with optional moving averages and geopolitical annotations. "
        "Use the sidebar to adjust the date range and toggle overlays."
    )

    # ── Build the hero chart ───────────────────────────────────────────────
    fig = go.Figure()

    # Price fill
    fig.add_trace(go.Scatter(
        x=df_filtered.index, y=df_filtered["price"],
        fill="tozeroy", fillcolor="rgba(50,102,173,0.07)",
        line=dict(color=C_BLUE, width=1.8),
        name="Monthly price",
        hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}/bbl<extra></extra>",
    ))

    # 12-month MA
    if show_moving_avg:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["rolling_mean_12"],
            line=dict(color=C_RED, width=2.2),
            name="12-month MA",
            hovertemplate="<b>%{x|%b %Y}</b><br>12M MA: $%{y:.2f}<extra></extra>",
        ))

    # 5-year MA
    if show_5yr_avg:
        fig.add_trace(go.Scatter(
            x=df_filtered.index, y=df_filtered["rolling_mean_60"],
            line=dict(color=C_AMBER, width=2, dash="dash"),
            name="5-year MA",
            hovertemplate="<b>%{x|%b %Y}</b><br>5Y MA: $%{y:.2f}<extra></extra>",
        ))

    # Regime shading
    if show_regimes and breaks and "breakpoint_dates" in breaks:
        regime_labels = breaks.get("regime_labels", [])
        bp_dates      = [pd.Timestamp(d) for d in breaks["breakpoint_dates"]]
        boundaries    = [df.index.min()] + bp_dates + [df.index.max()]
        seg_colors    = ["rgba(50,102,173,0.08)", "rgba(29,158,117,0.08)",
                         "rgba(212,147,10,0.08)",  "rgba(127,119,221,0.08)",
                         "rgba(100,153,34,0.08)",  "rgba(232,89,60,0.08)",
                         "rgba(136,135,128,0.08)", "rgba(212,83,126,0.08)",
                         "rgba(50,102,173,0.08)"]
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            start_f = max(start, df_filtered.index.min())
            end_f   = min(end,   df_filtered.index.max())
            if start_f >= end_f:
                continue
            label = regime_labels[i] if i < len(regime_labels) else f"Regime {i+1}"
            fig.add_vrect(
                x0=ts(start_f), x1=ts(end_f),
                fillcolor=seg_colors[i % len(seg_colors)],
                layer="below", line_width=0,
                annotation_text=label[:18],
                annotation_position="top left",
                annotation_font_size=9,
                annotation_font_color="#888",
            )

    # Event annotations
    if show_events:
        for ev in GEOPOLITICAL_EVENTS:
            ev_ts = pd.Timestamp(ev["date"])
            if ev_ts < df_filtered.index.min() or ev_ts > df_filtered.index.max():
                continue
            if ev_ts not in df.index:
                continue
            price_val = df.loc[ev_ts, "price"]
            fig.add_vline(
                x=ts(ev_ts), line_width=1, line_dash="dot",
                line_color=ev["color"], opacity=0.6,
            )
            fig.add_annotation(
                x=ts(ev_ts), y=price_val,
                text=ev["label"],
                showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowwidth=1, arrowcolor=ev["color"],
                ax=0, ay=-45,
                font=dict(size=9, color=ev["color"]),
                bgcolor="white", bordercolor=ev["color"],
                borderwidth=0.8, borderpad=3, opacity=0.9,
            )

    fig.update_layout(**PLOTLY_LAYOUT(
        height=500,
        title=dict(text="WTI Crude Oil Price 1970–2026", font=dict(size=16)),
        yaxis_title="USD / barrel",
        xaxis_title="",
        hovermode="x unified",
    ))
    st.plotly_chart(fig, width="stretch")

    # ── Decade stats ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Decade-by-Decade Statistics</div>',
                unsafe_allow_html=True)

    decade_stats = df_filtered.groupby("decade")["price"].agg(
        Mean="mean", Median="median", Std="std", Min="min", Max="max"
    ).round(2)
    decade_stats["CV (%)"]          = (decade_stats["Std"] / decade_stats["Mean"] * 100).round(1)
    decade_stats["Range ($)"]       = (decade_stats["Max"] - decade_stats["Min"]).round(2)
    decade_stats.index              = [f"{d}s" for d in decade_stats.index]
    decade_stats.index.name         = "Decade"

    col_a, col_b = st.columns([3, 2])

    with col_a:
        # Decade mean bar chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=decade_stats.index,
            y=decade_stats["Mean"],
            marker_color=C_BLUE, opacity=0.85,
            error_y=dict(type="data", array=decade_stats["Std"].values, visible=True,
                         color="#aaa", thickness=1.5),
            name="Mean ± std",
            hovertemplate="<b>%{x}</b><br>Mean: $%{y:.2f}/bbl<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=decade_stats.index, y=decade_stats["CV (%)"],
            mode="lines+markers", yaxis="y2",
            line=dict(color=C_RED, width=2),
            marker=dict(size=8),
            name="CV (%)",
            hovertemplate="CV: %{y:.1f}%<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT(
            height=320,
            title="Mean price and volatility by decade",
            yaxis_title="Mean price (USD/bbl)",
            yaxis2=dict(title="CV (%)", overlaying="y", side="right",
                        showgrid=False, zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        ))
        st.plotly_chart(fig2, width="stretch")

    with col_b:
        st.dataframe(
            decade_stats[["Mean", "Std", "Min", "Max", "CV (%)"]],
            width="stretch",
            height=320,
        )

    # ── Monthly return distribution ───────────────────────────────────────
    st.markdown('<div class="section-header">Return Distribution</div>',
                unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        log_ret = df_filtered["log_return"].dropna()
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=log_ret, nbinsx=50,
            marker_color=C_PURPLE, opacity=0.7,
            name="Log returns",
        ))
        fig3.update_layout(**PLOTLY_LAYOUT(
            height=300,
            title="Monthly log-return distribution",
            xaxis_title="Log return", yaxis_title="Count",
        ))
        st.plotly_chart(fig3, width="stretch")

    with col_d:
        # Heatmap: year × month avg price
        pivot = df_filtered.pivot_table(
            index="year", columns="month", values="price", aggfunc="mean"
        )
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig4 = px.imshow(
            pivot,
            color_continuous_scale="YlOrRd",
            labels=dict(color="USD/bbl"),
            title="Year × Month price heatmap",
            aspect="auto",
        )
        fig4.update_layout(**PLOTLY_LAYOUT(height=300))
        st.plotly_chart(fig4, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_events:

    st.markdown('<div class="section-header">Geopolitical Shock Events</div>',
                unsafe_allow_html=True)

    # Event type filter
    type_filter = st.radio(
        "Filter by event type",
        options=["All", "Supply shock", "Demand shock", "Financial crisis"],
        horizontal=True,
    )
    window_months = st.slider("Event window (months before / after shock)", 6, 24, 12)

    # Filter events
    filtered_events = [
        ev for ev in GEOPOLITICAL_EVENTS
        if type_filter == "All" or ev["type"] == type_filter
    ]

    # ── Event catalogue badges ────────────────────────────────────────────
    badge_class = {
        "Supply shock":    "badge-supply",
        "Demand shock":    "badge-demand",
        "Financial crisis":"badge-financial",
    }
    badges_html = ""
    for ev in filtered_events:
        cls = badge_class.get(ev["type"], "badge-supply")
        badges_html += (
            f'<span class="event-badge {cls}">'
            f'{ev["label"]} ({ev["date"][:4]})'
            f'</span> '
        )
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("")

    # ── Normalised event window overlay ──────────────────────────────────
    st.markdown('<div class="section-header">Normalised Price Trajectories</div>',
                unsafe_allow_html=True)
    st.caption(
        "Each event is normalised to 100 at the shock date. "
        "Lines above 100 = prices rose; below 100 = prices fell."
    )

    fig_ev = go.Figure()
    fig_ev.add_hline(y=100, line_dash="dot", line_color="#ccc", line_width=1)
    fig_ev.add_vline(x=0,   line_dash="dash", line_color="#555", line_width=1,
                    annotation_text="Shock date", annotation_font_size=10)
    fig_ev.add_vrect(x0=-window_months, x1=0, fillcolor="rgba(200,200,200,0.08)",
                    layer="below", line_width=0)

    for ev in filtered_events:
        shock_ts = pd.Timestamp(ev["date"])
        if shock_ts not in df.index:
            continue
        base = df.loc[shock_ts, "price"]
        if base == 0:
            continue
        start = shock_ts - pd.DateOffset(months=window_months)
        end   = shock_ts + pd.DateOffset(months=window_months)
        end   = min(end, df.index.max())
        window_data = df.loc[start:end, "price"]
        offsets     = [
            (d.year - shock_ts.year) * 12 + (d.month - shock_ts.month)
            for d in window_data.index
        ]
        normalised = (window_data.values / base * 100).tolist()

        fig_ev.add_trace(go.Scatter(
            x=offsets, y=normalised,
            name=f"{ev['label']} ({ev['date'][:4]})",
            line=dict(color=ev["color"], width=2),
            opacity=0.85,
            hovertemplate=(
                f"<b>{ev['label']}</b><br>"
                "Month offset: %{x}<br>"
                "Normalised: %{y:.1f}<extra></extra>"
            ),
        ))

    fig_ev.update_layout(**PLOTLY_LAYOUT(
        height=460,
        title="Price trajectories around shock events (shock month = 100)",
        xaxis_title=f"Months relative to shock (±{window_months})",
        yaxis_title="Normalised price (shock = 100)",
        hovermode="x unified",
    ))
    st.plotly_chart(fig_ev, width="stretch")

    # ── Impact metrics table ──────────────────────────────────────────────
    if not events_df.empty:
        st.markdown('<div class="section-header">Quantified Impact Metrics</div>',
                    unsafe_allow_html=True)

        display_cols = [
            "Event", "Date", "Type", "Direction",
            "Price at shock ($)", "Change from shock (%)",
            "Months to extreme", "Recovery months",
        ]
        available = [c for c in display_cols if c in events_df.columns]

        # Colour rows by direction
        def highlight_direction(row):
            if "Direction" in row.index:
                if str(row["Direction"]).lower() == "spike":
                    return ["background-color: #fff5f3"] * len(row)
                elif str(row["Direction"]).lower() == "crash":
                    return ["background-color: #f3f7ff"] * len(row)
            return [""] * len(row)

        st.dataframe(
            events_df[available].style.apply(highlight_direction, axis=1),
            width="stretch",
            height=380,
        )

        # Quick stats
        col_e, col_f, col_g = st.columns(3)
        numeric_chg = pd.to_numeric(events_df["Change from shock (%)"], errors="coerce")
        col_e.metric("Largest single shock",
                     f"{numeric_chg.abs().max():.1f}%",
                     events_df.loc[numeric_chg.abs().idxmax(), "Event"] if not numeric_chg.isna().all() else "")
        numeric_rec = pd.to_numeric(events_df["Recovery months"], errors="coerce")
        col_f.metric("Avg recovery time",
                     f"{numeric_rec.mean():.0f} months",
                     "across events with recovery in window")
        supply_n = (events_df["Type"] == "Supply shock").sum()
        col_g.metric("Supply vs demand shocks",
                     f"{supply_n} supply",
                     f"{len(events_df) - supply_n} demand/financial")
    else:
        st.info("Run notebook 02 to generate event impact metrics.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_forecast:

    st.markdown('<div class="section-header">12-Month Forward Forecast</div>',
                unsafe_allow_html=True)

    if fwd_df.empty:
        st.info("Run notebook 03 to generate forecasts. Forecast data not found.")
    else:
        # Context: recent history + forecast
        history_months = st.slider(
            "Months of history to show alongside forecast", 12, 60, 36
        )
        recent_hist = df["price"].iloc[-history_months:]

        fig_fwd = go.Figure()

        # Historical
        fig_fwd.add_trace(go.Scatter(
            x=recent_hist.index, y=recent_hist.values,
            line=dict(color="#222", width=2.2),
            name="Historical price",
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:.2f}/bbl<extra></extra>",
        ))

        # Each model's forecast
        for model in selected_models:
            if model not in fwd_df.columns:
                continue
            vals = fwd_df[model]
            fig_fwd.add_trace(go.Scatter(
                x=vals.index, y=vals.values,
                line=dict(color=MODEL_COLORS[model], width=2.2, dash="dash"),
                name=f"{model} forecast",
                hovertemplate=(
                    f"<b>%{{x|%b %Y}}</b><br>{model}: $%{{y:.2f}}/bbl<extra></extra>"
                ),
            ))
            # End-point marker
            fig_fwd.add_trace(go.Scatter(
                x=[vals.index[-1]], y=[vals.values[-1]],
                mode="markers+text",
                marker=dict(color=MODEL_COLORS[model], size=10),
                text=[f"${vals.values[-1]:.0f}"],
                textposition="top center",
                textfont=dict(size=10, color=MODEL_COLORS[model]),
                showlegend=False,
            ))

        # Forecast zone shading
        if not fwd_df.empty:
            fig_fwd.add_vrect(
                x0=ts(fwd_df.index[0]), x1=ts(fwd_df.index[-1]),
                fillcolor="rgba(200,200,200,0.08)",
                layer="below", line_width=0,
                annotation_text="Forecast zone",
                annotation_font_size=10,
                annotation_font_color="#888",
                annotation_position="top left",
            )
        fig_fwd.add_vline(
            x=ts(df.index[-1]),
            line_dash="dot", line_color="#555", line_width=1.5,
            annotation_text=f"Last data: {df.index[-1].date()}",
            annotation_font_size=10,
        )

        fig_fwd.update_layout(**PLOTLY_LAYOUT(
            height=480,
            title="12-Month Forward Forecast — All Selected Models",
            yaxis_title="USD / barrel",
            hovermode="x unified",
        ))
        st.plotly_chart(fig_fwd, width="stretch")

        # ── Forecast table ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Forecast Values Table</div>',
                    unsafe_allow_html=True)

        display_fwd = fwd_df[[m for m in selected_models if m in fwd_df.columns]].copy()
        display_fwd.index = display_fwd.index.strftime("%Y-%m")
        display_fwd.index.name = "Month"
        display_fwd = display_fwd.round(2)

        st.dataframe(display_fwd, width="stretch")

        # Download button
        csv_bytes = display_fwd.to_csv().encode()
        st.download_button(
            label="📥  Download forecast CSV",
            data=csv_bytes,
            file_name="oil_forecast_12m.csv",
            mime="text/csv",
        )

        # ── Forecast divergence note ──────────────────────────────────────
        if len(selected_models) > 1:
            avail = [m for m in selected_models if m in fwd_df.columns]
            if len(avail) > 1:
                last_vals = {m: fwd_df[m].iloc[-1] for m in avail}
                spread = max(last_vals.values()) - min(last_vals.values())
                highest_m = max(last_vals, key=last_vals.get)
                lowest_m  = min(last_vals, key=last_vals.get)
                st.markdown(
                    f'<div class="insight-box">'
                    f'<strong>Model divergence at 12-month horizon:</strong> '
                    f'${spread:.2f}/bbl spread between '
                    f'{highest_m} (${last_vals[highest_m]:.2f}) and '
                    f'{lowest_m} (${last_vals[lowest_m]:.2f}). '
                    f'Wider divergence signals higher uncertainty in the long-run trend.'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:

    st.markdown('<div class="section-header">Test-Set Performance</div>',
                unsafe_allow_html=True)

    if test_df.empty or metrics_df.empty:
        st.info("Run notebook 03 to generate model evaluation data.")
    else:
        # ── Metrics scoreboard ────────────────────────────────────────────
        if not metrics_df.empty and "RMSE" in metrics_df.columns:
            mc1, mc2, mc3 = st.columns(3)
            best_rmse = metrics_df["RMSE"].idxmin()
            best_mae  = metrics_df["MAE"].idxmin()
            best_mape = metrics_df["MAPE"].idxmin()
            mc1.metric("Best RMSE",
                       best_rmse,
                       f"${metrics_df.loc[best_rmse,'RMSE']:.2f}/bbl")
            mc2.metric("Best MAE",
                       best_mae,
                       f"${metrics_df.loc[best_mae,'MAE']:.2f}/bbl")
            mc3.metric("Best MAPE",
                       best_mape,
                       f"{metrics_df.loc[best_mape,'MAPE']:.2f}%")
            st.markdown("")

        # ── Actual vs predicted on test set ──────────────────────────────
        fig_test = go.Figure()

        if "actual" in test_df.columns:
            fig_test.add_trace(go.Scatter(
                x=test_df.index, y=test_df["actual"],
                line=dict(color="#111", width=2.5),
                name="Actual price",
                hovertemplate="<b>%{x|%b %Y}</b><br>Actual: $%{y:.2f}<extra></extra>",
            ))

        for model in selected_models:
            if model not in test_df.columns:
                continue
            fig_test.add_trace(go.Scatter(
                x=test_df.index, y=test_df[model],
                line=dict(color=MODEL_COLORS[model], width=2, dash="dash"),
                name=model,
                hovertemplate=f"<b>%{{x|%b %Y}}</b><br>{model}: $%{{y:.2f}}<extra></extra>",
            ))

        fig_test.update_layout(**PLOTLY_LAYOUT(
            height=380,
            title="Actual vs Predicted — Test Period",
            yaxis_title="USD / barrel",
            hovermode="x unified",
        ))
        st.plotly_chart(fig_test, width="stretch")

        # ── Residuals ─────────────────────────────────────────────────────
        if "actual" in test_df.columns:
            fig_res = go.Figure()
            fig_res.add_hline(y=0, line_color="#aaa", line_width=1)
            fig_res.add_hrect(y0=-5, y1=5, fillcolor="rgba(200,200,200,0.1)",
                              layer="below", line_width=0,
                              annotation_text="±$5/bbl band",
                              annotation_font_size=9, annotation_font_color="#888")

            for model in selected_models:
                if model not in test_df.columns:
                    continue
                residuals = test_df["actual"] - test_df[model]
                fig_res.add_trace(go.Scatter(
                    x=test_df.index, y=residuals,
                    line=dict(color=MODEL_COLORS[model], width=1.8),
                    name=model,
                    hovertemplate=(
                        f"<b>%{{x|%b %Y}}</b><br>{model} residual: $%{{y:.2f}}<extra></extra>"
                    ),
                ))

            fig_res.update_layout(**PLOTLY_LAYOUT(
                height=300,
                title="Prediction Residuals (Actual − Forecast)",
                yaxis_title="Residual (USD/bbl)",
                hovermode="x unified",
            ))
            st.plotly_chart(fig_res, width="stretch")

        # ── Metric bar charts ─────────────────────────────────────────────
        if not metrics_df.empty and "RMSE" in metrics_df.columns:
            col_m1, col_m2 = st.columns(2)

            with col_m1:
                fig_m = go.Figure()
                avail_m = [m for m in ["ARIMA", "Prophet", "LSTM"] if m in metrics_df.index]
                for metric, opacity in [("RMSE", 0.85), ("MAE", 0.55)]:
                    fig_m.add_trace(go.Bar(
                        x=avail_m,
                        y=[metrics_df.loc[m, metric] for m in avail_m],
                        name=metric, opacity=opacity,
                        marker_color=[MODEL_COLORS.get(m, C_GRAY) for m in avail_m],
                        hovertemplate=f"{metric}: $%{{y:.2f}}<extra></extra>",
                    ))
                fig_m.update_layout(**PLOTLY_LAYOUT(
                    height=320,
                    title="RMSE and MAE (lower = better)",
                    yaxis_title="Error (USD/bbl)",
                    barmode="group",
                ))
                st.plotly_chart(fig_m, width="stretch")

            with col_m2:
                avail_m = [m for m in ["ARIMA", "Prophet", "LSTM"] if m in metrics_df.index]
                fig_mape = go.Figure(go.Bar(
                    x=avail_m,
                    y=[metrics_df.loc[m, "MAPE"] for m in avail_m],
                    marker_color=[MODEL_COLORS.get(m, C_GRAY) for m in avail_m],
                    opacity=0.85,
                    text=[f"{metrics_df.loc[m,'MAPE']:.2f}%" for m in avail_m],
                    textposition="outside",
                    hovertemplate="MAPE: %{y:.2f}%<extra></extra>",
                ))
                fig_mape.update_layout(**PLOTLY_LAYOUT(
                    height=320,
                    title="MAPE — % of actual price (lower = better)",
                    yaxis_title="MAPE (%)",
                ))
                st.plotly_chart(fig_mape, width="stretch")

            # Full metrics table
            st.markdown('<div class="section-header">Full Evaluation Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                metrics_df[["RMSE", "MAE", "MAPE"]].style.highlight_min(
                    axis=0, color="#e8f5e9"
                ),
                width="stretch",
            )
            st.caption(
                "Green cells = best value for that metric. "
                "RMSE penalises large errors more heavily; "
                "MAPE is scale-independent and easiest to interpret."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:

    st.markdown('<div class="section-header">Data Explorer</div>',
                unsafe_allow_html=True)

    # Column selector
    all_cols = [c for c in df_filtered.columns if c != "regime"]
    selected_cols = st.multiselect(
        "Columns to display",
        options=all_cols,
        default=["price", "pct_change", "rolling_mean_12", "rolling_std_12"],
    )

    if selected_cols:
        # Show table (most recent first)
        st.dataframe(
            df_filtered[selected_cols].iloc[::-1].round(4),
            width="stretch",
            height=400,
        )

        # Download
        csv_bytes = df_filtered[selected_cols].to_csv().encode()
        st.download_button(
            label="📥  Download filtered dataset (CSV)",
            data=csv_bytes,
            file_name=f"oil_prices_{year_range[0]}_{year_range[1]}.csv",
            mime="text/csv",
        )

    # ── Summary stats ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Summary Statistics</div>',
                unsafe_allow_html=True)
    numeric_cols = [c for c in df_filtered.columns
                    if df_filtered[c].dtype in [float, int] and c != "regime"]
    st.dataframe(
        df_filtered[numeric_cols].describe().round(4),
        width="stretch",
    )

    # ── Correlation matrix ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Correlations</div>',
                unsafe_allow_html=True)

    corr_cols = ["price", "pct_change", "log_return",
                 "rolling_mean_12", "rolling_std_12", "rolling_mean_60"]
    corr_cols = [c for c in corr_cols if c in df_filtered.columns]
    corr_matrix = df_filtered[corr_cols].corr().round(3)

    fig_corr = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=True,
        title="Pearson correlation matrix",
        aspect="auto",
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT(height=380))
    st.plotly_chart(fig_corr, width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>"
    "Crude Oil Price Analysis & Forecasting · Built with Streamlit & Plotly · "
    "Data: WTI Monthly Prices 1970–2026 · "
    "Models: ARIMA · Facebook Prophet · LSTM"
    "</div>",
    unsafe_allow_html=True,
)
