
# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f0f0f; }
    .metric-card {
        background: #1a1a2e;
        border-left: 4px solid #e94560;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Risk Dashboard")
st.markdown("*Interactive VaR, Expected Shortfall & Drawdown Analysis — 2016–2025*")
st.markdown("---")

# ── Load Data ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    prices  = pd.read_csv(os.path.join(BASE_DIR, "../data/raw/prices.csv"),
                          index_col="Date", parse_dates=True)
    returns = pd.read_csv(os.path.join(BASE_DIR, "../data/processed/returns.csv"),
                          index_col="Date", parse_dates=True)
    return prices, returns

prices, returns = load_data()

TICKERS = {
    "SPY": "S&P 500 ETF",
    "TLT": "20Y US Treasury ETF",
    "GLD": "Gold ETF",
    "QQQ": "Nasdaq 100 ETF",
    "EEM": "Emerging Markets ETF"
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Portfolio Controls")
st.sidebar.markdown("Adjust weights and settings below. Charts update automatically.")
st.sidebar.markdown("---")

st.sidebar.subheader("📊 Asset Weights")
st.sidebar.markdown("*Weights must sum to 1.0*")

w_spy = st.sidebar.slider("SPY — S&P 500",       0.0, 1.0, 0.35, 0.05)
w_tlt = st.sidebar.slider("TLT — US Treasuries", 0.0, 1.0, 0.25, 0.05)
w_gld = st.sidebar.slider("GLD — Gold",          0.0, 1.0, 0.15, 0.05)
w_qqq = st.sidebar.slider("QQQ — Nasdaq 100",    0.0, 1.0, 0.15, 0.05)
w_eem = st.sidebar.slider("EEM — Emerging Mkts", 0.0, 1.0, 0.10, 0.05)

weights_raw = np.array([w_spy, w_tlt, w_gld, w_qqq, w_eem])
weight_sum  = weights_raw.sum()

# Normalise weights to always sum to 1
weights = weights_raw / weight_sum if weight_sum > 0 else weights_raw

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Risk Settings")
confidence = st.sidebar.selectbox(
    "Confidence Level",
    options=[0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{int(x*100)}%"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Normalised Weights:**")
for ticker, w in zip(TICKERS.keys(), weights):
    st.sidebar.markdown(f"- {ticker}: {w:.1%}")
st.sidebar.markdown(f"**Total: {weights.sum():.2f}**")

# ── Compute Portfolio Returns ─────────────────────────────────────────────────
portfolio_returns = returns.dot(weights)

# ── Risk Calculations ─────────────────────────────────────────────────────────
# Historical VaR & ES
hist_var = np.percentile(portfolio_returns, (1 - confidence) * 100)
tail     = portfolio_returns[portfolio_returns <= hist_var]
hist_es  = tail.mean()

# Parametric VaR
mu        = portfolio_returns.mean()
sigma     = portfolio_returns.std()
z_scores  = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
param_var = mu - z_scores[confidence] * sigma

# Monte Carlo VaR
np.random.seed(42)
sim       = np.random.normal(mu, sigma, 10_000)
mc_var    = np.percentile(sim, (1 - confidence) * 100)

# Drawdown
cumulative  = (1 + portfolio_returns).cumprod()
running_max = cumulative.cummax()
drawdown    = (cumulative - running_max) / running_max
max_dd      = drawdown.min()
max_dd_date = drawdown.idxmin()

# Annualised stats
ann_return = mu * 252
ann_vol    = sigma * np.sqrt(252)
sharpe     = ann_return / ann_vol if ann_vol > 0 else 0

# ── Metrics Row ───────────────────────────────────────────────────────────────
st.subheader(f"📐 Risk Metrics at {int(confidence*100)}% Confidence")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label=f"Historical VaR ({int(confidence*100)}%)",
    value=f"{abs(hist_var)*100:.2f}%",
    delta="daily loss threshold",
    delta_color="off"
)
col2.metric(
    label=f"Parametric VaR ({int(confidence*100)}%)",
    value=f"{abs(param_var)*100:.2f}%",
    delta="normal distribution",
    delta_color="off"
)
col3.metric(
    label=f"Monte Carlo VaR ({int(confidence*100)}%)",
    value=f"{abs(mc_var)*100:.2f}%",
    delta="10,000 simulations",
    delta_color="off"
)
col4.metric(
    label=f"Expected Shortfall ({int(confidence*100)}%)",
    value=f"{abs(hist_es)*100:.2f}%",
    delta="avg loss beyond VaR",
    delta_color="off"
)
col5.metric(
    label="Max Drawdown",
    value=f"{abs(max_dd)*100:.2f}%",
    delta=f"on {max_dd_date.strftime('%d %b %Y')}",
    delta_color="off"
)

st.markdown("---")

# ── Portfolio Stats Row ───────────────────────────────────────────────────────
st.subheader("📊 Portfolio Statistics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Annualised Return",   f"{ann_return*100:.2f}%")
col2.metric("Annualised Volatility", f"{ann_vol*100:.2f}%")
col3.metric("Sharpe Ratio",        f"{sharpe:.2f}")
col4.metric("Worst Single Day",    f"{portfolio_returns.min()*100:.2f}%")

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("📈 Charts")
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price History",
    "📉 Return Distribution",
    "📉 Drawdown",
    "📊 VaR Comparison"
])

# ── Tab 1: Normalised Price History ──────────────────────────────────────────
with tab1:
    st.markdown("**Normalised price performance (Base = 100) — 2016 to 2025**")
    normalised = (prices / prices.iloc[0]) * 100
    fig1 = go.Figure()
    colors_map = {
        "SPY": "#e94560", "TLT": "#a8dadc",
        "GLD": "#f4a261", "QQQ": "#ffffff", "EEM": "#c77dff"
    }
    for ticker, label in TICKERS.items():
        fig1.add_trace(go.Scatter(
            x=normalised.index,
            y=normalised[ticker],
            name=f"{ticker} — {label}",
            line=dict(color=colors_map[ticker], width=1.5)
        ))
    # Shade crisis periods
    fig1.add_vrect(x0="2020-02-01", x1="2020-04-30",
                   fillcolor="red", opacity=0.1,
                   annotation_text="COVID", annotation_position="top left")
    fig1.add_vrect(x0="2022-01-01", x1="2022-12-31",
                   fillcolor="orange", opacity=0.1,
                   annotation_text="Rate Hikes", annotation_position="top left")
    fig1.update_layout(
        template="plotly_dark",
        title="Normalised Price Performance (Base = 100)",
        xaxis_title="Date",
        yaxis_title="Normalised Price",
        height=500,
        legend=dict(orientation="v", x=0.01, y=0.99)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ── Tab 2: Return Distribution ────────────────────────────────────────────────
with tab2:
    st.markdown(f"**Portfolio daily return distribution with VaR thresholds marked**")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=portfolio_returns,
        nbinsx=100,
        name="Daily Returns",
        marker_color="#e94560",
        opacity=0.8
    ))
    fig2.add_vline(x=hist_var,  line_dash="dash", line_color="orange",
                   annotation_text=f"Hist VaR {int(confidence*100)}%: {abs(hist_var)*100:.2f}%",
                   annotation_position="top left")
    fig2.add_vline(x=param_var, line_dash="dash", line_color="cyan",
                   annotation_text=f"Param VaR {int(confidence*100)}%: {abs(param_var)*100:.2f}%",
                   annotation_position="top right")
    fig2.add_vline(x=hist_es,   line_dash="dot",  line_color="white",
                   annotation_text=f"ES {int(confidence*100)}%: {abs(hist_es)*100:.2f}%",
                   annotation_position="bottom left")
    fig2.update_layout(
        template="plotly_dark",
        title="Portfolio Daily Log Return Distribution",
        xaxis_title="Daily Log Return",
        yaxis_title="Frequency",
        height=500
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Drawdown ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("**Portfolio value and drawdown from peak over the full sample period**")
    fig3 = go.Figure()

    # Cumulative value
    fig3.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative.values,
        name="Portfolio Value",
        line=dict(color="#e94560", width=1.5),
        yaxis="y1"
    ))
    # Drawdown
    fig3.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="#f4a261", width=1),
        yaxis="y2"
    ))
    fig3.add_hline(
        y=max_dd, line_dash="dash", line_color="cyan",
        annotation_text=f"Max DD: {max_dd*100:.2f}%",
        yref="y2"
    )
    fig3.update_layout(
        template="plotly_dark",
        title="Cumulative Portfolio Value & Drawdown",
        height=550,
        yaxis=dict(title="Portfolio Value ($)", side="left"),
        yaxis2=dict(title="Drawdown", side="right", overlaying="y"),
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 4: VaR Comparison ─────────────────────────────────────────────────────
with tab4:
    st.markdown("**VaR estimates across all three methods and all confidence levels**")
    cls    = [0.90, 0.95, 0.99]
    labels = ["90%", "95%", "99%"]

    hist_vals  = [abs(np.percentile(portfolio_returns, (1-c)*100))*100 for c in cls]
    param_vals = [abs(mu - z_scores[c] * sigma)*100 for c in cls]
    mc_vals    = [abs(np.percentile(np.random.normal(mu, sigma, 10_000), (1-c)*100))*100 for c in cls]
    es_vals    = [abs(portfolio_returns[portfolio_returns <=
                  np.percentile(portfolio_returns, (1-c)*100)].mean())*100 for c in cls]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(name="Historical VaR",  x=labels, y=hist_vals,  marker_color="#e94560"))
    fig4.add_trace(go.Bar(name="Parametric VaR",  x=labels, y=param_vals, marker_color="#a8dadc"))
    fig4.add_trace(go.Bar(name="Monte Carlo VaR", x=labels, y=mc_vals,    marker_color="#f4a261"))
    fig4.add_trace(go.Bar(name="Expected Shortfall", x=labels, y=es_vals, marker_color="#c77dff"))

    fig4.update_layout(
        template="plotly_dark",
        barmode="group",
        title="VaR & ES Comparison Across Methods and Confidence Levels",
        xaxis_title="Confidence Level",
        yaxis_title="Loss Threshold (%)",
        height=500,
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Summary table
    st.markdown("**📋 Summary Table**")
    summary = pd.DataFrame({
        "Historical VaR"    : [f"{v:.3f}%" for v in hist_vals],
        "Parametric VaR"    : [f"{v:.3f}%" for v in param_vals],
        "Monte Carlo VaR"   : [f"{v:.3f}%" for v in mc_vals],
        "Expected Shortfall": [f"{v:.3f}%" for v in es_vals]
    }, index=["90%", "95%", "99%"])
    summary.index.name = "Confidence Level"
    st.dataframe(summary, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Portfolio Risk Dashboard · Built with Python, Streamlit & Plotly · 2026*"
)
