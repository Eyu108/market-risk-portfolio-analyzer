# app.py — Market Risk & Portfolio Analyzer
# UI: guided inputs, table editor (with pyarrow fallback)
# Data source options: Stooq (no key mentioned) or CSV upload only

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from risklab.beta import rolling_beta
from risklab.data import get_prices, pct_returns
from risklab.factors import factor_regression
from risklab.garch import fit_garch, forecast_vol
from risklab.metrics import (annualize_return, annualize_vol, drawdown,
                             hit_rate, max_drawdown, sharpe, sortino)
from risklab.portfolio import allocate_static
from risklab.var import (cvar_historical, var_cornish_fisher, var_historical,
                         var_parametric_normal)

# ---- detect pyarrow (Streamlit's data_editor prefers it) ----
try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

st.set_page_config(page_title="Market Risk & Portfolio Analyzer", layout="wide")
st.title("Market Risk & Portfolio Analyzer")

# ---------- palettes / templates ----------
PALETTES = {
    "Energy Pro": ["#22c55e", "#eab308", "#3b82f6", "#ef4444", "#a855f7"],
    "Quant Minimal": ["#06b6d4", "#94a3b8", "#22c55e", "#f59e0b", "#ef4444"],
    "Rose Pine": ["#eb6f92", "#31748f", "#9ccfd8", "#f6c177", "#c4a7e7"],
    "Citrus": ["#10b981", "#f97316", "#f59e0b", "#3b82f6", "#ef4444"],
}
TEMPLATES = ["plotly_dark", "plotly", "simple_white", "ggplot2", "seaborn"]

# ---------- defaults ----------
if "params" not in st.session_state:
    st.session_state.params = {
        "provider": "Stooq",  # <-- limited to Stooq or CSV only
        "tick_tbl": pd.DataFrame(
            {"Ticker": ["SPY", "XLE", "CVX"], "Weight": [0.5, 0.25, 0.25]}
        ),
        "tickers_fallback": "SPY, XLE, CVX",
        "weights_fallback": "0.5, 0.25, 0.25",
        "benchmark_choice": "SPY",
        "benchmark_custom": "",
        "start": pd.to_datetime("2020-01-01"),
        "end": pd.Timestamp.today().normalize(),
        "rebalance": "M",
        "alpha": 0.95,
        "interval": "1d",
        "retries": 2,
        "pause": 1.0,
        "template": "plotly_dark",
        "palette": "Energy Pro",
        "accent": "#22c55e",
        "normalize": True,
        "map_tsx_to_us": True,
    }
P = st.session_state.params

# keep uploaded prices across reruns
if "uploaded_prices_df" not in st.session_state:
    st.session_state["uploaded_prices_df"] = None

# ---------- sidebar UI ----------
with st.sidebar:
    st.subheader("Portfolio")

    provider = st.selectbox(
        "Data source",
        ["Stooq", "CSV only"],
        index=["Stooq", "CSV only"].index(P["provider"]),
        help="Choose live data from Stooq or work entirely from an uploaded CSV.",
    )

    st.caption("Enter tickers and weights as rows. Toggle **Normalize** to rescale weights to 1.0.")
    if HAS_PYARROW:
        tick_tbl = st.data_editor(
            P["tick_tbl"],
            num_rows="dynamic",
            column_config={
                "Ticker": st.column_config.TextColumn(
                    "Ticker",
                    help="Examples: SPY, XLE, CVX, IMO.TO (mapped to US duals for Stooq if mapping is ON)",
                    required=False,
                    width="medium",
                    validate="^[A-Za-z0-9^=._-]{1,15}$",
                ),
                "Weight": st.column_config.NumberColumn(
                    "Weight",
                    help="Portfolio weight (fraction). Use **Normalize** if they don't sum to 1.",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    format="%.4f",
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        tickers_fallback = P["tickers_fallback"]
        weights_fallback = P["weights_fallback"]
    else:
        st.info(
            "`pyarrow` not installed → falling back to simple inputs. "
            "Install with `python -m pip install pyarrow==17.0.0` for a table editor."
        )
        tickers_fallback = st.text_input("Tickers (comma-separated)", P["tickers_fallback"])
        weights_fallback = st.text_input("Weights (comma-separated)", P["weights_fallback"])
        tks = [t.strip() for t in tickers_fallback.split(",") if t.strip()]
        try:
            wts = [float(x.strip()) for x in weights_fallback.split(",")]
        except Exception:
            wts = []
        tick_tbl = pd.DataFrame({"Ticker": tks, "Weight": wts})

    c1, c2 = st.columns(2)
    with c1:
        normalize = st.toggle("Normalize", value=P["normalize"])
    with c2:
        map_tsx_to_us = st.toggle(
            "Map TSX (.TO) → US",
            value=P["map_tsx_to_us"],
            help="Maps e.g. IMO.TO → IMO / IMO.US when using Stooq.",
        )

    bench_opt = st.selectbox(
        "Benchmark",
        ["SPY", "None", "Custom…"],
        index=["SPY", "None", "Custom…"].index(P["benchmark_choice"]),
    )
    bench_custom = P["benchmark_custom"]
    if bench_opt == "Custom…":
        bench_custom = st.text_input("Custom benchmark ticker", P["benchmark_custom"])

    start = st.date_input("Start", value=P["start"])
    end = st.date_input("End", value=P["end"])
    rebalance_opt = st.selectbox(
        "Rebalance",
        ["M", "Q", "None"],
        index=["M", "Q", "None"].index(P["rebalance"]),
        help="Portfolio weights rebalanced Monthly, Quarterly, or not at all.",
    )
    alpha = st.slider("VaR Confidence (α)", 0.80, 0.99, P["alpha"], 0.01)

    with st.expander("Appearance", expanded=False):
        template = st.selectbox("Chart template", TEMPLATES, index=TEMPLATES.index(P["template"]))
        palette_name = st.selectbox(
            "Color palette", list(PALETTES.keys()), index=list(PALETTES.keys()).index(P["palette"])
        )
        accent = st.color_picker("Accent color", P["accent"])

    with st.expander("Advanced", expanded=False):
        interval = st.selectbox("Data interval", ["1d", "1wk", "1mo"], index=["1d", "1wk", "1mo"].index(P["interval"]))
        retries = st.slider("Retries per ticker", 0, 6, P["retries"])
        pause = st.slider("Pause between retries (sec)", 0.0, 3.0, float(P["pause"]), 0.1)

    with st.expander("Upload price history (CSV)"):
        st.caption("Wide CSV with a 'Date' column and one column per ticker (prices).")
        up_prices = st.file_uploader("Upload prices CSV", type=["csv"], key="price_csv")
        if up_prices is not None:
            try:
                dfp = pd.read_csv(up_prices)
                if "Date" not in dfp.columns:
                    st.error("CSV must include a 'Date' column.")
                else:
                    dfp["Date"] = pd.to_datetime(dfp["Date"], errors="coerce")
                    dfp = dfp.dropna(subset=["Date"]).set_index("Date").sort_index()
                    st.session_state["uploaded_prices_df"] = dfp
                    st.success("Loaded prices from CSV.")
            except Exception as e:
                st.error(f"Failed to parse price CSV: {e}")

    use_sample = st.toggle("Use sample data (offline demo)", value=False)
    submitted = st.button("Apply / Refresh", use_container_width=True)

# persist settings
if submitted:
    st.session_state.params = {
        "provider": provider,
        "tick_tbl": tick_tbl if HAS_PYARROW else P["tick_tbl"],
        "tickers_fallback": tickers_fallback,
        "weights_fallback": weights_fallback,
        "benchmark_choice": bench_opt,
        "benchmark_custom": bench_custom,
        "start": pd.to_datetime(start),
        "end": pd.to_datetime(end),
        "rebalance": None if rebalance_opt == "None" else rebalance_opt,
        "alpha": float(alpha),
        "interval": interval,
        "retries": int(retries),
        "pause": float(pause),
        "template": template,
        "palette": palette_name,
        "accent": accent,
        "normalize": normalize,
        "map_tsx_to_us": map_tsx_to_us,
    }
    P = st.session_state.params

# ---------- parse inputs ----------
tbl = P["tick_tbl"] if HAS_PYARROW else pd.DataFrame(
    {
        "Ticker": [t.strip() for t in P["tickers_fallback"].split(",") if t.strip()],
        "Weight": (
            [float(x.strip()) for x in P["weights_fallback"].split(",")]
            if P["weights_fallback"].strip()
            else []
        ),
    }
)
tbl["Ticker"] = tbl["Ticker"].astype(str).str.strip()
tbl = tbl[tbl["Ticker"] != ""]
tickers = tbl["Ticker"].tolist()
weights_vals = tbl["Weight"].astype(float).tolist() if "Weight" in tbl else []

if P["normalize"] and weights_vals:
    ssum = sum(weights_vals)
    weights_vals = [w / ssum if ssum != 0 else 0.0 for w in weights_vals]
weights = {t: w for t, w in zip(tickers, weights_vals)}

benchmark = None
if P["benchmark_choice"] == "SPY":
    benchmark = "SPY"
elif P["benchmark_choice"] == "Custom…" and P["benchmark_custom"].strip():
    benchmark = P["benchmark_custom"].strip()

# ---------- colors / template ----------
colors = [P["accent"]] + [c for c in PALETTES[P["palette"]] if c != P["accent"]]
template = P["template"]

# ---------- fetching ----------
@st.cache_data(show_spinner="Fetching prices...", ttl=3600)
def fetch_prices(all_syms, start, end, interval, retries, pause, provider, map_tsx_to_us):
    return get_prices(
        all_syms,
        start=start,
        end=end,
        interval=interval,
        retries=retries,
        pause=pause,
        provider=provider,          # "Stooq" or "CSV only" (we stop before calling when CSV only)
        map_tsx_to_us=map_tsx_to_us,
    )

uploaded_prices_df = st.session_state.get("uploaded_prices_df")

if uploaded_prices_df is not None:
    prices = uploaded_prices_df.copy()
else:
    if provider == "CSV only":
        st.warning("Data source is set to CSV only. Upload a prices CSV in the sidebar.")
        st.stop()
    all_syms = tickers + ([benchmark] if benchmark else [])
    if not all_syms:
        st.info("Add at least one ticker row, then click **Apply / Refresh**.")
        st.stop()
    prices = fetch_prices(
        all_syms,
        P["start"],
        P["end"],
        P["interval"],
        P["retries"],
        P["pause"],
        P["provider"],
        P["map_tsx_to_us"],
    )

rets = pct_returns(prices)

# sample fallback if live empty
if (prices is None or prices.empty or rets.empty) and not use_sample:
    st.warning("Live data unavailable — switching to sample data.")
    np.random.seed(42)
    dates = pd.date_range(P["start"], P["end"], freq="B")
    K = max(1, len(tickers))
    mu = np.full(K, 0.08 / 252.0)
    sig = np.full(K, 0.18 / np.sqrt(252.0))
    sim_rets = {
        tickers[i] if i < len(tickers) else f"A{i}": np.random.normal(
            mu[min(i, K - 1)], sig[min(i, K - 1)], size=len(dates)
        )
        for i in range(K)
    }
    prices = pd.DataFrame(sim_rets, index=dates).pipe(lambda df: (1.0 + df).cumprod())
    if benchmark:
        prices[benchmark] = (1.0 + np.random.normal(0.08 / 252.0, 0.15 / np.sqrt(252.0), size=len(dates))).cumprod()
    rets = pct_returns(prices)

if prices.empty or rets.empty:
    st.warning("No data available.")
    st.stop()

# ---------- portfolio ----------
tick_cols_present = [t for t in tickers if t in rets.columns]
if not tick_cols_present:
    st.warning("None of the requested tickers returned data. Adjust inputs and retry.")
    st.stop()

port = allocate_static(rets[tick_cols_present], weights, rebalance=P["rebalance"])
bench_rets = rets[benchmark].rename("bench") if (benchmark and benchmark in rets.columns) else None

# ---------- tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Performance", "Risk", "Beta", "Factors", "Volatility (GARCH)"])

# ===== Performance =====
with tab1:
    wealth = (1.0 + port).cumprod().rename("Portfolio")
    perf_df = pd.DataFrame(wealth)
    if bench_rets is not None and not bench_rets.empty:
        bench_wealth = (1.0 + bench_rets).cumprod().rename("Benchmark")
        perf_df = pd.concat([perf_df, bench_wealth], axis=1)

    perf_plot = perf_df.reset_index().rename(columns={"index": "Date"})
    st.plotly_chart(
        px.line(
            perf_plot,
            x="Date",
            y=perf_df.columns,
            title="Cumulative Wealth",
            template=template,
            color_discrete_sequence=colors,
        ),
        use_container_width=True,
    )

    # KPIs
    try:
        kpis = {
            "Ann. Return": f"{annualize_return(port):.2%}",
            "Ann. Vol": f"{annualize_vol(port):.2%}",
            "Sharpe": f"{sharpe(port):.2f}",
            "Sortino": f"{sortino(port):.2f}",
            "Hit Rate": f"{hit_rate(port):.2%}",
            "Max Drawdown": f"{max_drawdown(port):.2%}",
        }
    except Exception:
        kpis = {k: "N/A" for k in ["Ann. Return", "Ann. Vol", "Sharpe", "Sortino", "Hit Rate", "Max Drawdown"]}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for c, (k, v) in zip([c1, c2, c3, c4, c5, c6], kpis.items()):
        with c:
            st.write(f"**{k}**")
            st.write(v)

    dd = drawdown(port).rename("Drawdown").reset_index()
    dd.columns = ["Date", "Drawdown"]
    st.plotly_chart(
        px.area(dd, x="Date", y="Drawdown", title="Drawdown", template=template, color_discrete_sequence=[colors[0]]),
        use_container_width=True,
    )

# ===== Risk =====
with tab2:
    st.subheader("VaR & CVaR")
    v_h = var_historical(port, alpha=P["alpha"])
    cv_h = cvar_historical(port, alpha=P["alpha"])
    v_n = var_parametric_normal(port, alpha=P["alpha"])
    v_cf = var_cornish_fisher(port, alpha=P["alpha"])

    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Hist VaR", f"{-v_h:.2%}")
    rc2.metric("Hist CVaR", f"{-cv_h:.2%}")
    rc3.metric("Param VaR (Normal)", f"{-v_n:.2%}")
    rc4.metric("Cornish-Fisher VaR", f"{-v_cf:.2%}")

    hist_df = pd.DataFrame({"ret": port})
    st.plotly_chart(
        px.histogram(hist_df, x="ret", nbins=60, title="Return Distribution", template=template,
                     color_discrete_sequence=[colors[0]]),
        use_container_width=True,
    )

# ===== Beta =====
with tab3:
    if bench_rets is None or bench_rets.empty:
        st.info("Benchmark data unavailable — cannot compute rolling beta.")
    else:
        aligned = pd.concat([port.rename("port"), bench_rets.rename("bench")], axis=1).dropna()
        if aligned.empty:
            st.info("Insufficient overlap for rolling beta.")
        else:
            beta = rolling_beta(aligned["port"], aligned["bench"], window=63).dropna()
            beta_df = beta.reset_index()
            beta_df.columns = ["Date", "beta"]
            st.plotly_chart(
                px.line(
                    beta_df,
                    x="Date",
                    y="beta",
                    title="Rolling 3-Month Beta vs Benchmark",
                    template=template,
                    color_discrete_sequence=[colors[0]],
                ),
                use_container_width=True,
            )

# ===== Factors (ETF proxies) =====
with tab4:
    st.caption("Factor proxies (ETFs): Oil=USO, USD=UUP, Energy=XLE, Rates=IEF")
    factor_syms = ["USO", "UUP", "XLE", "IEF"]

    # If user picked CSV only, stay offline and skip live factors.
    if P["provider"] == "CSV only":
        st.info("Factors require live data. Switch to Stooq or include factor series in your CSV.")
    else:
        fact_prices = get_prices(
            factor_syms,
            start=P["start"],
            end=P["end"],
            interval=P["interval"],
            retries=P["retries"],
            pause=P["pause"],
            provider="Stooq",                 # always use Stooq for factor proxies here
            map_tsx_to_us=P["map_tsx_to_us"],
        )
        fact_rets = pct_returns(fact_prices)
        factors = fact_rets.rename(columns={"USO": "OIL", "UUP": "USD", "XLE": "XLE", "IEF": "RATES"}).dropna()

        if not factors.empty:
            try:
                res = factor_regression(port, factors)
                st.write(f"**R²:** {res['r2']:.3f}  ·  df_resid={res['df_resid']}")
                out = pd.concat([res["coef"], res["stderr"], res["tstat"]], axis=1)
                out.columns = ["coef", "stderr", "t-stat"]
                st.dataframe(out)
            except Exception as e:
                st.warning(f"Factor regression failed: {e}")
        else:
            st.info("Insufficient overlapping data for factor regression.")

    st.divider()
    st.subheader("Scenario Stress Test")
    c1, c2, c3, c4 = st.columns(4)
    oil = c1.slider("Oil (USO) shock", -0.30, 0.30, -0.10, 0.01)
    rates = c2.slider("Rates (IEF) shock", -0.10, 0.10, 0.02, 0.005)
    usd = c3.slider("USD (UUP) shock", -0.10, 0.10, 0.02, 0.005)
    xle = c4.slider("Energy (XLE) shock", -0.20, 0.20, -0.05, 0.01)

    try:
        coefs = res["coef"]  # exists if regression succeeded
        pred = (
            coefs.get("const", 0.0)
            + coefs.get("OIL", 0.0) * oil
            + coefs.get("RATES", 0.0) * rates
            + coefs.get("USD", 0.0) * usd
            + coefs.get("XLE", 0.0) * xle
        )
        st.metric("Predicted Portfolio Return (Scenario)", f"{pred:.2%}")
    except Exception:
        st.info("Run the regression first to enable scenario projection.")

# ===== Volatility (GARCH) =====
with tab5:
    st.caption("Fit GARCH(1,1) to portfolio daily returns (optional; requires `arch`).")
    if port.empty:
        st.info("Portfolio is empty — cannot fit GARCH.")
    else:
        try:
            res_g = fit_garch(port)
            vols = forecast_vol(res_g, horizon=20)
            vol_df = pd.DataFrame({"Day": np.arange(1, len(vols) + 1), "Daily Vol (Forecast)": vols})
            st.plotly_chart(
                px.line(
                    vol_df,
                    x="Day",
                    y="Daily Vol (Forecast)",
                    title="GARCH Forecasted Daily Volatility (next 20 days)",
                    template=template,
                    color_discrete_sequence=[colors[0]],
                ),
                use_container_width=True,
            )
            try:
                st.write(res_g.summary().tables[1].as_html(), unsafe_allow_html=True)
            except Exception:
                pass
        except Exception as e:
            st.warning(f"GARCH not available: {e}")
