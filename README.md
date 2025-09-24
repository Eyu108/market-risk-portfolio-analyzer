# 📊 Market Risk & Portfolio Analyzer — Streamlit App

> A fast, web app for **portfolio & market risk analytics** with professional visuals and pragmatic risk tooling. Built to showcase practical **quant + data engineering** skills for **energy, trading, and analytics** roles. **Jet-Black** theme included.

[![Live Demo – Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://market-risk-portfolio-analyzer.streamlit.app/)
*If the live demo isn’t deployed yet, clone and run locally.*


---

## Why this project (for reviewers)

- **Real-world focus**: portfolio construction, **VaR/CVaR**, rolling **beta**, **factor regression**, and **volatility** modeling.
- **Robust data plumbing**: default to **Stooq** (no keys), **CSV offline mode**, retry/caching, and warnings that help users fix inputs.
- **Readable, extendable code**: small `risklab/*` modules (data, metrics, var, factors, garch, portfolio) designed for reuse.
- **Professional UX**: guided sidebar inputs, informative toasts, and a cohesive **Jet‑Black** visual style.

> Complements my previous **Trading Strategy Backtester** by shifting from *signal testing* to *portfolio-level risk* (VaR, factor exposures, vol).

---

## 60‑second demo

1. **Data source**: keep **Stooq** (default) or use **CSV** in the sidebar.
2. Enter tickers + weights in the table editor (toggle **Normalize** to rescale).
3. Choose a **Benchmark** (SPY / Custom) and date range, then **Apply / Refresh**.
4. Explore tabs:
   - **Performance**: wealth curve, KPIs, drawdowns
   - **Risk**: Historical & Parametric VaR/CVaR
   - **Beta**: rolling beta vs benchmark
   - **Factors**: ETF proxies (USO/UUP/XLE/IEF) + scenario shocks
   - **Volatility**: **GARCH(1,1)** if available; otherwise **EWMA** fallback (with λ & horizon controls)

---

## Features

- **Portfolio builder**
  - Table editor for tickers & weights (supports **Normalize**)
  - Monthly / Quarterly / None **rebalancing**
  - **Benchmark**: SPY or custom
- **Performance**
  - Cumulative wealth vs benchmark
  - KPIs: CAGR, Annualized Vol, Sharpe, Sortino, Hit rate, Max Drawdown
  - Drawdown curve
- **Risk**
  - **VaR/CVaR (Historical)**
  - **Parametric VaR (Normal)** and **Cornish–Fisher**
- **Beta**
  - Rolling 3‑month beta vs benchmark
- **Factors**
  - Quick regression using ETF proxies (USO=Oil, UUP=USD, XLE=Energy, IEF=Rates)
  - Scenario sliders to stress portfolio return
- **Volatility**
  - **GARCH(1,1)** (if `arch` installed) with parameter summary
  - **EWMA** fallback (RiskMetrics‑style) with λ slider; horizon & annualization controls
- **Data**
  - **Stooq** (default; no API key)
  - **CSV offline mode** (wide format with `Date` column)
  - Optional mapping: **TSX .TO → US** duals
  - Sample data toggle when live fetch is unavailable
- **UI / Visuals**
  - Cohesive **Jet‑Black** theme via `.streamlit/config.toml`
  - Plotly interactive charts, tooltips, and responsive layout

---

## Screenshots
_Add screenshots to `docs/` and uncomment once you have them._
<!--
![Performance](docs/screenshot_performance.png)
![Risk](docs/screenshot_risk.png)
![Beta](docs/screenshot_beta.png)
![Factors](docs/screenshot_factors.png)
![Volatility](docs/screenshot_volatility.png)
-->

---

## Quickstart

### Windows (recommended path)
```bat
cd "C:\Users\User\Documents\market-risk-portfolio-analyzer"
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

> If your OS alias hijacks `python`, call the venv explicitly:
> `.\.venv\Scripts\python.exe -m streamlit run app.py`

### macOS / Linux
```bash
git clone https://github.com/Eyu108/market-risk-portfolio-analyzer.git
cd market-risk-portfolio-analyzer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**Optional**
```bash
# Nicer in‑app table editor
python -m pip install pyarrow==17.0.0

# Full GARCH (else EWMA fallback is used)
python -m pip install --only-binary=:all: scipy==1.14.1
python -m pip install --only-binary=:all: arch==7.2.0
```

---

## Project structure
```
market-risk-portfolio-analyzer/
├─ app.py                      # Streamlit UI
├─ risklab/                    # Analytics modules
│  ├─ beta.py                  # rolling_beta
│  ├─ data.py                  # get_prices (Stooq, CSV), pct_returns
│  ├─ factors.py               # factor_regression
│  ├─ garch.py                 # fit_garch, forecast_vol (optional arch)
│  ├─ metrics.py               # annualize, sharpe/sortino, drawdown, etc.
│  ├─ portfolio.py             # combine returns, rebalance
│  └─ var.py                   # VaR/CVaR implementations
├─ requirements.txt
└─ .streamlit/
   └─ config.toml              # Jet‑Black theme
```

---

## Notes & Troubleshooting
- **Flat vol forecast?** That’s the EWMA fallback (by design). Install `arch` to see GARCH mean‑reversion.
- **Windows alias error** (“Python was not found”): use the venv path `.\.venv\Scripts\python.exe`.
- **No data returned**: try shorter ranges, fewer tickers, switch to CSV, or toggle **Sample data**.
- **Table editor missing**: `pip install pyarrow==17.0.0`.

---

## License & Disclaimer
For educational/use-at-your-own-risk purposes. **Not financial advice.**
