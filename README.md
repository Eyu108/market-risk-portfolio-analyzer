# 📊 Market Risk & Portfolio Analyzer

Jet-black, production-ready **Streamlit** app for portfolio analytics:
- Performance & drawdowns
- VaR / CVaR (historical, parametric, Cornish–Fisher)
- Rolling beta vs benchmark
- Fast factor regression (ETF proxies) + scenario shocks
- Volatility modeling with **GARCH(1,1)** *(optional)* and an automatic **EWMA** fallback

> **No brokerage connection. No API keys required.**  
> Daily prices from **Stooq** (default) or work fully offline with a **CSV**.

---

## 🎯 Why this project (for recruiters & reviewers)

This app complements my previous **Trading Strategy Backtester**. Together they demonstrate:

- **Quant engineering**: factor modeling, risk decomposition, portfolio construction, VaR/CVaR, GARCH/EWMA volatility.
- **Production-oriented data plumbing**: provider fallbacks, caching, retry/backoff, CSV offline mode.
- **UX for decision support**: opinionated defaults, guided inputs, informative warnings, and a cohesive **Jet-Black** theme.
- **Good software practice**: clear module boundaries (`risklab/*`), reproducible env, CI-friendly start commands, safe `.gitignore`.

If you want a 60-second tour, open the app and:
1) Keep **Data source = Stooq**.  
2) Use the default `SPY, XLE, CVX` weights (Normalize on).  
3) Click **Apply / Refresh** → check **Performance** KPIs + **Drawdown**.  
4) Go to **Risk** → compare Historical vs Parametric VaR/CVaR at the slidered α.  
5) **Beta** tab → review rolling 3-month beta vs benchmark.  
6) **Factors** → see quick regression (USO/UUP/XLE/IEF) + try scenario sliders.  
7) **Volatility** → if `arch` missing you’ll see **EWMA** (flat forecast by design); install `arch` to see mean-reverting **GARCH** paths.

---

## 🧱 Tech Stack

- **Python**, **Streamlit**, **Plotly**
- **pandas / numpy / statsmodels**
- Optional: **arch** (GARCH), **pyarrow** (nicer table editor)

---

## 📦 Project Structure

```
market-risk-portfolio-analyzer/
├─ app.py                      # Streamlit UI (Jet-Black theme via config)
├─ risklab/
│  ├─ beta.py                  # rolling_beta
│  ├─ data.py                  # get_prices (Stooq, CSV), pct_returns
│  ├─ factors.py               # factor_regression
│  ├─ garch.py                 # fit_garch, forecast_vol (optional arch)
│  ├─ metrics.py               # annualize, sharpe/sortino, drawdown, etc.
│  ├─ portfolio.py             # combine returns, rebalance
│  └─ var.py                   # VaR/CVaR implementations
├─ requirements.txt
└─ .streamlit/
   └─ config.toml              # Jet-Black theme
```

---

## 🖤 Jet-Black Theme

`.streamlit/config.toml` (already included):

```toml
[theme]
base = "dark"
primaryColor = "#22c55e"
backgroundColor = "#000000"
secondaryBackgroundColor = "#0a0a0a"
textColor = "#e5e7eb"
```

---

## 📈 Data Sources

- **Stooq** (default): no API key. Works for many US tickers & common ETFs.  
  *Optional mapping*: toggle **“Map TSX (.TO) → US”** for simple duals.
- **CSV**: upload a wide CSV with a `Date` column + one column per ticker (prices).

**CSV example**
```csv
Date,SPY,XLE,CVX
2020-01-02,323.5,60.12,120.4
2020-01-03,321.9,59.70,118.9
```

---

## 🚀 Run Locally

> Use your venv’s Python explicitly on Windows to avoid the Microsoft Store alias.

### Windows
```bat
cd "C:\Users\User\Documents\market-risk-portfolio-analyzer"
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

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

**Optional goodies**
```bash
# Nicer in-app table editor
python -m pip install pyarrow==17.0.0

# Full GARCH (otherwise EWMA fallback is used)
python -m pip install --only-binary=:all: scipy==1.14.1
python -m pip install --only-binary=:all: arch==7.2.0
```

---

## ☁️ Deploy: Streamlit Community Cloud (free)

1. Push this repo to GitHub (already done ✔).
2. Go to **share.streamlit.io** → **New app**.
3. Select your repo **Eyu108/market-risk-portfolio-analyzer**, branch `main`, and file `app.py`.
4. Click **Deploy**.
5. *(Optional)* in **Advanced settings** set:
   - Python version: `3.12`
   - Secrets: **none required**
6. Wait for the build → your app URL goes live.

---

## 🐳 Deploy: Docker (local or any container host)

Create `Dockerfile` (optional if you want a containerized deploy):

```dockerfile
FROM python:3.12-slim

# System deps (optional, helps scipy wheels etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build & run:
```bash
docker build -t mrpa .
docker run -p 8501:8501 mrpa
```

---

## 🌐 Deploy: Render (free tier web service)

1) Create a new **Web Service** from your GitHub repo.  
2) **Build command**:
```bash
pip install --upgrade pip && pip install -r requirements.txt
```
3) **Start command**:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
4) Choose Python 3.12, deploy.

---

## 🧪 Reviewer’s Guide (quick demo flow)

- **Portfolio**: leave defaults `SPY, XLE, CVX` and click **Apply / Refresh**.  
- **Performance**: note KPIs and the drawdown chart.  
- **Risk**: move α slider; compare Historical vs Parametric VaR/CVaR.  
- **Beta**: observe rolling 3-month beta vs benchmark (SPY).  
- **Factors**: see regression (USO, UUP, XLE, IEF) and try the scenario sliders.  
- **Volatility**:  
  - Without `arch`: EWMA (flat forecast).  
  - With `arch`: GARCH (mean-reverting forecast) + parameter table.

---

## 🔒 What’s not committed

`.gitignore` excludes:
- `.venv/`
- `.streamlit/secrets.toml`
- `__pycache__/`, `*.pyc`, IDE folders

---

## ⚠️ Disclaimer

Educational purposes only — **not financial advice**.

---
