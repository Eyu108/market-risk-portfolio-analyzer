Streamlit app for portfolio analytics (performance, VaR/CVaR, factors, beta, GARCH).

## Quick start
1. python -m venv .venv
2. .\.venv\Scripts\activate.bat
3. python -m pip install -r requirements.txt
4. python -m streamlit run app.py

## Data sources
- Stooq (default): no API key required.
- CSV upload option.

## Project structure
- app.py | Streamlit UI
- risklab/ | analytics modules
- .streamlit/ | local secrets (not committed)

