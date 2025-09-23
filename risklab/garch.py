# risklab/garch.py
import pandas as pd

try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False

def fit_garch(rets: pd.Series, p=1, q=1, dist="normal"):
    if not _HAS_ARCH:
        raise ImportError("arch package not installed")
    am = arch_model(rets.dropna()*100, vol='GARCH', p=p, q=q, dist=dist, mean='Constant')
    res = am.fit(disp="off")
    return res

def forecast_vol(res, horizon=10):
    f = res.forecast(horizon=horizon, reindex=False)
    daily_vol = (f.variance.values[-1])**0.5 / 100.0
    return daily_vol
