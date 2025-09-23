# risklab/var.py
import numpy as np
import pandas as pd


# Inverse normal CDF (ppf) approximation so we don't need SciPy
def _norm_ppf(p: float) -> float:
    if p <= 0.0: return -np.inf
    if p >= 1.0: return  np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2*np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = np.sqrt(-2*np.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def var_historical(rets: pd.Series, alpha=0.95):
    r = rets.dropna()
    return np.quantile(r, 1 - alpha)  # negative = loss

def cvar_historical(rets: pd.Series, alpha=0.95):
    q = var_historical(rets, alpha)
    tail = rets.dropna()[rets.dropna() <= q]
    return tail.mean() if len(tail) else np.nan

def var_parametric_normal(rets: pd.Series, alpha=0.95):
    r = rets.dropna()
    mu, sigma = r.mean(), r.std(ddof=0)
    z = _norm_ppf(1 - alpha)
    return mu + sigma * z

def var_cornish_fisher(rets: pd.Series, alpha=0.95):
    r = rets.dropna()
    mu, sigma = r.mean(), r.std(ddof=0)
    z = _norm_ppf(1 - alpha)
    s = r.skew()   # pandas skewness
    k = r.kurt()   # excess kurtosis
    z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*(s**2)/36
    return mu + sigma * z_cf
