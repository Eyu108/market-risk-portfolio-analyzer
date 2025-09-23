# risklab/factors.py
import numpy as np
import pandas as pd

__all__ = ["factor_regression"]

def factor_regression(port_rets: pd.Series, factors: pd.DataFrame):
    """
    OLS: y = Xb + e with intercept (no SciPy/statsmodels).
    Returns dict with coef, stderr, tstat, r2, df_resid.
    """
    df = pd.concat([port_rets.rename("y"), factors], axis=1).dropna()
    if df.empty:
        raise ValueError("No overlapping data between portfolio and factors.")

    y = df["y"].to_numpy().reshape(-1, 1)
    Xf = df[factors.columns].to_numpy()
    X = np.column_stack([np.ones((Xf.shape[0], 1)), Xf])  # intercept
    colnames = ["const"] + list(factors.columns)

    # Solve by least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)         # (k+1, 1)
    y_hat = X @ beta
    resid = y - y_hat

    n, k = X.shape
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError("Not enough observations for regression.")

    rss = float((resid**2).sum())
    tss = float(((y - y.mean())**2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    sigma2 = rss / df_resid
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma2 * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta)).reshape(-1, 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        tstat = beta / se_beta
        tstat = np.where(np.isfinite(tstat), tstat, np.nan)

    return {
        "coef":  pd.Series(beta.flatten(), index=colnames, name="coef"),
        "stderr": pd.Series(se_beta.flatten(), index=colnames, name="stderr"),
        "tstat": pd.Series(tstat.flatten(), index=colnames, name="tstat"),
        "r2": r2,
        "df_resid": df_resid,
    }
