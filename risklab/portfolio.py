import numpy as np
import pandas as pd


def normalize_weights(w):
    s = pd.Series(w, dtype=float)
    total = s.sum()
    return s / total if total != 0 else s

def _to_resample_rule(rebalance):
    """Map friendly codes to pandas-end frequency codes."""
    if rebalance is None:
        return None
    if rebalance == "M":
        return "ME"  # month-end
    if rebalance == "Q":
        return "QE"  # quarter-end
    return rebalance  # allow custom codes

def allocate_static(returns: pd.DataFrame, weights: dict, rebalance="M"):
    """
    Build a simple static-weight portfolio from asset returns.
    - returns: wide DataFrame of daily returns (pct or log) with DateTimeIndex
    - weights: dict like {'IMO.TO':0.4,'SU.TO':0.3,'XLE':0.3}
    - rebalance: "M", "Q", None (buy&hold/drift), or a pandas offset alias
    Returns a Series of daily portfolio returns.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype="float64", name="portfolio")

    w = normalize_weights(weights)
    cols = [c for c in w.index if c in returns.columns]
    if not cols:
        return pd.Series(dtype="float64", name="portfolio")

    R = returns.loc[:, cols].copy()
    R.index = pd.to_datetime(R.index)

    rule = _to_resample_rule(rebalance)

    if rule is None:
        # Buy & hold (drift)
        wealth = (1.0 + R).cumprod()
        port_wealth = (wealth * w.loc[cols]).sum(axis=1)
        port_ret = port_wealth.pct_change().dropna()
        port_ret.index = pd.to_datetime(port_ret.index)
        return port_ret.rename("portfolio")

    # Periodic rebalance: compound within period, then reapply weights
    grouped = (1.0 + R).resample(rule).apply(lambda df: df.prod() - 1.0)
    port_periodic = (grouped * w.loc[cols]).sum(axis=1)

    # Stepwise hold between rebalances -> forward-fill to daily index
    port = port_periodic.reindex(R.index, method="ffill")
    port.index = pd.to_datetime(port.index)
    return port.rename("portfolio")
