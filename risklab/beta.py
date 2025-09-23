import numpy as np
import pandas as pd


def rolling_beta(asset_rets: pd.Series, bench_rets: pd.Series, window=63):
    # beta = cov(asset, bench)/var(bench)
    cov = asset_rets.rolling(window).cov(bench_rets)
    var = bench_rets.rolling(window).var()
    return (cov / var).rename("beta")
