import numpy as np
import pandas as pd


def annualize_return(rets, periods_per_year=252):
    return (1 + rets).prod()**(periods_per_year/len(rets)) - 1

def annualize_vol(rets, periods_per_year=252):
    return rets.std(ddof=0) * np.sqrt(periods_per_year)

def sharpe(rets, rf=0.0, periods_per_year=252):
    excess = rets - rf/periods_per_year
    vol = excess.std(ddof=0) * np.sqrt(periods_per_year)
    if vol == 0: return np.nan
    return excess.mean() * periods_per_year / vol

def sortino(rets, rf=0.0, periods_per_year=252):
    excess = rets - rf/periods_per_year
    downside = excess[excess < 0]
    dd = downside.std(ddof=0) * np.sqrt(periods_per_year)
    if dd == 0: return np.nan
    return excess.mean() * periods_per_year / dd

def drawdown(rets):
    wealth = (1 + rets).cumprod()
    peak = wealth.cummax()
    dd = wealth/peak - 1.0
    return dd

def max_drawdown(rets):
    return drawdown(rets).min()

def hit_rate(rets):
    return (rets > 0).mean()
