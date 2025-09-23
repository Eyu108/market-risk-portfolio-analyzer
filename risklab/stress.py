import numpy as np
import pandas as pd


def apply_factor_shocks(model, shocks: dict):
    """
    model: statsmodels OLS result with params ['const','OIL','RATES','USD','XLE']
    shocks: e.g. {'OIL': -0.1, 'RATES': +0.005, 'USD': +0.02, 'XLE': -0.05}
    returns predicted portfolio return impact for the scenario.
    """
    params = model.params
    x = {'const':1.0}
    x.update({k: shocks.get(k, 0.0) for k in params.index if k!='const'})
    # dot product
    return sum(params.get(k,0.0)*v for k,v in x.items())
