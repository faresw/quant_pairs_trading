import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def find_cointegration_pair(series1, series2, significance=0.05):
    """
    Performs an OLS regression of series1 ~ series2,
    then tests if residuals are stationary (ADF test).
    Returns (is_cointegrated, beta, pvalue).
    """
    # Align indices
    df = pd.concat([series1, series2], axis=1).dropna()
    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]

    # OLS: s1 = β * s2 + ε
    s2_const = sm.add_constant(s2)
    model = sm.OLS(s1, s2_const).fit()
    beta = model.params[1]  # slope
    intercept = model.params[0]

    resid = s1 - (intercept + beta * s2)

    # ADF test on residuals: null hypothesis = unit root (not stationary)
    adf_result = adfuller(resid)
    pvalue = adf_result[1]

    is_coint = pvalue < significance
    return is_coint, beta, intercept, pvalue

if __name__ == "__main__":
    # Quick sanity check
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200)
    x = pd.Series(np.cumsum(np.random.randn(200)), index=dates)
    y = x * 2 + np.random.randn(200) * 0.5
    coint, β, α, p = find_cointegration_pair(x, y)
    print(f"Cointegrated: {coint}, β={β:.3f}, intercept={α:.3f}, p={p:.4f}")
