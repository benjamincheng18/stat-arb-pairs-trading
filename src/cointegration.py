import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests


# ── Single Pair Test ───────────────────────────────────────────────────────────

def test_pair_direction(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """
    Test cointegration in one direction: regress y on x, then ADF on residuals.

    Returns:
        (adf_pvalue, hedge_ratio)
    """
    x_with_const = sm.add_constant(x)
    fitted_model = sm.OLS(y, x_with_const).fit()
    hedge_ratio = fitted_model.params.iloc[1]
    residuals = fitted_model.resid
    adf_pvalue = adfuller(residuals)[1]
    return adf_pvalue, hedge_ratio


# ── Screen All Pairs ───────────────────────────────────────────────────────────

def screen_all_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Run cointegration test in both directions for every pair of tickers.

    Returns:
        DataFrame with columns:
        [stock1, stock2, pvalue_1on2, hr_1on2, pvalue_2on1, hr_2on1]
    """
    results = []
    tickers = prices.columns.tolist()

    for s1, s2 in itertools.combinations(tickers, 2):
        pvalue_1on2, hr_1on2 = test_pair_direction(prices[s1], prices[s2])
        pvalue_2on1, hr_2on1 = test_pair_direction(prices[s2], prices[s1])
        results.append({
            "stock1": s1, "stock2": s2,
            "pvalue_1on2": pvalue_1on2, "hr_1on2": hr_1on2,
            "pvalue_2on1": pvalue_2on1, "hr_2on1": hr_2on1,
        })

    return pd.DataFrame(results)


# ── FDR Correction ─────────────────────────────────────────────────────────────

def apply_fdr_correction(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Apply BH-FDR correction across all 2N p-values jointly.
    Flag pair as cointegrated only if BOTH directions pass.
    """
    n = len(results_df)
    all_pvalues = np.concatenate([results_df["pvalue_1on2"], results_df["pvalue_2on1"]])

    _, pvals_corrected, _, _ = multipletests(all_pvalues, alpha=alpha, method="fdr_bh")

    results_df = results_df.copy()
    results_df["pvalue_1on2_adj"] = pvals_corrected[:n]
    results_df["pvalue_2on1_adj"] = pvals_corrected[n:]
    results_df["cointegrated"] = (
        (results_df["pvalue_1on2_adj"] < alpha) &
        (results_df["pvalue_2on1_adj"] < alpha)
    )

    return results_df