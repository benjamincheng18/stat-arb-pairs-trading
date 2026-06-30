import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def compute_zscore(
    kf_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Compute rolling z-score of the Kalman filter spread.

    Returns:
        DataFrame with columns [spread, zscore]
    """
    spread = kf_df["spread"]

    adf_pvalue = adfuller(spread)[1]
    if adf_pvalue > 0.05:
        print(f"Warning: spread may be non-stationary (ADF p-value = {adf_pvalue:.4f})")

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std

    return pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
    }).dropna()


def generate_signals(
    zscore_df: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """
    Generate long/short/exit trading signals from z-score.

    Signal convention:
         1 = long spread  (spread too low, buy y sell x)
        -1 = short spread (spread too high, sell y buy x)
         0 = no position / exit

    Returns:
        zscore_df with added column [signal]
    """
    zscore = zscore_df["zscore"]
    zscore_df = zscore_df.copy()
    zscore_df["signal"] = np.where(
        zscore < -entry_z, 1,
        np.where(
            zscore > entry_z, -1,
            np.where(
                abs(zscore) < exit_z, 0,
                np.nan  # between thresholds — hold previous position
            )
        )
    )

    zscore_df["signal"] = zscore_df["signal"].ffill().fillna(0)

    return zscore_df