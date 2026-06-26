import numpy as np
import pandas as pd


def kalman_filter(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    R: float = None,
    beta_0: float = 0.0,
    P_0: float = 1.0,
) -> pd.DataFrame:
    """
    Estimate time-varying hedge ratio via Kalman filter.

    Args:
        y      : dependent price series
        x      : independent price series
        delta  : process noise parameter (controls beta drift speed)
                 Q = delta / (1 - delta)
        R      : observation noise variance (default: var(y))

    Returns:
        DataFrame with columns [beta, P, spread] indexed by date
    """
    n = len(y)

    if R is None:
        R = np.var(y)

    Q = delta / (1 - delta)

    beta_prev = beta_0
    P_prev = P_0

    betas = np.zeros(n)
    Ps = np.zeros(n)
    spreads = np.zeros(n)

    for t in range(n):
        x_t = x.iloc[t]
        y_t = y.iloc[t]

        # Predict
        beta_pred = beta_prev
        P_pred = P_prev + Q

        # Update
        innovation = y_t - beta_pred * x_t
        S = x_t ** 2 * P_pred + R
        K = P_pred * x_t / S
        beta = beta_pred + K * innovation
        P = (1 - K * x_t) * P_pred

        # Store
        betas[t] = beta
        Ps[t] = P
        spreads[t] = y_t - beta * x_t

        # Advance
        beta_prev = beta
        P_prev = P

    return pd.DataFrame({
        "beta": betas,
        "P": Ps,
        "spread": spreads,
    }, index=y.index)