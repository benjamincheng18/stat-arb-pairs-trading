# src/backtest.py

import numpy as np
import pandas as pd
from src.kalman_filter import kalman_filter
from src.signals import compute_zscore, generate_signals
from src.cointegration import screen_all_pairs, apply_fdr_correction


# ── Daily P&L ─────────────────────────────────────────────────────────────────

def daily_pnl(
    signals_df: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """
    Compute daily normalized P&L from spread positions.

    Returns:
        pd.Series of daily returns (normalized, decimal form)
    """
    spread_change = signals_df["spread"].diff()
    lagged_signal = signals_df["signal"].shift(1)
    raw_pnl = lagged_signal * spread_change

    prev_spread = signals_df["spread"].shift(1).abs()
    prev_spread = prev_spread.replace(0, np.nan)
    prev_spread[prev_spread < 1.0] = np.nan
    normalized_pnl = (raw_pnl / prev_spread).clip(-0.5, 0.5)

    signal_changed = signals_df["signal"].diff().abs() > 0
    cost = (cost_bps / 10000) * signal_changed.astype(float)

    return (normalized_pnl - cost).dropna()


# ── Performance Metrics ───────────────────────────────────────────────────────

def eval_metrics(
    pnl: pd.Series,
    signal: pd.Series = None,
    trading_days_per_year: int = 250,
) -> dict:
    """
    Compute performance metrics from daily P&L series.

    Returns:
        dict with keys: cumulative_return, sharpe_ratio, max_drawdown,
                        hit_rate, avg_trade_duration
    """
    cum_return = (1 + pnl).prod() - 1

    sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(trading_days_per_year)

    cum_wealth = (1 + pnl).cumprod()
    rolling_max = cum_wealth.cummax()
    drawdown = (cum_wealth - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    active_pnl = pnl[pnl != 0]
    hit_rate = (active_pnl > 0).mean()

    if signal is not None:
        active = signal[signal != 0]
        groups = (active != active.shift()).cumsum()
        avg_trade_duration = active.groupby(groups).count().mean()
    else:
        avg_trade_duration = np.nan

    return {
        "cumulative_return": cum_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_trade_duration": avg_trade_duration,
    }


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_backtest_pipeline(
    prices: pd.DataFrame,
    coint_results: pd.DataFrame,
    delta: float = 1e-4,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Run full pipeline for all cointegrated pairs.

    Returns:
        DataFrame with one row per pair and columns:
        [stock1, stock2, cumulative_return, sharpe_ratio,
         max_drawdown, hit_rate, avg_trade_duration]
    """
    pairs = coint_results[coint_results["cointegrated"]].reset_index(drop=True)
    summary = []

    for _, row in pairs.iterrows():
        s1, s2 = row["stock1"], row["stock2"]
        try:
            beta_0 = row["hr_1on2"]
            kf_df = kalman_filter(prices[s1], prices[s2], delta=delta, beta_0=beta_0)
            zscore_df = compute_zscore(kf_df, window=window)
            signals_df = generate_signals(zscore_df, entry_z=entry_z, exit_z=exit_z)
            pnl = daily_pnl(signals_df, cost_bps=cost_bps)
            metrics = eval_metrics(pnl, signal=signals_df["signal"])
            summary.append({"stock1": s1, "stock2": s2, **metrics})
        except Exception as e:
            print(f"Skipping {s1}/{s2}: {e}")

    return pd.DataFrame(summary)


def walk_forward_backtest(
    prices: pd.DataFrame,
    train_days: int = 630,
    test_days: int = 60,
    step_days: int = 60,
    delta: float = 1e-4,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    cost_bps: float = 10.0,
    alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest: rescreen pairs and refit Kalman at each fold.

    Returns:
        pnl_series : continuous daily P&L Series across all test folds
        summary_df : metrics per fold
    """
    n = len(prices)
    fold = 0
    start = 0
    all_pnl = []
    fold_summaries = []

    while start + train_days + test_days <= n:
        train_prices = prices.iloc[start:start + train_days]
        test_prices  = prices.iloc[start + train_days:start + train_days + test_days]

        # Rescreen pairs on training window only
        coint_train = screen_all_pairs(train_prices)
        fdr_train = apply_fdr_correction(coint_train, alpha=alpha)
        coint_pairs = fdr_train[fdr_train["cointegrated"]].reset_index(drop=True)

        fold_pnls = []

        for _, row in coint_pairs.iterrows():
            s1, s2 = row["stock1"], row["stock2"]
            try:
                beta_0 = row["hr_1on2"]

                # Run Kalman on train+test
                full_window = prices.iloc[start:start + train_days + test_days]
                kf_full = kalman_filter(
                    full_window[s1], full_window[s2],
                    delta=delta, beta_0=beta_0
                )

                # Compute z-score on full window (rolling mean/std uses train period as warmup)
                zscore_full = compute_zscore(kf_full, window=window)

                # NOW slice to test period only
                zscore_test = zscore_full[zscore_full.index >= test_prices.index[0]]

                signals_df = generate_signals(zscore_test, entry_z=entry_z, exit_z=exit_z)
                pnl = daily_pnl(signals_df, cost_bps=cost_bps)
                fold_pnls.append(pnl)

            except Exception as e:
                print(f"Skipping {s1}/{s2}: {e}")

        # Skip fold if no pairs survived
        if not fold_pnls:
            print(f"Fold {fold}: no cointegrated pairs found, skipping.")
            fold  += 1
            start += step_days
            continue

        # Equal-weighted portfolio P&L for this fold
        aggregate_pnl = pd.concat(fold_pnls, axis=1).mean(axis=1)

        metrics = eval_metrics(aggregate_pnl)
        fold_summaries.append({
            "fold"      : fold,
            "start_date": test_prices.index[0],
            "end_date"  : test_prices.index[-1],
            "n_pairs"   : len(fold_pnls),
            **metrics,
        })
        all_pnl.append(aggregate_pnl)

        fold  += 1
        start += step_days

    pnl_series = pd.concat(all_pnl).sort_index()
    summary_df = pd.DataFrame(fold_summaries)

    return pnl_series, summary_df