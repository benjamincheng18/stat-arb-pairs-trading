import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Ticker Universe ────────────────────────────────────────────────────────────

def get_ticker_universe() -> dict[str, list[str]]:
    """Return AI infrastructure tickers grouped by theme."""
    return {
        "semiconductor": [
            "2330.TW", "2454.TW", "3711.TW", "2303.TW", "7769.TW",
            "2408.TW", "2344.TW", "5274.TWO", "3443.TW", "6223.TWO",
            "8299.TWO", "6488.TWO", "6515.TW", "3661.TW", "3189.TW",
            "6770.TW", "2449.TW", "2379.TW", "5347.TWO", "2337.TW",
            "3034.TW", "6239.TW", "3529.TWO", "3105.TWO", "6415.TW",
        ],
        "electronic_components": [
            "2308.TW", "2327.TW", "2383.TW", "3037.TW", "2368.TW",
            "2059.TW", "4958.TW", "3653.TW", "8046.TW", "6274.TWO",
            "2313.TW", "3044.TW", "3533.TW", "2492.TW",
        ],
        "computer_hardware": [
            "2382.TW", "3017.TW", "6669.TW", "2357.TW", "3231.TW",
            "2301.TW", "2395.TW", "2356.TW", "2376.TW", "4938.TW",
        ],
        "other_electronics": [
            "2317.TW", "2360.TW", "3665.TW", "2404.TW", "6139.TW",
        ],
        "optical": [
            "3008.TW", "3481.TW", "8069.TWO",
        ],
        "networking": [
            "2345.TW", "3081.TWO",
        ],
        "distribution": [
            "3036.TW",
        ],
        "us_anchors": [
            "NVDA", "AMD", "TSM", "MU", "DELL", "MSFT", "GOOGL", "AVGO",
        ],
    }


def get_all_tickers(universe: dict[str, list[str]]) -> list[str]:
    """Flatten universe dict into a deduplicated list of tickers."""
    flat = []
    for ticker_list in universe.values():
        for ticker in ticker_list:
            flat.append(ticker)

    deduped = list(dict.fromkeys(flat))
    return deduped


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: list[str],
    start: str = "2023-01-01",
    end: str = "2026-05-31",
) -> pd.DataFrame:
    """
    Fetch daily adjusted closing prices for all tickers via yfinance.

    Returns a DataFrame with dates as index, tickers as columns.
    """
    price = yf.download(
        tickers=tickers, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]
    return price


# ── Data Cleaning ──────────────────────────────────────────────────────────────

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw price DataFrame.

    Steps:
        1. Forward-fill sporadic NaNs (e.g. public holidays in one market).
        2. Drop any ticker column where NaNs remain (e.g. late IPO, bad data).
    """
    df = df.ffill()
    df = df.dropna(axis=1)
    print(f"Tickers after cleaning: {df.shape[1]}")
    return df


# ── Master Loader ──────────────────────────────────────────────────────────────

def load_prices(
    start: str = "2023-01-01",
    end: str = "2026-05-31",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Master function: fetch and clean prices for the full AI universe.

    Returns:
        prices : cleaned DataFrame (dates x tickers)
        universe : theme dict (for reference in later modules)
    """
    universe = get_ticker_universe()
    tickers = get_all_tickers(universe)
    raw = fetch_prices(tickers, start, end)
    prices = clean_prices(raw)
    return prices, universe