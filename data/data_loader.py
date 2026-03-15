"""
Data loading and caching module for the UK Systematic Trading Research Framework.

Downloads OHLCV data for LSE-listed securities from Yahoo Finance, caches
results locally as parquet files, and performs incremental updates to avoid
re-downloading historical data.

All tickers must use the Yahoo Finance .L suffix (e.g., "AZN.L").
All prices are in GBP as returned by Yahoo Finance for LSE securities.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import DataConfig

logger = logging.getLogger(__name__)

# Columns returned by yfinance OHLCV download
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


class DataLoader:
    """
    Downloads and caches price data for LSE-listed securities.

    Caching strategy: Each ticker is stored as an individual parquet file at
    ``{cache_dir}/{ticker}.parquet``. On load, the loader checks whether the
    cache covers the requested date range. If data is stale, only the missing
    tail is downloaded and appended (incremental update). This avoids
    re-downloading the full history on daily runs.

    Parameters
    ----------
    cache_dir : str
        Directory for parquet cache files.
    config : DataConfig
        Data pipeline configuration.
    """

    def __init__(self, cache_dir: str, config: DataConfig) -> None:
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialised. Cache: %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_price_data(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        price_field: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Load adjusted close prices for a list of LSE tickers.

        Returns a wide DataFrame with DatetimeIndex and ticker symbols as
        columns. Only LSE trading days are included in the index.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols with .L suffix.
        start_date : str
            Start date in ISO 8601 format (e.g., '2010-01-01').
        end_date : str
            End date in ISO 8601 format (e.g., '2023-12-31').
        price_field : str
            Column to extract: 'Adj Close', 'Close', 'Open', etc.

        Returns
        -------
        pd.DataFrame
            Wide DataFrame (index=DatetimeIndex, columns=tickers), GBP prices.
        """
        logger.info("Loading price data for %d tickers from %s to %s",
                    len(tickers), start_date, end_date)

        series_list: list[pd.Series] = []
        failed: list[str] = []

        for ticker in tqdm(tickers, desc="Loading prices", unit="ticker"):
            s = self._load_single_ticker(ticker, start_date, end_date, price_field)
            if s is not None and not s.empty:
                series_list.append(s.rename(ticker))
            else:
                failed.append(ticker)

        if failed:
            logger.warning("Failed to load %d tickers: %s", len(failed), failed[:10])

        if not series_list:
            raise RuntimeError("No price data loaded for any ticker.")

        prices = pd.concat(series_list, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # Align to LSE trading calendar
        lse_days = self._get_lse_calendar(start_date, end_date)
        prices = prices.reindex(lse_days)

        logger.info("Price data loaded: %d dates × %d tickers", *prices.shape)
        return prices

    def load_ohlcv(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Load full OHLCV DataFrames for a list of tickers.

        Required by EarningsRevisionDrift which needs overnight gaps
        (Close[t-1] vs Open[t]).

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols with .L suffix.
        start_date : str
            Start date (ISO 8601).
        end_date : str
            End date (ISO 8601).

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of ticker → DataFrame with columns
            [Open, High, Low, Close, Adj Close, Volume].
        """
        logger.info("Loading OHLCV for %d tickers", len(tickers))
        result: dict[str, pd.DataFrame] = {}

        for ticker in tqdm(tickers, desc="Loading OHLCV", unit="ticker"):
            df = self._load_ohlcv_single(ticker, start_date, end_date)
            if df is not None and not df.empty:
                result[ticker] = df

        logger.info("OHLCV loaded for %d/%d tickers", len(result), len(tickers))
        return result

    # ------------------------------------------------------------------
    # Internal: single ticker loading with cache
    # ------------------------------------------------------------------

    def _load_single_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        price_field: str = "Adj Close",
    ) -> Optional[pd.Series]:
        """Load a single price series from cache or download."""
        cache_path = self._cache_path(ticker)

        if self._is_cache_valid(ticker, end_date):
            df = self._load_from_cache(ticker)
            if price_field in df.columns:
                series = df[price_field]
                series = series[start_date:end_date]
                return series
            logger.warning("Field '%s' not in cache for %s", price_field, ticker)

        # Download full history if no cache, else incremental update
        if cache_path.exists():
            existing = self._load_from_cache(ticker)
            last_cached = existing.index.max()
            new_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.debug("Incremental update for %s from %s", ticker, new_start)
            new_df = self._download_ohlcv(ticker, new_start, end_date)
            if new_df is not None and not new_df.empty:
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = existing
        else:
            logger.debug("Full download for %s from %s", ticker, start_date)
            combined = self._download_ohlcv(ticker, start_date, end_date)

        if combined is None or combined.empty:
            return None

        self._save_to_cache(ticker, combined)

        if price_field in combined.columns:
            return combined[price_field][start_date:end_date]
        return None

    def _load_ohlcv_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Load OHLCV DataFrame for a single ticker, using cache."""
        cache_path = self._cache_path(ticker)

        if self._is_cache_valid(ticker, end_date):
            df = self._load_from_cache(ticker)
            return df[start_date:end_date]

        if cache_path.exists():
            existing = self._load_from_cache(ticker)
            last_cached = existing.index.max()
            new_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            new_df = self._download_ohlcv(ticker, new_start, end_date)
            if new_df is not None and not new_df.empty:
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                combined = existing
        else:
            combined = self._download_ohlcv(ticker, start_date, end_date)

        if combined is None or combined.empty:
            return None

        self._save_to_cache(ticker, combined)
        return combined[start_date:end_date]

    # ------------------------------------------------------------------
    # Internal: download with retry
    # ------------------------------------------------------------------

    def _download_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data from Yahoo Finance with exponential backoff retry.

        Parameters
        ----------
        ticker : str
            Ticker with .L suffix.
        start_date, end_date : str
            Date range.
        max_retries : int
            Number of retry attempts.

        Returns
        -------
        pd.DataFrame or None
            OHLCV DataFrame, or None on persistent failure.
        """
        wait = 2.0
        for attempt in range(max_retries):
            try:
                raw = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                )
                if raw is None or raw.empty:
                    logger.warning("Empty data for %s (attempt %d)", ticker, attempt + 1)
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        wait *= 2
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Keep only standard OHLCV columns that exist
                cols_to_keep = [c for c in OHLCV_COLUMNS if c in raw.columns]
                df = raw[cols_to_keep].copy()
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df

            except Exception as exc:  # noqa: BLE001
                logger.warning("Download error for %s (attempt %d): %s",
                               ticker, attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    wait *= 2

        logger.error("Failed to download %s after %d attempts", ticker, max_retries)
        return None

    # ------------------------------------------------------------------
    # Internal: cache management
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        """Return cache file path for a ticker."""
        safe_name = ticker.replace(".", "_").replace("/", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def _is_cache_valid(self, ticker: str, end_date: str) -> bool:
        """
        Return True if cache exists and covers up to end_date.

        Parameters
        ----------
        ticker : str
            Ticker symbol.
        end_date : str
            Required end date.
        """
        path = self._cache_path(ticker)
        if not path.exists():
            return False
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return False
            last_date = df.index.max()
            required = pd.Timestamp(end_date)
            # Allow 5 business days tolerance (weekends, recent holidays)
            return last_date >= required - pd.offsets.BDay(5)
        except Exception:  # noqa: BLE001
            return False

    def _load_from_cache(self, ticker: str) -> pd.DataFrame:
        """Read cached parquet file for a ticker."""
        path = self._cache_path(ticker)
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _save_to_cache(self, ticker: str, data: pd.DataFrame) -> None:
        """Write DataFrame to parquet cache."""
        path = self._cache_path(ticker)
        data.to_parquet(path, compression=self.config.parquet_compression)
        logger.debug("Cached %s (%d rows) → %s", ticker, len(data), path)

    # ------------------------------------------------------------------
    # Internal: LSE trading calendar
    # ------------------------------------------------------------------

    def _get_lse_calendar(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DatetimeIndex:
        """
        Return valid LSE trading days between start and end dates.

        Uses pandas_market_calendars with the 'LSE' calendar, which correctly
        handles UK bank holidays (Good Friday, Easter Monday, May Bank Holiday,
        Late Summer Bank Holiday, Christmas, Boxing Day, New Year's Day).

        Falls back to pandas business day frequency if the library is unavailable.

        Parameters
        ----------
        start_date : str
            Start date (ISO 8601).
        end_date : str
            End date (ISO 8601).

        Returns
        -------
        pd.DatetimeIndex
            DatetimeIndex of valid LSE trading days.
        """
        try:
            import pandas_market_calendars as mcal

            lse = mcal.get_calendar("LSE")
            schedule = lse.schedule(start_date=start_date, end_date=end_date)
            trading_days = mcal.date_range(schedule, frequency="1D")
            # date_range returns UTC timestamps; normalise to date-only
            return pd.DatetimeIndex(trading_days.normalize().unique().sort_values())
        except ImportError:
            logger.warning(
                "pandas_market_calendars not available. "
                "Falling back to pandas business days (UK holidays not excluded)."
            )
            return pd.bdate_range(start=start_date, end=end_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LSE calendar error: %s. Using business days.", exc)
            return pd.bdate_range(start=start_date, end=end_date)


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------


def clean_data(
    prices: pd.DataFrame,
    min_history_days: int = 252,
    max_forward_fill_days: int = 5,
) -> pd.DataFrame:
    """
    Clean and align a wide price DataFrame.

    Steps applied:
    1. Remove tickers with fewer than ``min_history_days`` of non-NaN data.
    2. Forward-fill gaps up to ``max_forward_fill_days`` consecutive days
       (handles trading halts, thin markets).
    3. Drop remaining NaN values.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price DataFrame (index=date, columns=tickers).
    min_history_days : int
        Minimum number of valid price observations required.
    max_forward_fill_days : int
        Maximum consecutive days to forward-fill.

    Returns
    -------
    pd.DataFrame
        Cleaned price DataFrame.
    """
    n_orig = prices.shape[1]

    # Step 1: drop tickers with insufficient history
    valid_count = prices.notna().sum()
    prices = prices.loc[:, valid_count >= min_history_days]
    removed = n_orig - prices.shape[1]
    if removed > 0:
        logger.info("clean_data: removed %d tickers with < %d days history",
                    removed, min_history_days)

    # Step 2: limited forward-fill
    prices = prices.ffill(limit=max_forward_fill_days)

    # Step 3: drop rows where all tickers are NaN (e.g., holidays not caught by calendar)
    prices = prices.dropna(how="all")

    logger.info("clean_data: %d dates × %d tickers after cleaning", *prices.shape)
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute daily returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price DataFrame (index=date, columns=tickers), GBP prices.
    method : str
        'simple' for arithmetic returns (pct_change),
        'log' for log returns (ln(P_t / P_{t-1})).

    Returns
    -------
    pd.DataFrame
        Returns DataFrame with same shape as input.
        First row is NaN (no prior day available).
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method '{method}'. Use 'simple' or 'log'.")

    logger.debug("compute_returns: method='%s', shape=%s", method, returns.shape)
    return returns
