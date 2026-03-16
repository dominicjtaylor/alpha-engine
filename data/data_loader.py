"""
Data loading and caching module for the UK Systematic Trading Research Framework.

Caching strategy (two levels):

  Per-ticker OHLCV cache
    data/cache/{TICKER}.parquet
    Used by the OHLCV loader (earnings strategy).  Incremental tail updates.

  Universe price cache  ← primary optimisation
    data/cache/prices/n{N}_{hash}_{start}.parquet
    One wide-format file (DatetimeIndex × tickers, adjusted close).
    Generated on first run via batch download; subsequent runs load a single
    parquet file instead of reading N individual ticker files.
    Tail-updated automatically when end_date extends past last cached date.

  Invalid ticker cache
    data/cache/invalid_tickers.json
    Tickers that returned no data are stored here.  Skipped automatically on
    future runs.  Cleared by force_refresh=True.

Batch downloading:
    yf.download(batch_of_50, ...) instead of one-by-one calls.
    Dramatically reduces wall-clock time for large universes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import date as _date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import DataConfig

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Suppress noisy per-ticker yfinance ERROR logs; we handle failures gracefully.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
BATCH_SIZE = 50   # tickers per yfinance batch request
BATCH_PAUSE = 0.3 # seconds between batches (be kind to Yahoo Finance)

# Lowercase substrings that identify permanent (non-retriable) yfinance errors.
_PERMANENT_ERROR_KEYWORDS = (
    "possibly delisted",
    "no price data found",
    "no timezone found",
    "data doesn't exist",
    "startdate",
)


class DataLoader:
    """
    Downloads and caches price data for LSE-listed securities.

    Parameters
    ----------
    cache_dir : str
        Root cache directory (per-ticker parquet files live here directly).
    config : DataConfig
        Data pipeline configuration.
    """

    def __init__(self, cache_dir: str, config: DataConfig) -> None:
        self.cache_dir = Path(cache_dir)
        self.prices_cache_dir = self.cache_dir / "prices"
        self.invalid_cache_path = self.cache_dir / "invalid_tickers.json"
        self.config = config

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.prices_cache_dir.mkdir(parents=True, exist_ok=True)

        self._invalid_cache: dict | None = None  # lazy-loaded
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
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load adjusted close prices as a wide DataFrame (DatetimeIndex × tickers).

        On first call for a universe / start-date combination the data is
        downloaded in batches and saved to a single parquet file.  Subsequent
        calls with the same universe load that file instantly.  If end_date
        extends past the cached range, only the missing tail is downloaded.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols with .L suffix.
        start_date, end_date : str
            Date range in ISO 8601 format.
        price_field : str
            Kept for API compatibility; always returns adjusted close prices.
        force_refresh : bool
            Delete the universe cache and invalid-ticker list, then re-download.

        Returns
        -------
        pd.DataFrame
            Wide adjusted-close prices aligned to the LSE trading calendar.
        """
        cache_path = self._universe_cache_path(tickers, start_date)

        if force_refresh:
            self._invalidate_cache(cache_path)

        if cache_path.exists():
            cached = self._load_universe_cache(cache_path)
            if cached is not None and not cached.empty:
                last_cached = cached.index.max()
                req_end = pd.Timestamp(end_date)

                if last_cached >= req_end - pd.offsets.BDay(5):
                    logger.info(
                        "Loaded prices from cache: %s (%d dates × %d tickers)",
                        cache_path.name, *cached.shape,
                    )
                    return self._finalize(cached, start_date, end_date)

                # Tail update: download only the missing portion
                logger.info(
                    "Cache covers up to %s — downloading tail to %s",
                    last_cached.date(), end_date,
                )
                updated = self._append_tail(cached, end_date)
                self._save_universe_cache(updated, cache_path)
                return self._finalize(updated, start_date, end_date)

        # Full download
        invalid = self._load_invalid_cache()
        valid_tickers = [t for t in tickers if t not in invalid]
        n_skipped_invalid = len(tickers) - len(valid_tickers)
        if n_skipped_invalid:
            logger.info(
                "Skipping %d previously-invalid tickers (use force_refresh=True to retry)",
                n_skipped_invalid,
            )

        adj_close_map = self._batch_download_adj_close(valid_tickers, start_date, end_date)

        # Persist newly discovered invalid tickers
        newly_invalid = [t for t in valid_tickers if t not in adj_close_map]
        if newly_invalid:
            self._mark_invalid_tickers(newly_invalid)

        if not adj_close_map:
            raise RuntimeError("No price data loaded for any ticker.")

        prices = pd.DataFrame(adj_close_map)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        lse_days = self._get_lse_calendar(start_date, end_date)
        prices = prices.reindex(lse_days)

        n_loaded = prices.columns.size
        n_total = len(tickers)
        logger.info(
            "Loaded price data: %d / %d tickers (%d skipped — no data)",
            n_loaded, n_total, n_total - n_loaded,
        )

        self._save_universe_cache(prices, cache_path)
        return prices

    def load_ohlcv(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Load full OHLCV DataFrames per ticker (required by EarningsRevisionDrift).

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols with .L suffix.
        start_date, end_date : str
            Date range (ISO 8601).

        Returns
        -------
        dict[str, pd.DataFrame]
            {ticker: OHLCV DataFrame with columns Open/High/Low/Close/Adj Close/Volume}
        """
        invalid = self._load_invalid_cache()
        valid_tickers = [t for t in tickers if t not in invalid]
        logger.info(
            "Loading OHLCV for %d tickers (%d known-invalid skipped)",
            len(valid_tickers), len(tickers) - len(valid_tickers),
        )

        result: dict[str, pd.DataFrame] = {}
        for ticker in tqdm(valid_tickers, desc="Loading OHLCV", unit="ticker"):
            df = self._load_ohlcv_single(ticker, start_date, end_date)
            if df is not None and not df.empty:
                result[ticker] = df

        logger.info("OHLCV loaded for %d / %d tickers", len(result), len(tickers))
        return result

    # ------------------------------------------------------------------
    # Internal: batch adj-close download
    # ------------------------------------------------------------------

    def _batch_download_adj_close(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.Series]:
        """Download adjusted close prices in batches of BATCH_SIZE."""
        results: dict[str, pd.Series] = {}
        n_batches = max(1, (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE)

        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i: i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.debug("Batch %d / %d (%d tickers)", batch_num, n_batches, len(batch))

            batch_results = self._download_adj_close_batch(batch, start_date, end_date)
            results.update(batch_results)

            if i + BATCH_SIZE < len(tickers):
                time.sleep(BATCH_PAUSE)

        return results

    def _download_adj_close_batch(
        self,
        batch_tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.Series]:
        """Download a single batch; fall back to one-by-one on failure."""
        try:
            raw = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
        except Exception as exc:
            logger.warning("Batch download failed (%s) — retrying individually", exc)
            return self._download_individually(batch_tickers, start_date, end_date)

        if raw is None or raw.empty:
            return {}

        return self._parse_adj_close(raw, batch_tickers)

    def _parse_adj_close(
        self,
        raw: pd.DataFrame,
        batch_tickers: list[str],
    ) -> dict[str, pd.Series]:
        """
        Extract per-ticker adjusted-close series from a yfinance batch result.

        yfinance 0.2.x with group_by="ticker" returns a MultiIndex:
          level-0 = ticker, level-1 = field  → raw[ticker]["Adj Close"]

        Without group_by (or single-ticker result):
          level-0 = field, level-1 = ticker  → raw["Adj Close"][ticker]
        """
        results: dict[str, pd.Series] = {}

        if not isinstance(raw.columns, pd.MultiIndex):
            # Single-ticker flat result
            if len(batch_tickers) == 1:
                ticker = batch_tickers[0]
                for field in ("Adj Close", "Close"):
                    if field in raw.columns:
                        s = raw[field].dropna()
                        if not s.empty:
                            results[ticker] = s
                        break
            return results

        level_0_vals = set(raw.columns.get_level_values(0))
        level_1_vals = set(raw.columns.get_level_values(1))

        # group_by="ticker" → level-0 contains ticker names
        if any(t in level_0_vals for t in batch_tickers):
            for ticker in batch_tickers:
                if ticker not in level_0_vals:
                    continue
                sub = raw[ticker]
                for field in ("Adj Close", "Close"):
                    if field in sub.columns:
                        s = sub[field].dropna()
                        if not s.empty:
                            results[ticker] = s
                        break
            return results

        # No group_by → level-0 contains field names
        for field in ("Adj Close", "Close"):
            if field in level_0_vals:
                sub = raw[field]
                for ticker in batch_tickers:
                    if ticker in sub.columns:
                        s = sub[ticker].dropna()
                        if not s.empty:
                            results[ticker] = s
                return results

        return results

    def _download_individually(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.Series]:
        """One-by-one fallback used when a batch request fails entirely."""
        results: dict[str, pd.Series] = {}
        for ticker in tickers:
            s = self._download_single_adj_close(ticker, start_date, end_date)
            if s is not None and not s.empty:
                results[ticker] = s
        return results

    def _download_single_adj_close(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.Series]:
        """Download adjusted close for one ticker; no retry on permanent errors."""
        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
            )
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            for field in ("Adj Close", "Close"):
                if field in raw.columns:
                    s = raw[field].dropna()
                    return s if not s.empty else None
        except Exception as exc:
            exc_lower = str(exc).lower()
            if any(kw in exc_lower for kw in _PERMANENT_ERROR_KEYWORDS):
                logger.debug("Skipping %s (permanent error: %s)", ticker, type(exc).__name__)
            else:
                logger.warning("Download error for %s: %s", ticker, exc)
        return None

    # ------------------------------------------------------------------
    # Internal: universe-level cache
    # ------------------------------------------------------------------

    def _universe_cache_path(self, tickers: list[str], start_date: str) -> Path:
        """Stable path based on ticker set and start date (end date excluded)."""
        key = hashlib.md5("|".join(sorted(tickers)).encode()).hexdigest()[:12]
        n = len(tickers)
        start = start_date.replace("-", "")
        return self.prices_cache_dir / f"n{n}_{key}_{start}.parquet"

    def _load_universe_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(cache_path)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as exc:
            logger.warning("Could not read universe cache %s: %s", cache_path.name, exc)
            return None

    def _save_universe_cache(self, prices: pd.DataFrame, cache_path: Path) -> None:
        try:
            prices.to_parquet(cache_path, compression=self.config.parquet_compression)
            logger.debug("Saved universe cache: %s (%d×%d)", cache_path.name, *prices.shape)
        except Exception as exc:
            logger.warning("Could not save universe cache: %s", exc)

    def _append_tail(self, cached: pd.DataFrame, end_date: str) -> pd.DataFrame:
        """Download missing tail for all tickers already in the universe cache."""
        last_cached = cached.index.max()
        new_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        tickers = list(cached.columns)

        tail_map = self._batch_download_adj_close(tickers, new_start, end_date)
        if not tail_map:
            return cached

        tail_df = pd.DataFrame(tail_map)
        tail_df.index = pd.to_datetime(tail_df.index)

        combined = pd.concat([cached, tail_df]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined

    def _finalize(
        self,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Align cached prices to the LSE calendar and slice to the requested range."""
        lse_days = self._get_lse_calendar(start_date, end_date)
        return prices.reindex(lse_days)

    def _invalidate_cache(self, cache_path: Path) -> None:
        """Delete the universe cache and the invalid-ticker list."""
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Deleted universe cache: %s", cache_path.name)
        if self.invalid_cache_path.exists():
            self.invalid_cache_path.unlink()
            logger.info("Deleted invalid ticker cache")
        self._invalid_cache = {}

    # ------------------------------------------------------------------
    # Internal: invalid ticker cache
    # ------------------------------------------------------------------

    def _load_invalid_cache(self) -> dict:
        if self._invalid_cache is not None:
            return self._invalid_cache
        if self.invalid_cache_path.exists():
            try:
                with open(self.invalid_cache_path) as f:
                    self._invalid_cache = json.load(f)
            except Exception:
                self._invalid_cache = {}
        else:
            self._invalid_cache = {}
        return self._invalid_cache

    def _mark_invalid_tickers(self, tickers: list[str]) -> None:
        cache = self._load_invalid_cache()
        today = _date.today().isoformat()
        for ticker in tickers:
            cache[ticker] = {"marked_invalid": today}
            logger.info("Skipping %s (no historical data for requested period)", ticker)
        self._invalid_cache = cache
        try:
            with open(self.invalid_cache_path, "w") as f:
                json.dump(cache, f, indent=2, sort_keys=True)
        except Exception as exc:
            logger.warning("Could not save invalid ticker cache: %s", exc)

    # ------------------------------------------------------------------
    # Internal: per-ticker OHLCV (earnings strategy)
    # ------------------------------------------------------------------

    def _load_ohlcv_single(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Load OHLCV from per-ticker parquet cache, updating if needed."""
        if self._is_ticker_cache_valid(ticker, start_date, end_date):
            df = self._load_ticker_cache(ticker)
            return df[start_date:end_date]

        combined = self._fetch_ticker_with_cache(ticker, start_date, end_date)
        if combined is None or combined.empty:
            return None

        self._save_ticker_cache(ticker, combined)
        return combined[start_date:end_date]

    def _fetch_ticker_with_cache(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Return full-coverage OHLCV by merging existing cache with missing downloads."""
        cache_path = self._ticker_cache_path(ticker)

        if not cache_path.exists():
            return self._download_ohlcv(ticker, start_date, end_date)

        existing = self._load_ticker_cache(ticker)
        first_cached = existing.index.min()
        last_cached = existing.index.max()
        req_start = pd.Timestamp(start_date)
        req_end = pd.Timestamp(end_date)
        parts: list[pd.DataFrame] = [existing]

        if first_cached > req_start + pd.offsets.BDay(5):
            hist_end = (first_cached - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            hist_df = self._download_ohlcv(ticker, start_date, hist_end)
            if hist_df is not None and not hist_df.empty:
                parts.append(hist_df)

        if last_cached < req_end - pd.offsets.BDay(5):
            new_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            tail_df = self._download_ohlcv(ticker, new_start, end_date)
            if tail_df is not None and not tail_df.empty:
                parts.append(tail_df)

        if len(parts) == 1:
            return existing

        parts = [p[~p.index.duplicated(keep="last")] for p in parts]
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def _download_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_retries: int = 2,
    ) -> Optional[pd.DataFrame]:
        """Download OHLCV for a single ticker; skip immediately on permanent errors."""
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
                    break  # permanent — don't retry empty

                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                cols = [c for c in OHLCV_COLUMNS if c in raw.columns]
                df = raw[cols].copy()
                df.index = pd.to_datetime(df.index)
                return df.sort_index()

            except Exception as exc:
                exc_lower = str(exc).lower()
                if any(kw in exc_lower for kw in _PERMANENT_ERROR_KEYWORDS):
                    logger.debug(
                        "Skipping OHLCV for %s (permanent error: %s)",
                        ticker, type(exc).__name__,
                    )
                    return None
                logger.debug("OHLCV error for %s (attempt %d): %s", ticker, attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    wait *= 2

        return None

    # ------------------------------------------------------------------
    # Internal: per-ticker parquet cache helpers
    # ------------------------------------------------------------------

    def _ticker_cache_path(self, ticker: str) -> Path:
        safe = ticker.replace(".", "_").replace("/", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _is_ticker_cache_valid(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> bool:
        path = self._ticker_cache_path(ticker)
        if not path.exists():
            return False
        try:
            df = pd.read_parquet(path)
            if df.empty:
                return False
            first = df.index.min()
            last = df.index.max()
            return (
                first <= pd.Timestamp(start_date) + pd.offsets.BDay(5)
                and last >= pd.Timestamp(end_date) - pd.offsets.BDay(5)
            )
        except Exception:
            return False

    def _load_ticker_cache(self, ticker: str) -> pd.DataFrame:
        df = pd.read_parquet(self._ticker_cache_path(ticker))
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _save_ticker_cache(self, ticker: str, data: pd.DataFrame) -> None:
        path = self._ticker_cache_path(ticker)
        data.to_parquet(path, compression=self.config.parquet_compression)
        logger.debug("Cached %s (%d rows) → %s", ticker, len(data), path)

    # ------------------------------------------------------------------
    # Internal: LSE trading calendar
    # ------------------------------------------------------------------

    def _get_lse_calendar(self, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Return valid LSE trading days, with fallback to pandas business days."""
        try:
            import pandas_market_calendars as mcal
            lse = mcal.get_calendar("LSE")
            schedule = lse.schedule(start_date=start_date, end_date=end_date)
            trading_days = mcal.date_range(schedule, frequency="1D")
            return pd.DatetimeIndex(trading_days.normalize().unique().sort_values())
        except ImportError:
            logger.warning(
                "pandas_market_calendars not available — "
                "falling back to pandas business days (UK holidays not excluded)."
            )
            return pd.bdate_range(start=start_date, end=end_date)
        except Exception as exc:
            logger.warning("LSE calendar error: %s — using business days.", exc)
            return pd.bdate_range(start=start_date, end=end_date)


# ---------------------------------------------------------------------------
# Module-level utility functions (public API — unchanged)
# ---------------------------------------------------------------------------


def clean_data(
    prices: pd.DataFrame,
    min_history_days: int = 252,
    max_forward_fill_days: int = 5,
) -> pd.DataFrame:
    """
    Clean a wide price DataFrame.

    1. Remove tickers with fewer than ``min_history_days`` non-NaN observations.
    2. Forward-fill gaps up to ``max_forward_fill_days`` consecutive days.
    3. Drop all-NaN rows.
    """
    n_orig = prices.shape[1]
    valid_count = prices.notna().sum()
    prices = prices.loc[:, valid_count >= min_history_days]
    removed = n_orig - prices.shape[1]
    if removed:
        logger.info(
            "clean_data: removed %d tickers with < %d days history",
            removed, min_history_days,
        )

    prices = prices.ffill(limit=max_forward_fill_days)
    prices = prices.dropna(how="all")
    logger.info("clean_data: %d dates × %d tickers after cleaning", *prices.shape)
    return prices


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute daily returns from a wide price DataFrame.

    Parameters
    ----------
    method : str
        'simple' for pct_change, 'log' for log returns.
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"Unknown return method '{method}'. Use 'simple' or 'log'.")

    logger.debug("compute_returns: method='%s', shape=%s", method, returns.shape)
    return returns
