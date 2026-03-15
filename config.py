"""
Central configuration module for the UK Systematic Trading Research Framework.

All numeric parameters and settings are defined here as dataclasses.
Components receive their relevant config sub-object via __init__ injection,
enabling parameter sweeps without modifying global state.

All capital and performance metrics are denominated in GBP (£).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration for the data pipeline."""

    universe: str = "ftse_all"
    """Universe to use: 'ftse100', 'ftse250', or 'ftse_all'."""

    universe_size: int = 350
    """Maximum number of tickers to include in the universe."""

    cache_dir: str = "data/cache"
    """Local directory for parquet-cached price data."""

    start_date: str = "2010-01-01"
    """Default start date for data downloads (ISO 8601)."""

    parquet_compression: str = "snappy"
    """Compression codec for cached parquet files."""

    min_avg_daily_volume: float = 1_000_000.0
    """Minimum average daily trading volume in GBP for liquidity filtering."""

    min_price: float = 0.50
    """Minimum share price in GBP to exclude penny stocks."""

    calendar: str = "LSE"
    """pandas_market_calendars key for the London Stock Exchange."""

    max_forward_fill_days: int = 5
    """Maximum consecutive NaN days to forward-fill in price data."""

    min_history_days: int = 252
    """Minimum trading days of history required to include a ticker."""


@dataclass
class StrategyConfig:
    """Configuration for all strategy signal and portfolio parameters."""

    # Cross-Sectional Momentum
    momentum_lookback: int = 252
    """Total lookback window in trading days for momentum signal."""

    momentum_skip: int = 21
    """Number of recent days to skip (avoids short-term reversal contamination)."""

    momentum_long_pct: float = 0.10
    """Fraction of universe to hold long and short (e.g., 0.10 = top/bottom decile)."""

    # Short-Term Mean Reversion
    mean_reversion_lookback: int = 5
    """Lookback window in trading days for mean reversion signal."""

    mean_reversion_long_pct: float = 0.20
    """Fraction of universe to hold long and short for mean reversion."""

    # Earnings Revision Drift
    earnings_gap_threshold: float = 0.05
    """Minimum overnight gap magnitude (as fraction) to trigger a signal."""

    earnings_hold_days: int = 10
    """Number of trading days to hold a position after an earnings gap."""


@dataclass
class RiskConfig:
    """Configuration for risk controls and transaction cost assumptions."""

    max_leverage: float = 1.5
    """Maximum gross portfolio leverage (sum of absolute weights)."""

    max_position_pct: float = 0.05
    """Maximum single-position weight as fraction of portfolio."""

    target_volatility: float = 0.10
    """Annualised portfolio volatility target for vol-scaling."""

    vol_lookback: int = 63
    """Lookback window in trading days for realised volatility estimation."""

    # Transaction costs (UK-specific)
    commission_bps: float = 10.0
    """Broker commission per side in basis points."""

    slippage_bps: float = 10.0
    """Market impact / slippage per side in basis points."""

    spread_bps: float = 5.0
    """Bid-ask spread cost per side in basis points."""

    stamp_duty_rate: float = 0.005
    """UK Stamp Duty Reserve Tax rate (0.5%) applied to long purchases only."""

    currency: str = "GBP"
    """Base currency for all portfolio calculations and reporting."""


@dataclass
class FrameworkConfig:
    """Root configuration object passed through the framework."""

    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    log_level: str = "INFO"
    """Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'."""

    initial_capital: float = 1_000_000.0
    """Starting portfolio capital in GBP (£1,000,000 default)."""

    output_dir: str = "results"
    """Directory for saving charts and performance summaries."""
