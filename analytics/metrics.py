"""
Performance metrics for the UK Systematic Trading Research Framework.

All metrics assume daily return series as input. Annualisation uses 252
trading days per year (London Stock Exchange). Returns are simple (arithmetic)
unless stated otherwise.

Performance is reported in GBP terms — the risk-free rate should reflect
UK rates (e.g., Bank of England base rate or gilt yield).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: int = 252
"""LSE trading days per year (standard for UK performance calculations)."""


def annual_return(returns: pd.Series) -> float:
    """
    Compute geometric annualised return.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Annualised return as a decimal (e.g., 0.12 = 12%).
    """
    if returns.empty or returns.isna().all():
        return float("nan")
    n = len(returns.dropna())
    total = (1.0 + returns.dropna()).prod()
    return float(total ** (TRADING_DAYS_PER_YEAR / n) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """
    Compute annualised return volatility (standard deviation).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Annualised volatility as a decimal.
    """
    if returns.empty or returns.isna().all():
        return float("nan")
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
) -> float:
    """
    Compute annualised Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    risk_free_rate : float
        Annualised risk-free rate. Default 4% (approximate UK gilt yield).

    Returns
    -------
    float
        Sharpe ratio. Positive = excess returns over risk-free rate.
    """
    ann_ret = annual_return(returns)
    ann_vol = annual_volatility(returns)
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    target: float = 0.0,
) -> float:
    """
    Compute annualised Sortino ratio using downside deviation.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    risk_free_rate : float
        Annualised risk-free rate.
    target : float
        Daily return target. Returns below this are considered downside.

    Returns
    -------
    float
        Sortino ratio.
    """
    ann_ret = annual_return(returns)
    downside = returns[returns < target]
    if downside.empty or downside.std() == 0:
        return float("nan")
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    return float((ann_ret - risk_free_rate) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from peak.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal. E.g., -0.15 = 15% drawdown.
    """
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    """
    Compute Calmar ratio (annual return / |max drawdown|).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Calmar ratio. Higher is better.
    """
    ann_ret = annual_return(returns)
    mdd = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(ann_ret / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """
    Compute fraction of days with positive returns.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Win rate in [0, 1].
    """
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float((clean > 0).sum() / len(clean))


def profit_factor(returns: pd.Series) -> float:
    """
    Compute profit factor (gross wins / |gross losses|).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.

    Returns
    -------
    float
        Profit factor. > 1 = profitable in aggregate.
    """
    clean = returns.dropna()
    gross_wins = clean[clean > 0].sum()
    gross_losses = abs(clean[clean < 0].sum())
    if gross_losses == 0:
        return float("inf")
    return float(gross_wins / gross_losses)


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 63,
    risk_free_rate: float = 0.04,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio over a sliding window.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns.
    window : int
        Rolling window in trading days (default 63 = ~3 months).
    risk_free_rate : float
        Annualised risk-free rate.

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio. First ``window`` values are NaN.
    """
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    rolling_sharpe = (roll_mean / roll_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling_sharpe


def compute_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Aggregate daily returns to a year × month pivot table.

    Suitable for heatmap visualisation. Values are monthly returns as
    percentages (e.g., 2.5 = 2.5%).

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Pivot table with years as rows, months (1–12) as columns.
        Values are monthly returns in percent.
    """
    monthly = (1 + returns).resample("ME").prod() - 1
    monthly_pct = monthly * 100

    pivot = monthly_pct.to_frame("returns")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.month
    return pivot.pivot(index="year", columns="month", values="returns")


def compute_turnover_stats(turnover: pd.Series) -> dict[str, float]:
    """
    Compute turnover summary statistics.

    Parameters
    ----------
    turnover : pd.Series
        Daily portfolio turnover as fraction of NAV.

    Returns
    -------
    dict[str, float]
        Keys: 'avg_daily_turnover', 'annual_turnover', 'max_daily_turnover'.
    """
    clean = turnover.dropna()
    if clean.empty:
        return {"avg_daily_turnover": float("nan"),
                "annual_turnover": float("nan"),
                "max_daily_turnover": float("nan")}
    return {
        "avg_daily_turnover": float(clean.mean()),
        "annual_turnover": float(clean.mean() * TRADING_DAYS_PER_YEAR),
        "max_daily_turnover": float(clean.max()),
    }


def compute_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.04,
    currency: str = "GBP",
) -> dict[str, float]:
    """
    Compute a comprehensive set of performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (DatetimeIndex).
    benchmark_returns : pd.Series, optional
        Daily benchmark returns for alpha/beta/information ratio.
    risk_free_rate : float
        Annualised risk-free rate (default 4% = approximate UK gilt yield).
    currency : str
        Base currency label (default 'GBP').

    Returns
    -------
    dict[str, float]
        Flat dictionary of metric names → values.
        All return/volatility metrics are annualised decimals.
    """
    clean = returns.dropna()

    metrics: dict[str, float] = {
        "annual_return": annual_return(clean),
        "annual_volatility": annual_volatility(clean),
        "sharpe_ratio": sharpe_ratio(clean, risk_free_rate),
        "sortino_ratio": sortino_ratio(clean, risk_free_rate),
        "max_drawdown": max_drawdown(clean),
        "calmar_ratio": calmar_ratio(clean),
        "win_rate": win_rate(clean),
        "profit_factor": profit_factor(clean),
        "avg_daily_return": float(clean.mean()),
        "avg_win": float(clean[clean > 0].mean()) if (clean > 0).any() else 0.0,
        "avg_loss": float(clean[clean < 0].mean()) if (clean < 0).any() else 0.0,
        "skewness": float(stats.skew(clean)) if len(clean) > 3 else float("nan"),
        "kurtosis": float(stats.kurtosis(clean)) if len(clean) > 3 else float("nan"),
        "var_95": float(clean.quantile(0.05)),
        "cvar_95": float(clean[clean <= clean.quantile(0.05)].mean()),
        "n_days": int(len(clean)),
    }

    # Benchmark-relative metrics
    if benchmark_returns is not None:
        bench_clean = benchmark_returns.dropna()
        common = clean.index.intersection(bench_clean.index)
        if len(common) > 30:
            r = clean.loc[common]
            b = bench_clean.loc[common]

            # Benchmark standalone stats
            bench_metrics = {
                "benchmark_cagr": annual_return(b),
                "benchmark_volatility": annual_volatility(b),
                "benchmark_sharpe": sharpe_ratio(b, risk_free_rate),
                "benchmark_max_drawdown": max_drawdown(b),
            }

            # Relative metrics (simple active-return definitions)
            active = r - b
            alpha = float(active.mean() * TRADING_DAYS_PER_YEAR)
            tracking_error = float(active.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            info_ratio = (alpha / tracking_error
                          if tracking_error != 0 else float("nan"))

            bench_metrics.update({
                "alpha": alpha,
                "tracking_error": tracking_error,
                "information_ratio": info_ratio,
            })
            metrics.update(bench_metrics)

    logger.debug("Metrics computed for %d-day return series.", len(clean))
    return metrics


def format_metrics_table(
    metrics_dict: dict[str, dict[str, float]],
    currency: str = "GBP",
) -> str:
    """
    Format a dictionary of strategy metrics as a console-friendly table.

    Parameters
    ----------
    metrics_dict : dict[str, dict[str, float]]
        Outer key = strategy name, inner dict = metrics from compute_metrics().
    currency : str
        Currency symbol label (default 'GBP').

    Returns
    -------
    str
        Formatted ASCII table string.
    """
    if not metrics_dict:
        return "No metrics to display."

    strategy_names = list(metrics_dict.keys())
    # Check if any strategy has benchmark metrics
    has_benchmark = any(
        "benchmark_cagr" in m for m in metrics_dict.values()
    )

    display_keys = [
        ("annual_return", "Annual Return", "{:.1%}"),
        ("annual_volatility", "Annual Vol", "{:.1%}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:.1%}"),
        ("calmar_ratio", "Calmar Ratio", "{:.2f}"),
        ("win_rate", "Win Rate", "{:.1%}"),
        ("profit_factor", "Profit Factor", "{:.2f}"),
        ("n_days", "Trading Days", "{:.0f}"),
    ]
    if has_benchmark:
        display_keys += [
            ("benchmark_cagr", "Benchmark CAGR", "{:.1%}"),
            ("benchmark_sharpe", "Benchmark Sharpe", "{:.2f}"),
            ("benchmark_max_drawdown", "Benchmark MaxDD", "{:.1%}"),
            ("alpha", "Alpha vs Bench", "{:.1%}"),
            ("tracking_error", "Tracking Error", "{:.1%}"),
            ("information_ratio", "Info Ratio", "{:.2f}"),
        ]

    col_width = max(16, max(len(n) for n in strategy_names) + 2)
    label_width = 18

    header = f"{'Metric':<{label_width}}" + "".join(
        f"{name:>{col_width}}" for name in strategy_names
    )
    separator = "-" * len(header)

    lines = [
        "",
        f"  Performance Summary ({currency})",
        separator,
        header,
        separator,
    ]

    for key, label, fmt in display_keys:
        row = f"{label:<{label_width}}"
        for name in strategy_names:
            val = metrics_dict[name].get(key, float("nan"))
            try:
                row += f"{fmt.format(val):>{col_width}}"
            except (ValueError, TypeError):
                row += f"{'N/A':>{col_width}}"
        lines.append(row)

    lines.append(separator)
    return "\n".join(lines)
