"""
Signal validation utilities for factor pipeline diagnostics.

Pre-backtest sanity checks that automatically detect common factor pipeline bugs:
  - Constant / non-dispersed signals (cross-sectional std ≈ 0)
  - Stale signals (no time variation)
  - Incorrect factor configuration (IC near zero vs. forward returns)
  - Degenerate portfolio weights (no active positions, wrong sums)
  - Portfolio engine bias (random factor sanity check)

Public API
----------
validate_signal(signal, forward_returns=None) -> dict
validate_weights(weights) -> dict
run_random_sanity_check(prices, returns, long_pct=0.10) -> dict
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Signal validation
# ---------------------------------------------------------------------------

def validate_signal(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame | None = None,
    cs_std_min: float = 1e-4,
    ts_std_min: float = 1e-4,
    ic_min: float = 0.005,
    nan_max: float = 0.95,
) -> dict:
    """
    Run all diagnostic checks on a factor signal DataFrame.

    Checks
    ------
    1. Cross-sectional dispersion — signal must vary across stocks each date.
    2. Time variation — signal must evolve over time.
    3. NaN / non-zero coverage — signal must have valid data.
    4. Distribution summary (mean, std, min, max).
    5. IC with forward returns — signal should correlate with returns.

    Parameters
    ----------
    signal : pd.DataFrame
        Factor scores (index=date, columns=tickers).  High value = long candidate.
    forward_returns : pd.DataFrame, optional
        Next-period returns for IC computation.
    cs_std_min : float
        Minimum mean cross-sectional std before warning.
    ts_std_min : float
        Minimum mean time-series std before warning.
    ic_min : float
        Minimum |mean IC| before warning.
    nan_max : float
        Maximum NaN fraction before warning.

    Returns
    -------
    dict
        Keys: mean_cs_std, mean_ts_std, nan_pct, nonzero_pct,
        signal_mean, signal_std, signal_min, signal_max,
        mean_ic (float or None), warnings (list[str]), passed (bool).
    """
    result: dict = {
        "mean_cs_std": float("nan"),
        "mean_ts_std": float("nan"),
        "nan_pct": float("nan"),
        "nonzero_pct": float("nan"),
        "signal_mean": float("nan"),
        "signal_std": float("nan"),
        "signal_min": float("nan"),
        "signal_max": float("nan"),
        "mean_ic": None,
        "warnings": [],
        "passed": False,
    }
    warnings: list[str] = []

    if signal is None or signal.empty:
        warnings.append("Signal DataFrame is empty or None.")
        result["warnings"] = warnings
        return result

    # 1. Cross-sectional dispersion (std across tickers per date)
    cs_std = signal.std(axis=1)
    result["mean_cs_std"] = float(cs_std.mean())
    if result["mean_cs_std"] < cs_std_min:
        warnings.append(
            f"Near-zero cross-sectional dispersion (mean CS std = {result['mean_cs_std']:.2e}). "
            "Signal is constant across assets — no ranking information."
        )

    # 2. Time-series variation (std across dates per ticker)
    ts_std = signal.std(axis=0)
    result["mean_ts_std"] = float(ts_std.mean())
    if result["mean_ts_std"] < ts_std_min:
        warnings.append(
            f"Near-zero time variation (mean TS std = {result['mean_ts_std']:.2e}). "
            "Signal is not evolving over time."
        )

    # 3. Coverage
    total = signal.size
    nan_count = int(signal.isna().sum().sum())
    nonzero_count = int((signal != 0).sum().sum())
    result["nan_pct"] = float(nan_count / max(total, 1))
    result["nonzero_pct"] = float(nonzero_count / max(total, 1))

    if result["nan_pct"] > nan_max:
        warnings.append(
            f"{result['nan_pct']:.1%} of signal values are NaN. "
            "Lookback may exceed the date range — shorten lookback or extend the date range."
        )
    if result["nonzero_pct"] < 0.01:
        warnings.append(
            f"Only {result['nonzero_pct']:.1%} of values are non-zero. "
            "Signal appears degenerate."
        )

    # 4. Distribution summary
    flat = signal.values.ravel()
    flat_valid = flat[~np.isnan(flat)]
    if len(flat_valid) > 0:
        result["signal_mean"] = float(np.mean(flat_valid))
        result["signal_std"] = float(np.std(flat_valid))
        result["signal_min"] = float(np.min(flat_valid))
        result["signal_max"] = float(np.max(flat_valid))

    # 5. IC with forward returns (sampled for speed)
    if forward_returns is not None and not forward_returns.empty:
        ic_values = _compute_sample_ic(signal, forward_returns, n_sample=500)
        if ic_values:
            result["mean_ic"] = float(np.mean(ic_values))
            if abs(result["mean_ic"]) < ic_min:
                warnings.append(
                    f"Mean IC ≈ {result['mean_ic']:.4f} is near zero. "
                    "Verify that factor configuration matches what was researched "
                    "(lookback, skip, and other parameters must be identical)."
                )

    result["warnings"] = warnings
    result["passed"] = len(warnings) == 0
    return result


def _compute_sample_ic(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_sample: int = 500,
) -> list[float]:
    """Compute Spearman IC on a uniformly spaced sample of dates."""
    dates = signal.index.intersection(forward_returns.index)
    if len(dates) == 0:
        return []
    step = max(1, len(dates) // n_sample)
    ic_values: list[float] = []
    for date in dates[::step]:
        s = signal.loc[date].dropna()
        r = forward_returns.loc[date].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 10:
            continue
        try:
            ic = float(s[common].corr(r[common], method="spearman"))
            if not np.isnan(ic):
                ic_values.append(ic)
        except Exception:
            pass
    return ic_values


# ---------------------------------------------------------------------------
# 2. Portfolio weight validation
# ---------------------------------------------------------------------------

def validate_weights(weights: pd.DataFrame) -> dict:
    """
    Validate portfolio weight DataFrame before backtesting.

    Checks
    ------
    1. Active days — must hold positions on a meaningful fraction of dates.
    2. Long / short book sums — each book should sum to ~1.0 / ~-1.0.
    3. Net exposure — should be ~0 for dollar-neutral long-short.

    Parameters
    ----------
    weights : pd.DataFrame
        Signed weights (index=date, columns=tickers).

    Returns
    -------
    dict
        Keys: mean_long_sum, mean_short_sum, mean_net_sum, mean_gross_sum,
        active_days, total_days, avg_long_positions, avg_short_positions,
        warnings (list[str]), passed (bool).
    """
    result: dict = {
        "mean_long_sum": float("nan"),
        "mean_short_sum": float("nan"),
        "mean_net_sum": float("nan"),
        "mean_gross_sum": float("nan"),
        "active_days": 0,
        "total_days": 0,
        "avg_long_positions": float("nan"),
        "avg_short_positions": float("nan"),
        "warnings": [],
        "passed": False,
    }
    warnings: list[str] = []

    if weights is None or weights.empty:
        warnings.append("Weight DataFrame is empty.")
        result["warnings"] = warnings
        return result

    result["total_days"] = len(weights)

    long_w = weights.clip(lower=0)
    short_w = weights.clip(upper=0)
    long_sum = long_w.sum(axis=1)
    short_sum = short_w.sum(axis=1)
    net_sum = weights.sum(axis=1)
    gross_sum = weights.abs().sum(axis=1)

    result["mean_long_sum"] = float(long_sum.mean())
    result["mean_short_sum"] = float(short_sum.mean())
    result["mean_net_sum"] = float(net_sum.mean())
    result["mean_gross_sum"] = float(gross_sum.mean())

    active_mask = (weights.abs() > 1e-6).any(axis=1)
    result["active_days"] = int(active_mask.sum())

    if result["active_days"] == 0:
        warnings.append(
            "Portfolio has no active positions on any date. "
            "Factor signal may be all-NaN — check data coverage and factor lookback."
        )
    elif result["active_days"] < result["total_days"] * 0.05:
        pct = result["active_days"] / max(result["total_days"], 1)
        warnings.append(
            f"Portfolio only active on {result['active_days']}/{result['total_days']} days "
            f"({pct:.0%}). Factor lookback may exceed available history."
        )

    if result["active_days"] > 0:
        active_long = long_sum[active_mask]
        if active_long.mean() > 0.01 and abs(active_long.mean() - 1.0) > 0.5:
            warnings.append(
                f"Mean long book sum = {active_long.mean():.2f} (expected ~1.0). "
                "Portfolio may be mis-scaled."
            )

    long_pos = (long_w > 1e-6).sum(axis=1)
    short_pos = (short_w < -1e-6).sum(axis=1)
    if result["active_days"] > 0:
        result["avg_long_positions"] = float(long_pos[active_mask].mean())
        result["avg_short_positions"] = float(short_pos[active_mask].mean())

    result["warnings"] = warnings
    result["passed"] = len(warnings) == 0
    return result


# ---------------------------------------------------------------------------
# 3. Random factor sanity check
# ---------------------------------------------------------------------------

def run_random_sanity_check(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    long_pct: float = 0.10,
) -> dict:
    """
    Backtest a purely random signal to verify portfolio engine integrity.

    A random factor should produce Sharpe ≈ 0 and modest negative return
    (transaction costs).  Extreme results (max drawdown < -50%, |Sharpe| > 2)
    indicate a portfolio construction or cost calculation bug — not the factor.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices.
    returns : pd.DataFrame
        Daily simple returns.
    long_pct : float
        Fraction of universe per long / short book.

    Returns
    -------
    dict
        Keys: sharpe, annual_return, max_drawdown, warnings (list[str]), passed (bool).
    """
    result: dict = {
        "sharpe": float("nan"),
        "annual_return": float("nan"),
        "max_drawdown": float("nan"),
        "warnings": [],
        "passed": False,
    }
    warnings: list[str] = []

    try:
        from config import RiskConfig
        from backtester.engine import BacktestEngine
        from analytics.metrics import compute_metrics

        rng = np.random.RandomState(42)

        # Vectorised random long-short weights (same construction as FactorPortfolioStrategy)
        noise = pd.DataFrame(
            rng.randn(*prices.shape),
            index=prices.index,
            columns=prices.columns,
        ).where(prices.notna())

        ranks = noise.rank(axis=1, pct=True, na_option="keep")
        long_mask = (ranks >= (1.0 - long_pct)) & noise.notna()
        short_mask = (ranks <= long_pct) & noise.notna()

        n_long = long_mask.sum(axis=1).replace(0, np.nan)
        n_short = short_mask.sum(axis=1).replace(0, np.nan)

        lw = long_mask.astype(float).div(n_long, axis=0).fillna(0.0)
        sw = short_mask.astype(float).div(n_short, axis=0).fillna(0.0) * -1.0
        random_weights = lw + sw

        risk_cfg = RiskConfig()
        engine = BacktestEngine(risk_cfg)
        bt = engine.run(random_weights, returns, strategy_name="Random")
        metrics = compute_metrics(bt.daily_returns)

        result["sharpe"] = float(metrics.get("sharpe_ratio", float("nan")))
        result["annual_return"] = float(metrics.get("annual_return", float("nan")))
        result["max_drawdown"] = float(metrics.get("max_drawdown", float("nan")))

        dd = result["max_drawdown"]
        sh = result["sharpe"]

        if not np.isnan(dd):
            if dd < -0.50:
                warnings.append(
                    f"Random signal max drawdown = {dd:.1%}. "
                    "Portfolio engine may have a structural issue. "
                    "Check that SDRT is only applied to NEW long buys (not full notional each day)."
                )
            elif dd < -0.25:
                warnings.append(
                    f"Random signal max drawdown = {dd:.1%}. "
                    "Slightly elevated for a random signal — "
                    "check transaction costs for high-turnover strategies."
                )

        if not np.isnan(sh) and abs(sh) > 2.0:
            warnings.append(
                f"Random signal Sharpe = {sh:.2f} (expected ≈ 0). "
                "Portfolio engine may have a systematic bias."
            )

    except Exception as exc:
        warnings.append(f"Random sanity check could not run: {exc}")
        log.debug("Random sanity check exception", exc_info=True)

    result["warnings"] = warnings
    result["passed"] = len(warnings) == 0
    return result
