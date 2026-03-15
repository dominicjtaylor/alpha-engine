"""
Systematic Trading Research Framework — UK Equities Edition

Main orchestration script. Downloads LSE price data, runs three evidence-based
equity strategies, backtests them with realistic UK transaction costs (including
stamp duty), and produces performance charts and tables in GBP.

Usage
-----
Full backtest (all strategies, FTSE All):
    python main.py

Single strategy:
    python main.py --strategy momentum
    python main.py --strategy mean_reversion
    python main.py --strategy earnings

Specific universe:
    python main.py --universe ftse100
    python main.py --universe ftse250
    python main.py --universe ftse_all

Date range:
    python main.py --start 2015-01-01 --end 2023-12-31

Walk-forward validation:
    python main.py --walk-forward --strategy momentum

Parameter sweep:
    python main.py --sweep --strategy momentum

Save outputs to custom directory:
    python main.py --output-dir my_results/

Designed for UK equity markets. All performance reported in GBP (£).
Data sourced from Yahoo Finance using .L suffix for LSE tickers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Setup logging before any imports that use the logger
# ---------------------------------------------------------------------------


def setup_logging(log_level: str = "INFO", output_dir: str = "logs") -> None:
    """
    Configure root logger with console (INFO) and file (DEBUG) handlers.

    Parameters
    ----------
    log_level : str
        Console log level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    output_dir : str
        Directory for log files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    from datetime import date
    log_file = Path(output_dir) / f"framework_{date.today().isoformat()}.log"

    fmt = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(fh)

    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="UK Systematic Trading Research Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean_reversion", "earnings", "all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    parser.add_argument(
        "--universe",
        choices=["ftse100", "ftse250", "ftse_all"],
        default="ftse_all",
        help="UK equity universe (default: ftse_all)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Backtest start date ISO 8601 (default: 2015-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2024-06-30",
        help="Backtest end date ISO 8601 (default: 2024-06-30)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital in GBP (default: £1,000,000)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for charts and summaries (default: results/)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward validation instead of full backtest",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep (grid search) for the selected strategy",
    )
    parser.add_argument(
        "--combination",
        choices=["equal_weight", "volatility_scale"],
        default="equal_weight",
        help="Strategy combination method for multi-strategy portfolio",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (faster for scripting)",
    )
    return parser


# ---------------------------------------------------------------------------
# Main pipelines
# ---------------------------------------------------------------------------


def run_full_backtest(
    config,
    args: argparse.Namespace,
) -> None:
    """
    Execute the full research pipeline.

    Steps:
    1. Load UK equity universe (FTSE tickers with .L suffix)
    2. Download/cache LSE price data via yfinance
    3. Apply liquidity filters
    4. Clean data and compute returns
    5. Run selected strategies
    6. Combine into multi-strategy portfolio
    7. Run BacktestEngine with UK costs (incl. stamp duty)
    8. Print GBP performance summary
    9. Save visualisations
    """
    from analytics.metrics import compute_metrics, format_metrics_table
    from analytics.visualizations import PerformanceVisualizer
    from backtester.engine import BacktestEngine
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import apply_liquidity_filters, load_ftse_universe
    from portfolio.portfolio_manager import CombinationMethod, PortfolioConfig, PortfolioManager
    from strategies.earnings_revision import EarningsRevisionDrift
    from strategies.mean_reversion import ShortTermMeanReversion
    from strategies.momentum import CrossSectionalMomentum

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Universe
    # ------------------------------------------------------------------
    logger.info("=== Step 1: Loading UK equity universe ('%s') ===", args.universe)
    tickers = load_ftse_universe(index=args.universe, top_n=config.data.universe_size)
    logger.info("Universe: %d tickers", len(tickers))

    # ------------------------------------------------------------------
    # 2. Data download / cache
    # ------------------------------------------------------------------
    logger.info("=== Step 2: Downloading LSE price data (%s → %s) ===", args.start, args.end)
    loader = DataLoader(cache_dir=config.data.cache_dir, config=config.data)

    prices_raw = loader.load_price_data(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
    )
    logger.info("Raw prices shape: %s", prices_raw.shape)

    # Load OHLCV for EarningsRevisionDrift
    ohlcv: dict = {}
    if args.strategy in ("earnings", "all"):
        logger.info("=== Step 2b: Downloading OHLCV for earnings strategy ===")
        ohlcv = loader.load_ohlcv(
            tickers=list(prices_raw.columns),
            start_date=args.start,
            end_date=args.end,
        )

    # ------------------------------------------------------------------
    # 3. Liquidity filter + clean + returns
    # ------------------------------------------------------------------
    logger.info("=== Step 3: Cleaning data and computing returns ===")
    prices = clean_data(
        prices_raw,
        min_history_days=config.data.min_history_days,
        max_forward_fill_days=config.data.max_forward_fill_days,
    )

    # Liquidity filter using price threshold
    liquid_tickers = apply_liquidity_filters(
        tickers=list(prices.columns),
        prices=prices,
        min_price_gbp=config.data.min_price,
        min_avg_volume_gbp=config.data.min_avg_daily_volume,
    )
    prices = prices[liquid_tickers]
    logger.info("After liquidity filter: %d tickers", len(prices.columns))

    returns = compute_returns(prices)

    # ------------------------------------------------------------------
    # 4. Run strategies
    # ------------------------------------------------------------------
    logger.info("=== Step 4: Running strategies ===")
    strategy_map = {
        "momentum": CrossSectionalMomentum(config.strategy),
        "mean_reversion": ShortTermMeanReversion(config.strategy),
        "earnings": EarningsRevisionDrift(config.strategy),
    }

    if args.strategy == "all":
        selected_strategies = strategy_map
    else:
        selected_strategies = {args.strategy: strategy_map[args.strategy]}

    from strategies.base import StrategyResult
    strategy_results: dict[str, StrategyResult] = {}

    for name, strategy in selected_strategies.items():
        logger.info("Running strategy: %s", name)
        kwargs = {}
        if name == "earnings" and ohlcv:
            kwargs["ohlcv"] = ohlcv
        result = strategy.run(prices, returns, **kwargs)
        strategy_results[name] = result
        logger.info("Strategy '%s' complete.", name)

    # ------------------------------------------------------------------
    # 5. Portfolio combination + risk controls
    # ------------------------------------------------------------------
    logger.info("=== Step 5: Portfolio combination and risk controls ===")
    from portfolio.portfolio_manager import PortfolioConfig, PortfolioManager, CombinationMethod

    combo_method = CombinationMethod(args.combination)
    port_config = PortfolioConfig(combination_method=combo_method)
    portfolio_mgr = PortfolioManager(port_config, config.risk)

    if len(strategy_results) > 1:
        combined_weights = portfolio_mgr.combine_strategies(strategy_results)
        combined_weights = portfolio_mgr.apply_risk_controls(combined_weights, returns)
        strategy_results["Combined"] = type("SR", (), {
            "weights": combined_weights, "signals": combined_weights, "name": "Combined"
        })()

    # ------------------------------------------------------------------
    # 6. Backtest each strategy + combined
    # ------------------------------------------------------------------
    logger.info("=== Step 6: Backtesting ===")
    engine = BacktestEngine(config=config.risk, initial_capital=args.capital)
    backtest_results = {}

    for name, st_result in strategy_results.items():
        bt = engine.run(
            weights=st_result.weights,
            returns=returns,
            strategy_name=name,
        )
        backtest_results[name] = bt

    # ------------------------------------------------------------------
    # 7. Performance metrics + table
    # ------------------------------------------------------------------
    logger.info("=== Step 7: Computing performance metrics ===")
    all_metrics = {}
    for name, bt in backtest_results.items():
        all_metrics[name] = compute_metrics(bt.daily_returns, currency="GBP")

    print(format_metrics_table(all_metrics, currency="GBP"))

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics).T
    csv_path = output_dir / "performance_metrics.csv"
    metrics_df.to_csv(csv_path)
    logger.info("Metrics saved: %s", csv_path)

    # ------------------------------------------------------------------
    # 8. Visualisations
    # ------------------------------------------------------------------
    if not args.no_charts:
        logger.info("=== Step 8: Generating charts ===")
        viz = PerformanceVisualizer()

        # Individual strategy dashboards
        for name, bt in backtest_results.items():
            if name != "Combined":
                saved = viz.save_all(bt, output_dir=str(output_dir))
                logger.info("Charts saved for %s: %s", name, saved)

        # Comparison chart (if multiple strategies)
        if len(backtest_results) > 1:
            try:
                fig_compare = viz.plot_strategy_comparison(backtest_results)
                compare_path = str(output_dir / "strategy_comparison.png")
                fig_compare.savefig(compare_path, dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt
                plt.close(fig_compare)
                logger.info("Comparison chart saved: %s", compare_path)

                # Overlaid equity curves
                equity_curves = {n: r.equity_curve for n, r in backtest_results.items()}
                fig_eq = viz.plot_equity_curve(equity_curves, normalise=True)
                eq_path = str(output_dir / "equity_curves_all.png")
                fig_eq.savefig(eq_path, dpi=150, bbox_inches="tight")
                plt.close(fig_eq)
                logger.info("Equity curve chart saved: %s", eq_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Comparison chart failed: %s", exc)

    logger.info("=== Pipeline complete. Results in: %s ===", output_dir)


def run_walk_forward(config, args: argparse.Namespace) -> None:
    """
    Run walk-forward validation for a single strategy.
    """
    from analytics.metrics import compute_metrics, format_metrics_table
    from backtester.engine import BacktestEngine
    from backtester.walk_forward import WalkForwardAnalysis
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import apply_liquidity_filters, load_ftse_universe
    from strategies.earnings_revision import EarningsRevisionDrift
    from strategies.mean_reversion import ShortTermMeanReversion
    from strategies.momentum import CrossSectionalMomentum

    strategy_name = args.strategy if args.strategy != "all" else "momentum"
    logger.info("=== Walk-Forward Analysis: %s ===", strategy_name)

    tickers = load_ftse_universe(index=args.universe, top_n=config.data.universe_size)
    loader = DataLoader(cache_dir=config.data.cache_dir, config=config.data)
    prices_raw = loader.load_price_data(tickers, args.start, args.end)
    prices = clean_data(prices_raw, config.data.min_history_days, config.data.max_forward_fill_days)
    prices = prices[apply_liquidity_filters(list(prices.columns), prices,
                                            min_price_gbp=config.data.min_price)]
    returns = compute_returns(prices)

    strategy_cls_map = {
        "momentum": CrossSectionalMomentum,
        "mean_reversion": ShortTermMeanReversion,
        "earnings": EarningsRevisionDrift,
    }
    strategy = strategy_cls_map[strategy_name](config.strategy)

    engine = BacktestEngine(config.risk, initial_capital=args.capital)
    wf = WalkForwardAnalysis(engine, is_months=36, oos_months=12, expanding=True)

    kwargs = {}
    if strategy_name == "earnings":
        kwargs["ohlcv"] = loader.load_ohlcv(list(prices.columns), args.start, args.end)

    wf_result = wf.run(strategy, prices, returns, **kwargs)

    metrics = compute_metrics(wf_result.combined_oos_returns)
    print(format_metrics_table({f"{strategy_name} (OOS)": metrics}))

    logger.info(
        "Walk-forward: %d folds | Combined OOS return: %.1f%%",
        len(wf_result.folds),
        (1 + wf_result.combined_oos_returns).prod() * 100 - 100,
    )

    if not args.no_charts:
        from analytics.visualizations import PerformanceVisualizer
        from backtester.engine import BacktestResult
        import numpy as np
        import matplotlib.pyplot as plt

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        oos_equity = wf_result.combined_oos_equity
        viz = PerformanceVisualizer()
        fig = viz.plot_equity_curve(
            {f"{strategy_name} OOS": oos_equity},
            title=f"Walk-Forward OOS Equity — {strategy_name} (GBP)",
            normalise=False,
        )
        path = str(output_dir / f"{strategy_name}_walk_forward_oos.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Walk-forward chart saved: %s", path)


def run_parameter_sweep(config, args: argparse.Namespace) -> None:
    """
    Run a parameter grid search for a single strategy.
    """
    from backtester.engine import BacktestEngine
    from backtester.parameter_sweep import ParameterSweep
    from data.data_loader import DataLoader, clean_data, compute_returns
    from data.universe import apply_liquidity_filters, load_ftse_universe
    from strategies.momentum import CrossSectionalMomentum

    strategy_name = args.strategy if args.strategy != "all" else "momentum"
    logger.info("=== Parameter Sweep: %s ===", strategy_name)

    # For sweep, use a shorter period / smaller universe for speed
    tickers = load_ftse_universe(index="ftse100", top_n=100)
    loader = DataLoader(cache_dir=config.data.cache_dir, config=config.data)
    prices_raw = loader.load_price_data(tickers, args.start, args.end)
    prices = clean_data(prices_raw, config.data.min_history_days, config.data.max_forward_fill_days)
    prices = prices[apply_liquidity_filters(list(prices.columns), prices,
                                            min_price_gbp=config.data.min_price)]
    returns = compute_returns(prices)

    engine = BacktestEngine(config.risk, initial_capital=args.capital)

    # Define parameter grid (expand as needed)
    param_grid = {
        "momentum_lookback": [126, 252],
        "momentum_skip": [0, 21],
        "momentum_long_pct": [0.10, 0.15, 0.20],
    }

    sweep = ParameterSweep(engine, CrossSectionalMomentum, config.strategy)
    results = sweep.run_grid(param_grid, prices, returns, n_jobs=1)
    summary = sweep.summarize(results)

    logger.info("Parameter sweep complete. Top 5 results by Sharpe:")
    print(summary.head(10).to_string(index=False))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{strategy_name}_sweep_results.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("Sweep results saved: %s", csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(log_level=args.log_level, output_dir="logs")

    logger.info("=" * 60)
    logger.info("UK Systematic Trading Research Framework")
    logger.info("Universe: %s | Strategy: %s | Period: %s → %s",
                args.universe, args.strategy, args.start, args.end)
    logger.info("Capital: £%,.0f | Output: %s", args.capital, args.output_dir)
    logger.info("=" * 60)

    from config import FrameworkConfig
    config = FrameworkConfig()
    config.data.universe = args.universe
    config.initial_capital = args.capital
    config.output_dir = args.output_dir

    try:
        if args.walk_forward:
            run_walk_forward(config, args)
        elif args.sweep:
            run_parameter_sweep(config, args)
        else:
            run_full_backtest(config, args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
