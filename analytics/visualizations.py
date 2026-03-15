"""
Performance visualisation suite for the UK Systematic Trading Research Framework.

All charts label monetary values in GBP (£). Axis annotations, titles, and
legends clearly identify this as a UK equity framework.

All public methods return ``matplotlib.Figure`` objects. The caller decides
whether to display (plt.show()), save to file (fig.savefig()), or embed in a
report. This decoupling allows the same chart code to work in scripts,
notebooks, and automated reporting.

Requires: matplotlib, seaborn
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from analytics.metrics import (
    compute_metrics,
    compute_monthly_returns,
    compute_rolling_sharpe,
)
from backtester.engine import BacktestResult

logger = logging.getLogger(__name__)

# Colour palette for multi-strategy charts
STRATEGY_COLOURS = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


class PerformanceVisualizer:
    """
    Creates the standard performance visualisation suite.

    Parameters
    ----------
    figsize : tuple[int, int]
        Default figure size (width, height) in inches.
    style : str
        Matplotlib/seaborn style. Default 'seaborn-v0_8-whitegrid'.
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (14, 10),
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        self.figsize = figsize
        self._apply_style(style)

    # ------------------------------------------------------------------
    # Public: dashboard and individual charts
    # ------------------------------------------------------------------

    def plot_performance_dashboard(
        self,
        result: BacktestResult,
        title: str = "",
        risk_free_rate: float = 0.04,
    ) -> plt.Figure:
        """
        Multi-panel performance dashboard.

        Layout (3 rows × 2 columns):
        - Row 1 (full width): Equity curve with drawdown shading + metrics box
        - Row 2 left: Rolling 63-day Sharpe ratio
        - Row 2 right: Monthly returns heatmap
        - Row 3 left: Daily return distribution histogram
        - Row 3 right: Rolling 63-day annualised volatility

        Parameters
        ----------
        result : BacktestResult
            Backtest output to visualise.
        title : str
            Chart title (defaults to strategy name).
        risk_free_rate : float
            Used for Sharpe ratio computation in the metrics box.

        Returns
        -------
        plt.Figure
        """
        strategy_name = result.strategy_name
        chart_title = title or f"{strategy_name} — Performance Dashboard (GBP)"

        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * 1.3))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_equity = fig.add_subplot(gs[0, :])     # Top full-width
        ax_sharpe = fig.add_subplot(gs[1, 0])
        ax_heatmap = fig.add_subplot(gs[1, 1])
        ax_dist = fig.add_subplot(gs[2, 0])
        ax_vol = fig.add_subplot(gs[2, 1])

        # Panel 1: Equity curve
        self._plot_equity_panel(ax_equity, result, risk_free_rate)

        # Panel 2: Rolling Sharpe
        rolling_sharpe = compute_rolling_sharpe(result.daily_returns, window=63,
                                                risk_free_rate=risk_free_rate)
        ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe.values,
                       color=STRATEGY_COLOURS[0], linewidth=1.2)
        ax_sharpe.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax_sharpe.axhline(1, color="green", linewidth=0.5, linestyle=":")
        ax_sharpe.set_title("Rolling 63-Day Sharpe Ratio", fontsize=10)
        ax_sharpe.set_ylabel("Sharpe Ratio")
        ax_sharpe.tick_params(axis="x", rotation=30)

        # Panel 3: Monthly returns heatmap
        try:
            monthly = compute_monthly_returns(result.daily_returns)
            month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            cols = [c for c in range(1, 13) if c in monthly.columns]
            monthly_subset = monthly[cols]
            monthly_subset.columns = [month_labels[c - 1] for c in cols]

            sns.heatmap(
                monthly_subset,
                ax=ax_heatmap,
                cmap="RdYlGn",
                center=0,
                annot=True,
                fmt=".1f",
                annot_kws={"size": 7},
                linewidths=0.3,
                cbar_kws={"label": "Return (%)"},
            )
            ax_heatmap.set_title("Monthly Returns (%)", fontsize=10)
            ax_heatmap.set_xlabel("")
            ax_heatmap.set_ylabel("Year")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Monthly heatmap failed: %s", exc)
            ax_heatmap.text(0.5, 0.5, "Insufficient data",
                            ha="center", va="center", transform=ax_heatmap.transAxes)

        # Panel 4: Return distribution
        daily_ret_pct = result.daily_returns.dropna() * 100
        ax_dist.hist(daily_ret_pct, bins=60, color=STRATEGY_COLOURS[0],
                     alpha=0.7, edgecolor="white", linewidth=0.3)
        ax_dist.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_dist.axvline(daily_ret_pct.mean(), color="red", linewidth=1.0,
                        linestyle="--", label=f"Mean: {daily_ret_pct.mean():.2f}%")
        ax_dist.set_title("Daily Return Distribution", fontsize=10)
        ax_dist.set_xlabel("Daily Return (%)")
        ax_dist.set_ylabel("Frequency")
        ax_dist.legend(fontsize=8)

        # Panel 5: Rolling volatility
        roll_vol = result.daily_returns.rolling(63).std() * np.sqrt(252) * 100
        ax_vol.plot(roll_vol.index, roll_vol.values,
                    color=STRATEGY_COLOURS[2], linewidth=1.2)
        ax_vol.set_title("Rolling 63-Day Annualised Volatility", fontsize=10)
        ax_vol.set_ylabel("Volatility (%)")
        ax_vol.tick_params(axis="x", rotation=30)

        fig.suptitle(chart_title, fontsize=13, fontweight="bold", y=1.01)
        return fig

    def plot_equity_curve(
        self,
        equity_curves: dict[str, pd.Series],
        title: str = "Strategy Equity Curves (GBP)",
        normalise: bool = True,
    ) -> plt.Figure:
        """
        Overlay multiple equity curves on a single chart.

        Parameters
        ----------
        equity_curves : dict[str, pd.Series]
            Mapping of strategy name → equity curve in GBP.
        title : str
            Chart title.
        normalise : bool
            If True, normalise each curve to 1.0 at start.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, curve) in enumerate(equity_curves.items()):
            colour = STRATEGY_COLOURS[i % len(STRATEGY_COLOURS)]
            if normalise:
                display = curve / curve.iloc[0]
                ylabel = "Normalised Value (base = 1.0)"
            else:
                display = curve
                ylabel = "Portfolio Value (£)"
            final_val = display.iloc[-1]
            ax.plot(display.index, display.values, label=f"{name} ({final_val:.2f}x)" if normalise
                    else f"{name} (£{final_val:,.0f})",
                    color=colour, linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, loc="upper left")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def plot_drawdown(
        self,
        drawdowns: dict[str, pd.Series],
        title: str = "Strategy Drawdowns",
    ) -> plt.Figure:
        """
        Drawdown chart with filled areas.

        Parameters
        ----------
        drawdowns : dict[str, pd.Series]
            Mapping of strategy name → drawdown series (values <= 0).
        title : str
            Chart title.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, dd) in enumerate(drawdowns.items()):
            colour = STRATEGY_COLOURS[i % len(STRATEGY_COLOURS)]
            dd_pct = dd * 100
            ax.fill_between(dd_pct.index, dd_pct.values, 0,
                            alpha=0.3, color=colour)
            ax.plot(dd_pct.index, dd_pct.values, color=colour,
                    linewidth=1.0, label=f"{name} (max: {dd_pct.min():.1f}%)")

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def plot_rolling_sharpe(
        self,
        returns_dict: dict[str, pd.Series],
        window: int = 63,
        risk_free_rate: float = 0.04,
        title: str = "Rolling Sharpe Ratio (63-day)",
    ) -> plt.Figure:
        """
        Rolling Sharpe ratio for multiple strategies.

        Parameters
        ----------
        returns_dict : dict[str, pd.Series]
            Mapping of strategy name → daily returns.
        window : int
            Rolling window in trading days.
        risk_free_rate : float
            Annualised risk-free rate.
        title : str
            Chart title.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        for i, (name, returns) in enumerate(returns_dict.items()):
            colour = STRATEGY_COLOURS[i % len(STRATEGY_COLOURS)]
            rs = compute_rolling_sharpe(returns, window, risk_free_rate)
            ax.plot(rs.index, rs.values, color=colour, linewidth=1.2, label=name)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.7,
                   label="Sharpe = 1.0")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def plot_monthly_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns (%)",
    ) -> plt.Figure:
        """
        Year-by-month heatmap of monthly returns.

        Parameters
        ----------
        returns : pd.Series
            Daily simple returns.
        title : str
            Chart title.

        Returns
        -------
        plt.Figure
        """
        monthly = compute_monthly_returns(returns)
        month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        cols = [c for c in range(1, 13) if c in monthly.columns]
        monthly.columns = [month_labels.get(c, str(c)) for c in monthly.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 0.9), max(6, len(monthly) * 0.5)))
        sns.heatmap(
            monthly,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            annot=True,
            fmt=".1f",
            annot_kws={"size": 8},
            linewidths=0.5,
            cbar_kws={"label": "Monthly Return (%)"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Year")
        fig.tight_layout()
        return fig

    def plot_strategy_comparison(
        self,
        results: dict[str, BacktestResult],
        risk_free_rate: float = 0.04,
    ) -> plt.Figure:
        """
        Side-by-side bar chart comparing key metrics across strategies.

        Parameters
        ----------
        results : dict[str, BacktestResult]
        risk_free_rate : float

        Returns
        -------
        plt.Figure
        """
        from analytics.metrics import compute_metrics

        metrics_list = {}
        for name, r in results.items():
            metrics_list[name] = compute_metrics(r.daily_returns,
                                                  risk_free_rate=risk_free_rate)

        metric_keys = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]
        metric_labels = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"]

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        axes_flat = axes.flatten()

        names = list(metrics_list.keys())
        colours = STRATEGY_COLOURS[:len(names)]

        for ax, key, label in zip(axes_flat, metric_keys, metric_labels):
            values = [metrics_list[n].get(key, 0) * 100
                      if key in ("annual_return", "annual_volatility", "max_drawdown")
                      else metrics_list[n].get(key, 0)
                      for n in names]

            bars = ax.bar(names, values, color=colours, alpha=0.8, edgecolor="white")
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_ylabel("%" if key in ("annual_return", "annual_volatility", "max_drawdown")
                          else "Ratio")
            ax.axhline(0, color="black", linewidth=0.5)

            # Value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.1f}{'%' if key != 'sharpe_ratio' else ''}",
                        ha="center", va="bottom", fontsize=8)
            ax.tick_params(axis="x", rotation=20)

        fig.suptitle("Strategy Comparison — UK Equity Framework (GBP)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def save_all(
        self,
        result: BacktestResult,
        output_dir: str,
        dpi: int = 150,
    ) -> list[str]:
        """
        Generate and save all standard charts for a single strategy.

        Parameters
        ----------
        result : BacktestResult
            Backtest output.
        output_dir : str
            Directory to save PNG files.
        dpi : int
            Image resolution.

        Returns
        -------
        list[str]
            File paths of saved charts.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        name = result.strategy_name.replace(" ", "_").lower()
        saved: list[str] = []

        charts = {
            f"{name}_dashboard.png": self.plot_performance_dashboard(result),
            f"{name}_equity.png": self.plot_equity_curve(
                {result.strategy_name: result.equity_curve}
            ),
            f"{name}_drawdown.png": self.plot_drawdown(
                {result.strategy_name: result.drawdowns}
            ),
            f"{name}_monthly.png": self.plot_monthly_heatmap(result.daily_returns),
        }

        for filename, fig in charts.items():
            path = str(out_dir / filename)
            try:
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
                logger.info("Saved: %s", path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to save %s: %s", path, exc)

        return saved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_style(self, style: str) -> None:
        """Apply matplotlib/seaborn style settings."""
        try:
            plt.style.use(style)
        except OSError:
            try:
                plt.style.use("seaborn-whitegrid")
            except OSError:
                pass  # Fall back to matplotlib defaults

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.dpi": 100,
        })

    def _plot_equity_panel(
        self,
        ax: plt.Axes,
        result: BacktestResult,
        risk_free_rate: float,
    ) -> None:
        """Draw equity curve with drawdown shading and metrics annotation."""
        equity = result.equity_curve
        dd = result.drawdowns

        # Equity curve
        colour = STRATEGY_COLOURS[0]
        ax.plot(equity.index, equity.values, color=colour, linewidth=1.8,
                label=f"{result.strategy_name}")

        # Drawdown shading (secondary y-axis)
        ax2 = ax.twinx()
        dd_pct = dd * 100
        ax2.fill_between(dd_pct.index, dd_pct.values, 0,
                         alpha=0.15, color="red", label="Drawdown")
        ax2.set_ylabel("Drawdown (%)", color="red", fontsize=8)
        ax2.tick_params(axis="y", colors="red", labelsize=7)
        ax2.set_ylim(dd_pct.min() * 2.5, 5)

        # GBP formatting on primary y-axis
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"£{x:,.0f}"
        ))
        ax.set_title(f"Equity Curve & Drawdown — {result.strategy_name} (GBP)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Portfolio Value (£)")
        ax.tick_params(axis="x", rotation=30)

        # Metrics text box
        metrics = compute_metrics(result.daily_returns, risk_free_rate=risk_free_rate)
        text = (
            f"Ann. Return: {metrics['annual_return']:.1%}\n"
            f"Volatility:  {metrics['annual_volatility']:.1%}\n"
            f"Sharpe:      {metrics['sharpe_ratio']:.2f}\n"
            f"Max DD:      {metrics['max_drawdown']:.1%}\n"
            f"Win Rate:    {metrics['win_rate']:.1%}"
        )
        ax.text(
            0.02, 0.97, text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                  "edgecolor": "grey", "alpha": 0.8},
            family="monospace",
        )
        ax.legend(loc="upper left", fontsize=9)
