"""
Paper trading portfolio manager.

Tracks a simulated portfolio with fake money. State is persisted to a JSON
file so the portfolio survives between Streamlit sessions.

All monetary values are in GBP (£). Costs model UK SDRT (stamp duty) and
broker commission/slippage, identical to the backtesting engine.

This module is intentionally separate from BacktestEngine — paper trading is
forward-looking (uses live prices) while backtesting is historical.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_STATE_FILE = "paper_trading/portfolio_state.json"
MIN_TRADE_NOTIONAL = 50.0   # GBP — ignore rebalances smaller than £50
EQUITY_HISTORY_LIMIT = 1000  # keep last N daily snapshots


@dataclass
class PaperTrade:
    """A single paper trade execution record."""
    date: str
    ticker: str
    action: str          # "BUY" | "SELL" | "SHORT" | "COVER"
    shares: float
    price: float
    notional: float      # abs(shares * price)
    commission: float
    slippage: float
    stamp_duty: float
    total_cost: float


@dataclass
class PositionSnapshot:
    """Current state of a single position."""
    ticker: str
    shares: float
    side: str            # "long" | "short"
    entry_price: float
    entry_date: str
    target_weight: float
    current_price: float = 0.0
    market_value: float = 0.0
    unrealised_pnl: float = 0.0
    unrealised_pct: float = 0.0


class PaperPortfolio:
    """
    Simulated paper trading portfolio backed by a JSON state file.

    Lifecycle
    ---------
    1. ``create()`` — initialise a fresh portfolio and persist it.
    2. ``refresh_prices()`` — update unrealised P&L from latest prices.
    3. ``preview_rebalance()`` — compute trades required to hit target weights.
    4. ``execute_rebalance()`` — materialise the trades, update state, persist.
    5. ``reset()`` — delete state file and start over.

    Parameters
    ----------
    state_file : str
        Path to the JSON state file (created automatically).
    """

    def __init__(self, state_file: str = DEFAULT_STATE_FILE) -> None:
        self._file = Path(state_file)
        self._state: Optional[dict] = self._load()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if a portfolio has been created (state file exists)."""
        return self._state is not None

    @property
    def initial_capital(self) -> float:
        return self._state["initial_capital"] if self._state else 0.0

    @property
    def config(self) -> dict:
        return self._state.get("config", {}) if self._state else {}

    @property
    def trade_log(self) -> list[dict]:
        return self._state.get("trade_log", []) if self._state else []

    @property
    def equity_history(self) -> pd.Series:
        """Equity history as a pd.Series with DatetimeIndex."""
        if not self._state or not self._state["equity_history"]:
            return pd.Series(dtype=float)
        records = self._state["equity_history"]
        s = pd.Series(
            {r["date"]: r["equity"] for r in records}
        )
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    # ------------------------------------------------------------------
    # Portfolio lifecycle
    # ------------------------------------------------------------------

    def create(
        self,
        initial_capital: float,
        strategy: str,
        universe: str,
        universe_size: int,
        commission_bps: float = 10.0,
        slippage_bps: float = 10.0,
        stamp_duty_rate: float = 0.005,
    ) -> None:
        """
        Initialise a fresh paper portfolio and persist it.

        Parameters
        ----------
        initial_capital : float
            Starting cash in GBP.
        strategy : str
            Strategy name (e.g. 'momentum').
        universe : str
            Universe identifier (e.g. 'ftse100').
        universe_size : int
            Max tickers in universe.
        commission_bps : float
            Broker commission per side in basis points.
        slippage_bps : float
            Slippage per side in basis points.
        stamp_duty_rate : float
            UK SDRT rate (0.005 = 0.5%) on long purchases.
        """
        today = date.today().isoformat()
        self._state = {
            "created_date": today,
            "last_rebalanced": None,
            "last_price_update": None,
            "initial_capital": initial_capital,
            "cash": initial_capital,
            # positions: ticker -> {shares, entry_price, entry_date, side, target_weight}
            "positions": {},
            "equity_history": [{"date": today, "equity": initial_capital}],
            "trade_log": [],
            "config": {
                "strategy": strategy,
                "universe": universe,
                "universe_size": universe_size,
                "commission_bps": commission_bps,
                "slippage_bps": slippage_bps,
                "stamp_duty_rate": stamp_duty_rate,
            },
        }
        self._save()
        logger.info(
            "Paper portfolio created: £%,.0f | strategy=%s | universe=%s",
            initial_capital, strategy, universe,
        )

    def reset(self) -> None:
        """Delete the portfolio state file and clear in-memory state."""
        if self._file.exists():
            self._file.unlink()
        self._state = None
        logger.info("Paper portfolio reset.")

    # ------------------------------------------------------------------
    # Price refresh and P&L
    # ------------------------------------------------------------------

    def refresh_prices(
        self,
        current_prices: pd.Series,
    ) -> tuple[list[PositionSnapshot], float]:
        """
        Update unrealised P&L on all positions using latest prices.

        Parameters
        ----------
        current_prices : pd.Series
            Latest prices indexed by ticker symbol (e.g. 'AZN.L').

        Returns
        -------
        tuple[list[PositionSnapshot], float]
            - List of ``PositionSnapshot`` objects with current P&L filled in.
            - Total current portfolio equity (cash + market value).
        """
        if not self._state:
            return [], 0.0

        snapshots: list[PositionSnapshot] = []
        total_market_value = 0.0

        for ticker, pos in self._state["positions"].items():
            price = float(current_prices.get(ticker, 0.0))
            if price <= 0:
                price = pos["entry_price"]  # stale fallback

            market_value = pos["shares"] * price
            entry_value = pos["shares"] * pos["entry_price"]
            unrealised_pnl = market_value - entry_value
            unrealised_pct = unrealised_pnl / entry_value if entry_value != 0 else 0.0

            total_market_value += market_value
            snapshots.append(PositionSnapshot(
                ticker=ticker,
                shares=pos["shares"],
                side=pos.get("side", "long"),
                entry_price=pos["entry_price"],
                entry_date=pos["entry_date"],
                target_weight=pos.get("target_weight", 0.0),
                current_price=price,
                market_value=market_value,
                unrealised_pnl=unrealised_pnl,
                unrealised_pct=unrealised_pct,
            ))

        cash = float(self._state["cash"])
        total_equity = cash + total_market_value

        # Record today's equity if not already recorded
        today = date.today().isoformat()
        history = self._state["equity_history"]
        if not history or history[-1]["date"] != today:
            history.append({"date": today, "equity": total_equity})
            if len(history) > EQUITY_HISTORY_LIMIT:
                self._state["equity_history"] = history[-EQUITY_HISTORY_LIMIT:]
            self._state["last_price_update"] = today
            self._save()

        return snapshots, total_equity

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def preview_rebalance(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
    ) -> list[dict]:
        """
        Compute trades needed to move from current to target weights.

        Does NOT modify portfolio state.

        Parameters
        ----------
        target_weights : pd.Series
            Desired portfolio weights indexed by ticker.
            Positive = long, negative = short.
        current_prices : pd.Series
            Latest prices indexed by ticker.

        Returns
        -------
        list[dict]
            List of pending trade dicts with keys:
            ticker, action, shares, price, notional, estimated_cost.
        """
        if not self._state:
            return []

        _, total_equity = self.refresh_prices(current_prices)
        cfg = self._state["config"]
        cost_rate = (cfg["commission_bps"] + cfg["slippage_bps"]) / 10_000

        pending: list[dict] = []

        for ticker in set(list(target_weights.index) + list(self._state["positions"].keys())):
            price = float(current_prices.get(ticker, 0.0))
            if price <= 0:
                continue

            target_weight = float(target_weights.get(ticker, 0.0))
            target_notional = target_weight * total_equity

            pos = self._state["positions"].get(ticker)
            current_notional = float(pos["shares"]) * price if pos else 0.0

            trade_notional = target_notional - current_notional
            if abs(trade_notional) < MIN_TRADE_NOTIONAL:
                continue

            shares = abs(trade_notional) / price
            action = _trade_action(current_notional, target_notional)
            linear_cost = abs(trade_notional) * cost_rate
            stamp = max(trade_notional, 0) * cfg["stamp_duty_rate"]
            total_cost = linear_cost + stamp

            pending.append({
                "ticker": ticker,
                "action": action,
                "shares": round(shares, 4),
                "price": price,
                "notional": abs(trade_notional),
                "estimated_cost": total_cost,
                "stamp_duty": stamp,
            })

        return sorted(pending, key=lambda t: t["notional"], reverse=True)

    def execute_rebalance(
        self,
        target_weights: pd.Series,
        current_prices: pd.Series,
    ) -> list[PaperTrade]:
        """
        Materialise a rebalance: compute trades, apply costs, update state.

        Parameters
        ----------
        target_weights : pd.Series
            Desired portfolio weights indexed by ticker.
        current_prices : pd.Series
            Latest prices indexed by ticker.

        Returns
        -------
        list[PaperTrade]
            Executed trades this rebalance.
        """
        if not self._state:
            raise RuntimeError("No active paper portfolio. Call create() first.")

        _, total_equity = self.refresh_prices(current_prices)
        cfg = self._state["config"]
        cost_rate_bps = cfg["commission_bps"] + cfg["slippage_bps"]
        today = date.today().isoformat()
        executed: list[PaperTrade] = []
        total_costs = 0.0

        new_positions: dict = {}

        for ticker in set(list(target_weights.index) + list(self._state["positions"].keys())):
            price = float(current_prices.get(ticker, 0.0))
            if price <= 0:
                continue

            target_weight = float(target_weights.get(ticker, 0.0))
            target_notional = target_weight * total_equity

            pos = self._state["positions"].get(ticker)
            current_notional = float(pos["shares"]) * price if pos else 0.0

            trade_notional = target_notional - current_notional
            if abs(trade_notional) < MIN_TRADE_NOTIONAL:
                # Keep existing position unchanged
                if pos and abs(target_weight) > 0.001:
                    new_positions[ticker] = pos
                continue

            shares_traded = abs(trade_notional) / price
            action = _trade_action(current_notional, target_notional)
            commission = abs(trade_notional) * cfg["commission_bps"] / 10_000
            slippage = abs(trade_notional) * cfg["slippage_bps"] / 10_000
            stamp = max(trade_notional, 0) * cfg["stamp_duty_rate"]
            total_cost = commission + slippage + stamp

            trade = PaperTrade(
                date=today,
                ticker=ticker,
                action=action,
                shares=round(shares_traded, 4),
                price=price,
                notional=abs(trade_notional),
                commission=commission,
                slippage=slippage,
                stamp_duty=stamp,
                total_cost=total_cost,
            )
            executed.append(trade)
            self._state["trade_log"].append(asdict(trade))
            total_costs += total_cost

            # Update position
            if abs(target_weight) > 0.001 and abs(target_notional) > MIN_TRADE_NOTIONAL:
                new_positions[ticker] = {
                    "shares": target_notional / price,
                    "entry_price": price,
                    "entry_date": today,
                    "side": "long" if target_weight > 0 else "short",
                    "target_weight": float(target_weight),
                }

        # Recompute cash: equity minus all position market values minus costs
        market_value = sum(
            pos["shares"] * float(current_prices.get(t, pos["entry_price"]))
            for t, pos in new_positions.items()
        )
        self._state["cash"] = total_equity - market_value - total_costs
        self._state["positions"] = new_positions
        self._state["last_rebalanced"] = today

        # Record equity post-rebalance
        new_equity = self._state["cash"] + market_value
        history = self._state["equity_history"]
        if history and history[-1]["date"] == today:
            history[-1]["equity"] = new_equity
        else:
            history.append({"date": today, "equity": new_equity})

        self._save()
        logger.info(
            "Paper rebalance executed: %d trades, total cost £%.2f, equity £%,.0f",
            len(executed), total_costs, new_equity,
        )
        return executed

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _load(self) -> Optional[dict]:
        if self._file.exists():
            try:
                with open(self._file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("Could not load paper portfolio state: %s", exc)
        return None

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade_action(current_notional: float, target_notional: float) -> str:
    """Determine trade action string from current and target notionals."""
    if current_notional >= 0 and target_notional > current_notional:
        return "BUY"
    if current_notional > 0 and target_notional < current_notional:
        return "SELL"
    if current_notional <= 0 and target_notional < current_notional:
        return "SHORT"
    return "COVER"
