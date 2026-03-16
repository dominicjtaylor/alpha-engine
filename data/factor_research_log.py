"""
Factor research log — persistence layer for the leaderboard.

Saves factor analysis results to a parquet file with:
  - Upsert by config hash (same config → update if IC IR improves)
  - Maximum leaderboard size of MAX_ENTRIES (weakest entries dropped)
  - Sorted by |IC t-stat| descending at all times
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LEADERBOARD_DIR = Path(__file__).parent / "factor_results"
LEADERBOARD_PATH = LEADERBOARD_DIR / "factor_leaderboard.parquet"

# Only keep the top N entries by IC IR (prevents unbounded growth)
MAX_ENTRIES = 100

_SCHEMA_COLS = [
    "config_hash",
    "factor_name",
    "lookback",
    "skip",
    "ic_horizon",
    "universe",
    "universe_size",
    "start_date",
    "end_date",
    "n_quantiles",
    "ic_mean",
    "ic_std",
    "ic_tstat",
    "ic_ir",
    "pct_positive_ic",
    "observations",
    "spread_annual_return",
    "monotonicity_score",
    "factor_kwargs",
    "timestamp",
]


def _make_config_hash(record: dict) -> str:
    """Stable 12-char MD5 hash over the analysis configuration fields."""
    key_fields = [
        str(record.get("factor_name", "")),
        str(record.get("lookback", "")),
        str(record.get("skip", "")),
        str(record.get("ic_horizon", "")),
        str(record.get("universe", "")),
        str(record.get("universe_size", "")),
        str(record.get("start_date", "")),
        str(record.get("end_date", "")),
        str(record.get("n_quantiles", "")),
    ]
    raw = "|".join(key_fields)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def save_result(record: dict, update_if_better: bool = True) -> bool:
    """
    Upsert a factor analysis result into the leaderboard parquet.

    Parameters
    ----------
    record : dict
        Factor analysis results. Must include ic_ir (or ic_mean as fallback).
    update_if_better : bool
        If True (default), only overwrite an existing entry when the new IC IR
        is higher than the stored one. Set False to always overwrite.

    Returns
    -------
    bool
        True if the record was written, False if it was skipped (not improved).
    """
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)

    record = dict(record)  # don't mutate caller's dict

    # Serialize factor_kwargs to JSON string
    if "factor_kwargs" in record and isinstance(record["factor_kwargs"], dict):
        record["factor_kwargs"] = json.dumps(record["factor_kwargs"])
    elif "factor_kwargs" not in record:
        record["factor_kwargs"] = "{}"

    # Compute config hash for dedup
    if "config_hash" not in record:
        record["config_hash"] = _make_config_hash(record)

    if "timestamp" not in record:
        record["timestamp"] = pd.Timestamp.utcnow()

    new_ic_ir = float(record.get("ic_ir", float("nan")))
    new_row = pd.DataFrame([record])

    if LEADERBOARD_PATH.exists():
        existing = pd.read_parquet(LEADERBOARD_PATH)
        matching = existing[existing["config_hash"] == record["config_hash"]]

        if update_if_better and not matching.empty:
            old_ic_ir = float(matching.iloc[0].get("ic_ir", float("nan")))
            import math
            if not math.isnan(old_ic_ir) and not math.isnan(new_ic_ir):
                if new_ic_ir <= old_ic_ir:
                    log.debug(
                        "Skipping '%s' (new IC IR %.3f <= existing %.3f)",
                        record.get("factor_name"), new_ic_ir, old_ic_ir,
                    )
                    return False

        existing = existing[existing["config_hash"] != record["config_hash"]]
        df = pd.concat([existing, new_row], ignore_index=True)
    else:
        df = new_row

    # Sort by IC IR descending, trim to MAX_ENTRIES
    if "ic_ir" in df.columns:
        df = df.sort_values("ic_ir", ascending=False)
    df = df.head(MAX_ENTRIES).reset_index(drop=True)

    df.to_parquet(LEADERBOARD_PATH, index=False)
    log.info(
        "Saved factor result '%s' to leaderboard (hash=%s, IC IR=%.3f)",
        record.get("factor_name"),
        record["config_hash"],
        new_ic_ir if not (new_ic_ir != new_ic_ir) else 0.0,
    )
    return True


def load_leaderboard() -> pd.DataFrame:
    """Return the full leaderboard DataFrame, sorted by IC IR descending."""
    if not LEADERBOARD_PATH.exists():
        return pd.DataFrame(columns=_SCHEMA_COLS)
    df = pd.read_parquet(LEADERBOARD_PATH)
    if "ic_ir" in df.columns:
        df = df.sort_values("ic_ir", ascending=False)
    return df.reset_index(drop=True)


def clear_leaderboard() -> None:
    """Delete the leaderboard parquet file."""
    if LEADERBOARD_PATH.exists():
        LEADERBOARD_PATH.unlink()
        log.info("Leaderboard cleared.")
