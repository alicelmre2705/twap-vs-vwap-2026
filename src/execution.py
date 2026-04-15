from typing import Optional
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"close", "volume"}


def _validate_window(df_window: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df_window.columns)
    if missing:
        raise ValueError(f"df_window is missing required columns: {sorted(missing)}")
    if df_window.empty:
        raise ValueError("df_window must not be empty")


def market_vwap(df_window: pd.DataFrame) -> float:
    """Market VWAP over the window: sum(price*volume)/sum(volume)."""
    _validate_window(df_window)
    vol = df_window["volume"].astype(float).values
    px = df_window["close"].astype(float).values
    v = vol.sum()
    return float((px * vol).sum() / v) if v > 0 else np.nan


def _finalize_execution(df_window: pd.DataFrame, executed: float, notional_paid: float, Q: float):
    avg_price = notional_paid / executed if executed > 0 else np.nan
    p0 = float(df_window.iloc[0]["close"])
    p_end = float(df_window.iloc[-1]["close"])
    is_cost = avg_price - p0 if executed > 0 else np.nan
    close_slippage = avg_price - p_end if executed > 0 else np.nan

    mvwap = market_vwap(df_window)
    slip_vs_mvwap = avg_price - mvwap if executed > 0 else np.nan

    return {
        "executed": executed,
        "fill_ratio": executed / Q if Q > 0 else np.nan,
        "avg_price": avg_price,
        "p0": p0,
        "p_end": p_end,
        "is_cost": is_cost,
        "close_slippage": close_slippage,
        "market_vwap": mvwap,
        "slippage_vs_market_vwap": slip_vs_mvwap,
    }


def simulate_twap_buy(df_window: pd.DataFrame, Q: float, max_participation: float = 0.05):
    """
    BUY TWAP execution:
    - execute equal target size each bar
    - cap by max_participation * market volume per bar
    - execute at bar close price
    Returns:
      - implementation shortfall vs initial price (IS)
      - slippage vs market VWAP
      - slippage vs end-of-window close
      - fill ratio
    """
    _validate_window(df_window)
    n = len(df_window)
    target_per_step = Q / n

    executed = 0.0
    notional_paid = 0.0

    for _, row in df_window.iterrows():
        vol = float(row["volume"])
        px = float(row["close"])

        cap = max_participation * vol
        qty = min(target_per_step, cap, Q - executed)

        executed += qty
        notional_paid += qty * px

        if executed >= Q:
            break

    return _finalize_execution(df_window, executed, notional_paid, Q)


def simulate_vwap_buy(
    df_window: pd.DataFrame,
    Q: float,
    max_participation: float = 0.05,
    target_weights: Optional[np.ndarray] = None,
):
    """
    BUY VWAP execution:
    - allocate proportionally to volume forecast weights inside the window
    - cap by max_participation * market volume per bar
    - execute at bar close price

    If target_weights is None, falls back to in-sample realized-volume VWAP.
    For research experiments, prefer passing ex-ante target_weights.
    """
    _validate_window(df_window)
    vols = df_window["volume"].astype(float).values
    total_vol = float(vols.sum())

    if target_weights is None:
        weights = vols / total_vol if total_vol > 0 else np.repeat(1.0 / len(df_window), len(df_window))
    else:
        weights = np.asarray(target_weights, dtype=float)
        if len(weights) != len(df_window):
            raise ValueError("target_weights length must match df_window length")
        weights = np.clip(weights, 0.0, None)
        wsum = weights.sum()
        if wsum <= 0:
            weights = np.repeat(1.0 / len(df_window), len(df_window))
        else:
            weights = weights / wsum

    executed = 0.0
    notional_paid = 0.0

    for weight, row in zip(weights, df_window.itertuples(index=False)):
        vol = float(row.volume)
        px = float(row.close)

        desired = Q * float(weight)
        cap = max_participation * vol
        qty = min(desired, cap, Q - executed)

        executed += qty
        notional_paid += qty * px

        if executed >= Q:
            break

    return _finalize_execution(df_window, executed, notional_paid, Q)
