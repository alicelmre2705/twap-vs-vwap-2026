"""
Visual demo: execute one 60-minute window and plot the schedule.

Reads a single session from cleaned data, runs TWAP / forecast-VWAP /
realized-VWAP, and produces a multi-panel figure that shows:
  1. Bar-level volume with scheduled vs executed quantities
  2. Cumulative execution progress
  3. Price path with arrival-price reference

Usage (from repo root):
    python3 -m src.run_one_window          # uses AAPL first session
    python3 -m src.run_one_window NVDA     # override ticker
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .execution import simulate_twap_buy, simulate_vwap_buy

# ── Config ───────────────────────────────────────────────────────────
DEFAULT_TICKER = "AAPL"
Q_SHARES = 5000
MAX_PARTICIPATION = 0.05
BARS_PER_WINDOW = 12

PALETTE = {
    "twap": "#5B7BA5",
    "forecast": "#E07A52",
    "oracle": "#7FB685",
    "volume": "#C8D6E5",
    "price": "#2D2D2D",
    "arrival": "#C0392B",
    "bg": "#FAFAFA",
    "grid": "#E0E0E0",
    "text": "#2D2D2D",
}

OUT_DIR = "results/figures"


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.family": "sans-serif",
        "font.size": 9,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": PALETTE["grid"],
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _bar_level_execution(
    window: pd.DataFrame,
    Q: float,
    weights: np.ndarray,
    max_participation: float,
) -> dict:
    """Return per-bar executed quantities for each strategy."""
    n = len(window)
    target_twap = Q / n
    vols = window["volume"].astype(float).values
    total_vol = vols.sum()
    real_weights = vols / total_vol if total_vol > 0 else np.ones(n) / n

    result = {
        "twap_qty": np.zeros(n),
        "forecast_qty": np.zeros(n),
        "oracle_qty": np.zeros(n),
    }

    for label, w in [
        ("twap_qty", np.ones(n) / n),
        ("forecast_qty", weights / weights.sum()),
        ("oracle_qty", real_weights),
    ]:
        executed = 0.0
        for i, (_, row) in enumerate(window.iterrows()):
            vol = float(row["volume"])
            desired = Q * w[i]
            cap = max_participation * vol
            qty = min(desired, cap, Q - executed)
            result[label][i] = qty
            executed += qty
            if executed >= Q:
                break

    return result


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    path = f"data/clean/{ticker}_5m_60d_clean.csv"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        print("Run the pipeline first:  bash run_all.sh")
        return

    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert("America/New_York")
    df["session_date"] = df["datetime"].dt.date
    first_session = df.iloc[0]["session_date"]
    day = df[df["session_date"] == first_session].copy().reset_index(drop=True)

    if len(day) < BARS_PER_WINDOW:
        print(f"Not enough bars for {ticker} on {first_session}")
        return

    window = day.iloc[:BARS_PER_WINDOW].copy()
    times = window["datetime"].dt.strftime("%H:%M").values
    prices = window["close"].astype(float).values
    volumes = window["volume"].astype(float).values
    p0 = prices[0]

    # Simple ex-ante volume profile: slight U-shape
    toy_weights = np.array(
        [0.10, 0.09, 0.08, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.10, 0.10]
    )

    bar_exec = _bar_level_execution(
        window, Q_SHARES, toy_weights, MAX_PARTICIPATION
    )

    # Run full simulators for summary stats
    twap_res = simulate_twap_buy(
        window, Q=Q_SHARES, max_participation=MAX_PARTICIPATION
    )
    vwap_oracle_res = simulate_vwap_buy(
        window, Q=Q_SHARES, max_participation=MAX_PARTICIPATION
    )
    vwap_fcst_res = simulate_vwap_buy(
        window, Q=Q_SHARES, max_participation=MAX_PARTICIPATION,
        target_weights=toy_weights,
    )

    # ── Build figure ─────────────────────────────────────────────────
    _apply_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(
        4, 1, figsize=(9, 11), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1]},
    )
    x = np.arange(BARS_PER_WINDOW)

    # Panel 1: market volume + cap line
    ax1 = axes[0]
    cap_line = MAX_PARTICIPATION * volumes

    ax1.bar(x, volumes, color=PALETTE["volume"], edgecolor="white",
            linewidth=0.5, label="Market volume", zorder=2, width=0.7)
    ax1.step(x, cap_line, where="mid", color=PALETTE["arrival"],
             linewidth=1.0, linestyle=":", label=f"Cap ({MAX_PARTICIPATION:.0%} \u00d7 vol)",
             zorder=4)

    ax1.set_ylabel("Shares (market)")
    ax1.set_title(
        f"Single-window execution demo \u2014 {ticker}, "
        f"BUY {Q_SHARES:,.0f} shares, "
        f"\u03c1 = {MAX_PARTICIPATION:.0%}"
    )
    ax1.legend(loc="upper right", fontsize=7)

    # Panel 2: execution schedule (zoomed)
    ax_exec = axes[1]
    offset = 0.25
    w = 0.22
    ax_exec.bar(x - offset, bar_exec["twap_qty"], width=w,
                color=PALETTE["twap"], edgecolor="white", linewidth=0.3,
                label="TWAP sched.", zorder=3)
    ax_exec.bar(x, bar_exec["forecast_qty"], width=w,
                color=PALETTE["forecast"], edgecolor="white", linewidth=0.3,
                label="Forecast VWAP sched.", zorder=3)
    ax_exec.bar(x + offset, bar_exec["oracle_qty"], width=w,
                color=PALETTE["oracle"], edgecolor="white", linewidth=0.3,
                label="Oracle VWAP sched.", zorder=3)

    twap_target = Q_SHARES / BARS_PER_WINDOW
    ax_exec.axhline(twap_target, color=PALETTE["text"], linewidth=0.8,
                    linestyle="--", alpha=0.5,
                    label=f"TWAP target = {twap_target:,.0f}")

    ax_exec.set_ylabel("Shares (executed)")
    ax_exec.set_title("Execution schedule per bar")
    ax_exec.legend(loc="upper right", ncol=2, fontsize=7)

    # Panel 3: cumulative execution
    ax2 = axes[2]
    for label, key, color, ls in [
        ("TWAP", "twap_qty", PALETTE["twap"], "-"),
        ("Forecast VWAP", "forecast_qty", PALETTE["forecast"], "-"),
        ("Oracle VWAP", "oracle_qty", PALETTE["oracle"], "--"),
    ]:
        cum = np.cumsum(bar_exec[key])
        ax2.plot(x, cum, color=color, linewidth=1.6, linestyle=ls,
                 marker="o", markersize=3, label=label, zorder=3)

    ax2.axhline(Q_SHARES, color=PALETTE["text"], linewidth=0.6,
                linestyle="--", alpha=0.4, label=f"Target Q = {Q_SHARES:,.0f}")
    ax2.set_ylabel("Cumulative shares executed")
    ax2.set_title("Cumulative fill progress")
    ax2.legend(loc="lower right", fontsize=7)

    # Panel 4: price path
    ax3 = axes[3]
    ax3.plot(x, prices, color=PALETTE["price"], linewidth=1.4,
             marker="s", markersize=3, label="Bar close price", zorder=3)
    ax3.axhline(p0, color=PALETTE["arrival"], linewidth=1.0, linestyle="--",
                alpha=0.7, label=f"Arrival price = ${p0:.2f}")

    ax3.set_ylabel("Price (USD)")
    ax3.set_xlabel("Bar (5-minute intervals)")
    ax3.set_title("Price path over execution window")
    ax3.set_xticks(x)
    ax3.set_xticklabels(times, rotation=45, ha="right")
    ax3.legend(loc="best", fontsize=7)

    fig.tight_layout(h_pad=1.0)

    out_path = os.path.join(OUT_DIR, f"single_window_demo_{ticker}.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    # Console summary
    print(f"\n{'='*55}")
    print(f"  {ticker} | BUY {Q_SHARES:,} shares | cap = {MAX_PARTICIPATION:.0%}")
    print(f"{'='*55}")
    fmt = "  {:<22s}  IS={:+.4f}  fill={:.3f}"
    print(fmt.format("TWAP", twap_res["is_cost"], twap_res["fill_ratio"]))
    print(fmt.format("Forecast VWAP", vwap_fcst_res["is_cost"], vwap_fcst_res["fill_ratio"]))
    print(fmt.format("Oracle VWAP", vwap_oracle_res["is_cost"], vwap_oracle_res["fill_ratio"]))
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
