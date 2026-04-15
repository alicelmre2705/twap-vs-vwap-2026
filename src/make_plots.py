"""
Summarise experiment results and generate publication-quality figures.

Reads results/tables/experiment_results.csv (or the bundled sample) and
produces summary tables and figures under results/.
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────
RESULTS_PATH = "results/tables/experiment_results.csv"
SAMPLE_FALLBACK = "results/tables/experiment_results_sample.csv"
OUT_FIG_DIR = "results/figures"
OUT_TAB_DIR = "results/tables"
REQUIRED_COLUMNS = {"ticker", "v60", "twap_is", "twap_fill"}

# ── Plot style ───────────────────────────────────────────────────────
PALETTE: Dict[str, str] = {
    "twap": "#5B7BA5",
    "forecast": "#E07A52",
    "oracle": "#7FB685",
    "neutral": "#8C8C8C",
    "bg": "#FAFAFA",
    "grid": "#E0E0E0",
    "text": "#2D2D2D",
    "accent_neg": "#3A7D44",
    "accent_pos": "#C0392B",
}

TICKER_ORDER = ["AAPL", "AMZN", "MSFT", "NVDA", "SPY"]


def _apply_style() -> None:
    """Apply a clean, professional matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "font.family": "sans-serif",
        "font.size": 10,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": PALETTE["grid"],
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT_FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")


# ── Data loading (unchanged logic) ───────────────────────────────────
def load_results() -> pd.DataFrame:
    path = RESULTS_PATH if os.path.exists(RESULTS_PATH) else SAMPLE_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No experiment results found. Run src/run_experiment.py first."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(
            f"Results file is missing required columns: {sorted(missing)}"
        )

    # Backward-compatible column aliases (sample file)
    if "vwap_forecast_is" not in df.columns and "vwap_is" in df.columns:
        df["vwap_forecast_is"] = df["vwap_is"]
    if "vwap_forecast_fill" not in df.columns and "vwap_fill" in df.columns:
        df["vwap_forecast_fill"] = df["vwap_fill"]
    if (
        "vwap_forecast_slip_mvwap" not in df.columns
        and "vwap_slip_mvwap" in df.columns
    ):
        df["vwap_forecast_slip_mvwap"] = df["vwap_slip_mvwap"]
    if df.empty:
        raise ValueError("Results file is empty")

    df["start_time"] = pd.to_datetime(df.get("start_time"), utc=True, errors="coerce")
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived columns if missing (backward compat)."""
    num_cols = [
        "v60", "realized_vol", "volume_cv",
        "twap_fill", "vwap_realized_fill", "vwap_forecast_fill",
        "twap_is", "vwap_realized_is", "vwap_forecast_is",
        "market_vwap", "twap_slip_mvwap",
        "vwap_realized_slip_mvwap", "vwap_forecast_slip_mvwap",
        "is_diff_realized_vwap_minus_twap",
        "is_diff_forecast_vwap_minus_twap",
        "slip_diff_realized_vwap_minus_twap",
        "slip_diff_forecast_vwap_minus_twap",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "forecast_vwap_better_is" not in df.columns:
        if {"vwap_forecast_is", "twap_is"}.issubset(df.columns):
            df["forecast_vwap_better_is"] = (
                df["vwap_forecast_is"] < df["twap_is"]
            )
    if "realized_vwap_better_is" not in df.columns:
        if {"vwap_realized_is", "twap_is"}.issubset(df.columns):
            df["realized_vwap_better_is"] = (
                df["vwap_realized_is"] < df["twap_is"]
            )

    if "is_diff_forecast_vwap_minus_twap" not in df.columns:
        if "is_diff_vwap_minus_twap" in df.columns:
            df["is_diff_forecast_vwap_minus_twap"] = df[
                "is_diff_vwap_minus_twap"
            ]
        elif {"vwap_forecast_is", "twap_is"}.issubset(df.columns):
            df["is_diff_forecast_vwap_minus_twap"] = (
                df["vwap_forecast_is"] - df["twap_is"]
            )

    if "slip_diff_forecast_vwap_minus_twap" not in df.columns:
        if "slip_diff_vwap_minus_twap" in df.columns:
            df["slip_diff_forecast_vwap_minus_twap"] = df[
                "slip_diff_vwap_minus_twap"
            ]
        elif {"vwap_forecast_slip_mvwap", "twap_slip_mvwap"}.issubset(
            df.columns
        ):
            df["slip_diff_forecast_vwap_minus_twap"] = (
                df["vwap_forecast_slip_mvwap"] - df["twap_slip_mvwap"]
            )

    return df


# ── Figure 1: delta-IS histogram ────────────────────────────────────
def plot_delta_is_histogram(df: pd.DataFrame) -> None:
    if "is_diff_forecast_vwap_minus_twap" not in df.columns:
        return
    delta = df["is_diff_forecast_vwap_minus_twap"].dropna()
    if delta.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    counts, bin_edges, patches = ax.hist(
        delta, bins=60, edgecolor="white", linewidth=0.3
    )
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        patch.set_facecolor(
            PALETTE["accent_neg"] if left_edge < 0 else PALETTE["accent_pos"]
        )
        patch.set_alpha(0.82)

    ax.axvline(
        0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.6
    )
    ax.axvline(
        delta.mean(),
        color=PALETTE["forecast"],
        linewidth=1.4,
        linestyle="-",
        label=f"mean = {delta.mean():.4f}",
    )

    ax.set_xlabel(
        "\u0394IS = Forecast VWAP \u2212 TWAP  (USD / share)"
    )
    ax.set_ylabel("Count")
    ax.set_title(
        "Implementation shortfall difference: Forecast VWAP vs TWAP"
    )
    ax.legend(frameon=True)

    pct_neg = (delta < 0).mean() * 100
    ax.annotate(
        f"Forecast VWAP wins\n{pct_neg:.1f} % of windows",
        xy=(delta.quantile(0.05), counts.max() * 0.85),
        fontsize=8.5,
        color=PALETTE["accent_neg"],
        fontweight="bold",
        ha="center",
    )

    _save(fig, "hist_delta_is_forecast_vwap_vs_twap.png")


# ── Figure 2: better-rate by ticker ─────────────────────────────────
def plot_better_rate_by_ticker(df: pd.DataFrame) -> None:
    summary = df.groupby("ticker").agg(
        n=("ticker", "count"),
        twap_is_mean=("twap_is", "mean"),
        forecast_vwap_is_mean=("vwap_forecast_is", "mean"),
        twap_fill_mean=("twap_fill", "mean"),
        forecast_vwap_fill_mean=("vwap_forecast_fill", "mean"),
        forecast_vwap_better_is_rate=("forecast_vwap_better_is", "mean"),
    )
    summary.to_csv(os.path.join(OUT_TAB_DIR, "summary_by_ticker_is.csv"))
    if summary.empty:
        return

    tickers = [t for t in TICKER_ORDER if t in summary.index]
    rates = summary.loc[tickers, "forecast_vwap_better_is_rate"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        tickers,
        rates,
        color=PALETTE["forecast"],
        edgecolor="white",
        linewidth=0.5,
        width=0.55,
        zorder=3,
    )

    for bar, val in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PALETTE["text"],
        )

    ax.axhline(
        0.5,
        color=PALETTE["neutral"],
        linewidth=0.8,
        linestyle="--",
        alpha=0.6,
        label="50 % baseline",
    )
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Win rate (Forecast VWAP < TWAP IS)")
    ax.set_title(
        "Forecast VWAP beat rate by ticker (implementation shortfall)"
    )
    ax.legend(loc="upper right", frameon=True)

    _save(fig, "forecast_vwap_better_rate_by_ticker_is.png")


# ── Figure 3: regime heatmap ────────────────────────────────────────
def plot_regime_analysis(df: pd.DataFrame) -> None:
    if "realized_vol" not in df.columns or df["realized_vol"].isna().all():
        df["realized_vol"] = np.nan

    vol_med = df["v60"].median()
    rv_med = df["realized_vol"].median()
    df["regime_volume"] = np.where(
        df["v60"] >= vol_med, "High volume", "Low volume"
    )
    df["regime_volatility"] = np.where(
        df["realized_vol"] >= rv_med, "High vol", "Low vol"
    )

    regime = df.groupby(["regime_volume", "regime_volatility"]).agg(
        n=("ticker", "count"),
        delta_is_mean=("is_diff_forecast_vwap_minus_twap", "mean"),
        forecast_vwap_better_is_rate=("forecast_vwap_better_is", "mean"),
        twap_fill_mean=("twap_fill", "mean"),
        forecast_vwap_fill_mean=("vwap_forecast_fill", "mean"),
    )
    regime.to_csv(os.path.join(OUT_TAB_DIR, "summary_by_regime_is.csv"))
    if regime.empty:
        return

    vol_labels = ["Low volume", "High volume"]
    rv_labels = ["Low vol", "High vol"]
    matrix = np.full((2, 2), np.nan)
    rate_matrix = np.full((2, 2), np.nan)

    for i, vl in enumerate(vol_labels):
        for j, rl in enumerate(rv_labels):
            if (vl, rl) in regime.index:
                matrix[i, j] = regime.loc[(vl, rl), "delta_is_mean"]
                rate_matrix[i, j] = regime.loc[
                    (vl, rl), "forecast_vwap_better_is_rate"
                ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    im1 = ax1.imshow(
        matrix, cmap="RdYlGn_r", aspect="auto", vmin=-vmax, vmax=vmax
    )
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(rv_labels)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(vol_labels)
    ax1.set_title("Mean \u0394IS (Forecast \u2212 TWAP)")
    ax1.set_xlabel("Realized volatility regime")
    ax1.set_ylabel("Volume regime")
    for i in range(2):
        for j in range(2):
            if not np.isnan(matrix[i, j]):
                ax1.text(
                    j, i, f"{matrix[i, j]:+.4f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=(
                        "white"
                        if abs(matrix[i, j]) > vmax * 0.5
                        else PALETTE["text"]
                    ),
                )
    fig.colorbar(im1, ax=ax1, shrink=0.8, label="USD / share")

    im2 = ax2.imshow(
        rate_matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=0.6
    )
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(rv_labels)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(vol_labels)
    ax2.set_title("Win rate (Forecast VWAP < TWAP)")
    ax2.set_xlabel("Realized volatility regime")
    ax2.set_ylabel("Volume regime")
    for i in range(2):
        for j in range(2):
            if not np.isnan(rate_matrix[i, j]):
                ax2.text(
                    j, i, f"{rate_matrix[i, j]:.1%}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color=(
                        "white"
                        if abs(rate_matrix[i, j] - 0.5) > 0.05
                        else PALETTE["text"]
                    ),
                )
    fig.colorbar(
        im2, ax=ax2, shrink=0.8, label="Win rate",
        format=mtick.PercentFormatter(xmax=1.0),
    )

    fig.suptitle(
        "Regime analysis: volume \u00d7 realized volatility",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "regime_heatmap.png")


# ── Figure 4: IS box plots by ticker ────────────────────────────────
def plot_is_boxplot_by_ticker(df: pd.DataFrame) -> None:
    tickers = [t for t in TICKER_ORDER if t in df["ticker"].unique()]
    if not tickers:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    data_twap = [
        df.loc[df["ticker"] == t, "twap_is"].dropna().values for t in tickers
    ]
    data_fcst = [
        df.loc[df["ticker"] == t, "vwap_forecast_is"].dropna().values
        for t in tickers
    ]

    positions_twap = np.arange(len(tickers)) - 0.18
    positions_fcst = np.arange(len(tickers)) + 0.18
    width = 0.32

    bp1 = ax.boxplot(
        data_twap, positions=positions_twap, widths=width,
        patch_artist=True, showfliers=False, showmeans=True,
        meanprops=dict(
            marker="D", markersize=4,
            markerfacecolor="white", markeredgecolor=PALETTE["twap"],
        ),
        medianprops=dict(color="white", linewidth=1.2),
    )
    bp2 = ax.boxplot(
        data_fcst, positions=positions_fcst, widths=width,
        patch_artist=True, showfliers=False, showmeans=True,
        meanprops=dict(
            marker="D", markersize=4,
            markerfacecolor="white", markeredgecolor=PALETTE["forecast"],
        ),
        medianprops=dict(color="white", linewidth=1.2),
    )

    for patch in bp1["boxes"]:
        patch.set_facecolor(PALETTE["twap"])
        patch.set_alpha(0.75)
    for patch in bp2["boxes"]:
        patch.set_facecolor(PALETTE["forecast"])
        patch.set_alpha(0.75)

    ax.axhline(
        0, color=PALETTE["neutral"], linewidth=0.8, linestyle="--", alpha=0.5
    )
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Implementation shortfall (USD / share)")
    ax.set_title("IS distribution by ticker")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=PALETTE["twap"], alpha=0.75, label="TWAP"),
            Patch(
                facecolor=PALETTE["forecast"], alpha=0.75,
                label="Forecast VWAP",
            ),
        ],
        loc="upper left", frameon=True,
    )

    _save(fig, "is_boxplot_by_ticker.png")


# ── Figure 5: mean IS grouped bar chart ─────────────────────────────
def plot_mean_is_comparison(df: pd.DataFrame) -> None:
    tickers = [t for t in TICKER_ORDER if t in df["ticker"].unique()]
    if not tickers:
        return

    means = df.groupby("ticker")[["twap_is", "vwap_forecast_is"]].mean()
    if "vwap_realized_is" in df.columns:
        means["vwap_realized_is"] = df.groupby("ticker")[
            "vwap_realized_is"
        ].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tickers))
    w = 0.22

    ax.bar(
        x - w, means.loc[tickers, "twap_is"], width=w, label="TWAP",
        color=PALETTE["twap"], edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax.bar(
        x, means.loc[tickers, "vwap_forecast_is"], width=w,
        label="Forecast VWAP",
        color=PALETTE["forecast"], edgecolor="white", linewidth=0.5,
        zorder=3,
    )
    if "vwap_realized_is" in means.columns:
        ax.bar(
            x + w, means.loc[tickers, "vwap_realized_is"], width=w,
            label="Realized VWAP (oracle)",
            color=PALETTE["oracle"], edgecolor="white", linewidth=0.5,
            zorder=3,
        )

    ax.axhline(0, color=PALETTE["text"], linewidth=0.6, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Mean IS (USD / share)")
    ax.set_title("Mean implementation shortfall by ticker and strategy")
    ax.legend(frameon=True)

    _save(fig, "mean_is_by_ticker.png")


# ── Figure 6: fill ratio by ticker ──────────────────────────────────
def plot_fill_comparison(df: pd.DataFrame) -> None:
    tickers = [t for t in TICKER_ORDER if t in df["ticker"].unique()]
    if not tickers:
        return

    means = df.groupby("ticker")[["twap_fill", "vwap_forecast_fill"]].mean()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(tickers))
    w = 0.28

    ax.bar(
        x - w / 2, means.loc[tickers, "twap_fill"], width=w, label="TWAP",
        color=PALETTE["twap"], edgecolor="white", linewidth=0.5, zorder=3,
    )
    ax.bar(
        x + w / 2, means.loc[tickers, "vwap_forecast_fill"], width=w,
        label="Forecast VWAP",
        color=PALETTE["forecast"], edgecolor="white", linewidth=0.5,
        zorder=3,
    )

    ax.set_ylim(0.88, 1.005)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Mean fill ratio")
    ax.set_title("Fill ratio by ticker (participation-cap induced underfill)")
    ax.legend(frameon=True, loc="lower left")

    _save(fig, "fill_ratio_by_ticker.png")


# ── Figure 7: scatter IS vs realized vol ─────────────────────────────
def plot_scatter_is_vs_vol(df: pd.DataFrame) -> None:
    if "realized_vol" not in df.columns:
        return
    col = "is_diff_forecast_vwap_minus_twap"
    if col not in df.columns:
        return

    sub = df[["realized_vol", col, "ticker"]].dropna()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ticker in TICKER_ORDER:
        t_data = sub[sub["ticker"] == ticker]
        if t_data.empty:
            continue
        ax.scatter(
            t_data["realized_vol"], t_data[col],
            s=12, alpha=0.45, label=ticker, zorder=3,
        )

    ax.axhline(
        0, color=PALETTE["text"], linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax.set_xlabel("Realized volatility (window)")
    ax.set_ylabel(
        "\u0394IS = Forecast VWAP \u2212 TWAP (USD / share)"
    )
    ax.set_title("\u0394IS vs window realized volatility")
    ax.legend(frameon=True, markerscale=2, fontsize=8)

    _save(fig, "scatter_delta_is_vs_realized_vol.png")


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_TAB_DIR, exist_ok=True)

    _apply_style()

    df = load_results()
    df = _ensure_columns(df)

    print("Generating figures...")
    plot_delta_is_histogram(df)
    plot_better_rate_by_ticker(df)
    plot_regime_analysis(df)
    plot_is_boxplot_by_ticker(df)
    plot_mean_is_comparison(df)
    plot_fill_comparison(df)
    plot_scatter_is_vs_vol(df)

    print(f"\nSaved figures to: {OUT_FIG_DIR}")
    print(f"Saved tables to:  {OUT_TAB_DIR}")

    summary = pd.read_csv(
        os.path.join(OUT_TAB_DIR, "summary_by_ticker_is.csv"), index_col=0
    )
    regime = pd.read_csv(
        os.path.join(OUT_TAB_DIR, "summary_by_regime_is.csv"), index_col=[0, 1]
    )

    print("\n=== Summary by ticker (implementation shortfall) ===")
    print(summary.to_string())
    print("\n=== Summary by regime (implementation shortfall) ===")
    print(regime.to_string())


if __name__ == "__main__":
    main()
