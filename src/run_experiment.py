import os

import numpy as np
import pandas as pd

from .execution import simulate_twap_buy, simulate_vwap_buy

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "SPY"]
BARS_PER_WINDOW = 12  # 12*5min = 60 minutes

N_WINDOWS_PER_TICKER = 200
ORDER_ALPHA = 0.03
MAX_PARTICIPATION = 0.05
RANDOM_SEED = 42

CLEAN_DIR = "data/clean"
OUT_TABLE = "results/tables/experiment_results.csv"
REQUIRED_COLUMNS = {"datetime", "close", "volume"}


def realized_vol_from_prices(prices: np.ndarray) -> float:
    prices = prices.astype(float)
    rets = np.diff(np.log(prices))
    if len(rets) == 0:
        return np.nan
    return float(np.sqrt(np.sum(rets**2)))


def coeff_var(x: np.ndarray) -> float:
    x = x.astype(float)
    m = x.mean()
    s = x.std(ddof=0)
    return float(s / m) if m > 0 else np.nan


def add_session_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_convert("America/New_York")
    out = out.sort_values("datetime").reset_index(drop=True)
    out["session_date"] = out["datetime"].dt.date
    out["bar_time"] = out["datetime"].dt.strftime("%H:%M")
    out["bar_index_in_day"] = out.groupby("session_date").cumcount()
    return out


def build_intraday_profile(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    day_total = tmp.groupby("session_date")["volume"].transform("sum")
    tmp = tmp[day_total > 0].copy()
    if tmp.empty:
        raise ValueError("Cannot build intraday profile from empty or zero-volume data")
    # day_total is aligned with tmp's index after filtering.
    tmp["vol_frac"] = tmp["volume"] / day_total
    profile = tmp.groupby("bar_index_in_day")["vol_frac"].mean()
    return profile / profile.sum()


def sample_session_windows(
    df: pd.DataFrame,
    n_windows: int,
    bars_per_window: int,
    rng: np.random.Generator,
):
    candidates = []
    for session_date, day_df in df.groupby("session_date", sort=True):
        if len(day_df) < bars_per_window:
            continue
        max_start = len(day_df) - bars_per_window
        day_idx = day_df.index.to_numpy()
        for i in range(max_start + 1):
            idx = day_idx[i : i + bars_per_window]
            candidates.append((session_date, idx))

    if not candidates:
        return []

    replace = len(candidates) < n_windows
    chosen = rng.choice(len(candidates), size=n_windows, replace=replace)
    return [candidates[int(i)] for i in chosen]


def load_clean_ticker_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(CLEAN_DIR, f"{ticker}_5m_60d_clean.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cleaned file for {ticker}: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Cleaned file is empty for {ticker}: {path}")
    return add_session_columns(df)


def main() -> None:
    os.makedirs("results/tables", exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    all_rows = []

    for ticker in TICKERS:
        df = load_clean_ticker_data(ticker)

        windows = sample_session_windows(df, N_WINDOWS_PER_TICKER, BARS_PER_WINDOW, rng)
        if not windows:
            raise ValueError(f"Not enough same-session data for {ticker}")

        full_profile = build_intraday_profile(df)

        for session_date, idx in windows:
            window = df.loc[idx].copy().reset_index(drop=True)

            v60 = float(window["volume"].sum())
            Q = ORDER_ALPHA * v60
            if Q <= 0:
                continue

            prices = window["close"].values
            vols = window["volume"].values
            rv = realized_vol_from_prices(prices)
            vcv = coeff_var(vols)

            start_bar = int(window.iloc[0]["bar_index_in_day"])
            hist_df = df[df["session_date"] < session_date]
            if hist_df["session_date"].nunique() >= 5:
                hist_profile = build_intraday_profile(hist_df)
            else:
                hist_profile = full_profile

            target_weights = hist_profile.reindex(range(start_bar, start_bar + BARS_PER_WINDOW))
            if target_weights.isna().any():
                target_weights = full_profile.reindex(range(start_bar, start_bar + BARS_PER_WINDOW))
            target_weights = target_weights.ffill().bfill()
            target_weights = target_weights.to_numpy(dtype=float)
            if not np.isfinite(target_weights).all() or target_weights.sum() <= 0:
                raise ValueError(f"Invalid target weights for {ticker} on session {session_date}")
            target_weights = target_weights / target_weights.sum()

            twap = simulate_twap_buy(window, Q=Q, max_participation=MAX_PARTICIPATION)
            vwap_realized = simulate_vwap_buy(window, Q=Q, max_participation=MAX_PARTICIPATION)
            vwap_forecast = simulate_vwap_buy(
                window,
                Q=Q,
                max_participation=MAX_PARTICIPATION,
                target_weights=target_weights,
            )

            row = {
                "ticker": ticker,
                "session_date": session_date,
                "start_time": window.iloc[0]["datetime"],
                "start_bar": start_bar,
                "Q_shares": Q,
                "v60": v60,
                "realized_vol": rv,
                "volume_cv": vcv,
                "twap_fill": twap["fill_ratio"],
                "vwap_realized_fill": vwap_realized["fill_ratio"],
                "vwap_forecast_fill": vwap_forecast["fill_ratio"],
                "twap_is": twap["is_cost"],
                "vwap_realized_is": vwap_realized["is_cost"],
                "vwap_forecast_is": vwap_forecast["is_cost"],
                "twap_close_slip": twap["close_slippage"],
                "vwap_realized_close_slip": vwap_realized["close_slippage"],
                "vwap_forecast_close_slip": vwap_forecast["close_slippage"],
                "market_vwap": twap["market_vwap"],
                "twap_slip_mvwap": twap["slippage_vs_market_vwap"],
                "vwap_realized_slip_mvwap": vwap_realized["slippage_vs_market_vwap"],
                "vwap_forecast_slip_mvwap": vwap_forecast["slippage_vs_market_vwap"],
                "twap_avg_price": twap["avg_price"],
                "vwap_realized_avg_price": vwap_realized["avg_price"],
                "vwap_forecast_avg_price": vwap_forecast["avg_price"],
                "p0": twap["p0"],
                "p_end": twap["p_end"],
            }
            all_rows.append(row)

    if not all_rows:
        raise ValueError("Experiment produced no rows")

    results = pd.DataFrame(all_rows)

    results["is_diff_realized_vwap_minus_twap"] = results["vwap_realized_is"] - results["twap_is"]
    results["is_diff_forecast_vwap_minus_twap"] = results["vwap_forecast_is"] - results["twap_is"]
    results["realized_vwap_better_is"] = results["is_diff_realized_vwap_minus_twap"] < 0
    results["forecast_vwap_better_is"] = results["is_diff_forecast_vwap_minus_twap"] < 0

    results["slip_diff_realized_vwap_minus_twap"] = results["vwap_realized_slip_mvwap"] - results["twap_slip_mvwap"]
    results["slip_diff_forecast_vwap_minus_twap"] = results["vwap_forecast_slip_mvwap"] - results["twap_slip_mvwap"]
    results["realized_vwap_better_mvwap"] = results["slip_diff_realized_vwap_minus_twap"] < 0
    results["forecast_vwap_better_mvwap"] = results["slip_diff_forecast_vwap_minus_twap"] < 0

    results["close_diff_realized_vwap_minus_twap"] = results["vwap_realized_close_slip"] - results["twap_close_slip"]
    results["close_diff_forecast_vwap_minus_twap"] = results["vwap_forecast_close_slip"] - results["twap_close_slip"]
    results["realized_vwap_better_close"] = results["close_diff_realized_vwap_minus_twap"] < 0
    results["forecast_vwap_better_close"] = results["close_diff_forecast_vwap_minus_twap"] < 0

    os.makedirs(os.path.dirname(OUT_TABLE), exist_ok=True)
    results.to_csv(OUT_TABLE, index=False)

    print("Saved results ->", OUT_TABLE)
    print(results.groupby("ticker")[["twap_is", "vwap_forecast_is", "twap_fill", "vwap_forecast_fill"]].mean())
    print("Forecast VWAP better rate vs IS (overall):", results["forecast_vwap_better_is"].mean())


if __name__ == "__main__":
    main()
