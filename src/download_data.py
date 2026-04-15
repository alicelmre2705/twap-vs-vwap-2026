from typing import Optional
import os

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "SPY"]
INTERVAL = "5m"
PERIOD = "60d"

OUT_DIR = "data/raw"


def flatten_columns(cols):
    """
    yfinance may return MultiIndex columns (tuples).
    Convert them into clean strings: ('Close', 'AAPL') -> 'close_aapl'.
    """
    flat = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if x not in (None, "", " ")]
            name = "_".join(parts).strip().lower().replace(" ", "_")
        else:
            name = str(c).strip().lower().replace(" ", "_")
        flat.append(name)
    return flat


def download_one_ticker(ticker: str) -> Optional[str]:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(
            tickers=ticker,
            period=PERIOD,
            interval=INTERVAL,
            auto_adjust=False,
            prepost=False,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to download data via yfinance. "
            "Please check your internet connection and try again."
        ) from e

    if df is None or df.empty:
        print(f"WARNING: no data for {ticker}")
        return None

    df = df.reset_index()
    df.columns = flatten_columns(df.columns)

    if "datetime" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        else:
            raise ValueError(
                f"Missing datetime column after reset_index for {ticker}. "
                f"Columns: {df.columns.tolist()}"
            )

    def pick(col_base: str) -> Optional[str]:
        if col_base in df.columns:
            return col_base
        cand = f"{col_base}_{ticker.lower()}"
        if cand in df.columns:
            return cand
        return None

    open_c = pick("open")
    high_c = pick("high")
    low_c = pick("low")
    close_c = pick("close")
    vol_c = pick("volume")

    needed = {"open": open_c, "high": high_c, "low": low_c, "close": close_c, "volume": vol_c}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise ValueError(f"Missing columns {missing} for {ticker}. Columns: {df.columns.tolist()}")

    out = df[["datetime", open_c, high_c, low_c, close_c, vol_c]].copy()
    out.columns = ["datetime", "open", "high", "low", "close", "volume"]

    out_path = os.path.join(OUT_DIR, f"{ticker}_{INTERVAL}_{PERIOD}.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved -> {out_path} | rows={len(out)}")
    return out_path


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ticker in TICKERS:
        download_one_ticker(ticker)


if __name__ == "__main__":
    main()
