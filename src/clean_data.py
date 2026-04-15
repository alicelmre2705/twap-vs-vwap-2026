import glob
import os
from datetime import time

import pandas as pd

RAW_DIR = "data/raw"
CLEAN_DIR = "data/clean"

TARGET_TZ = "America/New_York"
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
REQUIRED_COLUMNS = {"datetime", "volume"}


def clean_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")

    # Parse datetime with timezone (raw data is expected in UTC already)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Convert to New York time
    df["datetime"] = df["datetime"].dt.tz_convert(TARGET_TZ)

    # Sort + remove duplicate timestamps
    df = df.sort_values("datetime").drop_duplicates("datetime")

    # Keep only weekdays (Mon=0 ... Fri=4)
    df = df[df["datetime"].dt.dayofweek < 5]

    # Keep only regular trading hours
    # Using time objects avoids relying on string comparisons.
    open_t = time.fromisoformat(MARKET_OPEN)
    close_t = time.fromisoformat(MARKET_CLOSE)
    t = df["datetime"].dt.time
    df = df[(t >= open_t) & (t <= close_t)]

    # Remove zero/NaN volume rows
    df = df[df["volume"].fillna(0) > 0]

    return df.reset_index(drop=True)


def main() -> None:
    os.makedirs(CLEAN_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}")

    for path in paths:
        cleaned = clean_one(path)

        out_path = os.path.join(
            CLEAN_DIR,
            os.path.basename(path).replace(".csv", "_clean.csv"),
        )
        cleaned.to_csv(out_path, index=False)

        print(f"Cleaned -> {out_path} | rows={len(cleaned)}")


if __name__ == "__main__":
    main()
