import pandas as pd
import pytest

from src.clean_data import clean_one
from src.execution import market_vwap
from src.run_experiment import add_session_columns, build_intraday_profile


def test_market_vwap_rejects_empty_window():
    with pytest.raises(ValueError):
        market_vwap(pd.DataFrame(columns=["close", "volume"]))


def test_add_session_columns_rejects_missing_columns():
    with pytest.raises(ValueError):
        add_session_columns(pd.DataFrame({"datetime": ["2024-01-01"]}))


def test_build_intraday_profile_rejects_zero_volume_data():
    df = pd.DataFrame(
        {
            "session_date": ["2024-01-02", "2024-01-02"],
            "bar_index_in_day": [0, 1],
            "volume": [0, 0],
        }
    )
    with pytest.raises(ValueError):
        build_intraday_profile(df)


def test_clean_one_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"datetime": ["2024-01-01T14:30:00Z"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        clean_one(str(path))
