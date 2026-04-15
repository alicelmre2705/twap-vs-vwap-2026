import numpy as np
import pandas as pd

from src.execution import market_vwap, simulate_twap_buy, simulate_vwap_buy


def _df(prices, volumes):
    n = len(prices)
    dt = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": volumes,
        }
    )


def test_market_vwap_constant_price():
    df = _df([100.0, 100.0, 100.0], [10, 20, 30])
    assert market_vwap(df) == 100.0


def test_fill_ratio_bounds():
    df = _df([100.0] * 12, [100] * 12)
    Q = 100.0
    max_part = 0.10

    tw = simulate_twap_buy(df, Q=Q, max_participation=max_part)
    vw = simulate_vwap_buy(df, Q=Q, max_participation=max_part)

    assert 0.0 <= tw["fill_ratio"] <= 1.0
    assert 0.0 <= vw["fill_ratio"] <= 1.0


def test_never_overfill():
    df = _df([100.0] * 12, [100] * 12)
    Q = 50.0
    max_part = 1.0

    tw = simulate_twap_buy(df, Q=Q, max_participation=max_part)
    vw = simulate_vwap_buy(df, Q=Q, max_participation=max_part)

    assert tw["executed"] <= Q + 1e-9
    assert vw["executed"] <= Q + 1e-9


def test_custom_weights_are_normalized_and_used():
    df = _df(np.linspace(100, 111, 12), [100] * 12)
    Q = 12.0
    res = simulate_vwap_buy(df, Q=Q, max_participation=1.0, target_weights=np.arange(1, 13))
    assert abs(res["executed"] - Q) < 1e-9
    assert res["avg_price"] > df["close"].mean()
