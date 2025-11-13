# tests/test_price_store.py

import pytest
from datetime import datetime, timedelta, timezone
from bisect import bisect_left

from condorgame.prices import PriceStore, Asset, PriceEntry

# ---------------------------
# Helper function
# ---------------------------
def generate_price_series(start_ts: int, n: int, step_sec: int, start_price: float = 100.0):
    """
    Generate synthetic price data as a list of (timestamp, price) tuples.
    """
    series = []
    price = start_price
    ts = start_ts
    for _ in range(n):
        series.append((ts, price))
        price *= 1 + 0.01  # simple +1% per step
        ts += step_sec
    return series


# ---------------------------
# Tests
# ---------------------------
def test_add_and_get_last_price():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)
    
    store.add_prices("BTC", series)
    
    # Check last price
    last_ts, last_price = store.get_last_price("BTC")
    assert last_ts == series[-1][0]
    assert last_price == series[-1][1]


def test_get_closest_price():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)
    store.add_prices("BTC", series)

    # Exact timestamp
    closest_ts, closest_price = store.get_closest_price("BTC", series[2][0])
    assert closest_ts == series[2][0]
    assert closest_price == series[2][1]

    # Between timestamps
    mid_ts = (series[2][0] + series[3][0]) // 2
    closest_ts, closest_price = store.get_closest_price("BTC", mid_ts)
    # Should be closer to series[2]
    assert closest_ts == series[2][0]


def test_get_prices_with_resolution():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 10, 60)  # every 1 min
    store.add_prices("BTC", series)

    # Request prices every 2 minutes
    prices_resampled = store.get_prices("BTC", resolution=120)
    # Should skip every other point
    assert all(prices_resampled[i+1][0] - prices_resampled[i][0] >= 120 for i in range(len(prices_resampled)-1))


def test_add_bulk_and_deduplication():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())

    # Add first batch
    series1 = generate_price_series(ts, 5, 60)
    store.add_bulk({"BTC": series1})

    # Add overlapping batch with updated last price
    series2 = series1[-3:] + [(series1[-1][0] + 60, 200.0)]
    store.add_bulk({"BTC": series2})

    # Last price should be from series2
    last_ts, last_price = store.get_last_price("BTC")
    assert last_price == 200.0


# ---------------------------
# Run pytest directly
# ---------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
