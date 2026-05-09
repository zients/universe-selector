from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

fixture_dir = Path("tests/fixtures/sample_basic")
fixture_dir.mkdir(parents=True, exist_ok=True)

(fixture_dir / "metadata.json").write_text(
    json.dumps(
        {
            "data_mode": "fixture",
            "run_latest_bar_date": "2026-04-24",
            "market_timezone": "UTC",
            "listing_provider_id": "fixture-listings-v1",
            "listing_source_id": "sample_basic/listings.csv",
            "ohlcv_provider_id": "fixture-ohlcv-v1",
            "ohlcv_source_id": "sample_basic/ohlcv.csv",
            "provider_config_hash": "fixture-sample-basic",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

listing_rows = [
    ["US", "AAA", "AAA", "NASDAQ", "active", "common_stock", "fixture:AAA"],
    ["US", "BBB", "BBB", "NASDAQ", "active", "common_stock", "fixture:BBB"],
    ["US", "CCC", "CCC", "NYSE", "active", "common_stock", "fixture:CCC"],
    ["US", "LOWVOL", "LOWVOL", "NYSE", "active", "common_stock", "fixture:LOWVOL"],
    ["US", "SHORT", "SHORT", "NASDAQ", "active", "common_stock", "fixture:SHORT"],
    ["TW", "2317", "2317", "TWSE", "active", "common_stock", "fixture:2317"],
    ["TW", "2330", "2330", "TWSE", "active", "common_stock", "fixture:2330"],
    ["TW", "2454", "2454", "TWSE", "active", "common_stock", "fixture:2454"],
    ["TW", "TLOWVOL", "TLOWVOL", "TWSE", "active", "common_stock", "fixture:TLOWVOL"],
    ["TW", "TSHORT", "TSHORT", "TWSE", "active", "common_stock", "fixture:TSHORT"],
]
with (fixture_dir / "listings.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        ["market", "ticker", "listing_symbol", "exchange_segment", "listing_status", "instrument_type", "source_id"]
    )
    writer.writerows(listing_rows)

snapshot = date(2026, 4, 24)
dates = [snapshot - timedelta(days=273 - index) for index in range(274)]


def close_series(start: float, daily_step: float, amplitude: float) -> list[float]:
    return [round(start + daily_step * index + amplitude * math.sin(index / 4.0), 6) for index in range(274)]


series_by_market: dict[str, dict[str, list[float]]] = {
    "US": {
        "AAA": close_series(12.0, 0.060, 0.35),
        "BBB": close_series(14.0, 0.030, 0.30),
        "CCC": close_series(14.0, 0.030, 0.30),
        "LOWVOL": close_series(20.0, 0.00001, 0.00001),
        "SHORT": close_series(15.0, 0.040, 0.25)[:-1],
    },
    "TW": {
        "2317": close_series(50.0, 0.120, 0.70),
        "2330": close_series(90.0, 0.220, 0.90),
        "2454": close_series(70.0, 0.080, 0.60),
        "TLOWVOL": close_series(80.0, 0.00001, 0.00001),
        "TSHORT": close_series(60.0, 0.090, 0.40)[:-1],
    },
}

with (fixture_dir / "ohlcv.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["market", "ticker", "bar_date", "open", "high", "low", "close", "adjusted_close", "volume"])
    for market, series_by_ticker in series_by_market.items():
        for ticker, closes in series_by_ticker.items():
            ticker_dates = dates[-len(closes):]
            for bar_date, close in zip(ticker_dates, closes, strict=True):
                writer.writerow(
                    [
                        market,
                        ticker,
                        bar_date.isoformat(),
                        round(close * 0.995, 6),
                        round(close * 1.01, 6),
                        round(close * 0.99, 6),
                        close,
                        close,
                        2_000_000,
                    ]
                )
