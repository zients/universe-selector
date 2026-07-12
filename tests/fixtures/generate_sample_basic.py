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
            ticker_dates = dates[-len(closes) :]
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

fundamentals_rows = [
    [
        "US",
        "AAA",
        "USD",
        "2026-03-31",
        "2026-03-31",
        "ttm",
        100.0,
        60.0,
        30.0,
        20.0,
        200.0,
        100.0,
        50.0,
        10.0,
        25.0,
        5.0,
        20.0,
        0.20,
        0.10,
        0.60,
        0.30,
        0.20,
        0.50,
        0.20,
        0.0,
        0.0,
        0.0,
    ],
    [
        "US",
        "BBB",
        "USD",
        "2026-03-31",
        "2026-03-31",
        "annual",
        120.0,
        66.0,
        24.0,
        12.0,
        240.0,
        120.0,
        96.0,
        12.0,
        18.0,
        6.0,
        12.0,
        0.10,
        0.05,
        0.55,
        0.20,
        0.10,
        0.80,
        0.10,
        1.0,
        0.0,
        0.0,
    ],
    [
        "US",
        "CCC",
        "USD",
        "2024-01-01",
        "2024-01-01",
        "ttm",
        100.0,
        55.0,
        15.0,
        8.0,
        200.0,
        100.0,
        40.0,
        10.0,
        12.0,
        4.0,
        8.0,
        0.08,
        0.04,
        0.55,
        0.15,
        0.08,
        0.40,
        0.08,
        0.0,
        0.0,
        0.0,
    ],
    [
        "TW",
        "2317",
        "TWD",
        "2026-03-31",
        "2026-03-31",
        "ttm",
        300.0,
        90.0,
        45.0,
        30.0,
        600.0,
        300.0,
        120.0,
        30.0,
        42.0,
        12.0,
        30.0,
        0.10,
        0.05,
        0.30,
        0.15,
        0.10,
        0.40,
        0.10,
        0.0,
        0.0,
        0.0,
    ],
    [
        "TW",
        "2330",
        "TWD",
        "2026-03-31",
        "2026-03-31",
        "ttm",
        500.0,
        250.0,
        150.0,
        100.0,
        1000.0,
        500.0,
        100.0,
        100.0,
        130.0,
        30.0,
        100.0,
        0.20,
        0.10,
        0.50,
        0.30,
        0.20,
        0.20,
        0.20,
        0.0,
        0.0,
        0.0,
    ],
    [
        "TW",
        "2454",
        "TWD",
        "2024-01-01",
        "2024-01-01",
        "ttm",
        200.0,
        80.0,
        30.0,
        20.0,
        400.0,
        200.0,
        80.0,
        20.0,
        28.0,
        8.0,
        20.0,
        0.10,
        0.05,
        0.40,
        0.15,
        0.10,
        0.40,
        0.10,
        0.0,
        0.0,
        0.0,
    ],
]
with (fixture_dir / "fundamentals.csv").open("w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(
        [
            "market",
            "ticker",
            "currency",
            "fiscal_period_end",
            "balance_sheet_as_of",
            "fiscal_period_type",
            "revenue_ttm",
            "gross_profit_ttm",
            "operating_income_ttm",
            "net_income_ttm",
            "total_assets",
            "shareholders_equity",
            "total_debt",
            "cash_and_cash_equivalents",
            "operating_cash_flow_ttm",
            "capital_expenditures_ttm",
            "free_cash_flow_ttm",
            "roe",
            "roa",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "debt_to_equity",
            "fcf_margin",
            "tag_fundamentals_annual_fallback",
            "tag_negative_net_income",
            "tag_negative_fcf",
        ]
    )
    writer.writerows(fundamentals_rows)
