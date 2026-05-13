from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.models import ListingCandidate, ProviderMetadata, ProviderRunData


def test_provider_models_are_owned_by_provider_package() -> None:
    listing = ListingCandidate(
        market=Market.US,
        ticker="AAA",
        listing_symbol="AAA",
        exchange_segment="NASDAQ",
        listing_status="active",
        instrument_type="common_stock",
        source_id="fixture",
    )
    metadata = ProviderMetadata(
        data_mode="fixture",
        listing_provider_id="fixture",
        listing_source_id="fixture",
        ohlcv_provider_id="fixture",
        ohlcv_source_id="fixture",
        provider_config_hash="provider-hash",
        data_fetch_started_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
        market_timezone="America/New_York",
        run_latest_bar_date=date(2026, 5, 8),
    )
    bars = pl.DataFrame(
        {
            "market": ["US"],
            "ticker": ["AAA"],
            "bar_date": [date(2026, 5, 8)],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "adjusted_close": [10.5],
            "volume": [100],
        }
    )

    provider_data = ProviderRunData(metadata=metadata, listings=[listing], bars=bars)

    assert not hasattr(metadata, "run_id")
    assert provider_data.metadata is metadata
    assert provider_data.listings == [listing]
    assert provider_data.bars.height == 1
