from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from universe_selector.domain import Market
from universe_selector.providers.models import (
    FundamentalsCoverage,
    FundamentalsMetadata,
    FundamentalsUniverseRunData,
    ListingCandidate,
    ProviderDataRequirements,
    ProviderMetadata,
    ProviderRunData,
)


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
    assert provider_data.fundamentals is None
    assert metadata.fundamentals_provider_id is None
    assert metadata.fundamentals_requested_count is None


def test_provider_data_requirements_default_to_no_fundamentals() -> None:
    requirements = ProviderDataRequirements()

    assert requirements.fundamentals is False


def test_fundamentals_universe_run_data_owns_normalized_facts_and_coverage() -> None:
    facts = pl.DataFrame(
        {
            "market": ["US"],
            "ticker": ["AAA"],
            "currency": ["USD"],
            "fiscal_period_end": [date(2026, 3, 31)],
            "balance_sheet_as_of": [date(2026, 3, 31)],
            "fiscal_period_type": ["ttm"],
            "revenue_ttm": [100.0],
            "gross_profit_ttm": [60.0],
            "operating_income_ttm": [30.0],
            "net_income_ttm": [20.0],
            "total_assets": [200.0],
            "shareholders_equity": [100.0],
            "total_debt": [25.0],
            "cash_and_cash_equivalents": [10.0],
            "operating_cash_flow_ttm": [24.0],
            "capital_expenditures_ttm": [4.0],
            "free_cash_flow_ttm": [20.0],
            "roe": [0.20],
            "roa": [0.10],
            "gross_margin": [0.60],
            "operating_margin": [0.30],
            "net_margin": [0.20],
            "debt_to_equity": [0.25],
            "fcf_margin": [0.20],
            "tag_fundamentals_annual_fallback": [0.0],
            "tag_negative_net_income": [0.0],
            "tag_negative_fcf": [0.0],
        }
    )
    metadata = FundamentalsMetadata(
        data_mode="live",
        fundamentals_provider_id="unit_fundamentals",
        fundamentals_source_ids=("unit:fundamentals",),
        data_fetch_started_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
        latest_source_date=date(2026, 3, 31),
    )
    coverage = FundamentalsCoverage(
        requested_count=3,
        returned_count=1,
        missing_count=1,
        invalid_count=1,
    )

    data = FundamentalsUniverseRunData(metadata=metadata, facts=facts, coverage=coverage)

    assert data.metadata is metadata
    assert data.facts["ticker"].to_list() == ["AAA"]
    assert data.coverage.returned_count == 1
