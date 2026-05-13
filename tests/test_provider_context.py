from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.context import (
    build_provider_run_context,
    market_profile_for,
)


def test_market_profiles_define_market_timezones() -> None:
    assert market_profile_for(Market.TW).market_timezone == "Asia/Taipei"
    assert market_profile_for(Market.US).market_timezone == "America/New_York"


def test_unknown_markets_are_rejected_by_market_enum() -> None:
    with pytest.raises(ValueError):
        Market("JP")


def test_provider_run_context_derives_tw_market_fetch_date_from_utc_timestamp() -> None:
    context = build_provider_run_context(
        market=Market.TW,
        data_fetch_started_at=datetime(2026, 5, 3, 17, 30, tzinfo=timezone.utc),
        ticker_limit=25,
    )

    assert not hasattr(context, "run_id")
    assert context.data_fetch_started_at == datetime(2026, 5, 3, 17, 30, tzinfo=timezone.utc)
    assert context.market_timezone == "Asia/Taipei"
    assert context.market_fetch_date.isoformat() == "2026-05-04"
    assert context.ticker_limit == 25


def test_provider_run_context_derives_us_market_fetch_date_from_utc_timestamp() -> None:
    context = build_provider_run_context(
        market=Market.US,
        data_fetch_started_at=datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc),
        ticker_limit=None,
    )

    assert not hasattr(context, "run_id")
    assert context.data_fetch_started_at == datetime(2026, 5, 3, 13, 30, tzinfo=timezone.utc)
    assert context.market_timezone == "America/New_York"
    assert context.market_fetch_date.isoformat() == "2026-05-03"
    assert context.ticker_limit is None


def test_provider_run_context_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        build_provider_run_context(
            market=Market.TW,
            data_fetch_started_at=datetime(2026, 5, 3, 17, 30),
            ticker_limit=None,
        )


def test_provider_run_context_normalizes_non_utc_datetime_to_utc() -> None:
    context = build_provider_run_context(
        market=Market.TW,
        data_fetch_started_at=datetime(2026, 5, 4, 1, 30, tzinfo=ZoneInfo("Asia/Taipei")),
        ticker_limit=None,
    )

    assert not hasattr(context, "run_id")
    assert context.data_fetch_started_at == datetime(2026, 5, 3, 17, 30, tzinfo=timezone.utc)
    assert context.market_fetch_date.isoformat() == "2026-05-04"
