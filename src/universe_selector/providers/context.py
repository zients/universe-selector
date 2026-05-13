from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Mapping
from zoneinfo import ZoneInfo

from universe_selector.domain import Market
from universe_selector.errors import ValidationError


@dataclass(frozen=True)
class MarketProfile:
    market: Market
    market_timezone: str


@dataclass(frozen=True)
class ProviderRunContext:
    data_fetch_started_at: datetime
    market_timezone: str
    market_fetch_date: date
    ticker_limit: int | None


_MARKET_PROFILES: Mapping[Market, MarketProfile] = {
    Market.TW: MarketProfile(market=Market.TW, market_timezone="Asia/Taipei"),
    Market.US: MarketProfile(market=Market.US, market_timezone="America/New_York"),
}


def market_profile_for(market: Market) -> MarketProfile:
    try:
        return _MARKET_PROFILES[market]
    except KeyError as exc:
        raise ValidationError(f"unsupported market profile: {market}") from exc


def build_provider_run_context(
    *,
    market: Market,
    data_fetch_started_at: datetime,
    ticker_limit: int | None,
) -> ProviderRunContext:
    if data_fetch_started_at.tzinfo is None or data_fetch_started_at.utcoffset() is None:
        raise ValidationError("data_fetch_started_at must be timezone-aware")

    profile = market_profile_for(market)
    utc_started_at = data_fetch_started_at.astimezone(timezone.utc)
    market_started_at = utc_started_at.astimezone(ZoneInfo(profile.market_timezone))
    return ProviderRunContext(
        data_fetch_started_at=utc_started_at,
        market_timezone=profile.market_timezone,
        market_fetch_date=market_started_at.date(),
        ticker_limit=ticker_limit,
    )
