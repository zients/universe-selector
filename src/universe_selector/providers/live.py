from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.base import FundamentalsProvider, ListingProvider, MarketDataProvider, OhlcvProvider
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.models import ProviderDataRequirements, ProviderMetadata, ProviderRunData
from universe_selector.providers.registration import (
    FundamentalsProviderRegistration,
    ListingProviderRegistration,
    OhlcvProviderRegistration,
)
from universe_selector.providers.registry import (
    get_fundamentals_registration,
    get_listing_registration,
    get_ohlcv_registration,
)


def _join_source_ids(source_ids: tuple[str, ...]) -> str:
    return "+".join(source_ids)


def _clean_ohlcv_bars(bars: pl.DataFrame) -> pl.DataFrame:
    if bars.is_empty() or "bar_date" not in bars.columns:
        raise ProviderDataError("OHLCV provider returned no usable bars")
    cleaned = bars.filter(pl.col("bar_date").is_not_null())
    if cleaned.is_empty():
        raise ProviderDataError("OHLCV provider returned no usable bars")
    return cleaned


def _run_latest_bar_date(cleaned_bars: pl.DataFrame):
    if "ticker" not in cleaned_bars.columns:
        return cleaned_bars["bar_date"].max()
    ticker_latest_dates = cleaned_bars.group_by("ticker").agg(pl.col("bar_date").max().alias("ticker_latest_bar_date"))
    return ticker_latest_dates["ticker_latest_bar_date"].max()


class LiveMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        config: AppConfig,
        *,
        listing_registration_resolver: Callable[[str, Market], ListingProviderRegistration] | None = None,
        ohlcv_registration_resolver: Callable[[str], OhlcvProviderRegistration] | None = None,
        fundamentals_registration_resolver: Callable[[str, Market], FundamentalsProviderRegistration] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._listing_registration_resolver = listing_registration_resolver or get_listing_registration
        self._ohlcv_registration_resolver = ohlcv_registration_resolver or get_ohlcv_registration
        self._fundamentals_registration_resolver = fundamentals_registration_resolver or get_fundamentals_registration
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def load_run_data(self, market: Market, requirements: ProviderDataRequirements | None = None) -> ProviderRunData:
        requirements = requirements or ProviderDataRequirements()
        started_at = self._clock()
        context = build_provider_run_context(
            market=market,
            data_fetch_started_at=started_at,
            ticker_limit=self.config.live_ticker_limit,
        )

        listing_registration = self._listing_registration_resolver(
            self.config.live_listing_provider[market],
            market,
        )
        ohlcv_registration = self._ohlcv_registration_resolver(self.config.live_ohlcv_provider)
        listing_adapter: ListingProvider = listing_registration.factory(self.config)
        ohlcv_adapter: OhlcvProvider = ohlcv_registration.factory(self.config)

        listings = sorted(listing_adapter.load_listings(context, market), key=lambda item: item.ticker)
        if context.ticker_limit is not None:
            listings = listings[: context.ticker_limit]
        if not listings:
            raise ProviderDataError("listing provider returned no usable listings")

        bars = _clean_ohlcv_bars(ohlcv_adapter.load_ohlcv(context, market, listings))
        latest_bar_date = _run_latest_bar_date(bars)
        if latest_bar_date is None:
            raise ProviderDataError("OHLCV provider returned no usable bars")

        fundamentals = None
        fundamentals_provider_id = None
        fundamentals_source_id = None
        fundamentals_latest_source_date = None
        fundamentals_source_risk_note = None
        fundamentals_field_mapping_note = None
        fundamentals_requested_count = None
        fundamentals_returned_count = None
        fundamentals_missing_count = None
        fundamentals_invalid_count = None
        if requirements.fundamentals:
            fundamentals_registration = self._fundamentals_registration_resolver(
                self.config.live_fundamentals_provider,
                market,
            )
            fundamentals_adapter: FundamentalsProvider = fundamentals_registration.factory()
            fundamentals = fundamentals_adapter.load_fundamentals_for_listings(context, market, listings)
            fundamentals_provider_id = fundamentals.metadata.fundamentals_provider_id
            fundamentals_source_id = _join_source_ids(fundamentals.metadata.fundamentals_source_ids)
            fundamentals_latest_source_date = fundamentals.metadata.latest_source_date
            fundamentals_source_risk_note = fundamentals.metadata.source_risk_note
            fundamentals_field_mapping_note = fundamentals.metadata.field_mapping_note
            fundamentals_requested_count = fundamentals.coverage.requested_count
            fundamentals_returned_count = fundamentals.coverage.returned_count
            fundamentals_missing_count = fundamentals.coverage.missing_count
            fundamentals_invalid_count = fundamentals.coverage.invalid_count

        metadata = ProviderMetadata(
            data_mode="live",
            listing_provider_id=listing_adapter.provider_id,
            listing_source_id=_join_source_ids(listing_adapter.source_ids),
            ohlcv_provider_id=ohlcv_adapter.provider_id,
            ohlcv_source_id=_join_source_ids(ohlcv_adapter.source_ids),
            provider_config_hash=self.config.provider_config_hash(requirements),
            data_fetch_started_at=context.data_fetch_started_at,
            market_timezone=context.market_timezone,
            run_latest_bar_date=latest_bar_date,
            fundamentals_provider_id=fundamentals_provider_id,
            fundamentals_source_id=fundamentals_source_id,
            fundamentals_latest_source_date=fundamentals_latest_source_date,
            fundamentals_source_risk_note=fundamentals_source_risk_note,
            fundamentals_field_mapping_note=fundamentals_field_mapping_note,
            fundamentals_requested_count=fundamentals_requested_count,
            fundamentals_returned_count=fundamentals_returned_count,
            fundamentals_missing_count=fundamentals_missing_count,
            fundamentals_invalid_count=fundamentals_invalid_count,
        )
        return ProviderRunData(metadata=metadata, listings=listings, bars=bars, fundamentals=fundamentals)
