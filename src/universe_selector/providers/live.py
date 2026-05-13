from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import polars as pl

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.providers.base import ListingProvider, MarketDataProvider, OhlcvProvider
from universe_selector.providers.context import build_provider_run_context
from universe_selector.providers.models import ListingCandidate, ProviderMetadata, ProviderRunData
from universe_selector.providers.registration import ListingProviderRegistration, OhlcvProviderRegistration
from universe_selector.providers.registry import (
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
    ticker_latest_dates = cleaned_bars.group_by("ticker").agg(
        pl.col("bar_date").max().alias("ticker_latest_bar_date")
    )
    return ticker_latest_dates["ticker_latest_bar_date"].max()


class LiveMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        config: AppConfig,
        *,
        listing_registration_resolver: Callable[[str, Market], ListingProviderRegistration] | None = None,
        ohlcv_registration_resolver: Callable[[str], OhlcvProviderRegistration] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._listing_registration_resolver = listing_registration_resolver or get_listing_registration
        self._ohlcv_registration_resolver = ohlcv_registration_resolver or get_ohlcv_registration
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def load_run_data(self, market: Market) -> ProviderRunData:
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

        metadata = ProviderMetadata(
            data_mode="live",
            listing_provider_id=listing_adapter.provider_id,
            listing_source_id=_join_source_ids(listing_adapter.source_ids),
            ohlcv_provider_id=ohlcv_adapter.provider_id,
            ohlcv_source_id=_join_source_ids(ohlcv_adapter.source_ids),
            provider_config_hash=self.config.provider_config_hash(),
            data_fetch_started_at=context.data_fetch_started_at,
            market_timezone=context.market_timezone,
            run_latest_bar_date=latest_bar_date,
        )
        return ProviderRunData(metadata=metadata, listings=listings, bars=bars)
