from __future__ import annotations

from collections.abc import Mapping

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.nasdaq_trader import NASDAQ_TRADER_LISTING_REGISTRATION
from universe_selector.providers.registration import (
    FundamentalsProviderRegistration as _FundamentalsProviderRegistration,
    ListingProviderRegistration as _ListingProviderRegistration,
    OhlcvProviderRegistration as _OhlcvProviderRegistration,
    build_fundamentals_provider_registration_map as _build_fundamentals_provider_registration_map,
    build_listing_provider_registration_map as _build_listing_provider_registration_map,
    build_ohlcv_provider_registration_map as _build_ohlcv_provider_registration_map,
)
from universe_selector.providers.twse_isin import TWSE_ISIN_LISTING_REGISTRATION
from universe_selector.providers.yfinance_fundamentals import YFINANCE_FUNDAMENTALS_REGISTRATION
from universe_selector.providers.yfinance_ohlcv import YFINANCE_OHLCV_REGISTRATION


_FUNDAMENTALS_PROVIDER_REGISTRATIONS: tuple[_FundamentalsProviderRegistration, ...] = (
    YFINANCE_FUNDAMENTALS_REGISTRATION,
)
_LISTING_PROVIDER_REGISTRATIONS: tuple[_ListingProviderRegistration, ...] = (
    NASDAQ_TRADER_LISTING_REGISTRATION,
    TWSE_ISIN_LISTING_REGISTRATION,
)
_OHLCV_PROVIDER_REGISTRATIONS: tuple[_OhlcvProviderRegistration, ...] = (
    YFINANCE_OHLCV_REGISTRATION,
)

_FUNDAMENTALS_PROVIDER_REGISTRY: Mapping[str, _FundamentalsProviderRegistration] = (
    _build_fundamentals_provider_registration_map(_FUNDAMENTALS_PROVIDER_REGISTRATIONS)
)
_LISTING_PROVIDER_REGISTRY: Mapping[str, _ListingProviderRegistration] = _build_listing_provider_registration_map(
    _LISTING_PROVIDER_REGISTRATIONS
)
_OHLCV_PROVIDER_REGISTRY: Mapping[str, _OhlcvProviderRegistration] = _build_ohlcv_provider_registration_map(
    _OHLCV_PROVIDER_REGISTRATIONS
)


def supported_fundamentals_provider_ids(market: Market) -> tuple[str, ...]:
    return tuple(
        sorted(
            provider_id
            for provider_id, registration in _FUNDAMENTALS_PROVIDER_REGISTRY.items()
            if market in registration.supported_markets
        )
    )


def supported_listing_provider_ids(market: Market) -> tuple[str, ...]:
    return tuple(
        sorted(
            provider_id
            for provider_id, registration in _LISTING_PROVIDER_REGISTRY.items()
            if market in registration.supported_markets
        )
    )


def supported_ohlcv_provider_ids() -> tuple[str, ...]:
    return tuple(sorted(_OHLCV_PROVIDER_REGISTRY))


def _supported_message(supported_ids: tuple[str, ...]) -> str:
    return ", ".join(supported_ids) if supported_ids else "none"


def get_fundamentals_registration(provider_id: str, market: Market) -> _FundamentalsProviderRegistration:
    supported_ids = supported_fundamentals_provider_ids(market)
    registration = _FUNDAMENTALS_PROVIDER_REGISTRY.get(provider_id)
    if registration is None or market not in registration.supported_markets:
        raise ValidationError(
            f"unsupported fundamentals provider for {market.value}: {provider_id}; "
            f"supported ids: {_supported_message(supported_ids)}"
        )
    return registration


def get_listing_registration(provider_id: str, market: Market) -> _ListingProviderRegistration:
    supported_ids = supported_listing_provider_ids(market)
    registration = _LISTING_PROVIDER_REGISTRY.get(provider_id)
    if registration is None or market not in registration.supported_markets:
        raise ValidationError(
            f"unsupported listing provider for {market.value}: {provider_id}; "
            f"supported ids: {_supported_message(supported_ids)}"
        )
    return registration


def get_ohlcv_registration(provider_id: str) -> _OhlcvProviderRegistration:
    registration = _OHLCV_PROVIDER_REGISTRY.get(provider_id)
    if registration is None:
        raise ValidationError(
            f"unsupported OHLCV provider: {provider_id}; "
            f"supported ids: {_supported_message(supported_ohlcv_provider_ids())}"
        )
    return registration
