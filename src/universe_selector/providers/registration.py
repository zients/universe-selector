from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from universe_selector.domain import Market
from universe_selector.providers.base import FundamentalsProvider, ListingProvider, OhlcvProvider


FundamentalsProviderFactory = Callable[[], FundamentalsProvider]
ListingProviderFactory = Callable[[Any], ListingProvider]
OhlcvProviderFactory = Callable[[Any], OhlcvProvider]


@dataclass(frozen=True)
class ListingProviderRegistration:
    provider_id: str
    supported_markets: frozenset[Market]
    source_ids: tuple[str, ...]
    factory: ListingProviderFactory


@dataclass(frozen=True)
class OhlcvProviderRegistration:
    provider_id: str
    supported_markets: frozenset[Market]
    source_ids: tuple[str, ...]
    factory: OhlcvProviderFactory


@dataclass(frozen=True)
class FundamentalsProviderRegistration:
    provider_id: str
    supported_markets: frozenset[Market]
    source_ids: tuple[str, ...]
    factory: FundamentalsProviderFactory


def build_fundamentals_provider_registration_map(
    registrations: Iterable[FundamentalsProviderRegistration],
) -> Mapping[str, FundamentalsProviderRegistration]:
    result: dict[str, FundamentalsProviderRegistration] = {}
    for registration in registrations:
        if registration.provider_id in result:
            raise ValueError(f"duplicate fundamentals provider registration {registration.provider_id}")
        result[registration.provider_id] = registration
    return MappingProxyType(result)


def build_listing_provider_registration_map(
    registrations: Iterable[ListingProviderRegistration],
) -> Mapping[str, ListingProviderRegistration]:
    result: dict[str, ListingProviderRegistration] = {}
    for registration in registrations:
        if registration.provider_id in result:
            raise ValueError(f"duplicate listing provider registration {registration.provider_id}")
        result[registration.provider_id] = registration
    return MappingProxyType(result)


def build_ohlcv_provider_registration_map(
    registrations: Iterable[OhlcvProviderRegistration],
) -> Mapping[str, OhlcvProviderRegistration]:
    result: dict[str, OhlcvProviderRegistration] = {}
    for registration in registrations:
        if registration.provider_id in result:
            raise ValueError(f"duplicate OHLCV provider registration {registration.provider_id}")
        result[registration.provider_id] = registration
    return MappingProxyType(result)
