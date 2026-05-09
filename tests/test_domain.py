from __future__ import annotations

import pytest

import universe_selector.domain as domain
from universe_selector.domain import Market, canonical_market, canonical_ticker
from universe_selector.errors import ValidationError


def test_canonical_market_accepts_case_insensitive_values() -> None:
    assert canonical_market("tw") == Market.TW
    assert canonical_market("US") == Market.US


def test_canonical_market_rejects_unknown_value() -> None:
    with pytest.raises(ValidationError):
        canonical_market("jp")


def test_canonical_ticker_uppercases_and_strips() -> None:
    assert canonical_ticker(" aapl ") == "AAPL"
    assert canonical_ticker(" 2330 ") == "2330"


def test_domain_does_not_expose_stale_runtime_enums() -> None:
    assert not hasattr(domain, "Horizon")
    assert not hasattr(domain, "RunStatus")
