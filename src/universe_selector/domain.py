from __future__ import annotations

from enum import StrEnum

from universe_selector.errors import ValidationError


class Market(StrEnum):
    TW = "TW"
    US = "US"


def canonical_market(value: str) -> Market:
    normalized = value.strip().upper()
    try:
        return Market(normalized)
    except ValueError as exc:
        raise ValidationError(f"invalid market: {value}") from exc


def canonical_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if not ticker:
        raise ValidationError("ticker is required")
    return ticker
