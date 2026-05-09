from __future__ import annotations

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.identifiers import make_run_id, parse_run_id


def test_parse_run_id_requires_lowercase_prefix_and_uuid() -> None:
    parsed = parse_run_id("tw-123e4567-e89b-12d3-a456-426614174000")
    assert parsed.market == Market.TW
    assert parsed.run_id == "tw-123e4567-e89b-12d3-a456-426614174000"

    with pytest.raises(ValidationError):
        parse_run_id("TW-123e4567-e89b-12d3-a456-426614174000")

    with pytest.raises(ValidationError):
        parse_run_id("tw-not-a-uuid")


def test_make_run_id_uses_lowercase_market_prefix_and_uuid_suffix() -> None:
    run_id = make_run_id(Market.US)

    parsed = parse_run_id(run_id)

    assert run_id.startswith("us-")
    assert parsed.market is Market.US
    assert parsed.run_id == run_id
