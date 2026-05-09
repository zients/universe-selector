from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

from universe_selector.domain import Market
from universe_selector.errors import ValidationError


@dataclass(frozen=True)
class ParsedRunId:
    run_id: str
    market: Market


def make_run_id(market: Market) -> str:
    return f"{market.value.lower()}-{uuid4()}"


def parse_run_id(value: str) -> ParsedRunId:
    run_id = value.strip()
    if run_id.startswith("tw-"):
        market = Market.TW
        uuid_part = run_id[3:]
    elif run_id.startswith("us-"):
        market = Market.US
        uuid_part = run_id[3:]
    else:
        raise ValidationError("run_id must start with lowercase tw- or us-")

    try:
        UUID(uuid_part)
    except ValueError as exc:
        raise ValidationError("run_id must use uuid format after the market prefix") from exc

    return ParsedRunId(run_id=run_id, market=market)
