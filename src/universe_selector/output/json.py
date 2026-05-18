from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any


SNAPSHOT_CORE_KEYS = ("run_id", "market", "ticker", "close", "adjusted_close")
RANKING_CORE_KEYS = ("run_id", "market", "horizon", "ticker", "score", "rank")


def to_jsonable(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping | MappingProxyType):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]
    return value


def json_dumps(payload: object) -> str:
    return json.dumps(
        to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def split_profile_metrics(
    row: dict[str, object],
    *,
    core_keys: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> dict[str, object]:
    payload = {key: row[key] for key in core_keys if key in row}
    payload["metrics"] = {key: row[key] for key in metric_keys if key in row}
    return payload
