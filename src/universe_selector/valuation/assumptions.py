from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from universe_selector.domain import Market, canonical_ticker
from universe_selector.errors import ValidationError
from universe_selector.valuation.models import ValuationAssumptionSet
from universe_selector.valuation.registry import get_valuation_model


_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "market",
        "ticker",
        "purpose",
        "as_of",
        "currency",
        "amount_unit",
        "facts_overrides",
        "facts_override_notes",
        "assumption_source",
        "prepared_by",
        "source_note",
        "models",
    }
)
_REQUIRED_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "market",
        "ticker",
        "purpose",
        "as_of",
        "currency",
        "amount_unit",
        "assumption_source",
        "prepared_by",
        "source_note",
        "models",
    }
)
_FACT_OVERRIDE_KEYS = ("shares_outstanding", "net_debt", "reference_price")


def default_assumptions_path(market: Market, ticker: str) -> Path:
    return Path("valuation_assumptions") / market.value.lower() / f"{canonical_ticker(ticker)}.yaml"


def load_valuation_assumptions(
    market: Market,
    ticker: str,
    model_id: str,
    assumptions_path: Path | None,
) -> ValuationAssumptionSet:
    normalized_ticker = canonical_ticker(ticker)
    path = assumptions_path or default_assumptions_path(market, normalized_ticker)
    if not path.exists():
        raise ValidationError(f"missing valuation assumptions file: {path}; pass --assumptions to override")

    try:
        loaded = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid valuation assumptions YAML: {path}") from exc

    if not isinstance(loaded, Mapping):
        raise ValidationError("valuation assumptions root must be a mapping")

    payload = dict(loaded)
    _validate_keys(payload, _ROOT_KEYS, "unknown assumptions key")
    for key in sorted(_REQUIRED_ROOT_KEYS - set(payload)):
        raise ValidationError(f"missing required assumptions key: {key}")

    schema_version = payload["schema_version"]
    if schema_version != 1:
        raise ValidationError("schema_version must be 1")

    yaml_market = payload["market"]
    if yaml_market != market.value:
        raise ValidationError(f"market must match requested market {market.value}")

    yaml_ticker = payload["ticker"]
    if canonical_ticker(str(yaml_ticker)) != normalized_ticker:
        raise ValidationError(f"ticker must match requested ticker {normalized_ticker}")

    purpose = _require_non_empty_str(payload["purpose"], "purpose")
    as_of = _parse_date(payload["as_of"], "as_of")
    currency = _parse_currency(payload["currency"])
    amount_unit = _parse_amount_unit(payload["amount_unit"])
    assumption_source = _require_non_empty_str(payload["assumption_source"], "assumption_source")
    prepared_by = _require_non_empty_str(payload["prepared_by"], "prepared_by")
    source_note = _require_non_empty_str(payload["source_note"], "source_note")

    facts_overrides = _parse_optional_float_map(payload.get("facts_overrides"), "facts_overrides")
    facts_override_notes = _parse_optional_note_map(payload.get("facts_override_notes"), "facts_override_notes")
    for key, value in facts_overrides.items():
        if value is not None and not facts_override_notes[key]:
            raise ValidationError(f"facts_override_notes.{key} is required when facts_overrides.{key} is set")
    if facts_overrides["shares_outstanding"] is not None and facts_overrides["shares_outstanding"] <= 0:
        raise ValidationError("facts_overrides.shares_outstanding must be greater than zero")
    if facts_overrides["reference_price"] is not None and facts_overrides["reference_price"] <= 0:
        raise ValidationError("facts_overrides.reference_price must be greater than zero")

    models = payload["models"]
    if not isinstance(models, Mapping):
        raise ValidationError("models must be a mapping")
    model_payload = models.get(model_id)
    if not isinstance(model_payload, Mapping):
        raise ValidationError(f"missing model assumptions for {model_id}")

    parsed_model_assumptions = get_valuation_model(model_id).validate_assumptions(model_payload)
    normalized_payload = {
        "schema_version": schema_version,
        "market": market.value,
        "ticker": normalized_ticker,
        "purpose": purpose,
        "as_of": as_of.isoformat(),
        "currency": currency,
        "amount_unit": amount_unit,
        "facts_overrides": facts_overrides,
        "facts_override_notes": facts_override_notes,
        "assumption_source": assumption_source,
        "prepared_by": prepared_by,
        "source_note": source_note,
        "models": _normalize_models(models),
    }

    return ValuationAssumptionSet(
        schema_version=schema_version,
        market=market,
        ticker=normalized_ticker,
        purpose=purpose,
        as_of=as_of,
        currency=currency,
        amount_unit=amount_unit,
        assumption_source=assumption_source,
        prepared_by=prepared_by,
        source_note=source_note,
        assumption_path=str(path.resolve()),
        assumption_hash=hashlib.sha256(_canonical_json(normalized_payload).encode()).hexdigest(),
        facts_overrides=facts_overrides,
        facts_override_notes=facts_override_notes,
        model_id=model_id,
        model_assumptions=parsed_model_assumptions,
    )


def _validate_keys(payload: Mapping[str, object], allowed: frozenset[str], message: str) -> None:
    unknown_keys = sorted(set(payload) - allowed)
    if unknown_keys:
        raise ValidationError(f"{message}: {unknown_keys[0]}")


def _require_non_empty_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field} must be a non-empty string")
    return value


def _parse_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        raise ValidationError(f"{field} must be an ISO date")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise ValidationError(f"{field} must be an ISO date") from exc
        if parsed.isoformat() != value:
            raise ValidationError(f"{field} must be an ISO date")
        return parsed
    raise ValidationError(f"{field} must be a date")


def _parse_currency(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError("currency must be an ISO currency code")
    if value != value.upper() or len(value) != 3 or not value.isalpha():
        raise ValidationError("currency must be an uppercase ISO currency code")
    return value


def _parse_amount_unit(value: object) -> str:
    if value != "currency_units":
        raise ValidationError("amount_unit must be currency_units")
    return "currency_units"


def _parse_optional_float_map(value: object, field: str) -> dict[str, float | None]:
    if value is None:
        payload: Mapping[str, object] = {}
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise ValidationError(f"{field} must be a mapping")

    _validate_keys(payload, frozenset(_FACT_OVERRIDE_KEYS), f"unknown {field} key")
    result: dict[str, float | None] = {}
    for key in _FACT_OVERRIDE_KEYS:
        item = payload.get(key)
        if item is None:
            result[key] = None
        elif isinstance(item, bool) or not isinstance(item, int | float):
            raise ValidationError(f"{field}.{key} must be a number or null")
        else:
            number = float(item)
            if not math.isfinite(number):
                raise ValidationError(f"{field}.{key} must be finite")
            result[key] = number
    return result


def _parse_optional_note_map(value: object, field: str) -> dict[str, str | None]:
    if value is None:
        payload: Mapping[str, object] = {}
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise ValidationError(f"{field} must be a mapping")

    _validate_keys(payload, frozenset(_FACT_OVERRIDE_KEYS), f"unknown {field} key")
    result: dict[str, str | None] = {}
    for key in _FACT_OVERRIDE_KEYS:
        item = payload.get(key)
        if item is None:
            result[key] = None
        elif not isinstance(item, str) or not item.strip():
            raise ValidationError(f"{field}.{key} must be a non-empty string or null")
        else:
            result[key] = item
    return result


def _normalize_models(value: Mapping[object, object]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValidationError("model id must be a string")
        result[key] = _normalize_value(item)
    return result


def _normalize_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
