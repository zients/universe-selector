from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.registry import (
    get_fundamentals_provider_registration,
    get_listing_registration,
    get_ohlcv_registration,
)
from universe_selector.ranking_profiles import RankingProfile, get_ranking_profile


DEFAULT_CONFIG_PATH = "config.yaml"
CONFIG_EXAMPLE_PATH = "config.example.yaml"
PACKAGED_CONFIG_EXAMPLE_PATH = "universe_selector/config.example.yaml"

_REQUIRED_CONFIG_KEYS = (
    "data_mode",
    "duckdb_path",
    "lock_path",
    "fixture_dir",
    "live",
    "live.listing_provider",
    "live.listing_provider.US",
    "live.listing_provider.TW",
    "live.ohlcv_provider",
    "live.fundamentals_provider",
    "live.ticker_limit",
    "live.yfinance",
    "live.yfinance.batch_size",
    "ranking",
    "ranking.profile",
    "report",
    "report.top_n",
)

_REQUIRED_MAPPING_KEYS = frozenset(
    {
        "live",
        "live.listing_provider",
        "live.yfinance",
        "ranking",
        "report",
    }
)


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class AppConfig:
    data_mode: str = "live"
    duckdb_path: str = ".universe-selector/universe_selector.duckdb"
    lock_path: str = ".universe-selector/batch.lock"
    fixture_dir: str = "tests/fixtures/sample_basic"
    ranking_profile: str = "sample_price_trend_v1"
    report_top_n: int = 100
    live_listing_provider: Mapping[Market, str] = field(
        default_factory=lambda: {Market.US: "nasdaq_trader", Market.TW: "twse_isin"}
    )
    live_ohlcv_provider: str = "yfinance"
    live_fundamentals_provider: str = "yfinance_fundamentals"
    live_ticker_limit: int | None = None
    live_yfinance_batch_size: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "live_listing_provider", MappingProxyType(dict(self.live_listing_provider)))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AppConfig":
        ranking = data["ranking"]
        if not isinstance(ranking, dict):
            raise ValidationError("ranking must be a mapping with only ranking.profile")
        allowed_ranking_keys = {"profile"}
        extra_keys = set(ranking) - allowed_ranking_keys
        if extra_keys:
            key = sorted(extra_keys)[0]
            raise ValidationError(f"ranking.{key} is not configurable; only ranking.profile is supported")
        report = data["report"]
        live = data["live"]
        listing_provider = live["listing_provider"]
        yfinance = _parse_yfinance_mapping(live["yfinance"])
        return cls(
            data_mode=data["data_mode"],
            duckdb_path=data["duckdb_path"],
            lock_path=data["lock_path"],
            fixture_dir=data["fixture_dir"],
            ranking_profile=ranking["profile"],
            report_top_n=int(report["top_n"]),
            live_listing_provider={market: str(listing_provider[market.value]) for market in Market},
            live_ohlcv_provider=str(live["ohlcv_provider"]),
            live_fundamentals_provider=_parse_provider_id(
                live["fundamentals_provider"],
                label="live.fundamentals_provider",
            ),
            live_ticker_limit=_parse_live_ticker_limit(live["ticker_limit"], label="live.ticker_limit"),
            live_yfinance_batch_size=_parse_positive_int(
                yfinance["batch_size"],
                label="live.yfinance.batch_size",
            ),
        )

    @property
    def selected_ranking_profile(self) -> RankingProfile:
        return get_ranking_profile(self.ranking_profile)

    def validate(self) -> None:
        if self.data_mode not in {"fixture", "live"}:
            raise ValidationError("data_mode must be fixture or live")
        self.selected_ranking_profile.validate()
        for market in Market:
            get_listing_registration(self.live_listing_provider[market], market)
        get_ohlcv_registration(self.live_ohlcv_provider)
        get_fundamentals_provider_registration(self.live_fundamentals_provider)
        _parse_live_ticker_limit(self.live_ticker_limit, label="live.ticker_limit")
        _parse_positive_int(self.live_yfinance_batch_size, label="live.yfinance.batch_size")

    def ranking_config_payload(self) -> dict[str, Any]:
        return self.selected_ranking_profile.ranking_config_payload()

    def ranking_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.ranking_config_payload()).encode("utf-8")).hexdigest()

    def provider_config_payload(self) -> dict[str, Any]:
        listing_provider_payload = {}
        for market in Market:
            registration = get_listing_registration(self.live_listing_provider[market], market)
            listing_provider_payload[market.value] = {
                "provider_id": registration.provider_id,
                "source_ids": list(registration.source_ids),
            }
        ohlcv_registration = get_ohlcv_registration(self.live_ohlcv_provider)
        return {
            "listing_provider": listing_provider_payload,
            "ohlcv_provider": {
                "provider_id": ohlcv_registration.provider_id,
                "source_ids": list(ohlcv_registration.source_ids),
                "config": {
                    "yfinance": {
                        "batch_size": self.live_yfinance_batch_size,
                    },
                },
            },
            "ticker_limit": self.live_ticker_limit,
        }

    def provider_config_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.provider_config_payload()).encode("utf-8")).hexdigest()


def _parse_live_ticker_limit(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be null or a positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise ValidationError(f"{label} must be null or a positive integer")
        return value
    raise ValidationError(f"{label} must be null or a positive integer")


def _parse_provider_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be a provider id")
    return value


def _parse_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{label} must be a positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise ValidationError(f"{label} must be a positive integer")
        return value
    raise ValidationError(f"{label} must be a positive integer")


def _parse_yfinance_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ValidationError("live.yfinance.batch_size must be a positive integer")


def _missing_config_message() -> str:
    action = f"copy {CONFIG_EXAMPLE_PATH} to {DEFAULT_CONFIG_PATH}"
    return (
        f"config file not found: {DEFAULT_CONFIG_PATH}; {action}; "
        f"installed template: {PACKAGED_CONFIG_EXAMPLE_PATH}"
    )


def _validate_required_config_keys(data: dict[str, Any]) -> None:
    for key in _REQUIRED_CONFIG_KEYS:
        _validate_required_config_key(data, key)


def _validate_required_config_key(data: dict[str, Any], key: str) -> None:
    current: object = data
    parts = key.split(".")
    for index, part in enumerate(parts):
        parent_key = ".".join(parts[:index])
        if not isinstance(current, dict):
            raise ValidationError(f"config key {parent_key} must be a mapping")
        if part not in current:
            raise ValidationError(f"config missing required key: {key}")
        current = current[part]
    if key in _REQUIRED_MAPPING_KEYS and not isinstance(current, dict):
        raise ValidationError(f"config key {key} must be a mapping")


def load_config() -> AppConfig:
    loaded = _load_config_mapping()
    _validate_required_config_keys(loaded)

    config = AppConfig.from_mapping(loaded)
    config.validate()
    return config


def load_live_fundamentals_provider_id() -> str:
    loaded = _load_config_mapping()
    live = loaded.get("live")
    if not isinstance(live, dict):
        raise ValidationError("config key live must be a mapping")
    if "fundamentals_provider" not in live:
        raise ValidationError("config missing required key: live.fundamentals_provider")
    provider_id = _parse_provider_id(
        live["fundamentals_provider"],
        label="live.fundamentals_provider",
    )
    get_fundamentals_provider_registration(provider_id)
    return provider_id


def _load_config_mapping() -> dict[str, Any]:
    config_path = Path(DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise ValidationError(_missing_config_message())

    loaded = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, dict):
        raise ValidationError("YAML config root must be a mapping")
    return loaded


def ensure_runtime_dirs(config: AppConfig) -> None:
    Path(config.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.lock_path).parent.mkdir(parents=True, exist_ok=True)
