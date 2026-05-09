from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

import universe_selector.config as config_module
import universe_selector.ranking_profiles as ranking_profiles
from universe_selector.config import AppConfig, canonical_json, ensure_runtime_dirs, load_config
from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.ranking_profiles import (
    RankingProfileRegistration,
    get_ranking_profile,
    get_ranking_profile_registration,
    supported_ranking_profile_ids,
)
from universe_selector.ranking_profiles.registration import build_ranking_profile_registration_map
from universe_selector.ranking_profiles.sample_price_trend_v1 import (
    SAMPLE_PRICE_TREND_PROFILE_ID,
    SamplePriceTrendV1Profile,
)


COMPLETE_CONFIG: dict[str, object] = {
    "data_mode": "live",
    "duckdb_path": ".universe-selector/universe_selector.duckdb",
    "lock_path": ".universe-selector/batch.lock",
    "fixture_dir": "tests/fixtures/sample_basic",
    "live": {
        "listing_provider": {
            "US": "nasdaq_trader",
            "TW": "twse_isin",
        },
        "ohlcv_provider": "yfinance",
        "ticker_limit": None,
        "yfinance": {
            "batch_size": 200,
        },
    },
    "ranking": {"profile": "sample_price_trend_v1"},
    "report": {"top_n": 100},
}


def _deep_merge_for_test(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_for_test(result[key], value)
        else:
            result[key] = value
    return result


def _write_config(
    path: Path,
    override: dict[str, object] | None = None,
    *,
    raw_yaml: str | None = None,
) -> Path:
    if raw_yaml is not None:
        path.write_text(raw_yaml.strip() + "\n")
        return path

    data = _deep_merge_for_test(COMPLETE_CONFIG, override or {})
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _use_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    override: dict[str, object] | None = None,
    *,
    raw_yaml: str | None = None,
) -> Path:
    config_path = _write_config(tmp_path / "config.yaml", override, raw_yaml=raw_yaml)
    monkeypatch.chdir(tmp_path)
    return config_path


def test_config_hash_is_stable_and_ignores_runtime_paths() -> None:
    config_a = AppConfig(data_mode="fixture", duckdb_path="/tmp/a.duckdb", lock_path="/tmp/a.lock", fixture_dir="/tmp/a")
    config_b = AppConfig(data_mode="live", duckdb_path="/tmp/b.duckdb", lock_path="/tmp/b.lock", fixture_dir="/tmp/b")

    assert config_a.ranking_config_hash() == config_b.ranking_config_hash()
    assert config_a.ranking_config_hash() == "17845410100068fda4e91c946bb04487a3ee26d6c7a2c38b0fc105d9b43204d9"


def test_app_config_defaults_to_sample_price_trend_profile() -> None:
    config = AppConfig()
    profile = config.selected_ranking_profile

    assert isinstance(profile, SamplePriceTrendV1Profile)
    assert profile.profile_id == SAMPLE_PRICE_TREND_PROFILE_ID
    assert config.ranking_config_payload() == profile.ranking_config_payload()


def test_sample_price_trend_profile_public_api_and_payload() -> None:
    profile = SamplePriceTrendV1Profile()

    assert SAMPLE_PRICE_TREND_PROFILE_ID == "sample_price_trend_v1"
    assert profile.horizon_order == ("midterm", "longterm")
    assert profile.snapshot_metric_keys == (
        "avg_traded_value_20d_local",
        "return_60d",
        "return_120d",
    )
    assert profile.ranking_metric_keys == (
        "return_60d_rank_percentile",
        "return_120d_rank_percentile",
    )
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "sample_price_trend_v1",
        "min_history_bars": 121,
        "return_windows": {"midterm": 60, "longterm": 120},
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 20_000_000.0, "US": 5_000_000.0},
        "horizon_order": ["midterm", "longterm"],
        "snapshot_metric_keys": ["avg_traded_value_20d_local", "return_60d", "return_120d"],
        "ranking_metric_keys": ["return_60d_rank_percentile", "return_120d_rank_percentile"],
        "inspect_metric_keys": ["avg_traded_value_20d_local", "return_60d", "return_120d"],
        "percentile_method": "average_rank_100_times_rank_minus_half_over_n",
    }


def test_ranking_profiles_package_root_exposes_registry_contract_only() -> None:
    assert ranking_profiles.RankingProfileRegistration is RankingProfileRegistration
    assert ranking_profiles.get_ranking_profile is get_ranking_profile
    assert ranking_profiles.supported_ranking_profile_ids is supported_ranking_profile_ids
    assert "SamplePriceTrendV1Profile" not in ranking_profiles.__all__


def test_supported_profile_registry_is_sample_only() -> None:
    assert supported_ranking_profile_ids() == ("sample_price_trend_v1",)

    registration = get_ranking_profile_registration("sample_price_trend_v1")
    assert isinstance(registration, RankingProfileRegistration)
    assert isinstance(registration.create_profile(), SamplePriceTrendV1Profile)
    assert isinstance(get_ranking_profile("sample_price_trend_v1"), SamplePriceTrendV1Profile)


def test_supported_profile_registry_rejects_unknown_profile() -> None:
    with pytest.raises(ValidationError, match="unknown ranking profile unknown_profile"):
        get_ranking_profile("unknown_profile")


def test_profile_registration_map_rejects_duplicate_profile_ids() -> None:
    registrations = (
        RankingProfileRegistration(profile_id="duplicate_profile", factory=SamplePriceTrendV1Profile),
        RankingProfileRegistration(profile_id="duplicate_profile", factory=SamplePriceTrendV1Profile),
    )

    with pytest.raises(ValueError, match="duplicate ranking profile registration duplicate_profile"):
        build_ranking_profile_registration_map(registrations)


def test_sample_profile_is_immutable() -> None:
    profile = SamplePriceTrendV1Profile()

    with pytest.raises(FrozenInstanceError):
        profile.min_history_bars = 100  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.price_floor[Market.US] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (SamplePriceTrendV1Profile(min_history_bars=120), "min_history_bars"),
        (SamplePriceTrendV1Profile(return_windows={"midterm": 30, "longterm": 120}), "return_windows"),
        (SamplePriceTrendV1Profile(horizon_order=("longterm", "midterm")), "horizon_order"),
        (SamplePriceTrendV1Profile(snapshot_metric_keys=("avg_traded_value_20d_local",)), "snapshot metric"),
        (SamplePriceTrendV1Profile(ranking_metric_keys=("return_60d_rank_percentile",)), "ranking metric"),
        (SamplePriceTrendV1Profile(inspect_metric_keys=("close",)), "inspect metric"),
    ],
)
def test_sample_profile_rejects_invalid_contracts(profile: SamplePriceTrendV1Profile, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_load_config_reads_config_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_config(monkeypatch, tmp_path, {"data_mode": "fixture", "report": {"top_n": 25}})

    config = load_config()

    assert config.data_mode == "fixture"
    assert config.fixture_dir == "tests/fixtures/sample_basic"
    assert config.ranking_profile == "sample_price_trend_v1"
    assert config.report_top_n == 25


def test_load_config_rejects_missing_required_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    data = _deep_merge_for_test(COMPLETE_CONFIG, {})
    del data["ranking"]
    _write_config(tmp_path / "config.yaml", raw_yaml=yaml.safe_dump(data, sort_keys=False))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="config missing required key: ranking"):
        load_config()


def test_load_config_rejects_extra_ranking_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_config(monkeypatch, tmp_path, {"ranking": {"weight": 1}})

    with pytest.raises(ValidationError, match="ranking.weight is not configurable"):
        load_config()


def test_missing_config_message_points_to_example(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="config file not found: config.yaml"):
        load_config()


def test_canonical_json_is_stable() -> None:
    assert canonical_json({"b": 1, "a": "x"}) == '{"a":"x","b":1}'


def test_ensure_runtime_dirs_creates_parent_directories(tmp_path: Path) -> None:
    config = AppConfig(duckdb_path=str(tmp_path / "state" / "runs.duckdb"), lock_path=str(tmp_path / "locks" / "batch.lock"))

    ensure_runtime_dirs(config)

    assert (tmp_path / "state").is_dir()
    assert (tmp_path / "locks").is_dir()


def test_provider_config_hash_changes_when_live_provider_config_changes() -> None:
    default = AppConfig()
    limited = AppConfig(live_ticker_limit=25)

    assert default.provider_config_hash() != limited.provider_config_hash()
    assert config_module.DEFAULT_CONFIG_PATH == "config.yaml"
