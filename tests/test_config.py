from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

import universe_selector.config as config_module
import universe_selector.ranking_profiles as ranking_profiles
from universe_selector.config import (
    AppConfig,
    canonical_json,
    ensure_runtime_dirs,
    load_config,
    load_live_fundamentals_provider_id,
)
from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.ranking_profiles import (
    RankingProfileRegistration,
    get_ranking_profile,
    get_ranking_profile_registration,
    supported_ranking_profile_ids,
)
from universe_selector.ranking_profiles.liquidity_quality_v1 import (
    LIQUIDITY_QUALITY_PROFILE_ID,
    LIQUIDITY_QUALITY_SCORE_METHOD,
    LiquidityQualityV1Profile,
)
from universe_selector.ranking_profiles.momentum_v1 import (
    MOMENTUM_PROFILE_ID,
    MOMENTUM_SCORE_METHOD,
    MomentumV1Profile,
)
from universe_selector.ranking_profiles.registration import build_ranking_profile_registration_map
from universe_selector.ranking_profiles.sample_price_trend_v1 import (
    SAMPLE_PRICE_TREND_PROFILE_ID,
    SamplePriceTrendV1Profile,
)
from universe_selector.ranking_profiles.trend_quality_v1 import (
    TREND_QUALITY_PROFILE_ID,
    TREND_QUALITY_RANKING_METRIC_KEYS,
    TREND_QUALITY_RANK_INTERPRETATION_NOTE,
    TREND_QUALITY_SCORE_METHOD,
    TREND_QUALITY_SNAPSHOT_METRIC_KEYS,
    TrendQualityV1Profile,
)
from universe_selector.ranking_profiles.volatility_quality_v1 import (
    VOLATILITY_QUALITY_PROFILE_ID,
    VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE,
    VOLATILITY_QUALITY_SCORE_METHOD,
    VolatilityQualityV1Profile,
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
        "fundamentals_provider": "yfinance_fundamentals",
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
    assert config_a.ranking_config_hash() == "04ec9df4d31ac5b86a2ad79b7a4e973e10ebd3575a4f0f43fd7fc1f2745b669b"


def test_app_config_defaults_to_sample_price_trend_profile() -> None:
    config = AppConfig()
    profile = config.selected_ranking_profile

    assert isinstance(profile, SamplePriceTrendV1Profile)
    assert profile.profile_id == SAMPLE_PRICE_TREND_PROFILE_ID
    assert config.live_fundamentals_provider == "yfinance_fundamentals"
    assert config.ranking_config_payload() == profile.ranking_config_payload()


def test_load_live_fundamentals_provider_id_reads_minimal_value_config(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
live:
  fundamentals_provider: yfinance_fundamentals
""".lstrip()
    )
    monkeypatch.chdir(tmp_path)

    assert load_live_fundamentals_provider_id() == "yfinance_fundamentals"


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
        "score_return_60d",
        "score_return_120d",
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
        "ranking_metric_keys": ["score_return_60d", "score_return_120d"],
        "inspect_metric_keys": ["avg_traded_value_20d_local", "return_60d", "return_120d"],
    }


def test_momentum_profile_public_api_and_payload() -> None:
    profile = MomentumV1Profile()

    assert MOMENTUM_PROFILE_ID == "momentum_v1"
    assert profile.horizon_order == ("swing", "midterm")
    assert profile.snapshot_metric_keys == (
        "avg_traded_value_20d_local",
        "momentum_return_12_1",
        "momentum_return_6_1",
        "volatility_12_1",
        "volatility_6_1",
        "risk_adjusted_momentum_12_1",
        "risk_adjusted_momentum_6_1",
        "short_term_strength_20d",
    )
    assert profile.ranking_metric_keys == (
        "score_risk_adjusted_momentum_12_1",
        "score_risk_adjusted_momentum_6_1",
        "score_short_term_strength_20d",
    )
    assert profile.inspect_metric_keys == (
        "momentum_return_12_1",
        "momentum_return_6_1",
        "volatility_12_1",
        "volatility_6_1",
        "risk_adjusted_momentum_12_1",
        "risk_adjusted_momentum_6_1",
        "short_term_strength_20d",
    )
    assert profile.ranking_config_payload() == {
        "active_trading_min_days_274": 230,
        "horizon_weights": {
            "midterm": {
                "score_risk_adjusted_momentum_12_1": 0.5,
                "score_risk_adjusted_momentum_6_1": 0.5,
            },
            "swing": {
                "score_risk_adjusted_momentum_6_1": 0.5,
                "score_risk_adjusted_momentum_12_1": 0.3,
                "score_short_term_strength_20d": 0.2,
            },
        },
        "liquidity_floor": {"TW": 20_000_000.0, "US": 5_000_000.0},
        "min_history_bars": 274,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "ranking_profile": "momentum_v1",
        "ranking_metric_keys": list(profile.ranking_metric_keys),
        "score_input_columns": [
            "risk_adjusted_momentum_12_1",
            "risk_adjusted_momentum_6_1",
            "short_term_strength_20d",
            "volatility_6_1",
        ],
        "score_method": MOMENTUM_SCORE_METHOD,
        "stdev_ddof": 1,
        "volatility_floor": 0.001,
        "zero_volume_max_days_20": 2,
    }


def test_liquidity_quality_profile_public_api_and_payload() -> None:
    profile = LiquidityQualityV1Profile()

    assert LIQUIDITY_QUALITY_PROFILE_ID == "liquidity_quality_v1"
    assert profile.horizon_order == ("composite", "shortterm", "stable")
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "avg_traded_value_20d_local" in profile.snapshot_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "liquidity_quality_v1",
        "min_history_bars": 63,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "horizon_order": ["composite", "shortterm", "stable"],
        "snapshot_metric_keys": list(profile.snapshot_metric_keys),
        "ranking_metric_keys": list(profile.ranking_metric_keys),
        "inspect_metric_keys": list(profile.inspect_metric_keys),
        "score_method": LIQUIDITY_QUALITY_SCORE_METHOD,
    }


def test_volatility_quality_profile_public_api_and_payload() -> None:
    profile = VolatilityQualityV1Profile()

    assert VOLATILITY_QUALITY_PROFILE_ID == "volatility_quality_v1"
    assert profile.horizon_order == ("composite", "shortterm", "stable")
    assert VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE == (
        "Volatility quality scores rank market-local lower realized volatility, downside volatility, "
        "range tightness, and drawdown control; high scores do not imply future returns or lower future risk."
    )
    assert profile.rank_interpretation_note == VOLATILITY_QUALITY_RANK_INTERPRETATION_NOTE
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "avg_traded_value_20d_local" in profile.snapshot_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "volatility_quality_v1",
        "min_history_bars": 126,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "volatility_floor": 0.0001,
        "horizon_order": ["composite", "shortterm", "stable"],
        "snapshot_metric_keys": list(profile.snapshot_metric_keys),
        "ranking_metric_keys": list(profile.ranking_metric_keys),
        "inspect_metric_keys": list(profile.inspect_metric_keys),
        "stdev_ddof": 1,
        "score_method": VOLATILITY_QUALITY_SCORE_METHOD,
    }


def test_trend_quality_profile_public_api_and_payload() -> None:
    profile = TrendQualityV1Profile()

    assert TREND_QUALITY_PROFILE_ID == "trend_quality_v1"
    assert profile.horizon_order == ("composite", "shortterm", "midterm")
    assert profile.rank_interpretation_note == TREND_QUALITY_RANK_INTERPRETATION_NOTE
    assert profile.snapshot_metric_keys == TREND_QUALITY_SNAPSHOT_METRIC_KEYS
    assert profile.ranking_metric_keys == TREND_QUALITY_RANKING_METRIC_KEYS
    assert profile.inspect_metric_keys == profile.snapshot_metric_keys
    assert "avg_traded_value_20d_local" in profile.snapshot_metric_keys
    assert "asof_bar_date_yyyymmdd" in profile.snapshot_metric_keys
    assert "volume" not in profile.snapshot_metric_keys
    assert "tag_structure_cap_active" in profile.ranking_metric_keys
    assert "tag_structure_downtrend" not in profile.ranking_metric_keys
    assert profile.ranking_config_payload() == {
        "ranking_profile": "trend_quality_v1",
        "min_history_bars": 252,
        "price_floor": {"TW": 10.0, "US": 5.0},
        "liquidity_floor": {"TW": 50_000_000.0, "US": 10_000_000.0},
        "active_trading_min_days_60": {"TW": 50, "US": 55},
        "zero_volume_max_days_20": {"TW": 3, "US": 1},
        "volatility_floor": 0.0001,
        "horizon_order": ["composite", "shortterm", "midterm"],
        "snapshot_metric_keys": list(TREND_QUALITY_SNAPSHOT_METRIC_KEYS),
        "ranking_metric_keys": list(TREND_QUALITY_RANKING_METRIC_KEYS),
        "inspect_metric_keys": list(TREND_QUALITY_SNAPSHOT_METRIC_KEYS),
        "stdev_ddof": 1,
        "score_method": TREND_QUALITY_SCORE_METHOD,
    }


def test_volatility_quality_profile_is_immutable() -> None:
    profile = VolatilityQualityV1Profile()

    with pytest.raises(FrozenInstanceError):
        profile.min_history_bars = 100  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.price_floor[Market.US] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.liquidity_floor[Market.US] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.active_trading_min_days_60[Market.US] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.zero_volume_max_days_20[Market.US] = 1  # type: ignore[index]


def test_trend_quality_profile_is_immutable() -> None:
    profile = TrendQualityV1Profile()

    with pytest.raises(FrozenInstanceError):
        profile.min_history_bars = 100  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.price_floor[Market.US] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.liquidity_floor[Market.US] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.active_trading_min_days_60[Market.US] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        profile.zero_volume_max_days_20[Market.US] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (VolatilityQualityV1Profile(price_floor={Market.TW: 11.0, Market.US: 5.0}), "price_floor"),
        (
            VolatilityQualityV1Profile(
                liquidity_floor={Market.TW: 50_000_000.0, Market.US: 9_000_000.0}
            ),
            "liquidity_floor",
        ),
        (
            VolatilityQualityV1Profile(active_trading_min_days_60={Market.TW: 50, Market.US: 54}),
            "active_trading_min_days_60",
        ),
        (
            VolatilityQualityV1Profile(zero_volume_max_days_20={Market.TW: 2, Market.US: 1}),
            "zero_volume_max_days_20",
        ),
        (VolatilityQualityV1Profile(volatility_floor=0.001), "volatility_floor"),
    ],
)
def test_volatility_quality_profile_rejects_contract_changes(
    profile: VolatilityQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (TrendQualityV1Profile(min_history_bars=251), "min_history_bars"),
        (TrendQualityV1Profile(profile_id="bad"), "profile_id"),
        (TrendQualityV1Profile(price_floor={Market.TW: 11.0, Market.US: 5.0}), "price_floor"),
        (
            TrendQualityV1Profile(
                liquidity_floor={Market.TW: 50_000_000.0, Market.US: 9_000_000.0}
            ),
            "liquidity_floor",
        ),
        (
            TrendQualityV1Profile(active_trading_min_days_60={Market.TW: 50, Market.US: 54}),
            "active_trading_min_days_60",
        ),
        (
            TrendQualityV1Profile(zero_volume_max_days_20={Market.TW: 2, Market.US: 1}),
            "zero_volume_max_days_20",
        ),
        (TrendQualityV1Profile(volatility_floor=0.001), "volatility_floor"),
        (TrendQualityV1Profile(snapshot_metric_keys=("profile_metrics_version",)), "snapshot metric"),
        (TrendQualityV1Profile(ranking_metric_keys=("score_return_20d",)), "ranking metric"),
        (TrendQualityV1Profile(inspect_metric_keys=("return_20d",)), "inspect metric"),
        (TrendQualityV1Profile(horizon_order=("shortterm", "composite", "midterm")), "horizon order"),
        (TrendQualityV1Profile(stdev_ddof=0), "stdev_ddof"),
        (TrendQualityV1Profile(score_method="bad"), "score_method"),
        (TrendQualityV1Profile(rank_interpretation_note="bad"), "rank_interpretation_note"),
    ],
)
def test_trend_quality_profile_rejects_contract_changes(
    profile: TrendQualityV1Profile, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        profile.validate()


def test_ranking_profiles_package_root_exposes_registry_contract_only() -> None:
    assert ranking_profiles.RankingProfileRegistration is RankingProfileRegistration
    assert ranking_profiles.get_ranking_profile is get_ranking_profile
    assert ranking_profiles.supported_ranking_profile_ids is supported_ranking_profile_ids
    assert "LiquidityQualityV1Profile" not in ranking_profiles.__all__
    assert "MomentumV1Profile" not in ranking_profiles.__all__
    assert "SamplePriceTrendV1Profile" not in ranking_profiles.__all__
    assert "TrendQualityV1Profile" not in ranking_profiles.__all__


def test_supported_profile_registry_includes_public_profiles() -> None:
    assert supported_ranking_profile_ids() == (
        "sample_price_trend_v1",
        "momentum_v1",
        "trend_quality_v1",
        "volatility_quality_v1",
        "liquidity_quality_v1",
    )

    registration = get_ranking_profile_registration("sample_price_trend_v1")
    assert isinstance(registration, RankingProfileRegistration)
    assert isinstance(registration.create_profile(), SamplePriceTrendV1Profile)
    assert isinstance(get_ranking_profile("sample_price_trend_v1"), SamplePriceTrendV1Profile)

    momentum_registration = get_ranking_profile_registration("momentum_v1")
    assert isinstance(momentum_registration, RankingProfileRegistration)
    assert isinstance(momentum_registration.create_profile(), MomentumV1Profile)
    assert isinstance(get_ranking_profile("momentum_v1"), MomentumV1Profile)

    volatility_registration = get_ranking_profile_registration("volatility_quality_v1")
    assert isinstance(volatility_registration, RankingProfileRegistration)
    assert isinstance(volatility_registration.create_profile(), VolatilityQualityV1Profile)
    assert isinstance(get_ranking_profile("volatility_quality_v1"), VolatilityQualityV1Profile)

    trend_registration = get_ranking_profile_registration("trend_quality_v1")
    assert isinstance(trend_registration, RankingProfileRegistration)
    assert isinstance(trend_registration.create_profile(), TrendQualityV1Profile)
    assert isinstance(get_ranking_profile("trend_quality_v1"), TrendQualityV1Profile)

    liquidity_registration = get_ranking_profile_registration("liquidity_quality_v1")
    assert isinstance(liquidity_registration, RankingProfileRegistration)
    assert isinstance(liquidity_registration.create_profile(), LiquidityQualityV1Profile)
    assert isinstance(get_ranking_profile("liquidity_quality_v1"), LiquidityQualityV1Profile)


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
        (SamplePriceTrendV1Profile(ranking_metric_keys=("score_return_60d",)), "ranking metric"),
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


def test_load_config_rejects_invalid_fundamentals_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_config(monkeypatch, tmp_path, {"live": {"fundamentals_provider": "unknown"}})

    with pytest.raises(ValidationError, match="unsupported fundamentals provider: unknown"):
        load_config()


def test_load_config_rejects_null_fundamentals_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_config(monkeypatch, tmp_path, {"live": {"fundamentals_provider": None}})

    with pytest.raises(ValidationError, match="live.fundamentals_provider must be a provider id"):
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


def test_valuation_fundamentals_provider_is_not_part_of_batch_provider_hash() -> None:
    default = AppConfig()
    alternate_valuation_provider = AppConfig(live_fundamentals_provider="other_fundamentals")

    assert default.provider_config_hash() == alternate_valuation_provider.provider_config_hash()
    assert "fundamentals_provider" not in default.provider_config_payload()
