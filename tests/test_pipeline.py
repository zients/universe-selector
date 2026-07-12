from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

import universe_selector.pipeline as pipeline
from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError, ValidationError
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.pipeline import MultiProfileBatchError, run_batch, run_batch_profiles
from universe_selector.providers.fixture import FixtureProvider
from universe_selector.providers.models import FundamentalsCoverage, FundamentalsMetadata, FundamentalsUniverseRunData
from universe_selector.ranking_profiles import (
    RankingProfileRegistration,
    get_ranking_profile,
    get_ranking_profile_registration,
)
from universe_selector.ranking_profiles.base import RankingProfileDataRequirements


def _fixture_config(tmp_path: Path, fixture_dir: Path) -> AppConfig:
    return AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="sample_price_trend_v1",
        report_top_n=2,
    )


def _copy_fixture_with_mean_reversion_pullback(tmp_path: Path, fixture_dir: Path) -> Path:
    target_dir = tmp_path / "mean_reversion_fixture"
    shutil.copytree(fixture_dir, target_dir)
    ohlcv_path = target_dir / "ohlcv.csv"

    with ohlcv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    assert fieldnames is not None

    bbb_rows = [row for row in rows if row["market"] == "US" and row["ticker"] == "BBB"]
    assert len(bbb_rows) >= 2
    last_row = bbb_rows[-1]
    pullback_close = round(float(bbb_rows[-2]["close"]) * 0.95, 6)
    last_row.update(
        {
            "open": str(round(pullback_close * 0.995, 6)),
            "high": str(round(pullback_close * 1.01, 6)),
            "low": str(round(pullback_close * 0.99, 6)),
            "close": str(pullback_close),
            "adjusted_close": str(pullback_close),
        }
    )

    with ohlcv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return target_dir


def test_pipeline_runs_sample_profile_and_persists_report(tmp_path: Path, fixture_dir: Path) -> None:
    config = _fixture_config(tmp_path, fixture_dir)

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="sample_price_trend_v1")
    report = repo.read_report_markdown(result.run_id)
    json_report = repo.read_report_artifact(result.run_id, "json")
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=config.selected_ranking_profile)
    parsed_json_report = json.loads(json_report)

    assert resolved.run_id == result.run_id
    assert result.ranking_profile == "sample_price_trend_v1"
    assert resolved.ranking_profile == "sample_price_trend_v1"
    assert "# Universe Selector Report" in report
    assert "ranking_profile: sample_price_trend_v1" in report
    assert parsed_json_report["artifact_type"] == "universe_selector_report"
    assert parsed_json_report["ranking_profile"] == "sample_price_trend_v1"
    assert payload.snapshot["ticker"] == "AAA"
    assert "return_60d" in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == {"midterm", "longterm"}


def test_pipeline_runs_momentum_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="momentum_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="momentum_v1")
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=config.selected_ranking_profile)

    assert resolved.run_id == result.run_id
    assert result.ranking_profile == "momentum_v1"
    assert resolved.ranking_profile == "momentum_v1"
    assert "risk_adjusted_momentum_12_1" in payload.snapshot
    assert "score_risk_adjusted_momentum_12_1" in payload.rankings[0]
    assert {row["horizon"] for row in payload.rankings} == {"swing", "midterm"}


def test_pipeline_runs_liquidity_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="liquidity_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="liquidity_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=config.selected_ranking_profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "liquidity_quality_v1"
    assert "ranking_profile: liquidity_quality_v1" in report
    assert payload.snapshot["ticker"] == "AAA"
    assert payload.snapshot["profile_metrics_version"] == 1.0
    assert payload.snapshot["avg_traded_value_20d_local"] >= 10_000_000.0
    assert "volume" not in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == {"composite", "shortterm", "stable"}
    assert "depth_score" in payload.rankings[0]
    assert "tag_risk_thin_liquidity" in payload.rankings[0]


def test_pipeline_runs_fundamental_quality_profile_and_persists_metrics(
    tmp_path: Path,
    fixture_dir: Path,
) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="fundamental_quality_profitability_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    metadata = repo.read_provider_metadata(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=config.selected_ranking_profile)
    report = repo.read_report_markdown(result.run_id)
    json_report = json.loads(repo.read_report_artifact(result.run_id, "json"))

    assert result.ranking_profile == "fundamental_quality_profitability_v1"
    assert metadata.fundamentals_provider_id == "fixture-fundamentals-v1"
    assert metadata.fundamentals_requested_count == 5
    assert metadata.fundamentals_returned_count == 3
    assert json_report["provider_summary"]["fundamentals_provider_id"] == "fixture-fundamentals-v1"
    assert json_report["provider_summary"]["fundamentals_source_id"] == "sample_basic/fundamentals.csv"
    assert json_report["provider_summary"]["fundamentals_returned_count"] == "3"
    assert (
        "Fixture fundamentals are deterministic sample data"
        in json_report["provider_summary"]["fundamentals_source_risk_note"]
    )
    assert "ranking_profile: fundamental_quality_profitability_v1" in report
    assert "fundamentals_provider_id: fixture-fundamentals-v1" in report
    assert payload.snapshot["ticker"] == "AAA"
    assert payload.snapshot["roe"] == pytest.approx(0.20)
    assert payload.rankings[0]["horizon"] == "composite"
    assert "score_profitability" in payload.rankings[0]


def test_pipeline_runs_volatility_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="volatility_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="volatility_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)
    top_payload = repo.read_inspect_payload(result.run_id, "SHORT", profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "volatility_quality_v1"
    assert "ranking_profile: volatility_quality_v1" in report
    assert "| 1 | SHORT | 0.9583 |" in report
    assert "| 2 | AAA | 0.6750 |" in report

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["ticker"] == "AAA"
    assert set(payload.snapshot) == expected_snapshot_keys
    assert payload.snapshot["profile_metrics_version"] == 1.0
    assert payload.snapshot["avg_traded_value_20d_local"] >= 10_000_000.0
    assert payload.snapshot["active_trading_days_60d"] == 60.0
    assert payload.snapshot["zero_volume_days_20d"] == 0.0
    assert payload.snapshot["stale_close_days_20d"] == 0.0
    assert payload.snapshot["data_quality_extreme_return_flag"] == 0.0
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))

    for inspect_payload in (payload, top_payload):
        assert "volume" not in inspect_payload.snapshot
        assert {row["horizon"] for row in inspect_payload.rankings} == set(profile.horizon_order)
        for row in inspect_payload.rankings:
            assert "volume" not in row
            assert set(row) == expected_ranking_keys
            assert isinstance(row["rank"], int)
            assert math.isfinite(float(row["score"]))
            for key in profile.ranking_metric_keys:
                assert isinstance(row[key], int | float)
                assert math.isfinite(float(row[key]))

    aaa_by_horizon = {str(row["horizon"]): row for row in payload.rankings}
    assert aaa_by_horizon["composite"]["rank"] == 2
    assert aaa_by_horizon["composite"]["score"] == pytest.approx(0.675)
    assert aaa_by_horizon["shortterm"]["rank"] == 2
    assert aaa_by_horizon["shortterm"]["score"] == pytest.approx(0.65)
    assert aaa_by_horizon["stable"]["rank"] == 2
    assert aaa_by_horizon["stable"]["score"] == pytest.approx(2.0 / 3.0)
    assert aaa_by_horizon["composite"]["score_low_volatility_60d"] == pytest.approx(2.0 / 3.0)
    assert aaa_by_horizon["composite"]["score_range_tightness_20d"] == pytest.approx(1.0)
    assert aaa_by_horizon["composite"]["score_drawdown_control_120d"] == pytest.approx(2.0 / 3.0)
    assert aaa_by_horizon["composite"]["penalty_score"] == 0.0

    short_by_horizon = {str(row["horizon"]): row for row in top_payload.rankings}
    assert top_payload.snapshot["ticker"] == "SHORT"
    assert all(row["rank"] == 1 for row in short_by_horizon.values())
    assert short_by_horizon["composite"]["score"] == pytest.approx(0.9583333333333333)
    assert short_by_horizon["shortterm"]["score"] == pytest.approx(0.8833333333333333)
    assert short_by_horizon["stable"]["score"] == pytest.approx(1.0)
    assert short_by_horizon["composite"]["score_low_volatility_60d"] == pytest.approx(1.0)
    assert short_by_horizon["composite"]["volatility_control_score"] == pytest.approx(1.0)
    assert short_by_horizon["composite"]["trading_smoothness_score"] == pytest.approx(5.0 / 6.0)
    assert short_by_horizon["composite"]["drawdown_quality_score"] == pytest.approx(1.0)
    assert short_by_horizon["composite"]["penalty_score"] == 0.0


def test_pipeline_runs_trend_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="trend_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="trend_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "trend_quality_v1"
    assert "ranking_profile: trend_quality_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_structure_cap_active" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["ticker"] == "AAA"
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_runs_defensive_compounder_quality_profile_and_persists_metrics(
    tmp_path: Path, fixture_dir: Path
) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="defensive_compounder_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="defensive_compounder_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    snapshot_rows = (
        repo.connect(read_only=True)
        .execute(
            "select ticker from run_ticker_snapshot where run_id = ? order by ticker",
            [result.run_id],
        )
        .fetchall()
    )
    payload = repo.read_inspect_payload(result.run_id, snapshot_rows[0][0], profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "defensive_compounder_quality_v1"
    assert "ranking_profile: defensive_compounder_quality_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_positive_steady_compounder" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["return_120d"] > 0.0
    assert payload.snapshot["price_vs_sma_200d"] >= -0.05
    assert payload.snapshot["max_drawdown_252d"] > -0.35
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_runs_trend_pullback_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="trend_pullback_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="trend_pullback_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "trend_pullback_quality_v1"
    assert "ranking_profile: trend_pullback_quality_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_setup_healthy_pullback" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["ticker"] == "AAA"
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert payload.snapshot["pullback_from_120d_high"] <= 0.0
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_runs_base_breakout_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="base_breakout_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="base_breakout_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "base_breakout_quality_v1"
    assert "ranking_profile: base_breakout_quality_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_setup_valid_base" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["ticker"] == "AAA"
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    assert payload.snapshot["base_depth_60d"] <= 0.0
    assert payload.snapshot["pct_below_120d_high"] >= -0.08
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_runs_relative_strength_leader_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="relative_strength_leader_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="relative_strength_leader_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "relative_strength_leader_v1"
    assert "ranking_profile: relative_strength_leader_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_positive_rs_leader" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["ticker"] == "AAA"
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert payload.snapshot["return_60d"] > 0.0 or payload.snapshot["return_120d"] > 0.0
    assert payload.snapshot["price_vs_sma_200d"] > 0.0
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_runs_mean_reversion_quality_profile_and_persists_metrics(tmp_path: Path, fixture_dir: Path) -> None:
    mean_reversion_fixture_dir = _copy_fixture_with_mean_reversion_pullback(tmp_path, fixture_dir)
    config = AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(mean_reversion_fixture_dir),
        ranking_profile="mean_reversion_quality_v1",
        report_top_n=2,
    )

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    profile = config.selected_ranking_profile
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="mean_reversion_quality_v1")
    report = repo.read_report_markdown(result.run_id)
    snapshot_rows = (
        repo.connect(read_only=True)
        .execute(
            "select ticker from run_ticker_snapshot where run_id = ? order by ticker",
            [result.run_id],
        )
        .fetchall()
    )
    payload = repo.read_inspect_payload(result.run_id, snapshot_rows[0][0], profile=profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "mean_reversion_quality_v1"
    assert "ranking_profile: mean_reversion_quality_v1" in report
    ranking_content = report.split("## Methodology Notes", 1)[0]
    assert profile.rank_interpretation_note in report
    assert "tag_setup_oversold_quality" not in ranking_content

    expected_snapshot_keys = {
        "run_id",
        "market",
        "ticker",
        "close",
        "adjusted_close",
        *profile.snapshot_metric_keys,
    }
    expected_ranking_keys = {
        "run_id",
        "market",
        "horizon",
        "ticker",
        "score",
        "rank",
        *profile.ranking_metric_keys,
    }
    assert payload.snapshot["return_20d"] < 0.0 or payload.snapshot["distance_from_sma_20d"] < 0.0
    assert payload.snapshot["max_drawdown_120d"] > -0.40
    assert set(payload.snapshot) == expected_snapshot_keys
    assert "volume" not in payload.snapshot
    assert {row["horizon"] for row in payload.rankings} == set(profile.horizon_order)
    for key in profile.snapshot_metric_keys:
        assert isinstance(payload.snapshot[key], int | float)
        assert math.isfinite(float(payload.snapshot[key]))
    for row in payload.rankings:
        assert set(row) == expected_ranking_keys
        assert "volume" not in row
        assert isinstance(row["rank"], int)
        assert math.isfinite(float(row["score"]))
        for key in profile.ranking_metric_keys:
            assert isinstance(row[key], int | float)
            assert math.isfinite(float(row[key]))


def test_pipeline_marks_failed_run_when_provider_has_no_usable_rows(tmp_path: Path, fixture_dir: Path) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    empty_fixture_dir = tmp_path / "empty_fixture"
    empty_fixture_dir.mkdir()
    (empty_fixture_dir / "metadata.json").write_bytes((fixture_dir / "metadata.json").read_bytes())
    (empty_fixture_dir / "listings.csv").write_text((fixture_dir / "listings.csv").read_text().splitlines()[0] + "\n")
    (empty_fixture_dir / "ohlcv.csv").write_text(
        (fixture_dir / "ohlcv.csv").read_text().splitlines()[0] + "\nTW,ZZZ,2026-04-24,10.0,10.0,10.0,10.0,10.0,1000\n"
    )

    with pytest.raises(ProviderDataError, match="listing provider returned no usable listings"):
        run_batch(Market.US, replace(config, fixture_dir=str(empty_fixture_dir)))


def test_pipeline_runs_multiple_profiles_with_one_provider_load(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    load_calls = 0

    class CountingFixtureProvider:
        def __init__(self, fixture_dir: str) -> None:
            self._provider = FixtureProvider(fixture_dir)

        def load_run_data(self, market: Market, requirements=None):
            nonlocal load_calls
            load_calls += 1
            return self._provider.load_run_data(market)

    monkeypatch.setattr("universe_selector.pipeline.FixtureProvider", CountingFixtureProvider)

    results = run_batch_profiles(
        Market.US,
        config,
        ("sample_price_trend_v1", "momentum_v1"),
    )

    assert load_calls == 1
    assert [result.ranking_profile for result in results] == ["sample_price_trend_v1", "momentum_v1"]
    assert len({result.run_id for result in results}) == 2

    repo = DuckDbRepository(config.duckdb_path)
    for result in results:
        resolved = repo.resolve_successful_run(result.run_id)
        assert resolved.ranking_profile == result.ranking_profile
        metadata = repo.read_provider_metadata(result.run_id)
        assert not hasattr(metadata, "run_id")
        report = repo.read_report_markdown(result.run_id)
        assert f"ranking_profile: {result.ranking_profile}" in report
        profile = get_ranking_profile(result.ranking_profile)
        payload = repo.read_inspect_payload(result.run_id, "AAA", profile=profile)
        assert payload.snapshot["run_id"] == result.run_id
        assert payload.snapshot["ticker"] == "AAA"
        assert payload.rankings
        assert {row["run_id"] for row in payload.rankings} == {result.run_id}


def test_pipeline_aggregates_fundamentals_requirements_for_multi_profile_provider_load(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    _install_fundamentals_required_profile(config, monkeypatch)
    captured_requirements = []

    class CountingFixtureProvider:
        def __init__(self, fixture_dir: str) -> None:
            self._provider = FixtureProvider(fixture_dir)

        def load_run_data(self, market: Market, requirements=None):
            captured_requirements.append(requirements)
            provider_data = self._provider.load_run_data(market)
            return replace(provider_data, fundamentals=_pipeline_fundamentals())

    monkeypatch.setattr("universe_selector.pipeline.FixtureProvider", CountingFixtureProvider)

    results = run_batch_profiles(
        Market.US,
        config,
        ("sample_price_trend_v1", "fundamentals_required_profile"),
    )

    assert len(results) == 2
    assert len(captured_requirements) == 1
    assert captured_requirements[0].fundamentals is True


def test_pipeline_fails_fundamentals_profile_when_provider_data_has_no_fundamentals(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_fixture_config(tmp_path, fixture_dir), ranking_profile="fundamentals_required_profile")
    _install_fundamentals_required_profile(config, monkeypatch)

    class MissingFundamentalsProvider:
        def __init__(self, fixture_dir: str) -> None:
            self._provider = FixtureProvider(fixture_dir)

        def load_run_data(self, market: Market, requirements=None):
            return self._provider.load_run_data(market)

    monkeypatch.setattr("universe_selector.pipeline.FixtureProvider", MissingFundamentalsProvider)

    with pytest.raises(ProviderDataError, match="requires fundamentals"):
        run_batch(Market.US, config)

    rows = (
        DuckDbRepository(config.duckdb_path)
        .connect(read_only=True)
        .execute("select status, error_message from run_log")
        .fetchall()
    )
    assert len(rows) == 1
    assert rows[0][0] == "failed"
    assert "requires fundamentals" in rows[0][1]


def test_pipeline_marks_fixture_fundamentals_required_run_failed_when_csv_is_missing(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_fundamentals_dir = tmp_path / "missing_fundamentals"
    shutil.copytree(fixture_dir, missing_fundamentals_dir)
    (missing_fundamentals_dir / "fundamentals.csv").unlink()
    config = replace(
        _fixture_config(tmp_path, missing_fundamentals_dir),
        ranking_profile="fundamentals_required_profile",
    )
    _install_fundamentals_required_profile(config, monkeypatch)

    with pytest.raises(ProviderDataError, match="fixture fundamentals are required but unavailable"):
        run_batch(Market.US, config)

    rows = (
        DuckDbRepository(config.duckdb_path)
        .connect(read_only=True)
        .execute("select status, error_message from run_log")
        .fetchall()
    )
    assert len(rows) == 1
    assert rows[0][0] == "failed"
    assert "fixture fundamentals are required but unavailable" in rows[0][1]


def test_pipeline_marks_fundamentals_run_failed_when_profile_filters_all_rows(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_fixture_config(tmp_path, fixture_dir), ranking_profile="fundamental_quality_profitability_v1")

    class StaleFundamentalsFixtureProvider:
        def __init__(self, fixture_dir: str) -> None:
            self._provider = FixtureProvider(fixture_dir)

        def load_run_data(self, market: Market, requirements=None):
            provider_data = self._provider.load_run_data(market)
            return replace(provider_data, fundamentals=_stale_pipeline_fundamentals())

    monkeypatch.setattr("universe_selector.pipeline.FixtureProvider", StaleFundamentalsFixtureProvider)

    with pytest.raises(ProviderDataError, match="fundamentals provider returned no eligible fundamentals for US"):
        run_batch(Market.US, config)

    rows = (
        DuckDbRepository(config.duckdb_path)
        .connect(read_only=True)
        .execute("select status, error_message from run_log")
        .fetchall()
    )
    assert len(rows) == 1
    assert rows[0][0] == "failed"
    assert "fundamentals provider returned no eligible fundamentals for US" in rows[0][1]


def test_pipeline_rejects_duplicate_multi_profile_ids_before_provider_load(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)

    def fail_provider_for(config: AppConfig):
        raise AssertionError("provider must not be constructed for duplicate profiles")

    monkeypatch.setattr("universe_selector.pipeline._provider_for", fail_provider_for)

    with pytest.raises(ValidationError, match="duplicate ranking profile momentum_v1"):
        run_batch_profiles(Market.US, config, ("momentum_v1", "momentum_v1"))


def test_pipeline_rejects_unknown_multi_profile_id_before_provider_load(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)

    def fail_provider_for(config: AppConfig):
        raise AssertionError("provider must not be constructed for unknown profiles")

    monkeypatch.setattr("universe_selector.pipeline._provider_for", fail_provider_for)

    with pytest.raises(ValidationError, match="unknown ranking profile unknown_profile"):
        run_batch_profiles(Market.US, config, ("unknown_profile", "momentum_v1"))


def test_pipeline_multi_profile_provider_failure_writes_no_runs(tmp_path: Path, fixture_dir: Path) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    empty_fixture_dir = tmp_path / "empty_fixture"
    empty_fixture_dir.mkdir()
    (empty_fixture_dir / "metadata.json").write_bytes((fixture_dir / "metadata.json").read_bytes())
    (empty_fixture_dir / "listings.csv").write_text((fixture_dir / "listings.csv").read_text().splitlines()[0] + "\n")
    (empty_fixture_dir / "ohlcv.csv").write_text(
        (fixture_dir / "ohlcv.csv").read_text().splitlines()[0] + "\nTW,ZZZ,2026-04-24,10.0,10.0,10.0,10.0,10.0,1000\n"
    )
    config = replace(config, fixture_dir=str(empty_fixture_dir))

    with pytest.raises(ProviderDataError, match="listing provider returned no usable listings"):
        run_batch_profiles(Market.US, config, ("sample_price_trend_v1", "momentum_v1"))

    repo = DuckDbRepository(config.duckdb_path)
    rows = repo.connect(read_only=True).execute("select count(*) from run_log").fetchone()
    assert rows == (0,)


class _FailingProfile:
    profile_id = "failing_profile"
    snapshot_metric_keys = ("return_60d", "return_120d")
    ranking_metric_keys = ("score_return_60d", "score_return_120d")
    inspect_metric_keys = ("return_60d", "return_120d")
    horizon_order = ("midterm", "longterm")
    rank_interpretation_note = "failure fixture"

    def __init__(self, delegate_profile) -> None:
        self._delegate_profile = delegate_profile

    def validate(self) -> None:
        return None

    def ranking_config_payload(self) -> dict[str, object]:
        return {"profile": self.profile_id, "version": 1}

    def build_snapshot(self, **kwargs):
        return self._delegate_profile.build_snapshot(**kwargs)

    def assign_rankings(self, snapshot):
        raise ValidationError("profile failed intentionally")


def _install_failing_profile(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real_pipeline_get_ranking_profile = pipeline.get_ranking_profile
    real_pipeline_get_ranking_profile_registration = getattr(
        pipeline,
        "get_ranking_profile_registration",
        get_ranking_profile_registration,
    )
    real_config_get_ranking_profile = get_ranking_profile
    delegate_profile = get_ranking_profile("sample_price_trend_v1")
    registration = RankingProfileRegistration(
        profile_id="failing_profile",
        factory=lambda: _FailingProfile(delegate_profile),
        data_requirements=RankingProfileDataRequirements(),
    )

    def fake_get_ranking_profile(profile_id: str):
        if profile_id == "failing_profile":
            return _FailingProfile(delegate_profile)
        return real_pipeline_get_ranking_profile(profile_id)

    def fake_config_get_ranking_profile(profile_id: str):
        if profile_id == "failing_profile":
            return _FailingProfile(delegate_profile)
        return real_config_get_ranking_profile(profile_id)

    def fake_get_ranking_profile_registration(profile_id: str):
        if profile_id == "failing_profile":
            return registration
        return real_pipeline_get_ranking_profile_registration(profile_id)

    monkeypatch.setattr(pipeline, "get_ranking_profile", fake_get_ranking_profile)
    monkeypatch.setattr("universe_selector.config.get_ranking_profile", fake_config_get_ranking_profile)
    monkeypatch.setattr("universe_selector.persistence.repository.get_ranking_profile", fake_get_ranking_profile)
    monkeypatch.setattr(
        pipeline, "get_ranking_profile_registration", fake_get_ranking_profile_registration, raising=False
    )


class _FundamentalsRequiredProfile:
    profile_id = "fundamentals_required_profile"
    snapshot_metric_keys = ("return_60d", "return_120d")
    ranking_metric_keys = ("score_return_60d", "score_return_120d")
    inspect_metric_keys = ("return_60d", "return_120d")
    horizon_order = ("midterm", "longterm")
    rank_interpretation_note = "fundamentals required fixture"

    def __init__(self, delegate_profile) -> None:
        self._delegate_profile = delegate_profile

    def validate(self) -> None:
        return None

    def ranking_config_payload(self) -> dict[str, object]:
        return {"profile": self.profile_id, "version": 1}

    def build_snapshot(self, **kwargs):
        return self._delegate_profile.build_snapshot(**kwargs)

    def assign_rankings(self, snapshot):
        return self._delegate_profile.assign_rankings(snapshot)


def _install_fundamentals_required_profile(config: AppConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    real_pipeline_get_ranking_profile = pipeline.get_ranking_profile
    real_pipeline_get_ranking_profile_registration = getattr(
        pipeline,
        "get_ranking_profile_registration",
        get_ranking_profile_registration,
    )
    real_config_get_ranking_profile = get_ranking_profile
    delegate_profile = get_ranking_profile("sample_price_trend_v1")

    registration = RankingProfileRegistration(
        profile_id="fundamentals_required_profile",
        factory=lambda: _FundamentalsRequiredProfile(delegate_profile),
        data_requirements=RankingProfileDataRequirements(fundamentals=True),
    )

    def fake_get_ranking_profile(profile_id: str):
        if profile_id == "fundamentals_required_profile":
            return _FundamentalsRequiredProfile(delegate_profile)
        return real_pipeline_get_ranking_profile(profile_id)

    def fake_config_get_ranking_profile(profile_id: str):
        if profile_id == "fundamentals_required_profile":
            return _FundamentalsRequiredProfile(delegate_profile)
        return real_config_get_ranking_profile(profile_id)

    def fake_get_ranking_profile_registration(profile_id: str):
        if profile_id == "fundamentals_required_profile":
            return registration
        return real_pipeline_get_ranking_profile_registration(profile_id)

    monkeypatch.setattr(pipeline, "get_ranking_profile", fake_get_ranking_profile)
    monkeypatch.setattr("universe_selector.config.get_ranking_profile", fake_config_get_ranking_profile)
    monkeypatch.setattr("universe_selector.persistence.repository.get_ranking_profile", fake_get_ranking_profile)
    monkeypatch.setattr(
        pipeline, "get_ranking_profile_registration", fake_get_ranking_profile_registration, raising=False
    )


def _pipeline_fundamentals() -> FundamentalsUniverseRunData:
    return FundamentalsUniverseRunData(
        metadata=FundamentalsMetadata(
            data_mode="fixture",
            fundamentals_provider_id="fixture-fundamentals-v1",
            fundamentals_source_ids=("sample_basic/fundamentals.csv",),
            data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
            latest_source_date=date(2026, 3, 31),
        ),
        facts=pl.DataFrame({"market": ["US"], "ticker": ["AAA"]}),
        coverage=FundamentalsCoverage(requested_count=1, returned_count=1, missing_count=0, invalid_count=0),
    )


def _stale_pipeline_fundamentals() -> FundamentalsUniverseRunData:
    return FundamentalsUniverseRunData(
        metadata=FundamentalsMetadata(
            data_mode="fixture",
            fundamentals_provider_id="fixture-fundamentals-v1",
            fundamentals_source_ids=("sample_basic/fundamentals.csv",),
            data_fetch_started_at=datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc),
            latest_source_date=date(2024, 1, 1),
        ),
        facts=pl.DataFrame(
            {
                "market": ["US"],
                "ticker": ["AAA"],
                "currency": ["USD"],
                "fiscal_period_end": [date(2024, 1, 1)],
                "balance_sheet_as_of": [date(2024, 1, 1)],
                "fiscal_period_type": ["ttm"],
                "revenue_ttm": [100.0],
                "gross_profit_ttm": [60.0],
                "operating_income_ttm": [30.0],
                "net_income_ttm": [20.0],
                "total_assets": [200.0],
                "shareholders_equity": [100.0],
                "total_debt": [50.0],
                "cash_and_cash_equivalents": [10.0],
                "operating_cash_flow_ttm": [25.0],
                "capital_expenditures_ttm": [5.0],
                "free_cash_flow_ttm": [20.0],
                "roe": [0.20],
                "roa": [0.10],
                "gross_margin": [0.60],
                "operating_margin": [0.30],
                "net_margin": [0.20],
                "debt_to_equity": [0.50],
                "fcf_margin": [0.20],
                "tag_fundamentals_annual_fallback": [0.0],
                "tag_negative_net_income": [0.0],
                "tag_negative_fcf": [0.0],
            }
        ),
        coverage=FundamentalsCoverage(requested_count=1, returned_count=1, missing_count=0, invalid_count=0),
    )


def test_pipeline_multi_profile_partial_failure_carries_completed_and_failed_runs(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    _install_failing_profile(config, monkeypatch)

    with pytest.raises(MultiProfileBatchError) as exc_info:
        run_batch_profiles(
            Market.US,
            config,
            ("sample_price_trend_v1", "failing_profile"),
        )

    exc = exc_info.value
    assert [result.ranking_profile for result in exc.completed_results] == ["sample_price_trend_v1"]
    assert exc.failed_result.ranking_profile == "failing_profile"
    assert exc.exit_code == ValidationError.exit_code

    repo = DuckDbRepository(config.duckdb_path)
    assert repo.resolve_successful_run(exc.completed_results[0].run_id).ranking_profile == "sample_price_trend_v1"
    failed = (
        repo.connect(read_only=True)
        .execute(
            "select status, ranking_profile from run_log where run_id = ?",
            [exc.failed_result.run_id],
        )
        .fetchone()
    )
    assert failed == ("failed", "failing_profile")


def test_pipeline_multi_profile_first_failure_carries_failed_run_only(
    tmp_path: Path,
    fixture_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    _install_failing_profile(config, monkeypatch)

    with pytest.raises(MultiProfileBatchError) as exc_info:
        run_batch_profiles(
            Market.US,
            config,
            ("failing_profile", "sample_price_trend_v1"),
        )

    exc = exc_info.value
    assert exc.completed_results == ()
    assert exc.failed_result.ranking_profile == "failing_profile"
    assert exc.exit_code == ValidationError.exit_code

    repo = DuckDbRepository(config.duckdb_path)
    failed = (
        repo.connect(read_only=True)
        .execute(
            "select status, ranking_profile from run_log where run_id = ?",
            [exc.failed_result.run_id],
        )
        .fetchone()
    )
    assert failed == ("failed", "failing_profile")
    successful_count = (
        repo.connect(read_only=True).execute("select count(*) from run_log where status = 'successful'").fetchone()
    )
    assert successful_count == (0,)
