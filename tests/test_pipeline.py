from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from universe_selector.config import AppConfig
from universe_selector.domain import Market
from universe_selector.errors import ProviderDataError
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.pipeline import run_batch


def _fixture_config(tmp_path: Path, fixture_dir: Path) -> AppConfig:
    return AppConfig(
        data_mode="fixture",
        duckdb_path=str(tmp_path / "runs.duckdb"),
        lock_path=str(tmp_path / "batch.lock"),
        fixture_dir=str(fixture_dir),
        ranking_profile="sample_price_trend_v1",
        report_top_n=2,
    )


def test_pipeline_runs_sample_profile_and_persists_report(tmp_path: Path, fixture_dir: Path) -> None:
    config = _fixture_config(tmp_path, fixture_dir)

    result = run_batch(Market.US, config)

    repo = DuckDbRepository(config.duckdb_path)
    resolved = repo.resolve_latest_successful_run(Market.US, ranking_profile="sample_price_trend_v1")
    report = repo.read_report_markdown(result.run_id)
    payload = repo.read_inspect_payload(result.run_id, "AAA", profile=config.selected_ranking_profile)

    assert resolved.run_id == result.run_id
    assert resolved.ranking_profile == "sample_price_trend_v1"
    assert "# Universe Selector Report" in report
    assert "ranking_profile: sample_price_trend_v1" in report
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
    assert resolved.ranking_profile == "momentum_v1"
    assert "risk_adjusted_momentum_12_1" in payload.snapshot
    assert "score_risk_adjusted_momentum_12_1" in payload.rankings[0]
    assert {row["horizon"] for row in payload.rankings} == {"swing", "midterm"}


def test_pipeline_runs_liquidity_quality_profile_and_persists_metrics(
    tmp_path: Path, fixture_dir: Path
) -> None:
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


def test_pipeline_runs_volatility_quality_profile_and_persists_metrics(
    tmp_path: Path, fixture_dir: Path
) -> None:
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


def test_pipeline_marks_failed_run_when_provider_has_no_usable_rows(tmp_path: Path, fixture_dir: Path) -> None:
    config = _fixture_config(tmp_path, fixture_dir)
    empty_fixture_dir = tmp_path / "empty_fixture"
    empty_fixture_dir.mkdir()
    (empty_fixture_dir / "metadata.json").write_bytes((fixture_dir / "metadata.json").read_bytes())
    (empty_fixture_dir / "listings.csv").write_text((fixture_dir / "listings.csv").read_text().splitlines()[0] + "\n")
    (empty_fixture_dir / "ohlcv.csv").write_text(
        (fixture_dir / "ohlcv.csv").read_text().splitlines()[0]
        + "\nTW,ZZZ,2026-04-24,10.0,10.0,10.0,10.0,10.0,1000\n"
    )

    with pytest.raises(ProviderDataError, match="listing provider returned no usable listings"):
        run_batch(Market.US, replace(config, fixture_dir=str(empty_fixture_dir)))
