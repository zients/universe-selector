from __future__ import annotations

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
