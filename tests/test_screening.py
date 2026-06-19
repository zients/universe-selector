from __future__ import annotations

import duckdb
import pytest

from universe_selector.domain import Market
from universe_selector.errors import NotFoundError, ValidationError
from universe_selector.persistence.repository import DuckDbRepository
from universe_selector.persistence.schema import apply_migrations
from universe_selector.screening import ScreenResult, run_screen


def _seed_screen_runs(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        insert into run_log(run_id, market, status, created_at, ranking_profile, ranking_config_hash, error_message)
        values
          ('us-00000001', 'US', 'successful', timestamp '2026-01-01 00:00:00', 'sample_price_trend_v1', 'hash-a', null),
          ('us-00000002', 'US', 'successful', timestamp '2026-01-01 00:00:00', 'momentum_v1', 'hash-b', null)
        """
    )
    connection.execute(
        """
        insert into run_rankings(run_id, market, horizon, ticker, score, rank, metrics_json)
        values
          ('us-00000001', 'US', 'midterm', 'AAA', 0.9, 1, '{}'),
          ('us-00000001', 'US', 'midterm', 'BBB', 0.8, 2, '{}'),
          ('us-00000001', 'US', 'midterm', 'CCC', 0.7, 3, '{}'),
          ('us-00000002', 'US', 'swing', 'AAA', 0.95, 1, '{}'),
          ('us-00000002', 'US', 'swing', 'DDD', 0.85, 2, '{}'),
          ('us-00000002', 'US', 'swing', 'BBB', 0.75, 3, '{}')
        """
    )


def test_screen_cross_references_profiles(tmp_path) -> None:
    repo = DuckDbRepository(str(tmp_path / "test.duckdb"))
    connection = repo.connect()
    apply_migrations(connection)
    _seed_screen_runs(connection)

    result = run_screen(repo, Market.US, ("sample_price_trend_v1", "momentum_v1"), top_n=3)

    assert isinstance(result, ScreenResult)
    assert result.market is Market.US
    assert result.profile_ids == ("sample_price_trend_v1", "momentum_v1")
    assert result.top_n == 3

    tickers = [c.ticker for c in result.candidates]
    assert tickers[0] == "AAA"
    assert result.candidates[0].profile_count == 2
    assert result.candidates[0].avg_rank == 1.0
    assert result.candidates[0].profile_ranks == {"sample_price_trend_v1": 1, "momentum_v1": 1}

    assert "BBB" in tickers
    bbb = next(c for c in result.candidates if c.ticker == "BBB")
    assert bbb.profile_count == 2
    assert bbb.avg_rank == 2.5
    assert bbb.profile_ranks == {"sample_price_trend_v1": 2, "momentum_v1": 3}

    # DDD only in momentum_v1 top 3
    ddd = next(c for c in result.candidates if c.ticker == "DDD")
    assert ddd.profile_count == 1


def test_screen_sorts_by_profile_count_then_avg_rank(tmp_path) -> None:
    repo = DuckDbRepository(str(tmp_path / "test.duckdb"))
    connection = repo.connect()
    apply_migrations(connection)
    _seed_screen_runs(connection)

    result = run_screen(repo, Market.US, ("sample_price_trend_v1", "momentum_v1"), top_n=3)

    counts_and_ranks = [(c.profile_count, c.avg_rank) for c in result.candidates]
    assert counts_and_ranks == sorted(counts_and_ranks, key=lambda x: (-x[0], x[1]))


def test_screen_rejects_fewer_than_two_profiles(tmp_path) -> None:
    repo = DuckDbRepository(str(tmp_path / "test.duckdb"))
    connection = repo.connect()
    apply_migrations(connection)

    with pytest.raises(ValidationError, match="at least two"):
        run_screen(repo, Market.US, ("sample_price_trend_v1",), top_n=50)


def test_screen_rejects_duplicate_profiles(tmp_path) -> None:
    repo = DuckDbRepository(str(tmp_path / "test.duckdb"))
    connection = repo.connect()
    apply_migrations(connection)

    with pytest.raises(ValidationError, match="duplicate"):
        run_screen(repo, Market.US, ("sample_price_trend_v1", "sample_price_trend_v1"), top_n=50)


def test_screen_raises_not_found_when_profile_has_no_run(tmp_path) -> None:
    repo = DuckDbRepository(str(tmp_path / "test.duckdb"))
    connection = repo.connect()
    apply_migrations(connection)
    _seed_screen_runs(connection)

    with pytest.raises(NotFoundError):
        run_screen(repo, Market.US, ("sample_price_trend_v1", "missing_profile"), top_n=3)
