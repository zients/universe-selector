from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.persistence.repository import DuckDbRepository, ResolvedRun


@dataclass(frozen=True)
class ScreenCandidate:
    ticker: str
    profile_count: int
    avg_rank: float
    profile_ranks: dict[str, int]


@dataclass(frozen=True)
class ScreenResult:
    market: Market
    profile_ids: tuple[str, ...]
    top_n: int
    resolved_runs: dict[str, ResolvedRun]
    candidates: list[ScreenCandidate]


def _validate_screen_profiles(profile_ids: tuple[str, ...]) -> None:
    if len(profile_ids) < 2:
        raise ValidationError("provide at least two ranking profiles for screening")
    seen: set[str] = set()
    for profile_id in profile_ids:
        if profile_id in seen:
            raise ValidationError(f"duplicate ranking profile {profile_id}")
        seen.add(profile_id)


def run_screen(
    repo: DuckDbRepository,
    market: Market,
    profile_ids: tuple[str, ...],
    *,
    top_n: int,
) -> ScreenResult:
    _validate_screen_profiles(profile_ids)

    resolved_runs: dict[str, ResolvedRun] = {}
    for profile_id in profile_ids:
        resolved = repo.resolve_latest_successful_run(market, ranking_profile=profile_id)
        resolved_runs[profile_id] = resolved

    ticker_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    connection = repo.connect(read_only=True)
    for profile_id in profile_ids:
        run_id = resolved_runs[profile_id].run_id
        available_horizons = sorted(
            row[0] for row in connection.execute(
                "select distinct horizon from run_rankings where run_id = ?",
                [run_id],
            ).fetchall()
        )
        primary_horizon = "composite" if "composite" in available_horizons else (available_horizons[0] if available_horizons else "composite")
        rows = connection.execute(
            "select ticker, rank from run_rankings "
            "where run_id = ? and horizon = ? and rank <= ? "
            "order by rank",
            [run_id, primary_horizon, top_n],
        ).fetchall()
        for ticker, rank in rows:
            ticker_ranks[ticker][profile_id] = int(rank)

    candidates = []
    for ticker, profile_rank_map in ticker_ranks.items():
        count = len(profile_rank_map)
        avg_rank = sum(profile_rank_map.values()) / count
        candidates.append(
            ScreenCandidate(
                ticker=ticker,
                profile_count=count,
                avg_rank=avg_rank,
                profile_ranks=dict(profile_rank_map),
            )
        )

    candidates.sort(key=lambda c: (-c.profile_count, c.avg_rank))

    return ScreenResult(
        market=market,
        profile_ids=profile_ids,
        top_n=top_n,
        resolved_runs=resolved_runs,
        candidates=candidates,
    )
