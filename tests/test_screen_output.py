from __future__ import annotations

import json

from universe_selector.domain import Market
from universe_selector.output.screen import render_screen_json, render_screen_markdown
from universe_selector.persistence.repository import ResolvedRun
from universe_selector.screening import ScreenCandidate, ScreenResult


def _sample_result() -> ScreenResult:
    return ScreenResult(
        market=Market.US,
        profile_ids=("profile_a", "profile_b"),
        top_n=50,
        resolved_runs={
            "profile_a": ResolvedRun(
                run_id="us-00000001",
                market=Market.US,
                ranking_profile="profile_a",
                ranking_config_hash="hash-a",
            ),
            "profile_b": ResolvedRun(
                run_id="us-00000002",
                market=Market.US,
                ranking_profile="profile_b",
                ranking_config_hash="hash-b",
            ),
        },
        candidates=[
            ScreenCandidate(
                ticker="AAA",
                profile_count=2,
                avg_rank=1.5,
                profile_ranks={"profile_a": 1, "profile_b": 2},
            ),
            ScreenCandidate(
                ticker="BBB",
                profile_count=1,
                avg_rank=3.0,
                profile_ranks={"profile_a": 3},
            ),
        ],
    )


def test_markdown_contains_header_and_table() -> None:
    output = render_screen_markdown(_sample_result())

    assert "# Universe Selector Screen" in output
    assert "market: US" in output
    assert "profile_a" in output
    assert "profile_b" in output
    assert "AAA" in output
    assert "BBB" in output
    assert "| ticker" in output.lower() or "| Ticker" in output


def test_markdown_shows_rank_or_dash_per_profile() -> None:
    output = render_screen_markdown(_sample_result())

    lines = output.split("\n")
    aaa_line = next(l for l in lines if "AAA" in l)
    assert "1" in aaa_line
    assert "2" in aaa_line

    bbb_line = next(l for l in lines if "BBB" in l)
    assert "3" in bbb_line


def test_json_has_expected_structure() -> None:
    output = render_screen_json(_sample_result())
    payload = json.loads(output)

    assert payload["artifact_type"] == "universe_selector_screen"
    assert payload["market"] == "US"
    assert payload["profile_ids"] == ["profile_a", "profile_b"]
    assert payload["top_n"] == 50
    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["ticker"] == "AAA"
    assert payload["candidates"][0]["profile_count"] == 2
    assert payload["candidates"][0]["avg_rank"] == 1.5
    assert payload["candidates"][0]["profile_ranks"]["profile_a"] == 1


def test_json_includes_resolved_runs() -> None:
    output = render_screen_json(_sample_result())
    payload = json.loads(output)

    assert "resolved_runs" in payload
    assert payload["resolved_runs"]["profile_a"]["run_id"] == "us-00000001"
    assert payload["resolved_runs"]["profile_b"]["run_id"] == "us-00000002"
