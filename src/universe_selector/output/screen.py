from __future__ import annotations

from universe_selector.output.json import json_dumps
from universe_selector.screening import ScreenResult


SCREEN_RESEARCH_DISCLAIMER = "This screen is for quantitative research only and is not investment advice."


def render_screen_markdown(result: ScreenResult) -> str:
    profiles = result.profile_ids
    short_ids = {p: p for p in profiles}

    header_cols = ["Ticker", "#Prof", "AvgRk"] + [short_ids[p] for p in profiles]
    header = "| " + " | ".join(header_cols) + " |"
    separator = "|---|---:|---:|" + "|".join("---:" for _ in profiles) + "|"

    rows = []
    for c in result.candidates:
        rank_cells = []
        for p in profiles:
            if p in c.profile_ranks:
                rank_cells.append(str(c.profile_ranks[p]))
            else:
                rank_cells.append("—")
        row = f"| {c.ticker} | {c.profile_count} | {c.avg_rank:.1f} | " + " | ".join(rank_cells) + " |"
        rows.append(row)

    profile_lines = "\n".join(f"- {p}: {result.resolved_runs[p].run_id}" for p in profiles)

    return f"""# Universe Selector Screen

> {SCREEN_RESEARCH_DISCLAIMER}

## Screen Context

- market: {result.market.value}
- top_n per profile: {result.top_n}
- profiles screened: {len(profiles)}
- candidates found: {len(result.candidates)}

## Resolved Runs

{profile_lines}

## Cross-Profile Screening

{header}
{separator}
{chr(10).join(rows)}

## Methodology Notes

- Candidates are ranked by number of profiles in which they appear (descending), then by average composite rank (ascending).
- A dash (—) means the ticker was not in that profile's top {result.top_n} composite.
- Scores are not comparable across profiles; only ranks are used for screening.
- {SCREEN_RESEARCH_DISCLAIMER}
"""


def render_screen_json(result: ScreenResult) -> str:
    payload = {
        "artifact_type": "universe_selector_screen",
        "market": result.market.value,
        "profile_ids": list(result.profile_ids),
        "top_n": result.top_n,
        "resolved_runs": {
            profile_id: {
                "run_id": run.run_id,
                "ranking_profile": run.ranking_profile,
                "ranking_config_hash": run.ranking_config_hash,
            }
            for profile_id, run in result.resolved_runs.items()
        },
        "candidates": [
            {
                "ticker": c.ticker,
                "profile_count": c.profile_count,
                "avg_rank": c.avg_rank,
                "profile_ranks": c.profile_ranks,
            }
            for c in result.candidates
        ],
        "notes": [
            SCREEN_RESEARCH_DISCLAIMER,
            "Candidates are ranked by profile count (descending), then average composite rank (ascending).",
            "Scores are not comparable across profiles; only ranks are used for screening.",
        ],
    }
    return json_dumps(payload) + "\n"
