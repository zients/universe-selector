---
name: universe-selector-cross-profile
description: "Synthesize Universe Selector outputs across multiple ranking profiles using persisted report/inspect JSON. Use when comparing profiles, finding consensus or divergent tickers, explaining overlap across momentum/trend/quality/liquidity profiles, or preparing LLM-driven cross-profile research from existing ranking runs."
---

# Universe Selector Cross-Profile Synthesis

Use this skill for research synthesis across multiple Universe Selector ranking profiles. Do not add a CLI command for this workflow. Keep deterministic work in existing commands and use Codex/LLM reasoning only after collecting persisted evidence.

## Core Boundary

- `batch` computes and persists ranking runs.
- `report --json` and `inspect --json` read persisted successful runs.
- Cross-profile synthesis is an agent workflow, not a ranking model.
- Do not average raw scores across profiles. Scores are only meaningful within the same run, market, profile, and horizon.
- Treat rank overlap as a heuristic. Explain limitations clearly.

## Workflow

1. Identify the market, ranking profiles, and whether the user needs same-snapshot comparison.

2. If same-snapshot comparison matters and the runs may be stale or mixed, run one multi-profile batch:

   ```bash
   uv run universe-selector batch us \
     --ranking-profile momentum_quality_v1 \
     --ranking-profile trend_quality_v1 \
     --ranking-profile relative_strength_leader_v1
   ```

3. Collect persisted JSON reports for each profile:

   ```bash
   uv run universe-selector report us --ranking-profile momentum_quality_v1 --json
   uv run universe-selector report us --ranking-profile trend_quality_v1 --json
   uv run universe-selector report us --ranking-profile relative_strength_leader_v1 --json
   ```

4. Before comparing profiles, verify comparable run context across report payloads:

   - `market`
   - `mode_label`
   - `provider_summary.provider_config_hash`
   - `provider_summary.run_latest_bar_date`
   - listing and OHLCV provider IDs/source IDs

   If these differ, either rerun a multi-profile batch or state that the synthesis mixes snapshots.

5. Build a working comparison from report JSON:

   - Use `ranking_profile`, `ranking_config_hash`, and report run IDs for traceability.
   - Use each profile's documented horizon order; do not assume every profile has `composite`.
   - Find tickers that appear in multiple top-horizon lists.
   - Flag tickers that are strong in one profile but absent or weak in another.
   - Note profile families that may be correlated, such as momentum and relative strength.

6. Use `inspect --json` for tickers that need explanation:

   ```bash
   uv run universe-selector inspect us --ticker AAPL --ranking-profile momentum_quality_v1 --json
   uv run universe-selector inspect us --ticker AAPL --ranking-profile trend_quality_v1 --json
   ```

7. Synthesize a concise research brief:

   - Consensus candidates: tickers supported by multiple distinct profiles.
   - Divergent candidates: tickers where profiles disagree.
   - Evidence: ranks, horizons, key metrics, profile tags, run IDs, and config hashes.
   - Interpretation: why the profiles agree or disagree.
   - Limitations: snapshot consistency, profile correlation, missing fundamentals, and no investment advice.

## Output Shape

Prefer this structure unless the user asks for another format:

```text
## Context
- market:
- profiles:
- run_ids:
- snapshot consistency:

## Consensus Candidates
| ticker | evidence | interpretation | caveats |

## Divergent Candidates
| ticker | profile split | interpretation | next check |

## Follow-Up Inspect Targets
- ticker: reason

## Limitations
```

Keep conclusions tied to persisted evidence. If evidence is missing, say what command would gather it.
