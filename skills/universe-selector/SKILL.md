---
name: universe-selector
description: "Navigate the Universe Selector repo: CLI workflows, ranking profiles, providers, valuation models, persistence, fixtures, and quality gates."
---

# Universe Selector

Use this skill when operating or changing the `universe-selector` repo. Keep it focused on repo navigation, command workflows, and project-specific boundaries.

## Project Shape

Universe Selector is a run-centric quantitative universe selector. A ranking run is the durable unit of work: one market, one ranking profile, one provider snapshot, one config hash, and persisted report artifacts.

Key flows:

```text
config.yaml -> provider -> ranking profile -> DuckDB -> report / inspect
valuation_assumptions/{market}/{ticker}.yaml + fundamentals provider -> valuation model -> stdout
```

Important boundary:

- `batch` computes ranking runs and persists results.
- `report` and `inspect` read persisted successful runs; they do not recompute rankings.
- `value` is a live, ephemeral single-ticker valuation command; it does not read or write ranking runs.
- Do not add profile-specific DuckDB columns. Ranking profiles persist profile-specific values through declared metric keys and `metrics_json`.

## Standard Commands

Set up dependencies:

```bash
uv sync
```

Run quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

Run a network-free fixture smoke workflow:

```bash
cp config.example.yaml config.yaml
# edit config.yaml so data_mode: fixture
uv run universe-selector batch us
uv run universe-selector report us
uv run universe-selector report us --json
uv run universe-selector inspect us --ticker AAA
uv run universe-selector inspect us --ticker AAA --json
```

Run valuation examples:

```bash
uv run universe-selector value us --ticker AAPL --assumptions valuation_assumptions/us/AAPL.yaml
uv run universe-selector value tw --ticker 2330 --assumptions valuation_assumptions/tw/2330.yaml
uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1 --json
```

`config.yaml`, `.universe-selector/`, `.venv/`, `.worktrees/`, and `uv.lock` are local state or generated files and are ignored by git.

## Finding Ranking Profiles

Start here:

- `docs/ranking-profiles.md` for supported profile IDs, purpose, horizons, and required history.
- `src/universe_selector/ranking_profiles/registry.py` for the runtime registration list.
- `src/universe_selector/ranking_profiles/base.py` for the `RankingProfile` protocol.
- `src/universe_selector/ranking_profiles/registration.py` for registration shape.
- `src/universe_selector/ranking_profiles/sample_price_trend_v1.py` for the smallest reference implementation.

For a specific profile:

1. Find its module under `src/universe_selector/ranking_profiles/`.
2. Find the corresponding `tests/test_*_profile.py`.
3. Check profile constants for `profile_id`, `horizon_order`, metric keys, schemas, filters, and interpretation notes.
4. Check `build_snapshot()` for candidate filtering and persisted snapshot metrics.
5. Check `assign_rankings()` for horizon scores, sort order, rank assignment, and ranking metric output.

When changing or adding profile behavior, verify at least:

- profile validation
- snapshot construction
- ranking assignment
- registry support
- persistence/report/inspect behavior when the surfaced metrics change

## Finding Providers

Start here:

- `src/universe_selector/providers/base.py` for provider protocols.
- `src/universe_selector/providers/models.py` for listing, OHLCV, metadata, and fundamentals contracts.
- `src/universe_selector/providers/registry.py` for provider lookup.
- `src/universe_selector/providers/registration.py` for provider registration.
- `src/universe_selector/providers/fixture.py` for deterministic fixture-mode behavior.

Live provider modules:

- `src/universe_selector/providers/nasdaq_trader.py`
- `src/universe_selector/providers/twse_isin.py`
- `src/universe_selector/providers/yfinance_ohlcv.py`
- `src/universe_selector/providers/yfinance_fundamentals.py`

Provider tests usually follow `tests/test_*_provider.py`. Keep provider IDs and source IDs stable because they contribute to persisted provider metadata and config hashes.

## Finding Valuation Models

Start here:

- `docs/valuation.md` for command behavior, supported models, and assumption-file rules.
- `src/universe_selector/valuation/registry.py` for supported valuation models.
- `src/universe_selector/valuation/service.py` for orchestration.
- `src/universe_selector/valuation/assumptions.py` for YAML loading and validation.
- `src/universe_selector/valuation/models.py` for shared data models.

Valuation assumptions live at:

```text
valuation_assumptions/{market}/{ticker}.yaml
```

Sample files:

- `valuation_assumptions/us/AAPL.yaml`
- `valuation_assumptions/tw/2330.yaml`

When changing valuation behavior, inspect the matching model module in `src/universe_selector/valuation/` and the corresponding `tests/test_valuation_*` file. Present supported model blocks are validated even when unselected, so stale or malformed assumptions should fail closed.

## Finding CLI, Persistence, and Output

CLI entry points:

- `src/universe_selector/cli.py` for Typer commands and argument validation.
- `src/universe_selector/pipeline.py` for batch execution.
- `src/universe_selector/config.py` for `config.yaml` loading and config hashing.

Persistence:

- `src/universe_selector/persistence/schema.py`
- `src/universe_selector/persistence/repository.py`
- `src/universe_selector/persistence/migrations/`

Output:

- `src/universe_selector/output/report.py`
- `src/universe_selector/output/inspect.py`
- `src/universe_selector/output/json.py`
- `src/universe_selector/output/value.py`
- `src/universe_selector/valuation/output.py`

Report JSON reads persisted artifacts. If report JSON shape changes, check `tests/test_output.py`, `tests/test_valuation_output.py`, and CLI tests.

## Before Editing

Read the smallest relevant set of files first. Prefer existing patterns over new abstractions.

Common first reads:

- ranking work: `docs/ranking-profiles.md`, profile module, matching profile test, registry
- provider work: provider protocol/model files, provider module, matching provider test, registry
- valuation work: `docs/valuation.md`, assumptions loader, selected model module, matching valuation test
- CLI behavior: `src/universe_selector/cli.py`, `tests/test_cli.py`, relevant output/persistence files

Keep changes scoped. Do not refactor unrelated large profile modules unless the task requires it.

## Before Finishing

Run targeted tests for the touched area, then the full gates when practical:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
```

If only documentation or skill metadata changed, at minimum verify file paths and Markdown/YAML structure, then run a lightweight repository test such as:

```bash
uv run pytest -q tests/test_readme_contract.py tests/test_package_metadata.py
```
