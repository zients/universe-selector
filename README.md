# Universe Selector

Run-centric quantitative universe selector for repeatable market scans.

Universe Selector loads market listings and OHLCV data, runs one or more ranking
profiles, persists each full run into DuckDB, and renders reports or per-ticker
inspect output from persisted results.

A run is the durable unit of work: one market, one ranking profile, one provider
snapshot, one ranking config hash, and one rendered report. Reports and inspect
output read those persisted runs instead of recomputing rankings.

This project is an alpha-stage research tool. It is not investment advice.

## Current Status

- Public alpha quality.
- Command-line interface only.
- Supported markets: `US` and `TW`.
- Valuation command: `value` uses yfinance fundamentals for `US` and `TW` in v1.
- Runtime config source: `config.yaml` in the current working directory.
- Default example ranking profile: `sample_price_trend_v1`.
- Supported ranking profiles: `sample_price_trend_v1`, `momentum_v1`,
  `momentum_quality_v1`, `trend_quality_v1`, `trend_pullback_quality_v1`,
  `volatility_quality_v1`, `liquidity_quality_v1`,
  `base_breakout_quality_v1`, and `defensive_compounder_quality_v1`.
- Single-profile and multi-profile batch runs are supported.
- Persistence: local DuckDB database under `.universe-selector/` by default.

## Documentation

- [Valuation](docs/valuation.md)
- [Ranking profiles](docs/ranking-profiles.md)
- [Extending](docs/extending.md)
- [Data and output](docs/data-and-output.md)

## What It Does

The CLI has four command families:

- `batch MARKET`: fetch provider data, run one or more ranking profiles, persist
  each run, and render report artifacts.
- `report MARKET`: print the latest successful report for the current configured
  ranking profile, or for a `--ranking-profile` override.
- `report --run-id RUN_ID`: print one explicit persisted successful run.
- `inspect MARKET --ticker TICKER`: print persisted metrics and rankings for one
  ticker from the latest successful run for the current configured ranking
  profile, or for a `--ranking-profile` override.
- `inspect --run-id RUN_ID --ticker TICKER`: inspect one ticker from one explicit
  persisted successful run.
- `value MARKET --ticker TICKER`: run a live ephemeral single-ticker valuation
  analysis and print markdown or JSON to stdout.

`report` and `inspect` read persisted results. They do not recompute a batch run.
`value` does not read or write persisted ranking runs.
Add `--json` to `report`, `inspect`, or `value` to print machine-readable JSON.
`batch` persists both markdown and JSON report artifacts; `report --json` reads
the persisted JSON artifact for the selected run. Report JSON includes the full
persisted ticker snapshots and rankings plus a `top_horizons` report view.

## Architecture

The runtime flow is intentionally small and explicit:

```text
config.yaml -> provider -> ranking profile -> DuckDB -> report / inspect
```

Valuation is intentionally separate from ranking persistence:

```text
valuation_assumptions/{market}/{ticker}.yaml + fundamentals provider -> valuation model -> stdout
```

- `config.py` loads `config.yaml`, validates provider/profile IDs, and computes
  stable config hashes for persisted runs.
- `providers/` owns market data access. Listing providers return tradable listing
  candidates; OHLCV providers return canonical daily bar data.
- `ranking_profiles/` owns ranking logic. A profile builds a persisted ticker
  snapshot, assigns horizon rankings, and declares which metrics are persisted
  and inspectable.
- `pipeline.py` coordinates batch execution: load provider data, run the selected
  profile or profiles, persist each result, and render report artifacts.
- `persistence/` owns DuckDB migrations and read/write access for run logs,
  provider metadata, ticker snapshots, rankings, and report artifacts.
- `output/` renders markdown and JSON reports plus per-ticker inspect output from
  persisted data, plus thin command output adapters.
- `valuation/` owns valuation assumptions, model logic, orchestration, and
  valuation output.
- `cli.py` is the Typer command layer for `batch`, `report`, `inspect`, and
  `value`.

`batch` is the only command that computes and persists ranking runs. `report` and
`inspect` resolve a persisted successful run, then read DuckDB. This keeps output
reproducible for a specific run and prevents report rendering from silently
changing when provider data changes later.

`value` fetches fundamentals for valuation separately from ranking runs.

batch remains the only command that computes persisted rankings. `report` and
`inspect` still only read persisted ranking runs. `value` is a live ephemeral
single-ticker valuation analysis and is not persisted in v1.

## Requirements

- Python `>=3.11,<3.15`
- `uv`

The project is tested in CI on Python 3.11 and 3.14.

## Setup

Clone the repository, install dependencies, and create a local runtime config:

```bash
git clone https://github.com/zients/universe-selector.git
cd universe-selector
uv sync
cp config.example.yaml config.yaml
```

`config.yaml` is intentionally ignored by git. Edit it locally for runtime paths,
data mode, providers, ranking profile, and report size.

The default `.universe-selector/` directory is also ignored by git. It is local
runtime state and contains the DuckDB database with persisted runs, report
artifacts, inspect data, and the batch lock file.

## Fixture Smoke Run

For a network-free smoke run, copy `config.example.yaml` to `config.yaml` and set:

```yaml
data_mode: fixture
```

Then run:

```bash
uv run universe-selector batch us
uv run universe-selector report us
uv run universe-selector report us --json
uv run universe-selector inspect us --ticker AAA
uv run universe-selector inspect us --ticker AAA --json
```

## Configuration

The application reads only `config.yaml` from the current working directory.
Environment variables are not used for selecting or overriding runtime config.

The default example config is:

```yaml
data_mode: live
duckdb_path: .universe-selector/universe_selector.duckdb
lock_path: .universe-selector/batch.lock
fixture_dir: tests/fixtures/sample_basic

live:
  listing_provider:
    US: nasdaq_trader
    TW: twse_isin
  ohlcv_provider: yfinance
  fundamentals_provider: yfinance_fundamentals
  ticker_limit: null
  yfinance:
    batch_size: 200

ranking:
  profile: sample_price_trend_v1

report:
  top_n: 100
```

`ranking.profile` may be any supported profile id:

- `sample_price_trend_v1`
- `momentum_v1`
- `momentum_quality_v1`
- `trend_quality_v1`
- `trend_pullback_quality_v1`
- `volatility_quality_v1`
- `liquidity_quality_v1`
- `base_breakout_quality_v1`
- `defensive_compounder_quality_v1`

For quick smoke runs against a smaller live universe, set:

```yaml
live:
  ticker_limit: 25
```

Keep the rest of the `live` section intact when editing this value.

## Example Workflow

Run a US batch:

```bash
uv run universe-selector batch us
```

Use a specific ranking profile for a new run or latest-run lookup:

```bash
uv run universe-selector batch us --ranking-profile sample_price_trend_v1
uv run universe-selector report us --ranking-profile sample_price_trend_v1
uv run universe-selector inspect us --ticker AXTI --ranking-profile sample_price_trend_v1
uv run universe-selector inspect us --ticker AXTI --json
```

Run several ranking profiles from one provider data load:

```bash
uv run universe-selector batch us \
  --ranking-profile trend_quality_v1 \
  --ranking-profile momentum_v1 \
  --ranking-profile volatility_quality_v1
```

Each profile is persisted as its own run. `report` and `inspect` continue
to read one run at a time; use `--ranking-profile` to resolve the latest
run for a specific profile.

You can also read an explicit persisted run:

```bash
uv run universe-selector report --run-id us-00000000-0000-4000-8000-000000000001
uv run universe-selector inspect --run-id us-00000000-0000-4000-8000-000000000001 --ticker AXTI
```

`--run-id` reads one persisted run directly. Do not combine it with
`--ranking-profile`; the run already records its ranking profile.

Run an ephemeral valuation analysis:

```bash
uv run universe-selector value us --ticker AAPL
uv run universe-selector value us --ticker AAPL --json
uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1
uv run universe-selector value us --ticker AAPL --model exit_multiple_dcf_v1
uv run universe-selector value us --ticker AAPL --model reverse_dcf_v1
uv run universe-selector value us --ticker AAPL --model multiple_valuation_v1
uv run universe-selector value us --ticker AAPL --model two_stage_fcf_dcf_v1
uv run universe-selector value us --ticker AAPL --model implied_discount_rate_v1
uv run universe-selector value us --ticker AAPL \
  --assumptions valuation_assumptions/us/AAPL.yaml
uv run universe-selector value tw --ticker 2330 \
  --assumptions valuation_assumptions/tw/2330.yaml
```

Supported valuation models are `exit_multiple_dcf_v1`, `fcf_dcf_v1`,
`implied_discount_rate_v1`, `multiple_valuation_v1`, `reverse_dcf_v1`, and
`two_stage_fcf_dcf_v1`.

See [Valuation](docs/valuation.md) for model details and assumption semantics.
See [Ranking profiles](docs/ranking-profiles.md) for ranking profile behavior.

## Development

Run tests:

```bash
uv run pytest
```

Run local quality gates:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
```

Build package artifacts:

```bash
uv build --wheel --sdist
```

CI runs formatting, linting, type checks, tests on Python 3.11 and 3.14, and
builds both the wheel and source distribution.

## Disclaimer

Universe Selector is provided for research and software engineering purposes.
It does not provide financial, investment, trading, tax, or legal advice. You
are responsible for validating data quality, methodology, assumptions, and any
use of the output.
