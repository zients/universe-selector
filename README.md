# Universe Selector

Run-centric quantitative universe selector for repeatable market scans.

Universe Selector loads market listings and OHLCV data, runs a configured ranking
profile, persists the full run into DuckDB, and renders reports or per-ticker
inspect output from persisted results.

This project is an alpha-stage research tool. It is not investment advice.

## Current Status

- Public alpha quality.
- Command-line interface only.
- Supported markets: `US` and `TW`.
- Runtime config source: `config.yaml` in the current working directory.
- Default example ranking profile: `sample_price_trend_v1`.
- Persistence: local DuckDB database under `.universe-selector/` by default.

## What It Does

The CLI has three main commands:

- `batch MARKET`: fetch data, run the ranking profile, persist the run, and render a report artifact.
- `report MARKET`: print the latest successful report for the current configured ranking profile.
- `inspect MARKET --ticker TICKER`: print persisted metrics and rankings for one ticker.

`report` and `inspect` read persisted results. They do not recompute a batch run.

## Architecture

The runtime flow is intentionally small and explicit:

```text
config.yaml -> provider -> ranking profile -> DuckDB -> report / inspect
```

- `config.py` loads `config.yaml`, validates provider/profile IDs, and computes
  stable config hashes for persisted runs.
- `providers/` owns market data access. Listing providers return tradable listing
  candidates; OHLCV providers return canonical daily bar data.
- `ranking_profiles/` owns ranking logic. A profile builds a persisted ticker
  snapshot, assigns horizon rankings, and declares which metrics are persisted
  and inspectable.
- `pipeline.py` coordinates one batch run: load data, run the selected profile,
  persist the result, and render the report artifact.
- `persistence/` owns DuckDB migrations and read/write access for run logs,
  provider metadata, ticker snapshots, rankings, and report artifacts.
- `output/` renders markdown reports and per-ticker inspect output from persisted
  data.
- `cli.py` is the Typer command layer for `batch`, `report`, and `inspect`.

`batch` is the only command that fetches data or computes rankings. `report` and
`inspect` resolve a persisted successful run, then read DuckDB. This keeps output
reproducible for a specific run and prevents report rendering from silently
changing when provider data changes later.

The persistence schema stores stable run fields as columns and profile-specific
metrics in `metrics_json`. Ranking profiles declare their persisted metric keys,
so new profiles can add different metrics without adding profile-specific DuckDB
columns.

## Requirements

- Python `>=3.11,<3.15`
- `uv`

The project is tested in CI on Python 3.11 and 3.14.

## Setup

Clone the repository, install dependencies, and create a local runtime config:

```bash
git clone https://github.com/zients/universe-selector.git
cd universe-selector
uv sync --locked
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
uv run universe-selector inspect us --ticker AAA
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
  ticker_limit: null
  yfinance:
    batch_size: 200

ranking:
  profile: sample_price_trend_v1

report:
  top_n: 100
```

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

Print the latest successful US report for the current config profile:

```bash
uv run universe-selector report us
```

Inspect a ticker from the latest successful US run:

```bash
uv run universe-selector inspect us --ticker AXTI
```

Use a specific ranking profile for a new run or latest-run lookup:

```bash
uv run universe-selector batch us --ranking-profile sample_price_trend_v1
uv run universe-selector report us --ranking-profile sample_price_trend_v1
uv run universe-selector inspect us --ticker AXTI --ranking-profile sample_price_trend_v1
```

You can also read an explicit persisted run:

```bash
uv run universe-selector report --run-id us-00000000-0000-4000-8000-000000000001
uv run universe-selector inspect --run-id us-00000000-0000-4000-8000-000000000001 --ticker AXTI
```

`--run-id` reads one persisted run directly. Do not combine it with
`--ranking-profile`; the run already records its ranking profile.

## Ranking Profiles

### `sample_price_trend_v1`

`sample_price_trend_v1` is the default public example profile. It is intentionally
simple and is meant to demonstrate the ranking profile boundary.

It computes:

- 60-day adjusted-close return for the `midterm` horizon.
- 120-day adjusted-close return for the `longterm` horizon.
- 20-day average traded value for liquidity filtering.
- Raw adjusted-close return values are used as ranking scores for each horizon.

It is a sample profile, not a production strategy recommendation.

## Extending

Universe Selector is designed around two extension points: ranking profiles and
providers.

To add a ranking profile:

- Add a module under `src/universe_selector/ranking_profiles/`.
- Implement the `RankingProfile` protocol from `ranking_profiles/base.py`.
- Define a stable `profile_id` and include it in `ranking_config_payload()`.
- Implement `build_snapshot()` to turn provider data into one persisted row per
  surviving ticker.
- Implement `assign_rankings()` to produce one ranking row per ticker and
  profile horizon.
- Declare `snapshot_metric_keys`, `ranking_metric_keys`, and
  `inspect_metric_keys`; these keys control what is persisted in `metrics_json`
  and what `inspect` can print.
- Create a `RankingProfileRegistration` and add it to
  `ranking_profiles/registry.py`.
- Add tests for validation, snapshot construction, ranking assignment,
  persistence, report, and inspect behavior.

To add providers:

- Use `providers/models.py` for the data contracts.
- A listing provider returns `ListingCandidate` records for a market.
- An OHLCV provider returns canonical daily bars with `market`, `ticker`,
  `bar_date`, `open`, `high`, `low`, `close`, `adjusted_close`, and `volume`.
- Add a provider registration in the provider module and include it in
  `providers/registry.py`.
- Keep provider-specific source IDs stable, because they are part of persisted
  provider metadata and provider config hashes.
- Add tests for registration, provider parsing/normalization, and error cases.

Do not add profile-specific metrics as DuckDB columns. Keep profile-specific
values behind the metric key declarations so multiple profiles can coexist in
the same persistence model.

## Data Sources

Live mode currently uses:

- US listings: Nasdaq Trader files.
- Taiwan listings: TWSE ISIN HTML pages.
- OHLCV bars: Yahoo Finance through `yfinance`.

Third-party data can be delayed, incomplete, corrected, unavailable, or subject
to provider-specific terms and rate limits. A successful run only means the tool
completed with the data it received.

Supported market listings do not guarantee that every listed ticker can be
fetched from Yahoo Finance or included in rankings. The current `yfinance`
adapter skips unmappable symbols, normalizes some symbols for Yahoo Finance
requests, and can complete with partial OHLCV coverage when individual tickers
are missing or unavailable.

## Interpreting Output

Ranking profiles compute finite scores. Candidates are sorted by score
descending, and `rank` is the 1-based ordering derived from score within the
same run, market, profile, and horizon.

Scores are profile-defined ranking values. They may be 0-100 values, values
above 100, negative values, z-scores, composite model scores, or other finite
numeric values. Scores are only meaningful within the same run, market, profile,
and horizon unless a profile documents otherwise.

High scores do not guarantee positive absolute performance, future returns,
liquidity, or tradability.

## Development

Run tests:

```bash
uv run --locked pytest
```

Build package artifacts:

```bash
uv build --wheel --sdist
```

CI runs tests on Python 3.11 and 3.14, and builds both the wheel and source
distribution.

## Disclaimer

Universe Selector is provided for research and software engineering purposes.
It does not provide financial, investment, trading, tax, or legal advice. You
are responsible for validating data quality, methodology, assumptions, and any
use of the output.
