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

You can also read an explicit persisted run:

```bash
uv run universe-selector report --run-id us-00000000-0000-4000-8000-000000000001
uv run universe-selector inspect --run-id us-00000000-0000-4000-8000-000000000001 --ticker AXTI
```

## Ranking Profiles

### `sample_price_trend_v1`

`sample_price_trend_v1` is the default public example profile. It is intentionally
simple and is meant to demonstrate the ranking profile boundary.

It computes:

- 60-day adjusted-close return for the `midterm` horizon.
- 120-day adjusted-close return for the `longterm` horizon.
- 20-day average traded value for liquidity filtering.
- Run-local percentile ranks within the same market and run.

It is a sample profile, not a production strategy recommendation.

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

Rank percentiles are:

- Run-local.
- Market-local.
- Profile-specific.
- Not comparable across different runs, markets, or ranking profiles.

High rank percentiles do not guarantee positive absolute performance, future
returns, liquidity, or tradability.

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
