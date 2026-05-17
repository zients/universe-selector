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
  `trend_quality_v1`, `volatility_quality_v1`, and `liquidity_quality_v1`.
- Single-profile and multi-profile batch runs are supported.
- Persistence: local DuckDB database under `.universe-selector/` by default.

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
  analysis and print markdown to stdout.

`report` and `inspect` read persisted results. They do not recompute a batch run.
`value` does not read or write persisted ranking runs.

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
- `output/` renders markdown reports and per-ticker inspect output from persisted
  data, plus thin command output adapters.
- `valuation/` owns valuation assumptions, model logic, orchestration, and
  valuation markdown output.
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

When `batch` receives more than one `--ranking-profile`, provider data is loaded
once and then reused for each selected profile. Each profile is persisted as a
separate run with its own run id, ranking config hash, rankings, and report.

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
- `trend_quality_v1`
- `volatility_quality_v1`
- `liquidity_quality_v1`

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

If any profile fails during a multi-profile batch, completed profile runs remain
persisted and the CLI prints the completed run ids plus the failed run id, failed
profile, and error.

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
uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1
uv run universe-selector value us --ticker AAPL \
  --assumptions valuation_assumptions/us/AAPL.yaml
uv run universe-selector value tw --ticker 2330 \
  --assumptions valuation_assumptions/tw/2330.yaml
```

`value` v1 prints markdown only. It requires `config.yaml` only for selecting
`live.fundamentals_provider`, does not read DuckDB, and does not persist the result.
The default assumptions path is
`valuation_assumptions/{market}/{ticker}.yaml`; the committed
`valuation_assumptions/us/AAPL.yaml` and `valuation_assumptions/tw/2330.yaml`
are sample schemas only and are not investment advice. Each valuation assumptions
file declares a root `default_model`; `value` uses the assumptions file
`default_model` when `--model` is omitted. `--model` explicitly overrides the
assumptions file default model. Assumption schema `1` requires root
`default_model`. The committed valuation assumption files are repository
templates; installed wheels do not copy them into your working directory. Create
your own assumptions file in the working directory or pass `--assumptions`.

`fcf_dcf_v1` uses `models.fcf_dcf_v1.starting_fcf` to choose the DCF starting
FCF. The committed templates default to `starting_fcf.method: provider_ttm_fcf`,
which uses provider raw FCF as a starting proxy so the command can run directly.
Set `starting_fcf.method: override` with `value` and `note` when using an
analyst-normalized FCF.

`fcf_dcf_v1` is a simplified free-cash-flow DCF model. It uses starting FCF as
an enterprise cash-flow proxy, not verified unlevered FCFF, and computes
model-implied scenario results against a reference price. Results are highly
sensitive to starting FCF, share count, discount rate, terminal growth, and
terminal value assumptions. Scenarios are illustrative and are not forecasts,
expected outcomes, target cases, or recommendations.

`value` uses yfinance fundamentals for v1 `US` and `TW` live facts. TW tickers
default to the yfinance `.TW` request suffix. yfinance fundamentals are
third-party convenience data and may be stale, incomplete, restated, mapped
inconsistently, or unavailable. Independently verify provider facts and validate
or override assumptions before relying on model-implied outputs.

## Ranking Profiles

Ranking profiles are independent scoring lenses. They share the same persisted
run model but define their own candidate filters, metric keys, horizons, scores,
and interpretation notes.

| Profile | Purpose | Horizons | Required History |
|---|---|---|---:|
| `sample_price_trend_v1` | Minimal example profile for smoke runs and extension patterns. | `midterm`, `longterm` | 121 bars |
| `momentum_v1` | Raw weighted momentum profile using risk-adjusted medium-term momentum and short-term strength. | `swing`, `midterm` | 274 bars |
| `trend_quality_v1` | Market-relative trend profile using returns, trend slope, trend fit, moving-average structure, drawdown control, caps, and structure tags. | `composite`, `shortterm`, `midterm` | 252 bars |
| `volatility_quality_v1` | Market-relative quality profile favoring lower realized volatility, downside volatility control, range tightness, and drawdown control. | `composite`, `shortterm`, `stable` | 126 bars |
| `liquidity_quality_v1` | Market-relative liquidity profile using traded value depth, friction proxies, traded value stability, concentration, continuity, and range tightness. | `composite`, `shortterm`, `stable` | 63 bars |

All profile scores are ranking values, not return forecasts. Higher score ranks
better within the same run, market, profile, and horizon unless the profile
documents a narrower interpretation.

### `sample_price_trend_v1`

`sample_price_trend_v1` is the default public example profile. It is intentionally
simple and is meant to demonstrate the ranking profile boundary.

It computes:

- 60-day adjusted-close return for the `midterm` horizon.
- 120-day adjusted-close return for the `longterm` horizon.
- 20-day average traded value for liquidity filtering.
- Raw adjusted-close return values are used as ranking scores for each horizon.

It is a sample profile, not a production strategy recommendation.

### `momentum_v1`

`momentum_v1` is a raw weighted momentum profile. It computes 12-1 and 6-1
momentum returns, realized volatility over those same windows, risk-adjusted
momentum factors, and 20-day short-term strength.

It ranks two horizons:

- `swing`: shorter momentum lens emphasizing 6-1 risk-adjusted momentum and
  20-day strength.
- `midterm`: medium-term momentum lens emphasizing 12-1 and 6-1 risk-adjusted
  momentum.

Scores are raw weighted composites and are not bounded to 0-100.

### `trend_quality_v1`

`trend_quality_v1` is a market-relative trend quality profile. It combines
absolute and percentile components for recent returns, trend slope, trend fit,
moving-average structure, breakout position, drawdown control, and penalties.

It also persists non-exclusive structure tags such as `tag_structure_uptrend`,
`tag_structure_consistent_uptrend`, `tag_structure_negative_60d_return`,
`tag_structure_large_drawdown`, `tag_structure_weak_trend_component`, and
`tag_structure_cap_active`. These tags are intended to make high or low scores
easier to audit in `inspect`.

Top ranks are relative to the eligible candidates in the same run. In a weak
eligible universe, a top-ranked row can still be the least-bad candidate rather
than a clean upward trend. Scores may be negative or capped.

### `volatility_quality_v1`

`volatility_quality_v1` is a market-relative volatility quality profile. It
favors lower 20-day and 60-day realized volatility, lower downside volatility,
tighter daily ranges, more stable volatility, and better 120-day drawdown
control.

The profile is useful as a defensive or risk-control lens. High scores do not
guarantee lower future risk or positive future returns.

### `liquidity_quality_v1`

`liquidity_quality_v1` is a market-relative liquidity quality profile. It ranks
local-currency traded value depth, Amihud-style illiquidity proxies, traded value
stability and concentration, recent liquidity fade, trading continuity, and
range tightness.

The profile is useful for screening tradability and liquidity quality. Traded
value metrics are local currency amounts, so compare scores and ranks within the
same market, run, profile, and horizon.

### Choosing Profiles

Use `sample_price_trend_v1` for fixture smoke tests and as a reference
implementation for new profiles. Use `momentum_v1` when you want a raw momentum
candidate list. Use `trend_quality_v1` when you want a more structured trend
lens with audit tags. Use `volatility_quality_v1` and `liquidity_quality_v1` as
risk and tradability companions, either on their own or in a multi-profile batch
with momentum or trend profiles.

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
