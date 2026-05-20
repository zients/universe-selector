# Data and Output

## Batch Persistence

When `batch` receives more than one `--ranking-profile`, provider data is loaded
once and then reused for each selected profile. Each profile is persisted as a
separate run with its own run id, ranking config hash, rankings, and report.

If any profile fails during a multi-profile batch, completed profile runs remain
persisted and the CLI prints the completed run ids plus the failed run id, failed
profile, and error.

The persistence schema stores stable run fields as columns and profile-specific
metrics in `metrics_json`. Ranking profiles declare their persisted metric keys,
so new profiles can add different metrics without adding profile-specific DuckDB
columns.

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
