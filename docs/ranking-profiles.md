# Ranking Profiles

Ranking profiles are independent scoring lenses. They share the same persisted
run model but define their own candidate filters, metric keys, horizons, scores,
and interpretation notes.

| Profile | Purpose | Horizons | Required History |
|---|---|---|---:|
| `sample_price_trend_v1` | Minimal example profile for smoke runs and extension patterns. | `midterm`, `longterm` | 121 bars |
| `momentum_v1` | Raw weighted momentum profile using risk-adjusted medium-term momentum and short-term strength. | `swing`, `midterm` | 274 bars |
| `momentum_quality_v1` | Market-relative momentum quality profile using risk-adjusted momentum, moving-average structure, trend consistency, drawdown control, caps, and audit tags. | `composite`, `swing`, `midterm` | 274 bars |
| `trend_quality_v1` | Market-relative trend profile using returns, trend slope, trend fit, moving-average structure, drawdown control, caps, and structure tags. | `composite`, `shortterm`, `midterm` | 252 bars |
| `trend_pullback_quality_v1` | Market-relative pullback profile favoring strong prior trends that have corrected into orderly pullbacks while preserving trend, support, and liquidity. | `composite`, `near_support`, `trend_resume` | 252 bars |
| `volatility_quality_v1` | Market-relative quality profile favoring lower realized volatility, downside volatility control, range tightness, and drawdown control. | `composite`, `shortterm`, `stable` | 126 bars |
| `liquidity_quality_v1` | Market-relative liquidity profile using traded value depth, friction proxies, traded value stability, concentration, continuity, and range tightness. | `composite`, `shortterm`, `stable` | 63 bars |
| `base_breakout_quality_v1` | Market-relative base breakout profile favoring constructive bases near resistance or early breakout readiness with liquidity and failed-breakout risk tags. | `composite`, `near_breakout`, `breakout_readiness` | 252 bars |
| `relative_strength_leader_v1` | Market-relative leadership profile favoring persistent 20/60/120-day relative strength, risk-adjusted momentum, trend durability, and overheat audit tags. | `composite`, `shortterm_leader`, `midterm_leader` | 274 bars |
| `mean_reversion_quality_v1` | Market-relative mean reversion profile favoring short-term oversold candidates near support with rebound confirmation and falling-knife risk controls. | `composite`, `oversold_bounce`, `support_reversion` | 252 bars |
| `defensive_compounder_quality_v1` | OHLCV-only defensive compounder proxy favoring steady positive price behavior, downside volatility control, drawdown control, and intact long-trend structure. | `composite`, `steady_compounder`, `downside_control` | 252 bars |

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
- `midterm`: medium-term lens emphasizing 12-1 and 6-1 risk-adjusted
  momentum.

Scores are raw weighted composites and are not bounded to 0-100.

### `momentum_quality_v1`

`momentum_quality_v1` is a market-relative momentum quality profile. It combines
12-1 and 6-1 risk-adjusted momentum, short-term strength, moving-average
structure, trend consistency, drawdown control, and overheat penalties.

It ranks three horizons:

- `composite`: balanced momentum, trend quality, drawdown, and overheat control.
- `swing`: shorter momentum lens emphasizing 6-1 momentum and recent strength.
- `midterm`: medium-term lens emphasizing 12-1 and 6-1 risk-adjusted momentum.

It persists non-exclusive audit tags such as `tag_risk_overheated`,
`tag_risk_extended_from_ma20`, `tag_risk_high_volatility`,
`tag_risk_large_drawdown`, `tag_positive_strong_momentum`, and
`tag_positive_stable_uptrend`. High scores remain ranking values within the same
run and are not return forecasts or recommendations.

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

### `trend_pullback_quality_v1`

`trend_pullback_quality_v1` is a market-relative pullback profile for strong
stocks that have corrected into orderly pullbacks. It favors candidates with
positive prior strength, intact longer-term trend structure, controlled pullback
depth, support proximity around shorter moving averages, and recent liquidity
continuity.

It ranks three horizons:

- `composite`: balanced prior strength, intact trend, pullback setup, support,
  and risk control.
- `near_support`: emphasizes pullback setup and support proximity.
- `trend_resume`: emphasizes intact trend structure and signs of stabilization.

It persists setup and risk tags such as `tag_setup_healthy_pullback`,
`tag_setup_near_sma50`, `tag_setup_trend_resume`, `tag_risk_breakdown`,
`tag_risk_deep_drawdown`, `tag_risk_liquidity_fade`, and
`tag_risk_still_overheated`. Trend pullback quality is not a buy signal; use it
as a candidate-ranking lens that still requires independent review.

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

### `base_breakout_quality_v1`

`base_breakout_quality_v1` is a market-relative breakout setup profile for
constructive bases near or just through breakout. It favors intact trend
structure, controlled base depth, tighter recent ranges, proximity to recent
highs, and confirming traded value without requiring manual chart labels.

It ranks three horizons:

- `composite`: balanced base quality, trend structure, breakout setup, volume,
  and risk control.
- `near_breakout`: emphasizes clean bases close to a recent high without
  overextension.
- `breakout_readiness`: emphasizes candidates closest to early breakout
  confirmation with recent volume support and false-breakout caps.

It persists setup and risk tags such as `tag_setup_valid_base`,
`tag_setup_near_breakout`, `tag_setup_confirmed_breakout`,
`tag_risk_false_breakout`, `tag_risk_overextended_breakout`,
`tag_risk_weak_base`, and `tag_risk_liquidity_fade`. Base breakout quality is
not a buy signal; use it as a candidate-ranking lens that still requires
independent review.

### `relative_strength_leader_v1`

`relative_strength_leader_v1` is a market-relative leadership profile for
identifying the strongest names in the current market tape. It emphasizes
persistent 20-day, 60-day, and 120-day leadership, risk-adjusted 6-1 and 12-1
momentum, trend durability, proximity to recent highs, drawdown control, and
liquidity continuity.

It ranks three horizons:

- `composite`: balanced leadership persistence, trend durability, risk control,
  and proximity to highs.
- `shortterm_leader`: emphasizes recent 20-day and 60-day leadership.
- `midterm_leader`: emphasizes 60-day and 120-day leadership plus 6-1 and 12-1
  risk-adjusted momentum.

It persists positive and risk tags such as `tag_positive_rs_leader`,
`tag_positive_persistent_leader`, `tag_positive_new_high_leader`,
`tag_risk_chasing_extension`, `tag_risk_recent_rs_fade`,
`tag_risk_high_volatility`, and `tag_risk_liquidity_fade`. Relative strength
leader quality is not a buy signal; use it as a leadership-ranking lens that
still requires independent review.

### `mean_reversion_quality_v1`

`mean_reversion_quality_v1` is a market-relative mean reversion profile for
short-term oversold repair candidates. It favors controlled pullbacks below
shorter moving averages, support proximity, early rebound confirmation,
liquidity continuity, and preserved longer-term structure while penalizing
falling-knife and breakdown risk.

It ranks three horizons:

- `composite`: balanced oversold setup, support, rebound confirmation, and risk
  control.
- `oversold_bounce`: emphasizes short-term oversold depth and rebound
  confirmation.
- `support_reversion`: emphasizes support proximity and preserved structure.

It persists setup and risk tags such as `tag_setup_oversold_quality`,
`tag_setup_near_support`, `tag_setup_rebound_confirmation`,
`tag_risk_falling_knife`, `tag_risk_breakdown`,
`tag_risk_deep_drawdown`, `tag_risk_volatility_spike`, and
`tag_risk_liquidity_fade`. Mean reversion quality is not a buy signal; use it
as a repair-candidate ranking lens that still requires independent review.

### `defensive_compounder_quality_v1`

`defensive_compounder_quality_v1` is an OHLCV-only defensive compounder proxy.
It does not use fundamentals or imply business quality. It favors steady
positive price behavior, persistent rolling returns, lower realized and downside
volatility, drawdown control, range tightness, liquidity stability, and intact
long-term moving-average structure.

It ranks three horizons:

- `composite`: balanced steady return, downside control, trend quality, and
  risk control.
- `steady_compounder`: emphasizes steady positive return persistence and
  long-trend durability.
- `downside_control`: emphasizes downside volatility, drawdown control, low
  realized volatility, and range tightness.

It persists positive and risk tags such as `tag_positive_steady_compounder`,
`tag_positive_low_downside_volatility`, `tag_positive_drawdown_control`,
`tag_risk_flat_no_growth`, `tag_risk_broken_long_trend`,
`tag_risk_large_drawdown`, `tag_risk_volatility_spike`, and
`tag_risk_stale_or_illiquid`. Defensive compounder quality is not a buy signal;
use it as a price-behavior ranking proxy that still requires independent review.

## Choosing Profiles

Use `sample_price_trend_v1` for fixture smoke tests and as a reference
implementation for new profiles. Use `momentum_v1` when you want a raw momentum
candidate list. Use `momentum_quality_v1` for market-relative momentum quality
with audit tags. Use `trend_quality_v1` when you want a more structured trend
lens with audit tags. Use `trend_pullback_quality_v1` when you want strong
stocks that have corrected toward support without losing longer-term trend
structure. Use `base_breakout_quality_v1` when you want constructive bases near
or just through breakout. Use `relative_strength_leader_v1` when you want the
market's persistent leadership list with overheat and fade tags. Use
`mean_reversion_quality_v1` when you want short-term oversold repair candidates
that still preserve enough structure to avoid obvious falling-knife setups. Use
`defensive_compounder_quality_v1` when you want an OHLCV-only defensive
compounder proxy rather than a fundamental quality screen. Use
`volatility_quality_v1` and `liquidity_quality_v1` as risk and tradability
companions, either on their own or in a multi-profile batch with momentum or
trend profiles.

Use `universe-selector screen` with two or more `--ranking-profile` options to
cross-reference the latest persisted runs across profiles for one market. The
screen command reads persisted composite rankings, counts how many profiles each
ticker appears in within a configurable top-N, and outputs a cross-reference
table sorted by profile count then average rank.
