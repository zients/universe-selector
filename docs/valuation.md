# Valuation

`value` runs a live ephemeral single-ticker valuation analysis. It is separate
from persisted ranking runs, reads valuation assumptions from YAML, fetches
fundamental facts through the configured fundamentals provider, and prints
markdown or JSON to stdout.

`value` v1 prints markdown by default and JSON with `--json`. It requires
`config.yaml` only for selecting `live.fundamentals_provider`, does not read
DuckDB, and does not persist the result.

The default assumptions path is
`valuation_assumptions/{market}/{ticker}.yaml`; the committed
`valuation_assumptions/us/AAPL.yaml` and `valuation_assumptions/tw/2330.yaml`
are sample schemas only and are not investment advice. Each valuation assumptions
file declares a root `default_model`; `value` uses the assumptions file
`default_model` when `--model` is omitted. `--model` explicitly overrides the
assumptions file default model. Assumption schema `1` requires root
`default_model`, `share_basis: ordinary_share`, and a non-empty
`valuation_basis_note`. The basis note is rendered in Assumption Context with
the same markdown escaping and prohibited-term redaction as other assumption
text. The committed valuation assumption files are repository templates;
installed wheels do not copy them into your working directory. Create your own
assumptions file in the working directory or pass `--assumptions`.

Supported valuation models are `exit_multiple_dcf_v1`, `fcf_dcf_v1`,
`implied_discount_rate_v1`, `multiple_valuation_v1`, `reverse_dcf_v1`, and
`two_stage_fcf_dcf_v1`.
Assumptions YAML may omit supported model blocks that are not selected, but
present supported model blocks are validated even when unselected so stale or
malformed assumptions fail closed. Selecting a supported model whose block is
omitted fails with `missing model assumptions for <model_id>`.

## Commands

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

## `exit_multiple_dcf_v1`

`exit_multiple_dcf_v1` projects illustrative explicit-period FCF using scenario
assumptions and uses an analyst-supplied EV / FCF exit multiple to calculate
terminal value. The multiple is not peer-derived by the model. It reports
enterprise value, equity value, model-implied value per share, and descriptive
spread against reference price. The model requires positive starting FCF because
EV / FCF exit multiple terminal value is not meaningful when starting FCF is zero
or negative.

The raw provider TTM FCF may not be analyst-normalized, is not clean unlevered
FCFF, and may be affected by accounting classification, cyclicality, working
capital, capex, and capital-structure effects; normalized unlevered FCFF
requires `starting_fcf.method: override` with a note.

The output reports present-value terminal value and terminal-value share of EV
because exit-multiple terminal value can dominate enterprise value. Scenario
rows are illustrative assumption cases, not probabilities, forecasts, expected
outcomes, target cases, recommendations, or investment signals.

## `two_stage_fcf_dcf_v1`

`two_stage_fcf_dcf_v1` is a positive-FCF two-stage DCF with a
perpetual-growth terminal value. It uses provider-FCF-proxy EV/equity math
unless override FCF is supplied. The committed samples keep
`default_model: fcf_dcf_v1`, so this model is opt-in through
`--model two_stage_fcf_dcf_v1` or by changing a local assumptions file.

The raw provider TTM FCF may not be analyst-normalized, is not clean unlevered
FCFF, and may be affected by accounting classification, cyclicality, working
capital, capex, and capital-structure effects; normalized unlevered FCFF
requires `starting_fcf.method: override` with a note.

The output distinguishes undiscounted terminal value from present-value terminal
value and reports terminal-value share of EV. Scenario rows are not
probabilities, forecasts, expected outcomes, target cases, recommendations, or
investment signals. Model-implied value per share and spread are not target
prices, forecasts, expected returns, recommendations, or signals.

## `implied_discount_rate_v1`

`implied_discount_rate_v1` solves the nominal discount rate that reconciles the
reference price to a provider-FCF-proxy enterprise/equity bridge under stated
explicit FCF growth and terminal-growth assumptions. It is a diagnostic
reverse-DCF reconciliation bridge, not a company WACC estimate, not a hurdle
rate, not a required return, not an expected return, not a recommendation, and
not an investment signal.

The model uses raw provider TTM FCF as a starting proxy unless
`starting_fcf.method: override` supplies normalized unlevered FCFF with a note.
Raw provider TTM FCF may not be analyst-normalized, is not clean unlevered
FCFF, and may be affected by accounting classification, cyclicality, working
capital, capex, and capital-structure effects.

The output reports the reference equity value, net debt, reference-implied
enterprise value, solved discount rate, solver residual, and terminal-value
share of EV. No result is produced when reference-implied enterprise value is
outside solver bounds or when the solver cannot converge; the model does not
fall back to a midpoint or substitute rate. Scenario rows are diagnostic
assumption cases, not probabilities, forecasts, expected outcomes, target cases,
recommendations, or investment signals. The committed samples keep
`default_model: fcf_dcf_v1`, so this model is opt-in through
`--model implied_discount_rate_v1` or by changing a local assumptions file.

## `fcf_dcf_v1`

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

## `reverse_dcf_v1`

`reverse_dcf_v1` solves the explicit-period FCF growth required to reconcile
the model to the reference price under the stated discount-rate and terminal
growth assumptions. The solved growth applies only to the explicit forecast
period; terminal growth is a separate assumption. It is a reconciliation model,
not a forecast or investment signal.

## `multiple_valuation_v1`

`multiple_valuation_v1` applies analyst-supplied EV / FCF multiples to starting
FCF. The multiple is not peer-derived by the model. It reports enterprise
value, equity value, model-implied value per share, and descriptive spread
against reference price. EV / FCF multiple valuation is not meaningful when
starting FCF is zero or negative.

## Provider Facts

`value` uses yfinance fundamentals for v1 `US` and `TW` live facts. TW tickers
default to the yfinance `.TW` request suffix. yfinance fundamentals are
third-party convenience data and may be stale, incomplete, restated, mapped
inconsistently, or unavailable. Independently verify provider facts and validate
or override assumptions before relying on model-implied outputs.
TW valuation templates use TWD ordinary-share basis with no ADR-ratio,
board-lot, or currency adjustment.
