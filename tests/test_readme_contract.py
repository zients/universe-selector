from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DETAIL_DOCS = (
    REPO_ROOT / "docs/valuation.md",
    REPO_ROOT / "docs/ranking-profiles.md",
    REPO_ROOT / "docs/extending.md",
    REPO_ROOT / "docs/data-and-output.md",
)


def _documentation_corpus() -> str:
    return "\n\n".join(path.read_text() for path in (README, *DETAIL_DOCS) if path.exists())


def test_readme_is_concise_entrypoint_with_links_to_detail_docs() -> None:
    text = README.read_text()

    assert len(text.splitlines()) <= 360
    assert "- [Valuation](docs/valuation.md)" in text
    assert "- [Ranking profiles](docs/ranking-profiles.md)" in text
    assert "- [Extending](docs/extending.md)" in text
    assert "- [Data and output](docs/data-and-output.md)" in text
    for path in DETAIL_DOCS:
        assert path.is_file()


def test_readme_documents_value_as_ephemeral_not_persisted() -> None:
    text = _documentation_corpus()
    normalized = " ".join(text.split())

    assert "The CLI has four command families:" in text
    assert "`value MARKET --ticker TICKER`" in text
    assert "Valuation command: `value` uses yfinance fundamentals for `US` and `TW` in v1." in text
    assert ("valuation_assumptions/{market}/{ticker}.yaml + fundamentals provider -> valuation model -> stdout") in text
    assert "batch remains the only command that computes persisted rankings." in normalized
    assert "`report` and `inspect` still only read persisted ranking runs" in normalized
    assert "`value` is a live ephemeral single-ticker valuation analysis and is not persisted in v1" in normalized
    assert (
        "`output/` renders markdown and JSON reports plus per-ticker inspect output from persisted data, plus thin command output adapters."
        in normalized
    )
    assert "`valuation/` owns valuation assumptions, model logic, orchestration, and valuation output." in normalized
    assert "`batch` is the only command that fetches data or computes rankings." not in normalized
    assert "`batch` is the only command that computes and persists ranking runs." in normalized
    assert "`value` fetches fundamentals for valuation separately from ranking runs." in normalized
    assert "uv run universe-selector value us --ticker AAPL" in text
    assert "uv run universe-selector report us --json" in text
    assert "uv run universe-selector inspect us --ticker AXTI --json" in text
    assert "uv run universe-selector value us --ticker AAPL --json" in text
    assert (
        "Report JSON includes the full persisted ticker snapshots and rankings plus a `top_horizons` report view."
        in normalized
    )
    assert "uv run universe-selector value us --ticker AAPL --model exit_multiple_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model reverse_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model multiple_valuation_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model two_stage_fcf_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model implied_discount_rate_v1" in text
    assert "`value` uses the assumptions file `default_model` when `--model` is omitted" in normalized
    assert "`--model` explicitly overrides the assumptions file default model" in normalized
    assert (
        "Supported valuation models are `exit_multiple_dcf_v1`, `fcf_dcf_v1`, "
        "`implied_discount_rate_v1`, `multiple_valuation_v1`, `reverse_dcf_v1`, "
        "and `two_stage_fcf_dcf_v1`."
    ) in normalized
    assert "`exit_multiple_dcf_v1` projects illustrative explicit-period FCF using scenario assumptions" in normalized
    assert "uses an analyst-supplied EV / FCF exit multiple to calculate terminal value" in normalized
    assert "`two_stage_fcf_dcf_v1` is a positive-FCF two-stage DCF" in normalized
    assert "provider-FCF-proxy EV/equity math" in normalized
    assert "raw provider TTM FCF may not be analyst-normalized" in normalized
    assert "not clean unlevered FCFF" in normalized
    assert "accounting classification" in normalized
    assert "cyclicality" in normalized
    assert "working capital" in normalized
    assert "capex" in normalized
    assert "capital-structure effects" in normalized
    assert "normalized unlevered FCFF requires `starting_fcf.method: override` with a note" in normalized
    assert "terminal-value share of EV" in normalized
    assert "investment signal" in normalized
    assert "`implied_discount_rate_v1` solves the nominal discount rate" in normalized
    assert "diagnostic reverse-DCF reconciliation bridge" in normalized
    assert "not a company WACC estimate" in normalized
    assert "not a hurdle rate" in normalized
    assert "not a required return" in normalized
    assert "not an expected return" in normalized
    assert "not a recommendation" in normalized
    assert "not an investment signal" in normalized
    assert "reference-implied enterprise value is outside solver bounds" in normalized
    assert "`reverse_dcf_v1` solves the explicit-period FCF growth" in normalized
    assert "`multiple_valuation_v1` applies analyst-supplied EV / FCF multiples" in normalized
    assert "Assumptions YAML may omit supported model blocks that are not selected" in normalized
    assert "present supported model blocks are validated even when unselected" in normalized
    assert (
        "Assumption schema `1` requires root `default_model`, "
        "`share_basis: ordinary_share`, and a non-empty `valuation_basis_note`."
    ) in normalized
    assert "The basis note is rendered in Assumption Context" in normalized
    assert "--assumptions valuation_assumptions/us/AAPL.yaml" in text
    assert "uv run universe-selector value tw --ticker 2330" in text
    assert "--assumptions valuation_assumptions/tw/2330.yaml" in text
    assert "`value` v1 prints markdown by default and JSON with `--json`." in normalized
    assert "requires `config.yaml` only for selecting `live.fundamentals_provider`" in normalized
    assert "does not read DuckDB" in normalized
    assert "does not persist the result" in normalized
    assert "The default assumptions path is `valuation_assumptions/{market}/{ticker}.yaml`" in normalized
    assert (
        "`valuation_assumptions/us/AAPL.yaml` and `valuation_assumptions/tw/2330.yaml` are sample schemas only"
        in normalized
    )
    assert "Each valuation assumptions file declares a root `default_model`" in normalized
    assert "`fcf_dcf_v1` uses `models.fcf_dcf_v1.starting_fcf` to choose the DCF starting FCF" in normalized
    assert "templates default to `starting_fcf.method: provider_ttm_fcf`" in normalized
    assert "Set `starting_fcf.method: override` with `value` and `note`" in normalized
    assert "The committed valuation assumption files are repository templates" in normalized
    assert "installed wheels do not copy them into your working directory" in normalized
    assert "yfinance fundamentals are third-party convenience data" in normalized
    assert "TW tickers default to the yfinance `.TW` request suffix" in normalized
    assert "uses provider raw FCF as a starting proxy so the command can run directly" in normalized
    assert "model-implied scenario results" in normalized
    assert "not forecasts, expected outcomes, target cases, or recommendations" in normalized
    assert (
        "TW valuation templates use TWD ordinary-share basis with no ADR-ratio, board-lot, or currency adjustment"
        in normalized
    )


def test_readme_documents_registered_public_quality_ranking_profiles() -> None:
    text = _documentation_corpus()
    normalized = " ".join(text.split())

    assert "`momentum_quality_v1`" in text
    assert "`trend_pullback_quality_v1`" in text
    assert "`base_breakout_quality_v1`" in text
    assert "`relative_strength_leader_v1`" in text
    assert "`mean_reversion_quality_v1`" in text
    assert "`defensive_compounder_quality_v1`" in text
    assert "| `momentum_quality_v1` |" in text
    assert "| `trend_pullback_quality_v1` |" in text
    assert "| `base_breakout_quality_v1` |" in text
    assert "| `relative_strength_leader_v1` |" in text
    assert "| `mean_reversion_quality_v1` |" in text
    assert "| `defensive_compounder_quality_v1` |" in text
    assert "### `momentum_quality_v1`" in text
    assert "### `trend_pullback_quality_v1`" in text
    assert "### `base_breakout_quality_v1`" in text
    assert "### `relative_strength_leader_v1`" in text
    assert "### `mean_reversion_quality_v1`" in text
    assert "### `defensive_compounder_quality_v1`" in text
    assert "`momentum_quality_v1` for market-relative momentum quality with audit tags" in normalized
    assert "`trend_pullback_quality_v1` when you want strong stocks that have corrected toward support" in normalized
    assert "`base_breakout_quality_v1` when you want constructive bases near or just through breakout" in normalized
    assert "`relative_strength_leader_v1` when you want the market's persistent leadership list" in normalized
    assert "`mean_reversion_quality_v1` when you want short-term oversold repair candidates" in normalized
    assert "`defensive_compounder_quality_v1` when you want an OHLCV-only defensive compounder proxy" in normalized
    assert "Trend pullback quality is not a buy signal" in normalized
    assert "Base breakout quality is not a buy signal" in normalized
    assert "Relative strength leader quality is not a buy signal" in normalized
    assert "Mean reversion quality is not a buy signal" in normalized
    assert "Defensive compounder quality is not a buy signal" in normalized
