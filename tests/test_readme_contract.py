from __future__ import annotations

from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_documents_value_as_ephemeral_not_persisted() -> None:
    text = README.read_text()
    normalized = " ".join(text.split())

    assert "The CLI has four command families:" in text
    assert "`value MARKET --ticker TICKER`" in text
    assert "Valuation command: `value` uses yfinance fundamentals for `US` and `TW` in v1." in text
    assert (
        "valuation_assumptions/{market}/{ticker}.yaml + fundamentals provider -> "
        "valuation model -> stdout"
    ) in text
    assert "batch remains the only command that computes persisted rankings." in normalized
    assert "`report` and `inspect` still only read persisted ranking runs" in normalized
    assert "`value` is a live ephemeral single-ticker valuation analysis and is not persisted in v1" in normalized
    assert "`output/` renders markdown and JSON reports plus per-ticker inspect output from persisted data, plus thin command output adapters." in normalized
    assert "`valuation/` owns valuation assumptions, model logic, orchestration, and valuation output." in normalized
    assert "`batch` is the only command that fetches data or computes rankings." not in normalized
    assert "`batch` is the only command that computes and persists ranking runs." in normalized
    assert "`value` fetches fundamentals for valuation separately from ranking runs." in normalized
    assert "uv run universe-selector value us --ticker AAPL" in text
    assert "uv run universe-selector report us --json" in text
    assert "uv run universe-selector inspect us --ticker AXTI --json" in text
    assert "uv run universe-selector value us --ticker AAPL --json" in text
    assert "Report JSON includes the full persisted ticker snapshots and rankings plus a `top_horizons` report view." in normalized
    assert "uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model reverse_dcf_v1" in text
    assert "uv run universe-selector value us --ticker AAPL --model multiple_valuation_v1" in text
    assert "`value` uses the assumptions file `default_model` when `--model` is omitted" in normalized
    assert "`--model` explicitly overrides the assumptions file default model" in normalized
    assert "Supported valuation models are `fcf_dcf_v1`, `reverse_dcf_v1`, and `multiple_valuation_v1`." in normalized
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
    assert "`valuation_assumptions/us/AAPL.yaml` and `valuation_assumptions/tw/2330.yaml` are sample schemas only" in normalized
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
    assert "TW valuation templates use TWD ordinary-share basis with no ADR-ratio, board-lot, or currency adjustment" in normalized


def test_readme_documents_registered_public_quality_ranking_profiles() -> None:
    text = README.read_text()
    normalized = " ".join(text.split())

    assert "`momentum_quality_v1`" in text
    assert "`defensive_compounder_quality_v1`" in text
    assert "| `momentum_quality_v1` |" in text
    assert "| `defensive_compounder_quality_v1` |" in text
    assert "### `momentum_quality_v1`" in text
    assert "### `defensive_compounder_quality_v1`" in text
    assert "`momentum_quality_v1` for market-relative momentum quality with audit tags" in normalized
    assert "`defensive_compounder_quality_v1` when you want an OHLCV-only defensive compounder proxy" in normalized
    assert "Defensive compounder quality is not a buy signal" in normalized
