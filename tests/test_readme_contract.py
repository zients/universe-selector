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
    assert "`batch` is the only command that fetches data or computes rankings." not in normalized
    assert "`batch` is the only command that computes and persists ranking runs." in normalized
    assert "`value` fetches fundamentals for valuation separately from ranking runs." in normalized
    assert "uv run universe-selector value us --ticker AAPL" in text
    assert "uv run universe-selector value us --ticker AAPL --model fcf_dcf_v1" in text
    assert "--assumptions valuation_assumptions/us/AAPL.yaml" in text
    assert "uv run universe-selector value tw --ticker 2330" in text
    assert "--assumptions valuation_assumptions/tw/2330.yaml" in text
    assert "`value` v1 prints markdown only." in normalized
    assert "requires `config.yaml` only for selecting `live.fundamentals_provider`" in normalized
    assert "does not read DuckDB" in normalized
    assert "does not persist the result" in normalized
    assert "The default assumptions path is `valuation_assumptions/{market}/{ticker}.yaml`" in normalized
    assert "`valuation_assumptions/us/AAPL.yaml` and `valuation_assumptions/tw/2330.yaml` are sample schemas only" in normalized
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
