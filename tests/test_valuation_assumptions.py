from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.assumptions import default_assumptions_path, load_valuation_assumptions


FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "us" / "AAPL.yaml"


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "valuation_assumptions" / "us" / "AAPL.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, target)
    return target


def test_default_assumptions_path_uses_lowercase_market_and_canonical_ticker() -> None:
    assert default_assumptions_path(Market.US, "AAPL") == Path("valuation_assumptions/us/AAPL.yaml")
    assert default_assumptions_path(Market.US, "aapl") == Path("valuation_assumptions/us/AAPL.yaml")


def test_loads_default_assumptions_and_hash_is_path_independent(monkeypatch, tmp_path: Path) -> None:
    default_path = _copy_fixture(tmp_path)
    explicit_path = tmp_path / "moved.yaml"
    shutil.copyfile(default_path, explicit_path)
    monkeypatch.chdir(tmp_path)

    default_loaded = load_valuation_assumptions(
        market=Market.US,
        ticker="AAPL",
        model_id="fcf_dcf_v1",
        assumptions_path=None,
    )
    explicit_loaded = load_valuation_assumptions(
        market=Market.US,
        ticker="AAPL",
        model_id="fcf_dcf_v1",
        assumptions_path=explicit_path,
    )

    assert default_loaded.market is Market.US
    assert default_loaded.ticker == "AAPL"
    assert default_loaded.assumption_path == str(default_path)
    assert explicit_loaded.assumption_path == str(explicit_path)
    assert default_loaded.assumption_hash == explicit_loaded.assumption_hash
    assert default_loaded.model_id == "fcf_dcf_v1"
    assert default_loaded.model_assumptions.forecast_years == 5
    assert default_loaded.model_assumptions.scenario_order == ("conservative", "base", "upside")


def test_missing_default_assumptions_file_reports_expected_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match="valuation_assumptions/us/AAPL.yaml"):
        load_valuation_assumptions(
            market=Market.US,
            ticker="AAPL",
            model_id="fcf_dcf_v1",
            assumptions_path=None,
        )


def test_rejects_market_ticker_unknown_keys_and_missing_model(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("market: US", "market: TW"))
    with pytest.raises(ValidationError, match="market must match"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace("ticker: AAPL", "ticker: MSFT"))
    with pytest.raises(ValidationError, match="ticker must match"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text + "\nextra: nope\n")
    with pytest.raises(ValidationError, match="unknown assumptions key"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    with pytest.raises(ValidationError, match="missing model assumptions"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="unknown_model", assumptions_path=FIXTURE)


def test_rejects_missing_required_root_key(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    data = path.read_text().replace("prepared_by: universe-selector maintainers\n", "")
    path.write_text(data)

    with pytest.raises(ValidationError, match="prepared_by"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)


def test_rejects_unknown_override_and_note_keys(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    path.write_text(
        path.read_text().replace(
            "  reference_price: null\n\nfacts_override_notes:",
            "  reference_price: null\n  unknown_override: 1.0\n\nfacts_override_notes:",
        )
    )

    with pytest.raises(ValidationError, match="unknown facts_overrides key"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)

    path = _copy_fixture(tmp_path / "notes")
    path.write_text(
        path.read_text().replace(
            "  reference_price: null\n\nassumption_source:",
            "  reference_price: null\n  unknown_note: nope\n\nassumption_source:",
        )
    )

    with pytest.raises(ValidationError, match="unknown facts_override_notes key"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)


def test_rejects_non_finite_override_and_unknown_model_key(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    path.write_text(path.read_text().replace("normalized_fcf: null", "normalized_fcf: .nan", 1))

    with pytest.raises(ValidationError, match="normalized_fcf"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)

    path = _copy_fixture(tmp_path / "model")
    path.write_text(
        path.read_text().replace(
            "    terminal_method: perpetual_growth",
            "    terminal_method: perpetual_growth\n    unexpected: nope",
        )
    )

    with pytest.raises(ValidationError, match="unknown fcf_dcf_v1 key"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)


def test_rejects_invalid_rates_and_override_without_note(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("terminal_growth_rate: 0.025", "terminal_growth_rate: 0.08"))
    with pytest.raises(ValidationError, match="terminal_growth_rate"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace("discount_rate: 0.09", "discount_rate: 0.01"))
    with pytest.raises(ValidationError, match="discount_rate"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace("normalized_fcf: null", "normalized_fcf: 1000.0", 1))
    with pytest.raises(ValidationError, match="facts_override_notes.normalized_fcf"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)
