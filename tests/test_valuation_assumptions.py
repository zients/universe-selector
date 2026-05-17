from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.assumptions import default_assumptions_path, load_valuation_assumptions


US_FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "us" / "AAPL.yaml"
TW_FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "tw" / "2330.yaml"


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "valuation_assumptions" / "us" / "AAPL.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(US_FIXTURE, target)
    return target


def test_default_assumptions_path_uses_lowercase_market_and_canonical_ticker() -> None:
    assert default_assumptions_path(Market.US, "AAPL") == Path("valuation_assumptions/us/AAPL.yaml")
    assert default_assumptions_path(Market.US, "aapl") == Path("valuation_assumptions/us/AAPL.yaml")
    assert default_assumptions_path(Market.TW, "2330") == Path("valuation_assumptions/tw/2330.yaml")


def test_loads_default_assumptions_and_hash_is_path_independent(monkeypatch, tmp_path: Path) -> None:
    default_path = _copy_fixture(tmp_path)
    explicit_path = tmp_path / "moved.yaml"
    shutil.copyfile(default_path, explicit_path)
    monkeypatch.chdir(tmp_path)

    default_loaded = load_valuation_assumptions(
        market=Market.US,
        ticker="AAPL",
        model_id=None,
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
    assert default_loaded.currency == "USD"
    assert default_loaded.amount_unit == "currency_units"
    assert default_loaded.default_model == "fcf_dcf_v1"
    assert default_loaded.assumption_path == str(default_path)
    assert explicit_loaded.assumption_path == str(explicit_path)
    assert default_loaded.assumption_hash == explicit_loaded.assumption_hash
    assert default_loaded.model_id == "fcf_dcf_v1"
    assert default_loaded.model_assumptions.forecast_years == 5
    assert default_loaded.model_assumptions.starting_fcf.method == "provider_ttm_fcf"
    assert default_loaded.model_assumptions.starting_fcf.value is None
    assert default_loaded.model_assumptions.starting_fcf.note is None
    assert default_loaded.model_assumptions.scenario_order == ("conservative", "base", "upside")


def test_loads_tw_2330_assumptions_fixture() -> None:
    loaded = load_valuation_assumptions(
        market=Market.TW,
        ticker="2330",
        model_id=None,
        assumptions_path=TW_FIXTURE,
    )

    assert loaded.market is Market.TW
    assert loaded.ticker == "2330"
    assert loaded.currency == "TWD"
    assert loaded.amount_unit == "currency_units"
    assert loaded.default_model == "fcf_dcf_v1"
    assert loaded.assumption_path == str(TW_FIXTURE)
    assert loaded.model_id == "fcf_dcf_v1"
    assert loaded.model_assumptions.starting_fcf.method == "provider_ttm_fcf"


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

    with pytest.raises(ValidationError, match="unknown valuation model unknown_model"):
        load_valuation_assumptions(
            market=Market.US,
            ticker="AAPL",
            model_id="unknown_model",
            assumptions_path=US_FIXTURE,
        )


def test_rejects_invalid_default_model(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("default_model: fcf_dcf_v1", "default_model: missing_model"))
    with pytest.raises(ValidationError, match="default_model must exist in models"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id=None, assumptions_path=path)

    path.write_text(
        text.replace("default_model: fcf_dcf_v1", "default_model: unknown_model").replace(
            "models:\n  fcf_dcf_v1:",
            "models:\n  unknown_model: {}\n  fcf_dcf_v1:",
        )
    )
    with pytest.raises(ValidationError, match="unknown valuation model unknown_model"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id=None, assumptions_path=path)


def test_rejects_empty_requested_model_without_falling_back_to_default(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)

    with pytest.raises(ValidationError, match="model_id must be a non-empty string"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="", assumptions_path=path)


def test_rejects_missing_required_root_key(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    data = path.read_text().replace("prepared_by: universe-selector maintainers\n", "")
    path.write_text(data)

    with pytest.raises(ValidationError, match="prepared_by"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)

    path = _copy_fixture(tmp_path / "default-model")
    path.write_text(path.read_text().replace("default_model: fcf_dcf_v1\n", ""))

    with pytest.raises(ValidationError, match="default_model"):
        load_valuation_assumptions(Market.US, "AAPL", None, path)


def test_rejects_invalid_currency_and_amount_unit(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("currency: USD", "currency: usd"))
    with pytest.raises(ValidationError, match="currency"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)

    path.write_text(text.replace("amount_unit: currency_units", "amount_unit: millions"))
    with pytest.raises(ValidationError, match="amount_unit"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)


def test_rejects_datetime_as_of_values(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    path.write_text(path.read_text().replace("as_of: 2026-05-17", "as_of: 2026-05-17 12:34:56", 1))

    with pytest.raises(ValidationError, match="as_of must be an ISO date"):
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
    path.write_text(path.read_text().replace("reference_price: null", "reference_price: .nan", 1))

    with pytest.raises(ValidationError, match="reference_price"):
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


def test_rejects_invalid_starting_fcf_shapes(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("method: provider_ttm_fcf", "method: unknown_method", 1))
    with pytest.raises(ValidationError, match="starting_fcf.method"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace(
        "      method: provider_ttm_fcf\n",
        "      method: provider_ttm_fcf\n      value: 100.0\n",
        1,
    ))
    with pytest.raises(ValidationError, match="starting_fcf.value"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace(
        "      method: provider_ttm_fcf\n",
        "      method: override\n      value: 100.0\n",
        1,
    ))
    with pytest.raises(ValidationError, match="starting_fcf.note"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace(
        "      method: provider_ttm_fcf\n",
        "      method: override\n      value: .nan\n      note: Adjusted FCF.\n",
        1,
    ))
    with pytest.raises(ValidationError, match="starting_fcf.value"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)


def test_rejects_invalid_rates(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("terminal_growth_rate: 0.025", "terminal_growth_rate: 0.08"))
    with pytest.raises(ValidationError, match="terminal_growth_rate"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(text.replace("discount_rate: 0.09", "discount_rate: 0.01"))
    with pytest.raises(ValidationError, match="discount_rate"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)
