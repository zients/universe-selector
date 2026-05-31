from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.valuation.assumptions import default_assumptions_path, load_valuation_assumptions


US_FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "us" / "AAPL.yaml"
TW_FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "tw" / "2330.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
US_SAMPLE = REPO_ROOT / "valuation_assumptions" / "us" / "AAPL.yaml"
TW_SAMPLE = REPO_ROOT / "valuation_assumptions" / "tw" / "2330.yaml"


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
    assert default_loaded.share_basis == "ordinary_share"
    assert (
        default_loaded.valuation_basis_note
        == "Uses USD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied."
    )
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


@pytest.mark.parametrize(
    "model_id, expected_type_name",
    [
        ("exit_multiple_dcf_v1", "ExitMultipleDcfV1Assumptions"),
        ("fcf_dcf_v1", "FcfDcfV1Assumptions"),
        ("reverse_dcf_v1", "ReverseDcfV1Assumptions"),
        ("multiple_valuation_v1", "MultipleValuationV1Assumptions"),
        ("two_stage_fcf_dcf_v1", "TwoStageFcfDcfV1Assumptions"),
    ],
)
def test_us_fixture_supports_explicit_registered_valuation_models(
    model_id: str,
    expected_type_name: str,
) -> None:
    loaded = load_valuation_assumptions(
        market=Market.US,
        ticker="AAPL",
        model_id=model_id,
        assumptions_path=US_FIXTURE,
    )

    assert loaded.default_model == "fcf_dcf_v1"
    assert loaded.model_id == model_id
    assert type(loaded.model_assumptions).__name__ == expected_type_name


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
    assert loaded.share_basis == "ordinary_share"
    assert (
        loaded.valuation_basis_note
        == "Uses TWD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied."
    )
    assert loaded.default_model == "fcf_dcf_v1"
    assert loaded.assumption_path == str(TW_FIXTURE)
    assert loaded.model_id == "fcf_dcf_v1"
    assert loaded.model_assumptions.starting_fcf.method == "provider_ttm_fcf"


@pytest.mark.parametrize(
    "model_id, expected_type_name",
    [
        ("exit_multiple_dcf_v1", "ExitMultipleDcfV1Assumptions"),
        ("fcf_dcf_v1", "FcfDcfV1Assumptions"),
        ("reverse_dcf_v1", "ReverseDcfV1Assumptions"),
        ("multiple_valuation_v1", "MultipleValuationV1Assumptions"),
        ("two_stage_fcf_dcf_v1", "TwoStageFcfDcfV1Assumptions"),
    ],
)
def test_tw_fixture_supports_explicit_registered_valuation_models(
    model_id: str,
    expected_type_name: str,
) -> None:
    loaded = load_valuation_assumptions(
        market=Market.TW,
        ticker="2330",
        model_id=model_id,
        assumptions_path=TW_FIXTURE,
    )

    assert loaded.default_model == "fcf_dcf_v1"
    assert loaded.model_id == model_id
    assert type(loaded.model_assumptions).__name__ == expected_type_name
    assert loaded.share_basis == "ordinary_share"
    assert "no ADR ratio, board-lot, or currency adjustment" in loaded.valuation_basis_note


@pytest.mark.parametrize(
    "market, ticker, path, currency",
    [
        (Market.US, "AAPL", US_SAMPLE, "USD"),
        (Market.TW, "2330", TW_SAMPLE, "TWD"),
    ],
)
@pytest.mark.parametrize(
    "model_id",
    (
        "fcf_dcf_v1",
        "reverse_dcf_v1",
        "multiple_valuation_v1",
        "exit_multiple_dcf_v1",
        "two_stage_fcf_dcf_v1",
    ),
)
def test_committed_sample_assumptions_support_all_registered_models(
    market: Market,
    ticker: str,
    path: Path,
    currency: str,
    model_id: str,
) -> None:
    loaded = load_valuation_assumptions(market, ticker, model_id, path)

    assert loaded.default_model == "fcf_dcf_v1"
    assert loaded.model_id == model_id
    assert loaded.currency == currency
    assert loaded.share_basis == "ordinary_share"
    if market is Market.TW:
        assert (
            loaded.valuation_basis_note
            == "Uses TWD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied."
        )


def test_committed_reverse_dcf_sample_bounds_are_diagnostic_ranges() -> None:
    us = load_valuation_assumptions(Market.US, "AAPL", "reverse_dcf_v1", US_SAMPLE)
    tw = load_valuation_assumptions(Market.TW, "2330", "reverse_dcf_v1", TW_SAMPLE)

    assert us.model_assumptions.scenarios["conservative"].implied_growth_lower_bound == pytest.approx(-0.50)
    assert us.model_assumptions.scenarios["conservative"].implied_growth_upper_bound == pytest.approx(0.50)
    assert tw.model_assumptions.scenarios["conservative"].implied_growth_lower_bound == pytest.approx(-0.50)
    assert tw.model_assumptions.scenarios["conservative"].implied_growth_upper_bound == pytest.approx(0.80)


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
    with pytest.raises(ValidationError, match="unknown valuation model missing_model"):
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

    path = _copy_fixture(tmp_path / "share-basis")
    path.write_text(path.read_text().replace("share_basis: ordinary_share\n", ""))

    with pytest.raises(ValidationError, match="share_basis"):
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


def test_rejects_invalid_basis_metadata(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()

    path.write_text(text.replace("share_basis: ordinary_share", "share_basis: adr"))
    with pytest.raises(ValidationError, match="share_basis"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)

    path.write_text(
        text.replace(
            "valuation_basis_note: Uses USD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied.",
            "valuation_basis_note: '   '",
        )
    )
    with pytest.raises(ValidationError, match="valuation_basis_note"):
        load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)


def test_basis_metadata_participates_in_assumption_hash(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    changed_path = _copy_fixture(tmp_path / "changed")
    changed_path.write_text(
        changed_path.read_text().replace(
            "Uses USD ordinary-share basis; no ADR ratio, board-lot, or currency adjustment is applied.",
            "Uses USD ordinary-share basis for an alternate documentation test.",
        )
    )

    original = load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", path)
    changed = load_valuation_assumptions(Market.US, "AAPL", "fcf_dcf_v1", changed_path)

    assert original.assumption_hash != changed.assumption_hash


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

    path.write_text(
        text.replace(
            "      method: provider_ttm_fcf\n",
            "      method: provider_ttm_fcf\n      value: 100.0\n",
            1,
        )
    )
    with pytest.raises(ValidationError, match="starting_fcf.value"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(
        text.replace(
            "      method: provider_ttm_fcf\n",
            "      method: override\n      value: 100.0\n",
            1,
        )
    )
    with pytest.raises(ValidationError, match="starting_fcf.note"):
        load_valuation_assumptions(market=Market.US, ticker="AAPL", model_id="fcf_dcf_v1", assumptions_path=path)

    path.write_text(
        text.replace(
            "      method: provider_ttm_fcf\n",
            "      method: override\n      value: .nan\n      note: Adjusted FCF.\n",
            1,
        )
    )
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


class _FakeModel:
    def __init__(self, model_id: str, fail: bool = False) -> None:
        self.model_id = model_id
        self.fail = fail

    def validate_assumptions(self, assumptions):
        if self.fail:
            raise ValidationError(f"{self.model_id} parser failed")
        return {"model_id": self.model_id, "assumptions": assumptions}


def _write_fake_assumptions(path: Path, *, default_model: str, models: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""
schema_version: 1
market: US
ticker: AAPL
default_model: {default_model}
purpose: fake_schema_test
as_of: 2026-05-17
currency: USD
amount_unit: currency_units
share_basis: ordinary_share
valuation_basis_note: Uses USD ordinary-share basis for fake model tests.

facts_overrides:
  shares_outstanding: null
  net_debt: null
  reference_price: null

facts_override_notes:
  shares_outstanding: null
  net_debt: null
  reference_price: null

assumption_source: fake
prepared_by: test
source_note: Fake assumptions for loader tests.

models:
{models}
""".lstrip()
    )
    return path


def _install_fake_model_registry(monkeypatch: pytest.MonkeyPatch, failing_model_ids: set[str] | None = None) -> None:
    failing_model_ids = failing_model_ids or set()

    monkeypatch.setattr(
        "universe_selector.valuation.assumptions._supported_valuation_model_ids",
        lambda: ("fake_default", "fake_selected"),
    )
    monkeypatch.setattr(
        "universe_selector.valuation.assumptions._get_valuation_model",
        lambda model_id: _FakeModel(model_id, fail=model_id in failing_model_ids),
    )


def test_loader_allows_supported_default_block_omitted_when_cli_selects_other_model(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_model_registry(monkeypatch)
    path = _write_fake_assumptions(
        tmp_path / "AAPL.yaml",
        default_model="fake_default",
        models="  fake_selected:\n    ok: true\n",
    )

    loaded = load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)

    assert loaded.default_model == "fake_default"
    assert loaded.model_id == "fake_selected"
    assert loaded.model_assumptions == {"model_id": "fake_selected", "assumptions": {"ok": True}}


def test_loader_rejects_unknown_and_non_string_model_ids(monkeypatch, tmp_path: Path) -> None:
    _install_fake_model_registry(monkeypatch)

    path = _write_fake_assumptions(
        tmp_path / "unknown.yaml",
        default_model="fake_selected",
        models="  fake_selected:\n    ok: true\n  typo_model:\n    ok: true\n",
    )
    with pytest.raises(ValidationError, match="unknown valuation model typo_model"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)

    path = _write_fake_assumptions(
        tmp_path / "non-string.yaml",
        default_model="fake_selected",
        models="  fake_selected:\n    ok: true\n  123:\n    ok: true\n",
    )
    with pytest.raises(ValidationError, match="model id must be a string"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)


def test_loader_preserves_selected_missing_precedence_before_invalid_unselected_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _install_fake_model_registry(monkeypatch, failing_model_ids={"fake_default"})
    path = _write_fake_assumptions(
        tmp_path / "missing-selected.yaml",
        default_model="fake_default",
        models="  fake_default:\n    bad: true\n",
    )

    with pytest.raises(ValidationError, match="missing model assumptions for fake_selected"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)


def test_loader_rejects_selected_present_but_non_mapping(monkeypatch, tmp_path: Path) -> None:
    _install_fake_model_registry(monkeypatch)
    path = _write_fake_assumptions(
        tmp_path / "selected-list.yaml",
        default_model="fake_selected",
        models="  fake_selected:\n    - not-a-mapping\n",
    )

    with pytest.raises(ValidationError, match="invalid model assumptions for fake_selected"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)


def test_loader_validates_present_unselected_supported_model_blocks(monkeypatch, tmp_path: Path) -> None:
    _install_fake_model_registry(monkeypatch, failing_model_ids={"fake_default"})
    path = _write_fake_assumptions(
        tmp_path / "invalid-unselected.yaml",
        default_model="fake_default",
        models="  fake_default:\n    bad: true\n  fake_selected:\n    ok: true\n",
    )

    with pytest.raises(ValidationError, match="invalid model assumptions for fake_default"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)


def test_loader_rejects_unsupported_default_even_with_cli_override(monkeypatch, tmp_path: Path) -> None:
    _install_fake_model_registry(monkeypatch)
    path = _write_fake_assumptions(
        tmp_path / "unsupported-default.yaml",
        default_model="unknown_default",
        models="  fake_selected:\n    ok: true\n",
    )

    with pytest.raises(ValidationError, match="unknown valuation model unknown_default"):
        load_valuation_assumptions(Market.US, "AAPL", "fake_selected", path)


def test_loader_allows_real_yaml_to_omit_unselected_supported_models(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()
    if "  reverse_dcf_v1:" in text:
        text = text.split("  reverse_dcf_v1:", maxsplit=1)[0]
    path.write_text(text)

    loaded = load_valuation_assumptions(Market.US, "AAPL", None, path)

    assert loaded.model_id == "fcf_dcf_v1"


def test_loader_rejects_real_selected_supported_model_omitted(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()
    if "  reverse_dcf_v1:" in text:
        text = text.split("  reverse_dcf_v1:", maxsplit=1)[0]
    path.write_text(text)

    with pytest.raises(ValidationError, match="missing model assumptions for reverse_dcf_v1"):
        load_valuation_assumptions(Market.US, "AAPL", "reverse_dcf_v1", path)


def test_loader_rejects_invalid_present_but_unselected_real_supported_model(tmp_path: Path) -> None:
    path = _copy_fixture(tmp_path)
    text = path.read_text()
    if "ev_to_fcf_multiple: 16.0" not in text:
        pytest.skip("fixture multiple valuation block has not been added yet")
    path.write_text(text.replace("ev_to_fcf_multiple: 16.0", "ev_to_fcf_multiple: 0.0", 1))

    with pytest.raises(ValidationError, match="invalid model assumptions for multiple_valuation_v1"):
        load_valuation_assumptions(Market.US, "AAPL", None, path)
