from __future__ import annotations

import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata, FundamentalsRunData
from universe_selector.providers.registration import FundamentalsProviderRegistration
from universe_selector.valuation.models import ValuationModelInput, ValuationScenarioResult
from universe_selector.valuation.service import run_valuation


FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "us" / "AAPL.yaml"


class FakeFundamentalsProvider:
    provider_id = "fake_fundamentals"
    source_ids = ("fake-source",)

    def __init__(self, facts: FundamentalFacts) -> None:
        self._facts = facts
        self.requests: list[tuple[Market, str]] = []

    def load_fundamentals(self, market: Market, ticker: str) -> FundamentalsRunData:
        self.requests.append((market, ticker))
        return FundamentalsRunData(
            metadata=FundamentalsMetadata(
                data_mode="live",
                fundamentals_provider_id=self.provider_id,
                fundamentals_source_ids=self.source_ids,
                data_fetch_started_at=datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc),
                facts_as_of=date(2026, 5, 15),
            ),
            facts=self._facts,
        )


class SpyValuationModel:
    model_id = "fcf_dcf_v1"

    def __init__(self) -> None:
        self.inputs: list[ValuationModelInput] = []

    def validate_assumptions(self, assumptions):
        raise AssertionError("service model spy must not validate YAML assumptions")

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        self.inputs.append(model_input)
        assert not hasattr(model_input, "raw_facts")
        assert not hasattr(model_input, "fundamentals_metadata")
        effective = model_input.effective_inputs
        return (
            ValuationScenarioResult(
                scenario_id="base",
                projected_fcf=(effective.normalized_fcf,),
                present_value_projected_fcf=(effective.normalized_fcf,),
                terminal_value=0.0,
                present_value_terminal_value=0.0,
                enterprise_value=effective.normalized_fcf,
                equity_value=effective.normalized_fcf - effective.net_debt,
                model_implied_value_per_share=(effective.normalized_fcf - effective.net_debt)
                / effective.shares_outstanding,
                reference_price=effective.reference_price,
                model_implied_spread_to_reference_price=0.0,
            ),
        )


def _facts() -> FundamentalFacts:
    return FundamentalFacts(
        market=Market.US,
        ticker="AAPL",
        currency="USD",
        reference_price=190.0,
        reference_price_as_of=date(2026, 5, 15),
        reference_price_as_of_source="provider_reported",
        reference_price_as_of_note=None,
        shares_outstanding=10.0,
        cash_and_cash_equivalents=60.0,
        total_debt=110.0,
        balance_sheet_as_of=date(2025, 9, 30),
        net_debt=50.0,
        operating_cash_flow=120.0,
        capital_expenditures=10.0,
        free_cash_flow=110.0,
        fiscal_period_end=date(2025, 9, 30),
        fiscal_period_type="ttm",
    )


def _copy_fixture_with_overrides(tmp_path: Path) -> Path:
    target = tmp_path / "valuation_assumptions" / "us" / "AAPL.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, target)
    text = target.read_text()
    text = text.replace("normalized_fcf: null", "normalized_fcf: 100.0", 1)
    text = text.replace("reference_price: null", "reference_price: 185.0", 1)
    text = text.replace(
        "  normalized_fcf: null\n  shares_outstanding: null\n  net_debt: null\n  reference_price: null\n",
        "  normalized_fcf: Normalized for one-time working capital movement.\n"
        "  shares_outstanding: null\n"
        "  net_debt: null\n"
        "  reference_price: Reference price supplied for scenario review.\n",
        1,
    )
    target.write_text(text)
    return target


def _install_no_ranking_no_persistence_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "universe_selector.config.load_config",
        lambda: (_ for _ in ()).throw(AssertionError("load_config")),
    )
    monkeypatch.setattr(
        "universe_selector.config.ensure_runtime_dirs",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ensure_runtime_dirs")),
    )
    monkeypatch.setattr(
        "universe_selector.persistence.repository.DuckDbRepository",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("DuckDbRepository")),
    )
    monkeypatch.setattr(
        "universe_selector.persistence.schema.validate_schema",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("validate_schema")),
    )
    monkeypatch.setattr(
        "universe_selector.pipeline.run_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_batch")),
    )
    monkeypatch.setattr(
        "universe_selector.pipeline.run_batch_profiles",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run_batch_profiles")),
    )


def test_run_valuation_loads_facts_assumptions_applies_overrides_and_calls_model(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    assumptions_path = _copy_fixture_with_overrides(tmp_path)
    fake_provider = FakeFundamentalsProvider(_facts())
    fake_registration = FundamentalsProviderRegistration(
        provider_id=fake_provider.provider_id,
        supported_markets=frozenset({Market.US}),
        source_ids=fake_provider.source_ids,
        factory=lambda: fake_provider,
    )
    spy_model = SpyValuationModel()
    monkeypatch.setattr(
        "universe_selector.valuation.service.get_fundamentals_registration",
        lambda provider_id, market: fake_registration,
    )
    monkeypatch.setattr("universe_selector.valuation.service.get_valuation_model", lambda model_id: spy_model)

    result = run_valuation(Market.US, "aapl", "fcf_dcf_v1", assumptions_path)

    assert fake_provider.requests == [(Market.US, "AAPL")]
    assert len(spy_model.inputs) == 1
    assert spy_model.inputs[0].ticker == "AAPL"
    assert result.run_input.market is Market.US
    assert result.run_input.ticker == "AAPL"
    assert result.run_input.raw_facts.free_cash_flow == 110.0
    assert result.run_input.effective_inputs.normalized_fcf == 100.0
    assert result.run_input.effective_inputs.shares_outstanding == 10.0
    assert result.run_input.effective_inputs.net_debt == 50.0
    assert result.run_input.effective_inputs.reference_price == 185.0
    assert result.run_input.effective_inputs.reference_price_as_of == result.run_input.assumptions.as_of
    assert result.run_input.effective_inputs.reference_price_as_of_source == "assumption_override"
    assert result.run_input.effective_inputs.reference_price_as_of_note == "Reference price supplied for scenario review."
    assert result.run_input.raw_facts.reference_price_as_of_source == "provider_reported"
    assert result.run_input.input_provenance.normalized_fcf_source == "assumption_override"
    assert result.run_input.input_provenance.normalized_fcf_note == "Normalized for one-time working capital movement."
    assert result.run_input.input_provenance.shares_outstanding_source == "provider_fact"
    assert result.run_input.input_provenance.net_debt_source == "provider_fact"
    assert result.run_input.input_provenance.reference_price_source == "assumption_override"
    assert result.scenario_results[0].scenario_id == "base"


def test_run_valuation_uses_default_assumptions_path(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    _copy_fixture_with_overrides(tmp_path)
    monkeypatch.chdir(tmp_path)
    fake_provider = FakeFundamentalsProvider(_facts())
    fake_registration = FundamentalsProviderRegistration(
        provider_id=fake_provider.provider_id,
        supported_markets=frozenset({Market.US}),
        source_ids=fake_provider.source_ids,
        factory=lambda: fake_provider,
    )
    monkeypatch.setattr(
        "universe_selector.valuation.service.get_fundamentals_registration",
        lambda provider_id, market: fake_registration,
    )

    result = run_valuation(Market.US, "AAPL", "fcf_dcf_v1", None)

    assert result.run_input.assumptions.assumption_path.endswith("valuation_assumptions/us/AAPL.yaml")


def test_run_valuation_rejects_unknown_model_and_missing_assumptions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown valuation model"):
        run_valuation(Market.US, "AAPL", "unknown_model", tmp_path / "missing.yaml")

    with pytest.raises(ValidationError, match="missing valuation assumptions file"):
        run_valuation(Market.US, "AAPL", "fcf_dcf_v1", tmp_path / "missing.yaml")


def test_run_valuation_rejects_unsupported_fundamentals_market_before_loading_assumptions(monkeypatch, tmp_path: Path) -> None:
    def fail_load_assumptions(*args, **kwargs):
        raise AssertionError("assumptions must not load before unsupported fundamentals market is rejected")

    monkeypatch.setattr("universe_selector.valuation.service.load_valuation_assumptions", fail_load_assumptions)

    with pytest.raises(ValidationError, match="unsupported fundamentals provider for TW: yfinance_fundamentals"):
        run_valuation(Market.TW, "2330", "fcf_dcf_v1", tmp_path / "missing.yaml")
