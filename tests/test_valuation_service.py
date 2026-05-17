from __future__ import annotations

import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from universe_selector.domain import Market
from universe_selector.errors import ValidationError
from universe_selector.providers.models import FundamentalFacts, FundamentalsMetadata, FundamentalsRunData
from universe_selector.providers.registration import FundamentalsProviderRegistration
from universe_selector.valuation.models import (
    EffectiveValuationInputs,
    ValuationInputProvenance,
    ValuationModelInput,
    ValuationScenarioResult,
)
from universe_selector.valuation.service import run_valuation


FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "us" / "AAPL.yaml"
TW_FIXTURE = Path(__file__).parent / "fixtures" / "valuation_assumptions" / "tw" / "2330.yaml"


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
                latest_source_date=date(2026, 5, 15),
            ),
            facts=self._facts,
        )


class SpyValuationModel:
    model_id = "fcf_dcf_v1"

    def __init__(self) -> None:
        self.build_input_requests: list[tuple[FundamentalFacts, object]] = []
        self.inputs: list[ValuationModelInput] = []

    def validate_assumptions(self, assumptions):
        raise AssertionError("service model spy must not validate YAML assumptions")

    def build_inputs(self, *, facts: FundamentalFacts, assumptions: object):
        self.build_input_requests.append((facts, assumptions))
        return (
            EffectiveValuationInputs(
                starting_fcf=111.0,
                shares_outstanding=facts.shares_outstanding,
                net_debt=facts.net_debt,
                reference_price=185.0,
                currency=facts.currency,
                fiscal_period_type=facts.fiscal_period_type,
                fiscal_period_end=facts.fiscal_period_end,
                reference_price_as_of=date(2026, 5, 16),
                reference_price_as_of_source="model_resolved_override",
                reference_price_as_of_note="Resolved by spy valuation model.",
            ),
            ValuationInputProvenance(
                starting_fcf_source="model_resolved",
                shares_outstanding_source="provider_fact",
                net_debt_source="provider_fact",
                reference_price_source="model_resolved_override",
                starting_fcf_note="Resolved by spy valuation model.",
                shares_outstanding_note=None,
                net_debt_note=None,
                reference_price_note="Resolved by spy valuation model.",
            ),
        )

    def value(self, model_input: ValuationModelInput) -> tuple[ValuationScenarioResult, ...]:
        self.inputs.append(model_input)
        assert not hasattr(model_input, "raw_facts")
        assert not hasattr(model_input, "fundamentals_metadata")
        effective = model_input.effective_inputs
        return (
            ValuationScenarioResult(
                scenario_id="base",
                projected_fcf=(effective.starting_fcf,),
                present_value_projected_fcf=(effective.starting_fcf,),
                terminal_value=0.0,
                present_value_terminal_value=0.0,
                enterprise_value=effective.starting_fcf,
                equity_value=effective.starting_fcf - effective.net_debt,
                model_implied_value_per_share=(effective.starting_fcf - effective.net_debt)
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


def _copy_fixture_with_reference_price_override(tmp_path: Path) -> Path:
    target = tmp_path / "valuation_assumptions" / "us" / "AAPL.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, target)
    text = target.read_text()
    text = text.replace("reference_price: null", "reference_price: 185.0", 1)
    text = text.replace(
        "  shares_outstanding: null\n"
        "  net_debt: null\n"
        "  reference_price: null\n",
        "  shares_outstanding: null\n"
        "  net_debt: null\n"
        "  reference_price: Reference price supplied for scenario review.\n",
        1,
    )
    target.write_text(text)
    return target


def _copy_fixture_with_starting_fcf_override(tmp_path: Path) -> Path:
    target = _copy_fixture_with_reference_price_override(tmp_path)
    text = target.read_text()
    text = text.replace(
        "      method: provider_ttm_fcf\n",
        "      method: override\n"
        "      value: 100.0\n"
        "      note: Normalized for one-time working capital movement.\n",
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


def test_run_valuation_uses_provider_ttm_fcf_starting_fcf_and_calls_model(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    assumptions_path = _copy_fixture_with_reference_price_override(tmp_path)
    fake_provider = FakeFundamentalsProvider(_facts())
    fake_registration = FundamentalsProviderRegistration(
        provider_id=fake_provider.provider_id,
        supported_markets=frozenset({Market.US}),
        source_ids=fake_provider.source_ids,
        factory=lambda: fake_provider,
    )
    spy_model = SpyValuationModel()

    def resolve_registration(provider_id: str, market: Market) -> FundamentalsProviderRegistration:
        assert provider_id == "fake_fundamentals"
        assert market is Market.US
        return fake_registration

    monkeypatch.setattr(
        "universe_selector.valuation.service.get_fundamentals_registration",
        resolve_registration,
    )
    monkeypatch.setattr("universe_selector.valuation.service.get_valuation_model", lambda model_id: spy_model)

    result = run_valuation(Market.US, "aapl", None, assumptions_path, "fake_fundamentals")

    assert fake_provider.requests == [(Market.US, "AAPL")]
    assert len(spy_model.build_input_requests) == 1
    request_facts, request_assumptions = spy_model.build_input_requests[0]
    assert request_facts == _facts()
    assert request_assumptions is result.run_input.assumptions
    assert len(spy_model.inputs) == 1
    assert spy_model.inputs[0].ticker == "AAPL"
    assert result.run_input.market is Market.US
    assert result.run_input.ticker == "AAPL"
    assert result.run_input.model_id == "fcf_dcf_v1"
    assert result.run_input.raw_facts.free_cash_flow == 110.0
    assert result.run_input.effective_inputs.starting_fcf == 111.0
    assert result.run_input.effective_inputs.shares_outstanding == 10.0
    assert result.run_input.effective_inputs.net_debt == 50.0
    assert result.run_input.effective_inputs.reference_price == 185.0
    assert result.run_input.effective_inputs.reference_price_as_of == date(2026, 5, 16)
    assert result.run_input.effective_inputs.reference_price_as_of_source == "model_resolved_override"
    assert result.run_input.effective_inputs.reference_price_as_of_note == "Resolved by spy valuation model."
    assert result.run_input.raw_facts.reference_price_as_of_source == "provider_reported"
    assert result.run_input.input_provenance.starting_fcf_source == "model_resolved"
    assert result.run_input.input_provenance.starting_fcf_note == "Resolved by spy valuation model."
    assert result.run_input.input_provenance.shares_outstanding_source == "provider_fact"
    assert result.run_input.input_provenance.net_debt_source == "provider_fact"
    assert result.run_input.input_provenance.reference_price_source == "model_resolved_override"
    assert result.scenario_results[0].scenario_id == "base"


def test_run_valuation_uses_override_starting_fcf(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    assumptions_path = _copy_fixture_with_starting_fcf_override(tmp_path)
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

    result = run_valuation(Market.US, "AAPL", "fcf_dcf_v1", assumptions_path, "fake_fundamentals")

    assert fake_provider.requests == [(Market.US, "AAPL")]
    assert result.run_input.effective_inputs.starting_fcf == 100.0
    assert result.run_input.input_provenance.starting_fcf_source == "assumption_override"
    assert result.run_input.input_provenance.starting_fcf_note == "Normalized for one-time working capital movement."


def test_run_valuation_rejects_assumption_currency_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    assumptions_path = _copy_fixture_with_reference_price_override(tmp_path)
    assumptions_path.write_text(assumptions_path.read_text().replace("currency: USD", "currency: TWD"))
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

    with pytest.raises(ValidationError, match="assumptions currency TWD must match provider facts currency USD"):
        run_valuation(Market.US, "AAPL", "fcf_dcf_v1", assumptions_path, "fake_fundamentals")


def test_run_valuation_uses_default_assumptions_path(monkeypatch, tmp_path: Path) -> None:
    _install_no_ranking_no_persistence_guards(monkeypatch)
    _copy_fixture_with_reference_price_override(tmp_path)
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

    result = run_valuation(Market.US, "AAPL", None, None, "fake_fundamentals")

    assert result.run_input.assumptions.assumption_path.endswith("valuation_assumptions/us/AAPL.yaml")


def test_run_valuation_rejects_unknown_model_and_missing_assumptions(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown valuation model"):
        run_valuation(Market.US, "AAPL", "unknown_model", tmp_path / "missing.yaml", "yfinance_fundamentals")

    with pytest.raises(ValidationError, match="missing valuation assumptions file"):
        run_valuation(Market.US, "AAPL", "fcf_dcf_v1", tmp_path / "missing.yaml", "yfinance_fundamentals")


def test_run_valuation_loads_tw_assumptions_after_resolving_supported_provider(monkeypatch, tmp_path: Path) -> None:
    fake_provider = FakeFundamentalsProvider(
        FundamentalFacts(
            market=Market.TW,
            ticker="2330",
            currency="TWD",
            reference_price=800.0,
            reference_price_as_of=date(2026, 5, 15),
            reference_price_as_of_source="provider_reported",
            reference_price_as_of_note=None,
            shares_outstanding=25_900_000_000.0,
            cash_and_cash_equivalents=2_000_000_000_000.0,
            total_debt=1_000_000_000_000.0,
            balance_sheet_as_of=date(2026, 3, 31),
            net_debt=-1_000_000_000_000.0,
            operating_cash_flow=1_000_000_000_000.0,
            capital_expenditures=300_000_000_000.0,
            free_cash_flow=700_000_000_000.0,
            fiscal_period_end=date(2025, 12, 31),
            fiscal_period_type="ttm",
        )
    )
    fake_registration = FundamentalsProviderRegistration(
        provider_id=fake_provider.provider_id,
        supported_markets=frozenset({Market.TW}),
        source_ids=fake_provider.source_ids,
        factory=lambda: fake_provider,
    )

    monkeypatch.setattr(
        "universe_selector.valuation.service.get_fundamentals_registration",
        lambda provider_id, market: fake_registration,
    )

    result = run_valuation(Market.TW, "2330", "fcf_dcf_v1", TW_FIXTURE, "fake_fundamentals")

    assert fake_provider.requests == [(Market.TW, "2330")]
    assert result.run_input.market is Market.TW
    assert result.run_input.ticker == "2330"
    assert result.run_input.assumptions.assumption_path.endswith("valuation_assumptions/tw/2330.yaml")
    assert result.run_input.effective_inputs.currency == "TWD"
