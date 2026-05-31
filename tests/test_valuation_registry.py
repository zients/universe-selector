from __future__ import annotations

from universe_selector.errors import ValidationError
from universe_selector.valuation.registry import (
    ValuationModelRegistration,
    _build_registrations,
    get_valuation_model,
    get_valuation_output_renderer,
    supported_valuation_model_ids,
)


class DummyModel:
    def __init__(self, model_id: str = "dummy") -> None:
        self.model_id = model_id


class DummyRenderer:
    def __init__(self, model_id: str = "dummy") -> None:
        self.model_id = model_id


def test_registry_returns_fresh_model_and_renderer_instances() -> None:
    assert supported_valuation_model_ids() == (
        "exit_multiple_dcf_v1",
        "fcf_dcf_v1",
        "multiple_valuation_v1",
        "reverse_dcf_v1",
        "two_stage_fcf_dcf_v1",
    )

    first_model = get_valuation_model("fcf_dcf_v1")
    second_model = get_valuation_model("fcf_dcf_v1")
    first_renderer = get_valuation_output_renderer("fcf_dcf_v1")
    second_renderer = get_valuation_output_renderer("fcf_dcf_v1")

    assert first_model.model_id == "fcf_dcf_v1"
    assert second_model.model_id == "fcf_dcf_v1"
    assert first_model is not second_model
    assert first_renderer.model_id == "fcf_dcf_v1"
    assert second_renderer.model_id == "fcf_dcf_v1"
    assert first_renderer is not second_renderer

    assert get_valuation_model("exit_multiple_dcf_v1").model_id == "exit_multiple_dcf_v1"
    assert get_valuation_output_renderer("exit_multiple_dcf_v1").model_id == "exit_multiple_dcf_v1"
    assert get_valuation_model("reverse_dcf_v1").model_id == "reverse_dcf_v1"
    assert get_valuation_output_renderer("reverse_dcf_v1").model_id == "reverse_dcf_v1"
    assert get_valuation_model("multiple_valuation_v1").model_id == "multiple_valuation_v1"
    assert get_valuation_output_renderer("multiple_valuation_v1").model_id == "multiple_valuation_v1"
    assert get_valuation_model("two_stage_fcf_dcf_v1").model_id == "two_stage_fcf_dcf_v1"
    assert get_valuation_output_renderer("two_stage_fcf_dcf_v1").model_id == "two_stage_fcf_dcf_v1"


def test_registry_builder_rejects_duplicate_model_ids() -> None:
    registration = ValuationModelRegistration(
        model_id="dummy",
        model_factory=DummyModel,
        output_renderer_factory=DummyRenderer,
    )

    try:
        _build_registrations((registration, registration))
    except ValidationError as exc:
        assert "duplicate valuation model id dummy" in str(exc)
    else:
        raise AssertionError("duplicate registrations should fail")


def test_registry_registration_keeps_legacy_model_field_compatibility() -> None:
    model = DummyModel()
    registration = ValuationModelRegistration(
        model_id="dummy",
        model=model,
        output_renderer_factory=DummyRenderer,
    )
    registrations = _build_registrations((registration,))

    assert registration.model is model
    assert registrations["dummy"].model is model
    assert registrations["dummy"].model_factory() is model


def test_registry_builder_rejects_factory_model_id_mismatches() -> None:
    try:
        _build_registrations(
            (
                ValuationModelRegistration(
                    model_id="dummy",
                    model_factory=lambda: DummyModel("other"),
                    output_renderer_factory=DummyRenderer,
                ),
            )
        )
    except ValidationError as exc:
        assert "model factory for dummy returned model_id other" in str(exc)
    else:
        raise AssertionError("model factory mismatch should fail")

    try:
        _build_registrations(
            (
                ValuationModelRegistration(
                    model_id="dummy",
                    model_factory=DummyModel,
                    output_renderer_factory=lambda: DummyRenderer("other"),
                ),
            )
        )
    except ValidationError as exc:
        assert "renderer factory for dummy returned model_id other" in str(exc)
    else:
        raise AssertionError("renderer factory mismatch should fail")


def test_registry_registration_rejects_ambiguous_model_and_factory() -> None:
    try:
        ValuationModelRegistration(
            model_id="dummy",
            model=DummyModel(),
            model_factory=DummyModel,
            output_renderer_factory=DummyRenderer,
        )
    except TypeError as exc:
        assert "model and model_factory cannot both be provided" in str(exc)
    else:
        raise AssertionError("ambiguous model registration should fail")
