from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from universe_selector.errors import ValidationError
from universe_selector.valuation.base import ValuationModel
from universe_selector.valuation.fcf_dcf_v1 import FcfDcfV1Model


_MODELS: Mapping[str, ValuationModel] = MappingProxyType({"fcf_dcf_v1": FcfDcfV1Model()})


def supported_valuation_model_ids() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))


def get_valuation_model(model_id: str) -> ValuationModel:
    model = _MODELS.get(model_id)
    if model is None:
        supported = ", ".join(supported_valuation_model_ids())
        raise ValidationError(f"unknown valuation model {model_id}; supported models: {supported}")
    return model
