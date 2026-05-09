from __future__ import annotations

from collections.abc import Mapping

from universe_selector.errors import ValidationError
from universe_selector.ranking_profiles.base import RankingProfile
from universe_selector.ranking_profiles.registration import (
    RankingProfileRegistration,
    build_ranking_profile_registration_map,
)
from universe_selector.ranking_profiles.sample_price_trend_v1 import SAMPLE_PRICE_TREND_V1_REGISTRATION


_REGISTRATIONS: tuple[RankingProfileRegistration, ...] = (
    SAMPLE_PRICE_TREND_V1_REGISTRATION,
)

_REGISTRATION_BY_ID: Mapping[str, RankingProfileRegistration] = build_ranking_profile_registration_map(_REGISTRATIONS)


def supported_ranking_profile_ids() -> tuple[str, ...]:
    return tuple(_REGISTRATION_BY_ID)


def get_ranking_profile_registration(profile_id: str) -> RankingProfileRegistration:
    registration = _REGISTRATION_BY_ID.get(profile_id)
    if registration is not None:
        return registration
    supported = ", ".join(supported_ranking_profile_ids())
    raise ValidationError(f"unknown ranking profile {profile_id}; supported profiles: {supported}")


def get_ranking_profile(profile_id: str) -> RankingProfile:
    return get_ranking_profile_registration(profile_id).create_profile()
