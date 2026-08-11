"""Normalize prediction result units for model-specific API contracts."""

from __future__ import annotations

from typing import Any

from util import predict_constants as pc


LOG_KOC_MODEL_IDS = (1754, 1756, 1757, 1758, 1763)
LOG_KOC_PROPERTY_NAMES = frozenset({pc.KOC, pc.LOG_KOC})


def _is_known_log_koc_model(model_id: Any) -> bool:
    try:
        return int(model_id) in LOG_KOC_MODEL_IDS
    except (TypeError, ValueError):
        return False


def is_log_koc_model_details(model_details: Any, model_id: Any = None) -> bool:
    if _is_known_log_koc_model(model_id):
        return True

    if not isinstance(model_details, dict):
        return False

    return (
        model_details.get("propertyName") in LOG_KOC_PROPERTY_NAMES
        and model_details.get("unitsModel") == pc.LOG_L_KG
    )


def normalize_log_koc_model_details(model_details: Any, model_id: Any = None):
    if not isinstance(model_details, dict) or not is_log_koc_model_details(model_details, model_id):
        return model_details

    normalized = dict(model_details)
    previous_display_units = normalized.get("unitsDisplay")
    normalized["unitsLinear"] = normalized.get("unitsLinear") or (
        previous_display_units
        if previous_display_units and previous_display_units != normalized.get("unitsModel")
        else pc.L_KG
    )
    normalized["unitsDisplay"] = normalized.get("unitsModel") or pc.LOG_L_KG
    return normalized


def normalize_log_koc_prediction(prediction: Any, model_id: Any = None):
    if not isinstance(prediction, dict):
        return prediction

    model_details = prediction.get("modelDetails")
    if not is_log_koc_model_details(model_details, model_id):
        return prediction

    model_results = prediction.get("modelResults")
    if not isinstance(model_results, dict):
        return prediction

    normalized_results = dict(model_results)
    normalized_results.setdefault(
        "experimentalValueUnitsLinear",
        normalized_results.get("experimentalValueUnitsDisplay"),
    )
    normalized_results["experimentalValueUnitsDisplay"] = normalized_results.get(
        "experimentalValueUnitsModel"
    )
    normalized_results.setdefault(
        "predictionValueUnitsLinear",
        normalized_results.get("predictionValueUnitsDisplay"),
    )
    normalized_results["predictionValueUnitsDisplay"] = normalized_results.get(
        "predictionValueUnitsModel"
    )
    previous_display_units = normalized_results.get("unitsDisplay")
    normalized_results.setdefault(
        "unitsLinear",
        previous_display_units
        if previous_display_units and previous_display_units != normalized_results.get("unitsModel")
        else pc.L_KG,
    )
    normalized_results["unitsDisplay"] = normalized_results.get("unitsModel") or pc.LOG_L_KG

    normalized = dict(prediction)
    normalized["modelResults"] = normalized_results
    if isinstance(model_details, dict):
        normalized["modelDetails"] = normalize_log_koc_model_details(model_details, model_id)
    return normalized
