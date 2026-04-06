"""
Helpers for migrating legacy pickled model objects to the current runtime.
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
import pickle
import tempfile
from typing import Any

from xgboost import Booster, XGBModel


def serialize_model(model: Any) -> bytes:
    """
    Serialize model objects with the highest available pickle protocol.
    """
    return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)


def refresh_legacy_serialized_model(
    model: Any,
    *,
    logger: logging.Logger | None = None,
) -> tuple[Any, dict[str, int]]:
    """
    Walk a deserialized model object graph and refresh legacy XGBoost objects
    using the current runtime's native save/load path.

    Returns the potentially updated root object together with migration stats.
    """
    logger = logger or logging.getLogger(__name__)
    stats = {"xgboost_objects": 0}

    if hasattr(model, "model_obj"):
        refreshed_model_obj = _refresh_object_graph(
            model.model_obj,
            seen=set(),
            stats=stats,
            logger=logger,
        )
        if refreshed_model_obj is not model.model_obj:
            model.model_obj = refreshed_model_obj
        return model, stats

    refreshed = _refresh_object_graph(model, seen=set(), stats=stats, logger=logger)
    return refreshed, stats


def _refresh_object_graph(
    obj: Any,
    *,
    seen: set[int],
    stats: dict[str, int],
    logger: logging.Logger,
) -> Any:
    if obj is None:
        return None

    obj_id = id(obj)
    if obj_id in seen:
        return obj
    seen.add(obj_id)

    if isinstance(obj, XGBModel):
        refreshed = _refresh_xgb_estimator(obj, logger=logger)
        if refreshed is not obj:
            stats["xgboost_objects"] += 1
        return refreshed

    if isinstance(obj, Booster):
        refreshed = _refresh_booster(obj, logger=logger)
        if refreshed is not obj:
            stats["xgboost_objects"] += 1
        return refreshed

    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            obj[idx] = _refresh_object_graph(item, seen=seen, stats=stats, logger=logger)
        return obj

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            obj[key] = _refresh_object_graph(value, seen=seen, stats=stats, logger=logger)
        return obj

    if isinstance(obj, tuple):
        return tuple(
            _refresh_object_graph(item, seen=seen, stats=stats, logger=logger)
            for item in obj
        )

    if isinstance(obj, set):
        new_items = {
            _refresh_object_graph(item, seen=seen, stats=stats, logger=logger)
            for item in obj
        }
        if new_items != obj:
            obj.clear()
            obj.update(new_items)
        return obj

    values = getattr(obj, "__dict__", None)
    if not values:
        return obj

    for attr_name, attr_value in list(values.items()):
        refreshed_value = _refresh_object_graph(
            attr_value,
            seen=seen,
            stats=stats,
            logger=logger,
        )
        if refreshed_value is not attr_value:
            setattr(obj, attr_name, refreshed_value)

    return obj


def _refresh_xgb_estimator(estimator: XGBModel, *, logger: logging.Logger) -> XGBModel:
    temp_path = None
    try:
        _ensure_init_attributes(estimator)
        temp_path = _make_temp_model_path()
        estimator.save_model(temp_path)
        refreshed = estimator.__class__()
        _ensure_init_attributes(refreshed)
        refreshed.load_model(temp_path)
        _copy_init_attributes(estimator, refreshed)

        for attr_name in (
            "classes_",
            "feature_names_in_",
            "n_features_in_",
            "n_classes_",
            "best_iteration",
            "best_score",
            "feature_types",
            "evals_result_",
        ):
            if hasattr(estimator, attr_name) and not hasattr(refreshed, attr_name):
                setattr(refreshed, attr_name, getattr(estimator, attr_name))

        return refreshed
    except Exception:
        logger.exception(
            "Failed to refresh legacy XGBoost estimator %s; keeping original object",
            type(estimator).__name__,
        )
        return estimator
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _refresh_booster(booster: Booster, *, logger: logging.Logger) -> Booster:
    temp_path = None
    try:
        refreshed = Booster()
        temp_path = _make_temp_model_path()
        booster.save_model(temp_path)
        refreshed.load_model(temp_path)
        return refreshed
    except Exception:
        logger.exception("Failed to refresh legacy XGBoost booster; keeping original object")
        return booster
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _make_temp_model_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        return handle.name


def _ensure_init_attributes(estimator: XGBModel) -> None:
    for attr_name, default_value in _iter_init_defaults(estimator.__class__):
        if not hasattr(estimator, attr_name):
            setattr(estimator, attr_name, copy.deepcopy(default_value))


def _copy_init_attributes(source: XGBModel, target: XGBModel) -> None:
    for attr_name, default_value in _iter_init_defaults(source.__class__):
        if hasattr(source, attr_name):
            setattr(target, attr_name, copy.deepcopy(getattr(source, attr_name)))
        elif not hasattr(target, attr_name):
            setattr(target, attr_name, copy.deepcopy(default_value))


def _iter_init_defaults(cls):
    signature = inspect.signature(cls.__init__)
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect.Parameter.empty:
            continue
        yield parameter.name, parameter.default
