import json
import logging
import os
import re
import threading
import time

from bson.errors import InvalidDocument
from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.errors import PyMongoError


predictor_models_cache = None
_mongo_client = None
in_memory_cache = {}
_mongo_init_done = False
_mongo_unavailable_reason = None
_mongo_last_init_attempt_monotonic = None
_mongo_init_lock = threading.Lock()


def _reset_mongo_state_after_fork():
    global predictor_models_cache, _mongo_client, _mongo_init_done
    global _mongo_unavailable_reason, _mongo_last_init_attempt_monotonic
    global _mongo_init_lock

    inherited_client = _mongo_client
    predictor_models_cache = None
    _mongo_client = None
    _mongo_init_done = False
    _mongo_unavailable_reason = None
    _mongo_last_init_attempt_monotonic = None
    _mongo_init_lock = threading.Lock()

    if inherited_client is not None:
        try:
            inherited_client.close()
        except Exception:
            logging.debug("Failed to close inherited Mongo client after fork", exc_info=True)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_mongo_state_after_fork)


def _has_meaningful_error_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _prediction_has_error(prediction) -> bool:
    if isinstance(prediction, (bytes, bytearray)):
        try:
            prediction = json.loads(prediction.decode("utf-8"))
        except Exception:
            return False

    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except Exception:
            return False

    if isinstance(prediction, dict):
        if _has_meaningful_error_value(prediction.get("error")):
            return True

        if _has_meaningful_error_value(prediction.get("predictionError")):
            return True

        for value in prediction.values():
            if _prediction_has_error(value):
                return True

        return False

    if isinstance(prediction, list):
        for item in prediction:
            if _prediction_has_error(item):
                return True

    return False


def _preview_value(value, max_len: int = 300) -> str:
    if isinstance(value, dict):
        return f"dict keys={list(value.keys())[:10]}"
    if isinstance(value, list):
        return f"list len={len(value)}"

    text = str(value).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _find_prediction_error_reason(prediction, path="prediction"):
    prediction = _prediction_to_obj(prediction)

    if isinstance(prediction, dict):
        for field_name in ("error", "predictionError"):
            field_value = prediction.get(field_name)
            if _has_meaningful_error_value(field_value):
                return f"{path}.{field_name}={_preview_value(field_value)}"

        for key, value in prediction.items():
            nested_reason = _find_prediction_error_reason(value, f"{path}.{key}")
            if nested_reason is not None:
                return nested_reason
        return None

    if isinstance(prediction, list):
        for index, item in enumerate(prediction):
            nested_reason = _find_prediction_error_reason(item, f"{path}[{index}]")
            if nested_reason is not None:
                return nested_reason

    return None


def _warn_mongo_cache_skip(
    key: str,
    reason: str,
    fallback: str | None = None,
    log_warning: bool = True,
):
    if not log_warning:
        return
    if fallback:
        logging.warning("Mongo cache skipped for key=%r: %s; fallback=%s", key, reason, fallback)
    else:
        logging.warning("Mongo cache skipped for key=%r: %s", key, reason)


def _prediction_to_obj(prediction):
    if isinstance(prediction, (bytes, bytearray)):
        try:
            return json.loads(prediction.decode("utf-8"))
        except Exception:
            return prediction.decode("utf-8", errors="replace")

    if isinstance(prediction, str):
        try:
            return json.loads(prediction)
        except Exception:
            return prediction

    return prediction


def _coerce_bson_safe(value):
    if isinstance(value, dict):
        return {str(key): _coerce_bson_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_coerce_bson_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_coerce_bson_safe(item) for item in value]

    if isinstance(value, set):
        return [_coerce_bson_safe(item) for item in value]

    # Handle numpy/pandas scalar-like values without importing those packages here.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar_value = item_method()
        except Exception:
            scalar_value = None
        else:
            if scalar_value is not value:
                return _coerce_bson_safe(scalar_value)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            list_value = tolist_method()
        except Exception:
            list_value = None
        else:
            if list_value is not value:
                return _coerce_bson_safe(list_value)

    return value


def _normalize_prediction_for_storage(prediction):
    prediction_obj = _prediction_to_obj(prediction)
    return _coerce_bson_safe(prediction_obj)


def _wildcard_key_parts(key):
    if not isinstance(key, str) or key.count("*") != 1:
        return None

    prefix, suffix = key.split("*", 1)
    if not prefix:
        return None

    return prefix, suffix


def _wildcard_key_matches(key, prefix, suffix=""):
    if not isinstance(key, str) or not key.startswith(prefix):
        return False
    return not suffix or key.endswith(suffix)


def _wildcard_key_query(prefix, suffix=""):
    regex = f"^{re.escape(prefix)}"
    if suffix:
        regex = f"{regex}.*{re.escape(suffix)}$"
    return {"key": {"$regex": regex}}


def _find_one_by_wildcard(collection, prefix, suffix=""):
    return collection.find_one(
        _wildcard_key_query(prefix, suffix),
        sort=[("key", ASCENDING)],
    )


def _get_in_memory_by_wildcard(prefix, suffix=""):
    for key, prediction in in_memory_cache.items():
        if _wildcard_key_matches(key, prefix, suffix):
            return key, prediction
    return None


def _populate_wildcard_hits_from_exact_hits(cached_predictions, wildcard_keys):
    for wildcard_key in wildcard_keys:
        if wildcard_key in cached_predictions:
            continue

        wildcard_parts = _wildcard_key_parts(wildcard_key)
        if wildcard_parts is None:
            continue
        prefix, suffix = wildcard_parts

        for cached_key, prediction in list(cached_predictions.items()):
            if _wildcard_key_parts(cached_key) is not None:
                continue
            if _wildcard_key_matches(cached_key, prefix, suffix):
                cached_predictions[wildcard_key] = prediction
                break


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logging.warning("Invalid boolean value for %s=%r; using default=%s", name, raw, default)
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError:
        logging.warning("Invalid float value for %s=%r; using default=%s", name, raw, default)
        return default


def _init_mongo():
    global predictor_models_cache, _mongo_client, _mongo_init_done, _mongo_unavailable_reason, _mongo_last_init_attempt_monotonic
    _mongo_last_init_attempt_monotonic = time.monotonic()

    if not _env_bool("MONGO_CACHE_ENABLED", True):
        predictor_models_cache = None
        _mongo_client = None
        _mongo_unavailable_reason = "Mongo cache disabled via MONGO_CACHE_ENABLED"
        _mongo_init_done = True
        logging.info(_mongo_unavailable_reason)
        return

    try:
        client = MongoClient(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            username=os.getenv("MONGO_USER", "root"),
            password=os.getenv("MONGO_PASSWORD"),
            authSource="admin",
            appname=os.getenv("PREDICTOR_MODELS_MONGO_APP_NAME", os.getenv("MONGO_APP_NAME", "predictor_models")),
            serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "3000")),
            connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "3000")),
            socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "3000")),
        )

        client.admin.command("ping")
        _mongo_client = client

        db = client[os.getenv("MONGO_DATABASE", "predictor")]
        predictor_models_cache = db["predictor_models_cache"]
        _mongo_unavailable_reason = None

        try:
            predictor_models_cache.create_index([("key", ASCENDING)], unique=True, name="key_idx")
            logging.info("Index predictor_models_cache.key_idx created or already exists.")
        except PyMongoError as exc:
            logging.warning("Could not create index (continuing with Mongo anyway): %s", exc)

    except PyMongoError as exc:
        predictor_models_cache = None
        _mongo_client = None
        _mongo_unavailable_reason = f"Mongo unavailable: {exc}"
        logging.warning("Mongo unavailable; falling back to in-memory cache: %s", exc)
    finally:
        _mongo_init_done = True


def _should_retry_init() -> bool:
    if predictor_models_cache is not None:
        return False

    if not _env_bool("MONGO_CACHE_ENABLED", True):
        return False

    retry_seconds = _env_float("MONGO_INIT_RETRY_SECONDS", 5.0)
    if retry_seconds <= 0:
        return True

    if _mongo_last_init_attempt_monotonic is None:
        return True

    return (time.monotonic() - _mongo_last_init_attempt_monotonic) >= retry_seconds


def _ensure_init():
    should_init = not _mongo_init_done
    should_retry = _mongo_init_done and _should_retry_init()

    if not should_init and not should_retry:
        return

    with _mongo_init_lock:
        should_init = not _mongo_init_done
        should_retry = _mongo_init_done and _should_retry_init()
        if should_init or should_retry:
            _init_mongo()


def get_cached_prediction(key: str):
    _ensure_init()
    wildcard_parts = _wildcard_key_parts(key)
    if predictor_models_cache is not None:
        try:
            if wildcard_parts is None:
                doc = predictor_models_cache.find_one({"key": key})
            else:
                doc = _find_one_by_wildcard(predictor_models_cache, *wildcard_parts)
            return doc.get("prediction", None) if doc else None
        except PyMongoError as exc:
            logging.warning("Mongo read failed; using in-memory fallback: %s", exc)
    if wildcard_parts is None:
        return in_memory_cache.get(key)
    cached_prediction = _get_in_memory_by_wildcard(*wildcard_parts)
    if cached_prediction is not None:
        return cached_prediction[1]
    return None


def get_cached_predictions(keys):
    key_list = list(dict.fromkeys(keys))
    if not key_list:
        return {}

    _ensure_init()
    cached_predictions = {}
    wildcard_keys = [
        key for key in key_list if _wildcard_key_parts(key) is not None
    ]
    exact_keys = [
        key for key in key_list if _wildcard_key_parts(key) is None
    ]

    if predictor_models_cache is not None:
        try:
            if exact_keys:
                for doc in predictor_models_cache.find({"key": {"$in": exact_keys}}):
                    key = doc.get("key")
                    if key is not None:
                        cached_predictions[key] = doc.get("prediction")

            _populate_wildcard_hits_from_exact_hits(cached_predictions, wildcard_keys)
            missing_wildcard_keys = [
                key for key in wildcard_keys if key not in cached_predictions
            ]
            wildcard_hits = {}
            for wildcard_key in missing_wildcard_keys:
                wildcard_parts = _wildcard_key_parts(wildcard_key)
                if wildcard_parts is None:
                    continue
                if wildcard_parts in wildcard_hits:
                    cached_predictions[wildcard_key] = wildcard_hits[wildcard_parts][1]
                    continue

                doc = _find_one_by_wildcard(predictor_models_cache, *wildcard_parts)
                if not doc:
                    continue

                actual_key = doc.get("key")
                prediction = doc.get("prediction")
                wildcard_hits[wildcard_parts] = (actual_key, prediction)
                cached_predictions[wildcard_key] = prediction
                if actual_key is not None:
                    cached_predictions.setdefault(actual_key, prediction)
        except PyMongoError as exc:
            logging.warning("Mongo batch read failed; using in-memory fallback: %s", exc)

    for key in exact_keys:
        if key not in cached_predictions and key in in_memory_cache:
            cached_predictions[key] = in_memory_cache[key]
    for wildcard_key in wildcard_keys:
        if wildcard_key in cached_predictions:
            continue
        wildcard_parts = _wildcard_key_parts(wildcard_key)
        if wildcard_parts is None:
            continue
        cached_prediction = _get_in_memory_by_wildcard(*wildcard_parts)
        if cached_prediction is not None:
            cached_predictions[wildcard_key] = cached_prediction[1]

    return cached_predictions


def cache_prediction(key: str, prediction):
    normalized_prediction = _normalize_prediction_for_storage(prediction)

    if _prediction_has_error(normalized_prediction):
        reason = _find_prediction_error_reason(normalized_prediction) or "prediction contains error"
        _warn_mongo_cache_skip(key, reason, fallback="not cached", log_warning=False)
        return

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            predictor_models_cache.replace_one(
                {"key": key}, {"key": key, "prediction": normalized_prediction}, upsert=True
            )
            return
        except (PyMongoError, InvalidDocument) as exc:
            _warn_mongo_cache_skip(key, f"mongo write failed: {exc}", fallback="in_memory_cache")
    else:
        _warn_mongo_cache_skip(
            key,
            _mongo_unavailable_reason or "mongo cache collection is not initialized",
            fallback="in_memory_cache",
        )
    in_memory_cache[key] = normalized_prediction


def cache_predictions(items):
    normalized_items = []

    for key, prediction in items:
        normalized_prediction = _normalize_prediction_for_storage(prediction)
        if _prediction_has_error(normalized_prediction):
            reason = _find_prediction_error_reason(normalized_prediction) or "prediction contains error"
            _warn_mongo_cache_skip(key, reason, fallback="not cached", log_warning=False)
            continue
        normalized_items.append((key, normalized_prediction))

    if not normalized_items:
        return

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            predictor_models_cache.bulk_write(
                [
                    ReplaceOne(
                        {"key": key},
                        {"key": key, "prediction": normalized_prediction},
                        upsert=True,
                    )
                    for key, normalized_prediction in normalized_items
                ],
                ordered=False,
            )
            return
        except (PyMongoError, InvalidDocument) as exc:
            for key, _ in normalized_items:
                _warn_mongo_cache_skip(key, f"mongo batch write failed: {exc}", fallback="in_memory_cache")
    else:
        for key, _ in normalized_items:
            _warn_mongo_cache_skip(
                key,
                _mongo_unavailable_reason or "mongo cache collection is not initialized",
                fallback="in_memory_cache",
            )

    for key, normalized_prediction in normalized_items:
        in_memory_cache[key] = normalized_prediction
