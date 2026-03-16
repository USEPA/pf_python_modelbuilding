import json
import logging
import os

from bson.errors import InvalidDocument
from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.errors import PyMongoError


predictor_models_cache = None
in_memory_cache = {}
_mongo_init_done = False


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


def _init_mongo():
    global predictor_models_cache, _mongo_init_done

    if not _env_bool("MONGO_CACHE_ENABLED", True):
        predictor_models_cache = None
        _mongo_init_done = True
        logging.info("Mongo cache disabled via MONGO_CACHE_ENABLED")
        return

    try:
        client = MongoClient(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            username=os.getenv("MONGO_USER", "root"),
            password=os.getenv("MONGO_PASSWORD"),
            authSource="admin",
            serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "500")),
            connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "500")),
            socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "500")),
        )

        client.admin.command("ping")

        db = client[os.getenv("MONGO_DATABASE", "predictor")]
        predictor_models_cache = db["predictor_models_cache"]

        try:
            predictor_models_cache.create_index([("key", ASCENDING)], unique=True, name="key_idx")
            logging.info("Index predictor_models_cache.key_idx created or already exists.")
        except PyMongoError as exc:
            logging.warning("Could not create index (continuing with Mongo anyway): %s", exc)

    except PyMongoError as exc:
        predictor_models_cache = None
        logging.warning("Mongo unavailable; falling back to in-memory cache: %s", exc)
    finally:
        _mongo_init_done = True


def _ensure_init():
    global _mongo_init_done

    if not _mongo_init_done:
        _init_mongo()


def get_cached_prediction(key: str):
    _ensure_init()
    if predictor_models_cache is not None:
        try:
            doc = predictor_models_cache.find_one({"key": key})
            return doc.get("prediction", None) if doc else None
        except PyMongoError as exc:
            logging.warning("Mongo read failed; using in-memory fallback: %s", exc)
    return in_memory_cache.get(key)


def get_cached_predictions(keys):
    key_list = list(dict.fromkeys(keys))
    if not key_list:
        return {}

    _ensure_init()
    cached_predictions = {}

    if predictor_models_cache is not None:
        try:
            for doc in predictor_models_cache.find({"key": {"$in": key_list}}):
                key = doc.get("key")
                if key is not None:
                    cached_predictions[key] = doc.get("prediction")
        except PyMongoError as exc:
            logging.warning("Mongo batch read failed; using in-memory fallback: %s", exc)

    for key in key_list:
        if key not in cached_predictions and key in in_memory_cache:
            cached_predictions[key] = in_memory_cache[key]

    return cached_predictions


def cache_prediction(key: str, prediction):
    normalized_prediction = _normalize_prediction_for_storage(prediction)

    if _prediction_has_error(normalized_prediction):
        logging.debug("Skipping cache for key=%r: prediction contains error", key)
        return

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            predictor_models_cache.replace_one(
                {"key": key}, {"key": key, "prediction": normalized_prediction}, upsert=True
            )
            return
        except (PyMongoError, InvalidDocument) as exc:
            logging.warning("Mongo write failed; caching in memory: %s", exc)
    in_memory_cache[key] = normalized_prediction


def cache_predictions(items):
    normalized_items = []

    for key, prediction in items:
        normalized_prediction = _normalize_prediction_for_storage(prediction)
        if _prediction_has_error(normalized_prediction):
            logging.debug("Skipping cache for key=%r: prediction contains error", key)
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
            logging.warning("Mongo batch write failed; caching in memory: %s", exc)

    for key, normalized_prediction in normalized_items:
        in_memory_cache[key] = normalized_prediction
