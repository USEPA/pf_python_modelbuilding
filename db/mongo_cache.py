import json
import logging
import os
import threading

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


predictor_models_cache = None
in_memory_cache = {}
_FAILED_STANDARDIZATION_LOG_LOCK = threading.Lock()


def _cache_disabled() -> bool:
    return os.getenv("DISABLE_PREDICTION_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}


def _init_mongo():

    global predictor_models_cache
    
    try:
        client = MongoClient(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            username=os.getenv("MONGO_USER", "root"),
            password=os.getenv("MONGO_PASSWORD"),
            authSource="admin",
            # Keep these short so app startup isn’t delayed if Mongo is down
            serverSelectionTimeoutMS=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "500")),
            connectTimeoutMS=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "500")),
            socketTimeoutMS=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "500")),
        )

        # Verify connection
        client.admin.command("ping")

        db = client[os.getenv("MONGO_DATABASE", "predictor")]
        predictor_models_cache = db["predictor_models_cache"]

        try:
            predictor_models_cache.create_index([('key', ASCENDING)], unique=True, name='key_idx')
            logging.info("Index predictor_models_cache.key_idx created or already exists.")
        except PyMongoError as e:
            logging.warning(f"Could not create index (continuing with Mongo anyway): {e}")

    except PyMongoError as e:
        # Any connection issue: fall back to in-memory
        predictor_models_cache = None
        logging.warning(f"Mongo unavailable; falling back to in-memory cache: {e}")


def _ensure_init():
    # Lazy initialize on first use to avoid network I/O at import, but safe if called multiple times
    global predictor_models_cache

    if _cache_disabled():
        return
    
    if predictor_models_cache is None:
        _init_mongo()


def get_cached_prediction(key: str):
    if _cache_disabled():
        return None

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            doc = predictor_models_cache.find_one({"key": key})
            return doc.get("prediction", None) if doc else None
        except PyMongoError as e:
            logging.warning(f"Mongo read failed; using in-memory fallback: {e}")
    return in_memory_cache.get(key)


def delete_cached_prediction(key: str):
    if _cache_disabled():
        return

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            predictor_models_cache.delete_one({"key": key})
        except PyMongoError as e:
            logging.warning(f"Mongo delete failed for key {key}: {e}")

    in_memory_cache.pop(key, None)


def _is_failed_standardization_error(value) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower().endswith("failed standardization")


def _prediction_to_obj(prediction):
    if not isinstance(prediction, str):
        return prediction

    try:
        return json.loads(prediction)
    except ValueError:
        return prediction


def _iter_prediction_errors(prediction):
    prediction_obj = _prediction_to_obj(prediction)

    if isinstance(prediction_obj, str):
        yield prediction_obj
        return

    if isinstance(prediction_obj, dict):
        model_results = prediction_obj.get("modelResults")

        if isinstance(model_results, dict):
            yield model_results.get("predictionError")
            return

        if isinstance(model_results, list):
            for item in model_results:
                if isinstance(item, dict):
                    yield item.get("predictionError")
            return

    if isinstance(prediction_obj, list):
        for item in prediction_obj:
            if isinstance(item, dict):
                model_results = item.get("modelResults")
                if isinstance(model_results, dict):
                    yield model_results.get("predictionError")


def get_failed_standardization_error(prediction):
    for error in _iter_prediction_errors(prediction):
        if _is_failed_standardization_error(error):
            return error
    return None


def has_failed_standardization(prediction) -> bool:
    return get_failed_standardization_error(prediction) is not None


def log_failed_standardization_key(key: str, prediction=None):
    log_path = os.getenv("FAILED_STANDARDIZATION_LOG_FILE", "failed_standardization_keys.txt")
    error = get_failed_standardization_error(prediction) or "failed standardization"

    log_dir = os.path.dirname(log_path)
    try:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with _FAILED_STANDARDIZATION_LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{key}\t{error}\n")
    except OSError:
        logging.exception("Failed to append key %s to failed standardization log", key)


def cache_prediction(key: str, prediction):
    if _cache_disabled():
        return

    if has_failed_standardization(prediction):
        logging.info("Skipping cache for key=%s due to failed standardization", key)
        return

    _ensure_init()
    if predictor_models_cache is not None:
        try:
            # Upsert avoids duplicate key errors when index is unique
            predictor_models_cache.replace_one(
                {"key": key}, {"key": key, "prediction": prediction}, upsert=True
            )
            return
        except PyMongoError as e:
            logging.warning(f"Mongo write failed; caching in memory: {e}")
    in_memory_cache[key] = prediction
