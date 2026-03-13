import logging
import os
import json

from pymongo import MongoClient, ASCENDING
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


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    logging.warning(f"Invalid boolean value for {name}={raw!r}; using default={default}")
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
    finally:
        _mongo_init_done = True


def _ensure_init():
    # Lazy initialize on first use to avoid network I/O at import, but safe if called multiple times
    global _mongo_init_done
    
    if not _mongo_init_done:
        _init_mongo()


def get_cached_prediction(key: str):
    _ensure_init()
    if predictor_models_cache is not None:
        try:
            doc = predictor_models_cache.find_one({"key": key})
            return doc.get("prediction", None) if doc else None
        except PyMongoError as e:
            logging.warning(f"Mongo read failed; using in-memory fallback: {e}")
    return in_memory_cache.get(key)


def cache_prediction(key: str, prediction):
    if _prediction_has_error(prediction):
        logging.debug("Skipping cache for key=%r: prediction contains error", key)
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
