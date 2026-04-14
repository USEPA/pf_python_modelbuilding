import logging
import os

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError


predictor_models_cache = None
in_memory_cache = {}


def _init_mongo():

    global predictor_models_cache
    
    try:
        client = MongoClient(
            host=os.getenv("MONGO_HOST", "localhost"),
            port=int(os.getenv("MONGO_PORT", "27017")),
            username=os.getenv("MONGO_USER"),
            password=os.getenv("MONGO_PASSWORD"),
            authSource="admin",
            appname=os.getenv("WEBTEST_V2_MONGO_APP_NAME", os.getenv("MONGO_APP_NAME", "WebTEST v2")),
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
    
    if predictor_models_cache is None:
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