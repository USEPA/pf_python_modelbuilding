import logging
import os

from pymongo import MongoClient, ASCENDING
from pymongo.errors import OperationFailure

predictor_models_cache = None
in_memory_cache = {}

try:
    mongo_client = MongoClient(
        host=os.getenv("MONGO_HOST", "localhost"),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_USER", "root"),
        password=os.getenv("MONGO_PASSWORD"),
        authSource="admin"  # Database to authenticate against
    )

    predictor_db = mongo_client[os.getenv("MONGO_DATABASE", "predictor")]
    predictor_models_cache = predictor_db["predictor_models_cache"]
except Exception as e:
    logging.error(f"Falling back to in-memory cache as could not connect to MongoDB: {e}")

if predictor_models_cache is not None:
    try:
        predictor_models_cache.create_index([('key', ASCENDING)], unique=True, name='key_idx')
        logging.info("Index predictor_models_cache.key_idx created successfully or already exists.")
    except OperationFailure as e:
        logging.error(f"Could not create index: {e}")


def get_cached_prediction(key: str):
    if predictor_models_cache is not None:
        pred = predictor_models_cache.find_one({"key": key})
        return pred.get("prediction", None) if pred else None
    elif key in in_memory_cache:
        return in_memory_cache[key]
    else:
        return None


def cache_prediction(key: str, prediction):
    if predictor_models_cache is not None:
        predictor_models_cache.insert_one({"key": key, "prediction": prediction})
    else:
        in_memory_cache[key] = prediction
