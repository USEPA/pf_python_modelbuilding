import os
import sys
import types
import unittest
from unittest.mock import patch


bson_module = types.ModuleType("bson")
bson_errors_module = types.ModuleType("bson.errors")
bson_errors_module.InvalidDocument = Exception
bson_module.errors = bson_errors_module

pymongo_module = types.ModuleType("pymongo")
pymongo_errors_module = types.ModuleType("pymongo.errors")
pymongo_module.ASCENDING = 1
pymongo_module.MongoClient = object
pymongo_module.ReplaceOne = object
pymongo_errors_module.PyMongoError = Exception

sys.modules.setdefault("bson", bson_module)
sys.modules.setdefault("bson.errors", bson_errors_module)
sys.modules.setdefault("pymongo", pymongo_module)
sys.modules.setdefault("pymongo.errors", pymongo_errors_module)

from db import mongo_cache


class TestMongoCacheInitRetry(unittest.TestCase):
    def setUp(self):
        self.original_state = {
            "predictor_models_cache": mongo_cache.predictor_models_cache,
            "in_memory_cache": dict(mongo_cache.in_memory_cache),
            "_mongo_init_done": mongo_cache._mongo_init_done,
            "_mongo_unavailable_reason": mongo_cache._mongo_unavailable_reason,
            "_mongo_last_init_attempt_monotonic": mongo_cache._mongo_last_init_attempt_monotonic,
        }
        mongo_cache.predictor_models_cache = None
        mongo_cache.in_memory_cache = {}
        mongo_cache._mongo_init_done = False
        mongo_cache._mongo_unavailable_reason = None
        mongo_cache._mongo_last_init_attempt_monotonic = None

    def tearDown(self):
        mongo_cache.predictor_models_cache = self.original_state["predictor_models_cache"]
        mongo_cache.in_memory_cache = self.original_state["in_memory_cache"]
        mongo_cache._mongo_init_done = self.original_state["_mongo_init_done"]
        mongo_cache._mongo_unavailable_reason = self.original_state["_mongo_unavailable_reason"]
        mongo_cache._mongo_last_init_attempt_monotonic = self.original_state["_mongo_last_init_attempt_monotonic"]

    def test_ensure_init_retries_after_cooldown(self):
        mongo_cache._mongo_init_done = True
        mongo_cache._mongo_unavailable_reason = "Mongo unavailable"
        mongo_cache._mongo_last_init_attempt_monotonic = 10.0

        with patch.dict(os.environ, {"MONGO_CACHE_ENABLED": "true", "MONGO_INIT_RETRY_SECONDS": "5"}, clear=False):
            with patch("db.mongo_cache.time.monotonic", return_value=16.0):
                with patch("db.mongo_cache._init_mongo") as init_mongo:
                    mongo_cache._ensure_init()

        init_mongo.assert_called_once_with()

    def test_ensure_init_skips_retry_before_cooldown(self):
        mongo_cache._mongo_init_done = True
        mongo_cache._mongo_unavailable_reason = "Mongo unavailable"
        mongo_cache._mongo_last_init_attempt_monotonic = 10.0

        with patch.dict(os.environ, {"MONGO_CACHE_ENABLED": "true", "MONGO_INIT_RETRY_SECONDS": "5"}, clear=False):
            with patch("db.mongo_cache.time.monotonic", return_value=12.0):
                with patch("db.mongo_cache._init_mongo") as init_mongo:
                    mongo_cache._ensure_init()

        init_mongo.assert_not_called()

    def test_ensure_init_does_not_retry_when_disabled(self):
        mongo_cache._mongo_init_done = True
        mongo_cache._mongo_unavailable_reason = "Mongo cache disabled via MONGO_CACHE_ENABLED"
        mongo_cache._mongo_last_init_attempt_monotonic = 10.0

        with patch.dict(os.environ, {"MONGO_CACHE_ENABLED": "false", "MONGO_INIT_RETRY_SECONDS": "0"}, clear=False):
            with patch("db.mongo_cache._init_mongo") as init_mongo:
                mongo_cache._ensure_init()

        init_mongo.assert_not_called()

    def test_get_cached_predictions_matches_trailing_star_from_memory(self):
        mongo_cache._mongo_init_done = True
        mongo_cache.in_memory_cache = {
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N-1065": {"value": 1},
        }

        cached_predictions = mongo_cache.get_cached_predictions(["LFQSCWFLJHTTHZ*"])

        self.assertEqual(
            cached_predictions,
            {"LFQSCWFLJHTTHZ*": {"value": 1}},
        )

    def test_get_cached_predictions_matches_wildcard_suffix_from_memory(self):
        mongo_cache._mongo_init_done = True
        mongo_cache.in_memory_cache = {
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N-1065": {"value": 1},
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N-1066": {"value": 2},
        }

        cached_predictions = mongo_cache.get_cached_predictions(["LFQSCWFLJHTTHZ*-1066"])

        self.assertEqual(
            cached_predictions,
            {"LFQSCWFLJHTTHZ*-1066": {"value": 2}},
        )


if __name__ == "__main__":
    unittest.main()
