import os
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
