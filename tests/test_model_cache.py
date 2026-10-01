import unittest
from unittest.mock import patch

from db import model_cache


class _FakeReplaceOne:
    def __init__(self, filter_doc, replacement, upsert=False):
        self.filter_doc = filter_doc
        self.replacement = replacement
        self.upsert = upsert


class _FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, field_name, _direction):
        self.rows.sort(key=lambda row: row[field_name])
        return self

    def __iter__(self):
        return iter(self.rows)


class _FakeChunkCollection:
    def __init__(self):
        self.rows = {}

    def delete_many(self, filter_doc):
        owner_key = filter_doc["owner_key"]
        min_chunk_index = filter_doc["chunk_index"]["$gte"]
        for key in list(self.rows):
            if key[0] == owner_key and key[1] >= min_chunk_index:
                del self.rows[key]

    def bulk_write(self, operations, ordered=True):
        for operation in operations:
            owner_key = operation.replacement["owner_key"]
            chunk_index = operation.replacement["chunk_index"]
            self.rows[(owner_key, chunk_index)] = dict(operation.replacement)

    def find(self, filter_doc, _projection):
        owner_key = filter_doc["owner_key"]
        return _FakeCursor(
            row
            for (row_owner_key, _), row in self.rows.items()
            if row_owner_key == owner_key
        )


class _FakeDetailsCollection:
    def __init__(self):
        self.rows = {}
        self.indexes = []

    def create_index(self, keys, unique=False, name=None):
        self.indexes.append((keys, unique, name))

    def replace_one(self, filter_doc, replacement, upsert=False):
        self.rows[filter_doc["key"]] = dict(replacement)

    def find_one(self, filter_doc, projection=None):
        row = self.rows.get(filter_doc["key"])
        if row is None or row.get("schema_version") != filter_doc.get("schema_version"):
            return None
        if projection:
            return {key: row[key] for key in projection if key in row and key != "_id"}
        return dict(row)


class ModelCacheTests(unittest.TestCase):
    def test_blob_round_trip_uses_chunks(self):
        chunk_collection = _FakeChunkCollection()
        payload = b"abcdef" * 100

        with patch.object(model_cache, "ReplaceOne", _FakeReplaceOne), patch.object(
            model_cache,
            "_chunk_size_bytes",
            return_value=64,
        ), patch.object(model_cache.zlib, "compress", side_effect=lambda value: value), patch.object(
            model_cache.zlib,
            "decompress",
            side_effect=lambda value: value,
        ):
            metadata = model_cache._write_blob(chunk_collection, "model:1065", payload)
            restored = model_cache._read_blob(
                chunk_collection,
                "model:1065",
                metadata["chunk_count"],
            )

        self.assertEqual(restored, payload)
        self.assertGreater(metadata["chunk_count"], 1)

    def test_model_details_round_trip_uses_lightweight_collection(self):
        details_collection = _FakeDetailsCollection()

        class _FakeClient:
            def close(self):
                pass

        with patch.object(model_cache, "connect_mongo", return_value=(_FakeClient(), None, None, None)), patch.object(
            model_cache,
            "_get_model_details_collection",
            return_value=details_collection,
        ):
            payload = {
                "modelId": 1065,
                "propertyName": "Water solubility",
                "embedding": ("a", "b"),
                "performance": {"train": {"R2": float("nan")}},
            }

            model_cache.write_model_details(1065, "/api/predictor_models/model/file/", payload)
            restored = model_cache.read_model_details(1065, "/api/predictor_models/model/file/")

        self.assertEqual(restored["modelId"], 1065)
        self.assertEqual(restored["embedding"], ["a", "b"])
        self.assertIsNone(restored["performance"]["train"]["R2"])


if __name__ == "__main__":
    unittest.main()
