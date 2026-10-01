import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


class _DummyUpdateOne:
    def __init__(self, update_filter, update):
        self.update_filter = update_filter
        self.update = update


class _FakeCursor:
    def __init__(self, docs):
        self.docs = list(docs)
        self.closed = False
        self._limit = None

    def sort(self, field, direction):
        self.docs.sort(key=lambda doc: doc[field], reverse=direction < 0)
        return self

    def limit(self, value):
        self._limit = value
        return self

    def batch_size(self, _value):
        return self

    def __iter__(self):
        docs = self.docs
        if self._limit is not None:
            docs = docs[: self._limit]
        return iter(docs)

    def close(self):
        self.closed = True


class _FakeCollection:
    def __init__(self, docs):
        self.docs = list(docs)
        self.cursors = []
        self.find_queries = []

    def find(self, query, projection=None):
        self.find_queries.append(query)
        cursor = _FakeCursor([doc for doc in self.docs if self._matches(doc, query)])
        self.cursors.append(cursor)
        return cursor

    def _matches(self, doc, query):
        if not query:
            return True
        if "$and" in query:
            return all(self._matches(doc, clause) for clause in query["$and"])
        id_clause = query.get("_id")
        if isinstance(id_clause, dict) and "$gt" in id_clause:
            return doc["_id"] > id_clause["$gt"]
        return True


class _CapturingMongoClient:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Session = object
    sys.modules["requests"] = requests_stub

if "pymongo" not in sys.modules:
    pymongo_stub = types.ModuleType("pymongo")
    pymongo_stub.MongoClient = object
    pymongo_stub.UpdateOne = _DummyUpdateOne
    sys.modules["pymongo"] = pymongo_stub

if "pymongo.errors" not in sys.modules:
    pymongo_errors_stub = types.ModuleType("pymongo.errors")
    pymongo_errors_stub.PyMongoError = Exception
    sys.modules["pymongo.errors"] = pymongo_errors_stub


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backfill_cache_chemical_identifiers.py"
)
SPEC = importlib.util.spec_from_file_location("backfill_cache_chemical_identifiers", SCRIPT_PATH)
backfill = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backfill
SPEC.loader.exec_module(backfill)


class TestBackfillCacheChemicalIdentifiers(unittest.TestCase):
    def _candidate_doc(self, doc_id, smiles="CCC"):
        return {
            "_id": doc_id,
            "key": f"model:{doc_id}",
            "prediction": {
                "chemicalIdentifiers": {
                    "smiles": smiles,
                    "cid": None,
                    "sid": None,
                    "casrn": None,
                    "name": None,
                    "inchi": None,
                    "inchiKey": None,
                }
            },
        }

    def test_build_resolver_payload_uses_batch_lookup_shape(self):
        payload = backfill.build_resolver_payload(["CCC", "CCO"])

        self.assertEqual(payload["ids"], ["CCC", "CCO"])
        self.assertEqual(payload["idsType"], "SMILES")
        self.assertEqual(payload["fuzzy"], "Not")
        self.assertFalse(payload["mol"])

    def test_candidate_from_doc_extracts_smiles_and_missing_fields(self):
        doc = {
            "_id": "doc-1",
            "key": "model:CCC",
            "prediction": {
                "chemicalIdentifiers": {
                    "smiles": "CCC",
                    "cid": None,
                    "sid": None,
                    "casrn": None,
                    "name": None,
                    "inchi": None,
                    "inchiKey": None,
                }
            },
        }

        candidate = backfill._candidate_from_doc(doc)

        self.assertEqual(candidate.doc_id, "doc-1")
        self.assertEqual(candidate.key, "model:CCC")
        self.assertEqual(candidate.smiles, "CCC")
        self.assertEqual(candidate.missing_fields, backfill.IDENTIFIER_FIELDS)

    def test_candidate_from_doc_honors_all_null_mode(self):
        doc = {
            "_id": "doc-1",
            "prediction": {
                "chemicalIdentifiers": {
                    "smiles": "CCC",
                    "cid": "DTXCID006386",
                    "sid": None,
                    "casrn": None,
                    "name": None,
                    "inchi": None,
                    "inchiKey": None,
                }
            },
        }

        self.assertIsNone(backfill._candidate_from_doc(doc, "all-null"))

    def test_candidate_from_doc_honors_any_null_mode(self):
        doc = {
            "_id": "doc-1",
            "prediction": {
                "chemicalIdentifiers": {
                    "smiles": "CCC",
                    "cid": "DTXCID006386",
                    "sid": None,
                    "casrn": "74-98-6",
                    "name": "Propane",
                    "inchi": "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3",
                    "inchiKey": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
                }
            },
        }

        candidate = backfill._candidate_from_doc(doc, "any-null")

        self.assertEqual(candidate.missing_fields, ("sid",))

    def test_build_scan_query_defaults_to_lightweight_client_scan(self):
        self.assertEqual(backfill.build_scan_query("client", "all-null"), {})
        self.assertIn("$and", backfill.build_scan_query("server", "all-null"))

    def test_build_resume_scan_query_adds_id_resume_clause(self):
        self.assertEqual(backfill.build_resume_scan_query({}, 10), {"_id": {"$gt": 10}})
        self.assertEqual(
            backfill.build_resume_scan_query({"foo": "bar"}, 10),
            {"$and": [{"foo": "bar"}, {"_id": {"$gt": 10}}]},
        )

    def test_build_mongo_client_leaves_socket_timeout_unset_by_default(self):
        original_mongo_client = backfill.MongoClient
        original_env_value = os.environ.pop("BACKFILL_MONGO_SOCKET_TIMEOUT_MS", None)
        backfill.MongoClient = _CapturingMongoClient
        try:
            args = types.SimpleNamespace(
                mongo_uri="mongodb://example.test",
                mongo_socket_timeout_ms=None,
            )

            client = backfill.build_mongo_client(args)

            self.assertNotIn("socketTimeoutMS", client.kwargs)
        finally:
            backfill.MongoClient = original_mongo_client
            if original_env_value is not None:
                os.environ["BACKFILL_MONGO_SOCKET_TIMEOUT_MS"] = original_env_value

    def test_build_mongo_client_allows_explicit_socket_timeout(self):
        original_mongo_client = backfill.MongoClient
        backfill.MongoClient = _CapturingMongoClient
        try:
            args = types.SimpleNamespace(
                mongo_uri="mongodb://example.test",
                mongo_socket_timeout_ms=600000,
            )

            client = backfill.build_mongo_client(args)

            self.assertEqual(client.kwargs["socketTimeoutMS"], 600000)
        finally:
            backfill.MongoClient = original_mongo_client

    def test_iter_candidate_batches_closes_scan_cursor_before_yielding(self):
        collection = _FakeCollection(
            [
                self._candidate_doc(1, "CCC"),
                self._candidate_doc(2, "CCO"),
                self._candidate_doc(3, "CCN"),
            ]
        )
        stats = backfill.BackfillStats()

        iterator = backfill.iter_candidate_batches(
            collection,
            batch_size=2,
            limit=None,
            match_mode="all-null",
            query_mode="client",
            cursor_batch_size=3,
            progress_every=0,
            cursor_retries=0,
            cursor_retry_sleep=0,
            stats=stats,
        )
        first_batch = next(iterator)

        self.assertEqual([candidate.doc_id for candidate in first_batch], [1, 2])
        self.assertTrue(collection.cursors[0].closed)

        second_batch = next(iterator)

        self.assertEqual([candidate.doc_id for candidate in second_batch], [3])

    def test_iter_candidate_batches_pages_by_id(self):
        collection = _FakeCollection([self._candidate_doc(doc_id) for doc_id in range(1, 6)])
        stats = backfill.BackfillStats()

        batches = list(
            backfill.iter_candidate_batches(
                collection,
                batch_size=10,
                limit=None,
                match_mode="all-null",
                query_mode="client",
                cursor_batch_size=2,
                progress_every=0,
                cursor_retries=0,
                cursor_retry_sleep=0,
                stats=stats,
            )
        )

        self.assertEqual(
            [[candidate.doc_id for candidate in batch] for batch in batches],
            [[1, 2, 3, 4, 5]],
        )
        self.assertEqual(
            collection.find_queries,
            [{}, {"_id": {"$gt": 2}}, {"_id": {"$gt": 4}}],
        )

    def test_parse_resolver_payload_accepts_positional_chemical_list(self):
        payload = {
            "chemicals": [
                {
                    "chemical": {
                        "cid": "DTXCID006386",
                        "sid": "DTXSID5026386",
                        "casrn": "74-98-6",
                        "name": "Propane",
                        "smiles": "CCC",
                    }
                }
            ]
        }

        resolved = backfill.parse_resolver_payload(payload, ["CCC"])

        self.assertEqual(resolved["CCC"]["sid"], "DTXSID5026386")

    def test_build_update_fields_only_fills_requested_missing_fields(self):
        resolved_chemical = {
            "cid": "DTXCID006386",
            "sid": "DTXSID5026386",
            "name": "N/A",
            "inchiKey": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
        }

        update_fields = backfill.build_update_fields(
            resolved_chemical,
            ("cid", "name", "inchiKey"),
        )

        self.assertEqual(
            update_fields,
            {
                "prediction.chemicalIdentifiers.cid": "DTXCID006386",
                "prediction.chemicalIdentifiers.inchiKey": "ATUOYWHBWRKTHZ-UHFFFAOYSA-N",
            },
        )

    def test_build_update_filter_guards_against_concurrent_non_null_changes(self):
        update_filter = backfill.build_update_filter("doc-1", ("cid", "sid"))

        self.assertEqual(update_filter["_id"], "doc-1")
        self.assertIn(
            {
                "$or": [
                    {"prediction.chemicalIdentifiers.cid": None},
                    {"prediction.chemicalIdentifiers.cid": ""},
                    {"prediction.chemicalIdentifiers.cid": "N/A"},
                ]
            },
            update_filter["$and"],
        )
        self.assertIn(
            {
                "$or": [
                    {"prediction.chemicalIdentifiers.sid": None},
                    {"prediction.chemicalIdentifiers.sid": ""},
                    {"prediction.chemicalIdentifiers.sid": "N/A"},
                ]
            },
            update_filter["$and"],
        )


if __name__ == "__main__":
    unittest.main()
