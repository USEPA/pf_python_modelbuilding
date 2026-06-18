import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


class _DummyDeleteOne:
    def __init__(self, delete_filter):
        self.delete_filter = delete_filter


class _DummyUpdateOne:
    def __init__(self, update_filter, update):
        self.update_filter = update_filter
        self.update = update


class _DummyResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests_stub.RequestException(f"HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        return self.payload


class _DummySession:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.posts = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return _DummyResponse({})


class _FakeCursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, *_args, **_kwargs):
        self.docs.sort(key=lambda doc: doc.get("key", ""))
        return self

    def hint(self, *_args, **_kwargs):
        return self

    def limit(self, limit):
        self.docs = self.docs[:limit]
        return self

    def batch_size(self, *_args, **_kwargs):
        return self

    def __iter__(self):
        return iter(self.docs)

    def close(self):
        pass


class _FakeBulkWriteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


class _FakeCollection:
    def __init__(self, docs):
        self.docs = list(docs)
        self.bulk_writes = []

    def find(self, query, projection=None):
        if "key" in query and isinstance(query["key"], dict) and "$in" in query["key"]:
            keys = set(query["key"]["$in"])
            return _FakeCursor([doc for doc in self.docs if doc.get("key") in keys])
        return _FakeCursor(self.docs)

    def bulk_write(self, operations, ordered=False):
        self.bulk_writes.append({"operations": operations, "ordered": ordered})
        delete_filters = [operation.delete_filter for operation in operations]
        deleted_count = 0
        remaining_docs = []
        for doc in self.docs:
            should_delete = False
            for delete_filter in delete_filters:
                if (
                    doc.get("_id") == delete_filter["_id"]
                    and doc.get("key") == delete_filter["key"]
                    and doc.get("prediction", {})
                    .get("chemicalIdentifiers", {})
                    .get("inchiKey")
                    == delete_filter["prediction.chemicalIdentifiers.inchiKey"]
                ):
                    should_delete = True
                    break
            if should_delete:
                deleted_count += 1
            else:
                remaining_docs.append(doc)
        self.docs = remaining_docs
        return _FakeBulkWriteResult(deleted_count)


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    class _DummyRequestException(Exception):
        pass

    class _DummyTimeout(_DummyRequestException):
        pass

    requests_stub.RequestException = _DummyRequestException
    requests_stub.Timeout = _DummyTimeout
    requests_stub.Session = _DummySession
    sys.modules["requests"] = requests_stub
else:
    requests_stub = sys.modules["requests"]

if not hasattr(requests_stub, "RequestException"):
    class _DummyRequestException(Exception):
        pass

    requests_stub.RequestException = _DummyRequestException
if not hasattr(requests_stub, "Timeout"):
    class _DummyTimeout(requests_stub.RequestException):
        pass

    requests_stub.Timeout = _DummyTimeout

if "requests.exceptions" not in sys.modules:
    requests_exceptions_stub = types.ModuleType("requests.exceptions")
    requests_exceptions_stub.RequestException = requests_stub.RequestException
    requests_exceptions_stub.Timeout = requests_stub.Timeout
    sys.modules["requests.exceptions"] = requests_exceptions_stub


if "pymongo" not in sys.modules:
    pymongo_stub = types.ModuleType("pymongo")
    sys.modules["pymongo"] = pymongo_stub
else:
    pymongo_stub = sys.modules["pymongo"]

if not hasattr(pymongo_stub, "DeleteOne"):
    pymongo_stub.DeleteOne = _DummyDeleteOne
if not hasattr(pymongo_stub, "UpdateOne"):
    pymongo_stub.UpdateOne = _DummyUpdateOne
if not hasattr(pymongo_stub, "MongoClient"):
    pymongo_stub.MongoClient = object

if "pymongo.errors" not in sys.modules:
    pymongo_errors_stub = types.ModuleType("pymongo.errors")
    pymongo_errors_stub.PyMongoError = Exception
    sys.modules["pymongo.errors"] = pymongo_errors_stub


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "cleanup_prediction_cache_identity_mismatches.py"
)
SPEC = importlib.util.spec_from_file_location("cleanup_prediction_cache_identity_mismatches", SCRIPT_PATH)
cleanup = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


class TestCleanupPredictionCacheIdentityMismatches(unittest.TestCase):
    def _args(self, temp_dir, **overrides):
        defaults = {
            "key": [],
            "model_id": [],
            "only_without_standardized": False,
            "limit": None,
            "batch_size": 100,
            "repair_mode": "recompute",
            "resolver_url": "https://example.test/api/resolver/lookup",
            "predict_url": "https://example.test/api/predictor_models/predict",
            "prediction_batch_size": 100,
            "prediction_gateway_split_threshold": 10,
            "resolver_timeout": 10,
            "predict_timeout": 10,
            "mongo_cursor_batch_size": 1000,
            "mongo_cursor_retries": 0,
            "mongo_cursor_retry_sleep": 0,
            "progress_every": 0,
            "sample_size": 0,
            "report_jsonl": Path(temp_dir) / "mismatches.jsonl",
            "mismatch_keys_file": Path(temp_dir) / "mismatch_keys.txt",
            "write": True,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    def test_mismatch_from_doc_detects_poisoned_cache_record(self):
        stats = cleanup.CleanupStats()
        doc = {
            "_id": "doc-1",
            "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
            "prediction": {
                "chemicalIdentifiers": {
                    "inchiKey": "FXHOOIRPVKKKFG-UHFFFAOYNA-N",
                    "smiles": "CN(C)C(C)=O",
                    "canonicalSmiles": "CC(=O)N(C)C",
                    "name": "N,N-Dimethylacetamide",
                }
            },
        }

        candidate = cleanup.mismatch_from_doc(
            doc,
            model_ids=set(),
            only_without_standardized=False,
            stats=stats,
        )

        self.assertEqual(candidate.key_inchi_key, "WOZVHXUHUFLZGK-UHFFFAOYSA-N")
        self.assertEqual(candidate.chemical_inchi_key, "FXHOOIRPVKKKFG-UHFFFAOYNA-N")
        self.assertEqual(candidate.model_id, "1066")
        self.assertEqual(candidate.chemical_name, "N,N-Dimethylacetamide")

    def test_scan_mismatches_to_files_writes_keys_after_full_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(temp_dir, write=False, batch_size=1)
            collection = _FakeCollection(
                [
                    {
                        "_id": "doc-1",
                        "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
                        "prediction": {
                            "chemicalIdentifiers": {
                                "inchiKey": "FXHOOIRPVKKKFG-UHFFFAOYNA-N",
                                "smiles": "CN(C)C(C)=O",
                            }
                        },
                    },
                    {
                        "_id": "doc-2",
                        "key": "AAIRZZFLVBAKJP-UHFFFAOYSA-N-1066",
                        "prediction": {
                            "chemicalIdentifiers": {
                                "inchiKey": "AAIRZZFLVBAKJP-UHFFFAOYSA-N",
                            }
                        },
                    },
                ]
            )

            stats = cleanup.CleanupStats()
            file_records = cleanup.scan_mismatches_to_files(collection, args, stats)

            self.assertEqual(file_records, 1)
            self.assertEqual(stats.scanned_documents, 2)
            self.assertEqual(stats.candidates, 1)
            self.assertEqual(
                args.mismatch_keys_file.read_text(encoding="utf-8").splitlines(),
                ["WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066"],
            )
            report_row = args.report_jsonl.read_text(encoding="utf-8").strip()
            self.assertIn('"key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066"', report_row)

    def test_repair_from_keys_file_reads_batches_deletes_and_recomputes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self._args(temp_dir, batch_size=1, prediction_batch_size=100)
            args.mismatch_keys_file.write_text(
                "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066\n"
                "NJPLREDBSJQKKX-UHFFFAOYSA-N-1066\n",
                encoding="utf-8",
            )
            collection = _FakeCollection(
                [
                    {
                        "_id": "doc-1",
                        "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
                        "prediction": {
                            "chemicalIdentifiers": {
                                "inchiKey": "FXHOOIRPVKKKFG-UHFFFAOYNA-N",
                            }
                        },
                    },
                    {
                        "_id": "doc-2",
                        "key": "NJPLREDBSJQKKX-UHFFFAOYSA-N-1066",
                        "prediction": {
                            "chemicalIdentifiers": {
                                "inchiKey": "FXHOOIRPVKKKFG-UHFFFAOYNA-N",
                            }
                        },
                    },
                ]
            )
            session = _DummySession(
                [
                    _DummyResponse(
                        [
                            {
                                "query": "WOZVHXUHUFLZGK-UHFFFAOYSA-N",
                                "chemical": {
                                    "smiles": "COC(C1C=CC(C(OC)=O)=CC=1)=O",
                                    "inchiKey": "WOZVHXUHUFLZGK-UHFFFAOYSA-N",
                                },
                            }
                        ]
                    ),
                    _DummyResponse({"results": [{"chemical": {"smiles": "COC(C1C=CC(C(OC)=O)=CC=1)=O"}}]}),
                    _DummyResponse(
                        [
                            {
                                "query": "NJPLREDBSJQKKX-UHFFFAOYSA-N",
                                "chemical": {
                                    "smiles": "CCOC(C)=O",
                                    "inchiKey": "NJPLREDBSJQKKX-UHFFFAOYSA-N",
                                },
                            }
                        ]
                    ),
                    _DummyResponse({"results": [{"chemical": {"smiles": "CCOC(C)=O"}}]}),
                ]
            )
            stats = cleanup.CleanupStats(candidates=2, planned_deletes=2)

            cleanup.repair_from_keys_file(collection, session, args, stats, file_records=2)

            self.assertEqual(stats.deleted_documents, 2)
            self.assertEqual(stats.recomputed_predictions, 2)
            predict_posts = [
                post
                for post in session.posts
                if post["url"] == "https://example.test/api/predictor_models/predict"
            ]
            self.assertEqual(len(predict_posts), 2)
            self.assertEqual(predict_posts[0]["json"]["smiles"], ["COC(C1C=CC(C(OC)=O)=CC=1)=O"])
            self.assertEqual(predict_posts[1]["json"]["smiles"], ["CCOC(C)=O"])

    def test_mismatch_from_doc_allows_same_connectivity_block(self):
        doc = {
            "_id": "doc-1",
            "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
            "prediction": {
                "chemicalIdentifiers": {
                    "inchiKey": "WOZVHXUHUFLZGK-UHFFFAOYNA-N",
                    "canonicalSmiles": "COC(=O)C1C=CC(=CC=1)C(=O)OC",
                }
            },
        }

        candidate = cleanup.mismatch_from_doc(
            doc,
            model_ids=set(),
            only_without_standardized=False,
        )

        self.assertIsNone(candidate)

    def test_mismatch_from_doc_honors_model_id_filter(self):
        doc = {
            "_id": "doc-1",
            "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
            "prediction": {
                "chemicalIdentifiers": {
                    "inchiKey": "FXHOOIRPVKKKFG-UHFFFAOYNA-N",
                }
            },
        }

        candidate = cleanup.mismatch_from_doc(
            doc,
            model_ids={"1065"},
            only_without_standardized=False,
        )

        self.assertIsNone(candidate)

    def test_build_delete_filter_guards_identity_fields(self):
        candidate = cleanup.MismatchCandidate(
            doc_id="doc-1",
            key="WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
            model_id="1066",
            key_inchi_key="WOZVHXUHUFLZGK-UHFFFAOYSA-N",
            chemical_inchi_key="FXHOOIRPVKKKFG-UHFFFAOYNA-N",
            chemical_inchi_key_raw="fxhooirpvkkkfg-uhfffaoyna-n",
            chemical_smiles="CN(C)C(C)=O",
            chemical_canonical_smiles="CC(=O)N(C)C",
            chemical_name="N,N-Dimethylacetamide",
            has_standardized_chemical=False,
        )

        self.assertEqual(
            cleanup.build_delete_filter(candidate),
            {
                "_id": "doc-1",
                "key": "WOZVHXUHUFLZGK-UHFFFAOYSA-N-1066",
                "prediction.chemicalIdentifiers.inchiKey": "fxhooirpvkkkfg-uhfffaoyna-n",
            },
        )

    def test_build_inchi_key_resolver_payload_uses_inchi_key_type(self):
        payload = cleanup.build_inchi_key_resolver_payload(["WOZVHXUHUFLZGK-UHFFFAOYSA-N"])

        self.assertEqual(payload["ids"], ["WOZVHXUHUFLZGK-UHFFFAOYSA-N"])
        self.assertEqual(payload["idsType"], "InChIKey")
        self.assertEqual(payload["fuzzy"], "Not")
        self.assertFalse(payload["mol"])

    def test_parse_inchi_key_resolver_payload_maps_query_to_chemical(self):
        payload = [
            {
                "query": "WOZVHXUHUFLZGK-UHFFFAOYSA-N",
                "chemical": {
                    "smiles": "COC(C1C=CC(C(OC)=O)=CC=1)=O",
                    "inchiKey": "WOZVHXUHUFLZGK-UHFFFAOYNA-N",
                },
            }
        ]

        resolved = cleanup.parse_inchi_key_resolver_payload(
            payload,
            ["WOZVHXUHUFLZGK-UHFFFAOYSA-N"],
        )

        self.assertEqual(
            resolved["WOZVHXUHUFLZGK-UHFFFAOYSA-N"]["smiles"],
            "COC(C1C=CC(C(OC)=O)=CC=1)=O",
        )

    def test_predict_smiles_with_fallback_posts_legacy_batch_shape(self):
        stats = cleanup.CleanupStats()
        session = _DummySession(
            [
                _DummyResponse(
                    {
                        "results": [
                            {"chemical": {"smiles": "CCC"}},
                            {"chemical": {"smiles": "CCO"}},
                        ]
                    }
                )
            ]
        )

        success_count = cleanup.predict_smiles_with_fallback(
            session,
            "https://example.test/api/predictor_models/predict",
            "1066",
            ["CCC", "CCO"],
            10,
            10,
            stats,
        )

        self.assertEqual(success_count, 2)
        self.assertEqual(stats.prediction_batches, 1)
        self.assertEqual(stats.recomputed_predictions, 2)
        self.assertEqual(
            session.posts[0]["json"],
            {"model_id": 1066, "smiles": ["CCC", "CCO"]},
        )

    def test_gateway_timeout_failure_does_not_split_below_threshold(self):
        stats = cleanup.CleanupStats()
        session = _DummySession([_DummyResponse({}, status_code=504)])

        success_count = cleanup.predict_smiles_with_fallback(
            session,
            "https://example.test/api/predictor_models/predict",
            "1068",
            ["CCC", "CCO"],
            10,
            10,
            stats,
        )

        self.assertEqual(success_count, 0)
        self.assertEqual(len(session.posts), 1)
        self.assertEqual(stats.prediction_failures, 2)

    def test_gateway_timeout_failure_splits_above_threshold(self):
        stats = cleanup.CleanupStats()
        session = _DummySession(
            [
                _DummyResponse({}, status_code=504),
                _DummyResponse({"results": [{"chemical": {"smiles": "CCC"}}]}),
                _DummyResponse({"results": [{"chemical": {"smiles": "CCO"}}]}),
            ]
        )

        success_count = cleanup.predict_smiles_with_fallback(
            session,
            "https://example.test/api/predictor_models/predict",
            "1068",
            ["CCC", "CCO"],
            10,
            1,
            stats,
        )

        self.assertEqual(success_count, 2)
        self.assertEqual(len(session.posts), 3)
        self.assertEqual(stats.recomputed_predictions, 2)


if __name__ == "__main__":
    unittest.main()
