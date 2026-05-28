import importlib.util
import sys
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
            raise Exception(f"HTTP {self.status_code}")

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


if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.RequestException = Exception
    requests_stub.Session = _DummySession
    sys.modules["requests"] = requests_stub

if "requests.exceptions" not in sys.modules:
    requests_exceptions_stub = types.ModuleType("requests.exceptions")
    requests_exceptions_stub.RequestException = Exception
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
            stats,
        )

        self.assertEqual(success_count, 2)
        self.assertEqual(stats.prediction_batches, 1)
        self.assertEqual(stats.recomputed_predictions, 2)
        self.assertEqual(
            session.posts[0]["json"],
            {"model_id": 1066, "smiles": ["CCC", "CCO"]},
        )


if __name__ == "__main__":
    unittest.main()
