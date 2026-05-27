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


if __name__ == "__main__":
    unittest.main()
