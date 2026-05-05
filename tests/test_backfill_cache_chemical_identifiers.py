import importlib.util
import sys
import types
import unittest
from pathlib import Path


class _DummyUpdateOne:
    def __init__(self, update_filter, update):
        self.update_filter = update_filter
        self.update = update


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
