import unittest

from util.prediction_cache_key_utils import (
    build_prediction_cache_key,
    ensure_chemical_inchi_key,
    normalize_inchi_key,
)


class TestPredictionCacheKeyUtils(unittest.TestCase):
    def test_normalize_inchi_key_uppercases_valid_value(self):
        self.assertEqual(
            normalize_inchi_key("lfqscwfljhtthz-uhfffaoysa-n"),
            "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )

    def test_normalize_inchi_key_rejects_invalid_value(self):
        self.assertIsNone(normalize_inchi_key("not-an-inchikey"))
        self.assertIsNone(normalize_inchi_key("N/A"))
        self.assertIsNone(normalize_inchi_key(None))

    def test_ensure_chemical_inchi_key_prefers_canonical_smiles(self):
        def smiles_to_inchi_key(smiles):
            return {
                "canonical": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                "raw": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
            }.get(smiles)

        chemical = ensure_chemical_inchi_key(
            {"canonicalSmiles": "canonical", "smiles": "raw"},
            smiles_to_inchi_key,
        )

        self.assertEqual(chemical["inchiKey"], "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")

    def test_ensure_chemical_inchi_key_normalizes_existing_value(self):
        chemical = ensure_chemical_inchi_key(
            {"inchiKey": "lfqscwfljhtthz-uhfffaoysa-n"},
            lambda smiles: None,
        )

        self.assertEqual(chemical["inchiKey"], "LFQSCWFLJHTTHZ-UHFFFAOYSA-N")

    def test_build_prediction_cache_key_returns_none_without_inchi_key(self):
        key = build_prediction_cache_key(
            1065,
            lambda smiles: None,
            smiles="CCO",
        )

        self.assertIsNone(key)

    def test_build_prediction_cache_key_uses_inchi_key_model_id_format(self):
        key = build_prediction_cache_key(
            1065,
            lambda smiles: "lfqscwfljhtthz-uhfffaoysa-n",
            smiles="CCO",
        )

        self.assertEqual(key, "LFQSCWFLJHTTHZ-UHFFFAOYSA-N-1065")


if __name__ == "__main__":
    unittest.main()
