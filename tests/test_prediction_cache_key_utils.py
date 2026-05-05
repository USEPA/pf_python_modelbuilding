import unittest

from util.prediction_cache_key_utils import (
    build_prediction_cache_key,
    ensure_chemical_inchi_key,
    inchi_key_connectivity_block,
    inchi_keys_match_connectivity,
    normalize_inchi_key,
    standardized_chemical_changes_identity,
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

    def test_inchi_key_connectivity_block_returns_first_block(self):
        self.assertEqual(
            inchi_key_connectivity_block("ZWBAMYVPMDSJGQ-PKSOQXRJNA-N"),
            "ZWBAMYVPMDSJGQ",
        )

    def test_inchi_keys_match_connectivity_ignores_fixed_h_difference(self):
        self.assertTrue(
            inchi_keys_match_connectivity(
                "ZWBAMYVPMDSJGQ-UHFFFAOYSA-N",
                "ZWBAMYVPMDSJGQ-PKSOQXRJNA-N",
            )
        )

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

    def test_standardized_chemical_changes_identity_uses_inchi_key_equivalence(self):
        def smiles_to_inchi_key(smiles):
            return {
                "CC(O)=O": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
                "CC(=O)O": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
            }.get(smiles)

        changed = standardized_chemical_changes_identity(
            "CC(O)=O",
            {"canonicalSmiles": "CC(=O)O"},
            smiles_to_inchi_key,
        )

        self.assertFalse(changed)

    def test_standardized_chemical_changes_identity_detects_different_inchi_key(self):
        changed = standardized_chemical_changes_identity(
            "CCO",
            {
                "canonicalSmiles": "CC(=O)O",
                "inchiKey": "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
            },
            lambda smiles: "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        )

        self.assertTrue(changed)

    def test_standardized_chemical_changes_identity_allows_same_connectivity_block(self):
        changed = standardized_chemical_changes_identity(
            "OC(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)=O",
            {
                "canonicalSmiles": "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
                "inchiKey": "ZWBAMYVPMDSJGQ-PKSOQXRJNA-N",
            },
            lambda smiles: "ZWBAMYVPMDSJGQ-UHFFFAOYSA-N",
        )

        self.assertFalse(changed)


if __name__ == "__main__":
    unittest.main()
