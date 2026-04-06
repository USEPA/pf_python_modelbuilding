import os
from unittest import TestCase, skipIf
from unittest.mock import patch

from util.chemical_image_utils import (
    DEFAULT_CIM_RENDER_URL,
    build_render_image_url,
    get_render_smiles,
    resolve_report_image_src,
)

try:
    from model_ws_db_utilities import ModelPredictor
    MODEL_PREDICTOR_IMPORT_ERROR = None
except Exception as exc:
    ModelPredictor = None
    MODEL_PREDICTOR_IMPORT_ERROR = exc


class TestChemicalImageUtils(TestCase):
    def test_build_render_image_url_uses_default_and_encodes_query(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CIM_RENDER_URL", None)
            image_url = build_render_image_url("C/C=C\\C", width=150, height=120)

        self.assertIsNotNone(image_url)
        self.assertTrue(image_url.startswith(DEFAULT_CIM_RENDER_URL + "?"))
        self.assertIn("smiles=C%2FC%3DC%5CC", image_url)
        self.assertIn("format=PNG", image_url)
        self.assertIn("width=150", image_url)
        self.assertIn("height=120", image_url)

    def test_build_render_image_url_respects_env_override(self):
        with patch.dict(os.environ, {"CIM_RENDER_URL": "https://example.org/render"}, clear=False):
            image_url = build_render_image_url("C1CCCCC1")

        self.assertEqual(image_url, "https://example.org/render?smiles=C1CCCCC1&format=PNG")

    def test_get_render_smiles_prefers_smiles_then_canonical_smiles(self):
        self.assertEqual(get_render_smiles({"smiles": "C"}), "C")
        self.assertEqual(get_render_smiles({"canonicalSmiles": "CC"}), "CC")
        self.assertIsNone(get_render_smiles({"cid": "DTXCID123"}))

    def test_resolve_report_image_src_preserves_existing_value(self):
        self.assertEqual(
            resolve_report_image_src({"imageSrc": "N/A", "smiles": "C1CCCCC1"}),
            "N/A",
        )

    @skipIf(ModelPredictor is None, f"ModelPredictor unavailable: {MODEL_PREDICTOR_IMPORT_ERROR}")
    def test_prepare_chemical_for_report_keeps_indigo_branch(self):
        predictor = ModelPredictor()

        with patch.object(ModelPredictor, "smiles_to_base64", return_value="abc123") as smiles_to_base64:
            chemical = predictor._prepare_chemical_for_report({"smiles": "C1CCCCC1"})

        smiles_to_base64.assert_called_once_with("C1CCCCC1")
        self.assertEqual(chemical["imageSrc"], "data:image/png;base64,abc123")

    @skipIf(ModelPredictor is None, f"ModelPredictor unavailable: {MODEL_PREDICTOR_IMPORT_ERROR}")
    def test_prepare_chemical_for_report_uses_render_service_for_cid_branch(self):
        predictor = ModelPredictor()

        with patch.object(ModelPredictor, "smiles_to_base64", return_value="abc123") as smiles_to_base64:
            chemical = predictor._prepare_chemical_for_report({"cid": "DTXCID123", "smiles": "C1CCCCC1"})

        smiles_to_base64.assert_not_called()
        self.assertEqual(
            chemical["imageSrc"],
            build_render_image_url("C1CCCCC1", width=400, height=400),
        )
