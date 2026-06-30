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
    from model_ws_db_utilities import ModelPredictor, _sanitize_api_chemical_identifiers
    MODEL_PREDICTOR_IMPORT_ERROR = None
except Exception as exc:
    ModelPredictor = None
    _sanitize_api_chemical_identifiers = None
    MODEL_PREDICTOR_IMPORT_ERROR = exc

try:
    from util.helpers import _coerce_json_safe, _format_prediction_for_response
    PREDICT_HELPERS_IMPORT_ERROR = None
except Exception as exc:
    _coerce_json_safe = None
    _format_prediction_for_response = None
    PREDICT_HELPERS_IMPORT_ERROR = exc


class TestChemicalImageUtils(TestCase):
    def test_build_render_image_url_uses_same_origin_default_and_encodes_query(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CIM_RENDER_URL", None)
            image_url = build_render_image_url("C/C=C\\C", width=150, height=120)

        self.assertIsNotNone(image_url)
        self.assertTrue(image_url.startswith("/api/resolver/render?"))
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

    def test_resolve_report_image_src_replaces_legacy_comptox_url(self):
        image_url = resolve_report_image_src(
            {
                "imageSrc": "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-files/image/by-dtxcid/",
                "smiles": "C1CCCCC1",
            },
            width=150,
            height=150,
        )

        self.assertEqual(
            image_url,
            build_render_image_url("C1CCCCC1", width=150, height=150),
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

    @skipIf(ModelPredictor is None, f"ModelPredictor unavailable: {MODEL_PREDICTOR_IMPORT_ERROR}")
    def test_build_minimal_chemical_does_not_add_image_src(self):
        predictor = ModelPredictor()

        with patch.object(ModelPredictor, "smiles_to_base64", return_value="abc123") as smiles_to_base64:
            chemical = predictor._build_minimal_chemical("C1CCCCC1")

        smiles_to_base64.assert_not_called()
        self.assertNotIn("imageSrc", chemical)

    @skipIf(ModelPredictor is None, f"ModelPredictor unavailable: {MODEL_PREDICTOR_IMPORT_ERROR}")
    def test_sanitize_api_chemical_identifiers_removes_image_src(self):
        chemical = _sanitize_api_chemical_identifiers({
            "chemId": "C1CCCCC1",
            "imageSrc": "data:image/png;base64,abc123",
            "name": "N/A",
        })

        self.assertNotIn("imageSrc", chemical)
        self.assertIsNone(chemical["name"])

    @skipIf(_format_prediction_for_response is None, f"predict helpers unavailable: {PREDICT_HELPERS_IMPORT_ERROR}")
    def test_format_prediction_for_response_strips_chemical_image_src(self):
        formatted = _format_prediction_for_response({
            "chemicalIdentifiers": {
                "chemId": "C1CCCCC1",
                "imageSrc": "data:image/png;base64,abc123",
            },
            "standardizedChemical": {
                "chemId": "DTXSID123",
                "imageSrc": "https://example.org/render.png",
            },
            "modelResults": {"predictionError": None},
        })

        self.assertNotIn("imageSrc", formatted["chemical"])
        self.assertNotIn("imageSrc", formatted["standardizedChemical"])
        self.assertEqual(formatted["result"], {"predictionError": None})

    @skipIf(_coerce_json_safe is None, f"predict helpers unavailable: {PREDICT_HELPERS_IMPORT_ERROR}")
    def test_coerce_json_safe_strips_nested_image_src(self):
        sanitized = _coerce_json_safe({
            "result": {
                "chemical": {"chemId": "C1CCCCC1", "imageSrc": "abc123"},
                "nested": [{"imageSrc": "def456", "name": "kept"}],
            },
        })

        self.assertNotIn("imageSrc", sanitized["result"]["chemical"])
        self.assertNotIn("imageSrc", sanitized["result"]["nested"][0])
        self.assertEqual(sanitized["result"]["nested"][0]["name"], "kept")
