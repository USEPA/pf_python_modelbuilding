from unittest import TestCase

from util import predict_constants as pc
from util.model_result_units import (
    LOG_KOC_MODEL_IDS,
    normalize_log_koc_model_details,
    normalize_log_koc_prediction,
)


class LogKocResultUnitsTests(TestCase):
    def test_normalizes_every_log_koc_model(self):
        for model_id in LOG_KOC_MODEL_IDS:
            with self.subTest(model_id=model_id):
                prediction = {
                    "modelResults": {
                        "predictionValueUnitsModel": 2.0,
                        "predictionValueUnitsDisplay": 100.0,
                        "unitsModel": pc.LOG_L_KG,
                        "unitsDisplay": pc.L_KG,
                    }
                }

                normalized = normalize_log_koc_prediction(prediction, model_id=model_id)

                self.assertEqual(2.0, normalized["modelResults"]["predictionValueUnitsDisplay"])
                self.assertEqual(100.0, normalized["modelResults"]["predictionValueUnitsLinear"])

    def test_moves_linear_display_values_to_explicit_linear_fields(self):
        prediction = {
            "modelResults": {
                "experimentalValueUnitsModel": 2.415,
                "experimentalValueUnitsDisplay": 260.0159563165272,
                "experimentalValueSet": "Training",
                "predictionValueUnitsModel": 2.355494602101728,
                "unitsModel": pc.LOG_L_KG,
                "predictionValueUnitsDisplay": 226.72248973595177,
                "unitsDisplay": pc.L_KG,
                "predictionError": None,
            }
        }

        normalized = normalize_log_koc_prediction(prediction, model_id=1754)

        self.assertEqual(2.415, normalized["modelResults"]["experimentalValueUnitsDisplay"])
        self.assertEqual(
            260.0159563165272,
            normalized["modelResults"]["experimentalValueUnitsLinear"],
        )
        self.assertEqual(2.355494602101728, normalized["modelResults"]["predictionValueUnitsDisplay"])
        self.assertEqual(
            226.72248973595177,
            normalized["modelResults"]["predictionValueUnitsLinear"],
        )
        self.assertEqual(pc.LOG_L_KG, normalized["modelResults"]["unitsDisplay"])
        self.assertEqual(pc.L_KG, normalized["modelResults"]["unitsLinear"])

    def test_normalization_is_idempotent_for_cached_results(self):
        prediction = {
            "modelResults": {
                "experimentalValueUnitsModel": 2.415,
                "experimentalValueUnitsDisplay": 2.415,
                "experimentalValueUnitsLinear": 260.0159563165272,
                "predictionValueUnitsModel": 2.355494602101728,
                "predictionValueUnitsDisplay": 2.355494602101728,
                "predictionValueUnitsLinear": 226.72248973595177,
                "unitsModel": pc.LOG_L_KG,
                "unitsDisplay": pc.LOG_L_KG,
                "unitsLinear": pc.L_KG,
            }
        }

        normalized = normalize_log_koc_prediction(prediction, model_id="1758")

        self.assertEqual(prediction, normalized)

    def test_normalizes_log_koc_model_details_without_mutating_input(self):
        model_details = {
            "modelId": "1763",
            "propertyName": pc.KOC,
            "unitsModel": pc.LOG_L_KG,
            "unitsDisplay": pc.L_KG,
        }

        normalized = normalize_log_koc_model_details(model_details, model_id=1763)

        self.assertEqual(pc.LOG_L_KG, normalized["unitsDisplay"])
        self.assertEqual(pc.L_KG, normalized["unitsLinear"])
        self.assertEqual(pc.L_KG, model_details["unitsDisplay"])

    def test_leaves_non_koc_results_unchanged(self):
        prediction = {
            "modelDetails": {
                "propertyName": pc.WATER_SOLUBILITY,
                "unitsModel": pc.NEG_LOG_M,
            },
            "modelResults": {
                "predictionValueUnitsModel": 2.0,
                "predictionValueUnitsDisplay": 0.01,
            },
        }

        self.assertIs(prediction, normalize_log_koc_prediction(prediction, model_id=1066))

    def test_missing_cached_model_details_remain_missing(self):
        self.assertIsNone(normalize_log_koc_model_details(None, model_id=1754))
