import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from model_ws_db_utilities import ModelPredictor


class DescriptorInputValidationTests(unittest.TestCase):
    def test_only_required_descriptors_are_checked(self):
        frame = pd.DataFrame({"a": [2], "irrelevant": [np.nan]})
        self.assertIsNone(ModelPredictor._descriptor_input_error(frame, ["a"]))

    def test_missing_and_nonfinite_required_values_are_rejected(self):
        for value in [np.nan, np.inf, -np.inf, None, "invalid"]:
            with self.subTest(value=value):
                self.assertIsNotNone(ModelPredictor._descriptor_input_error(pd.DataFrame({"a": [value]}), ["a"]))
        self.assertIsNotNone(ModelPredictor._descriptor_input_error(pd.DataFrame({"a": [1]}), ["missing"]))

    def test_bad_row_does_not_fail_or_reorder_valid_neighbors(self):
        predictor = ModelPredictor.__new__(ModelPredictor)
        model = SimpleNamespace(embedding=["a"], descriptorService="webtest")
        model.do_predictions = Mock(side_effect=lambda frame: frame["a"].to_numpy() * 2)
        frames = [pd.DataFrame({"id": [str(i)], "exp": [None], "a": [value]})
                  for i, value in enumerate([2, np.nan, 4])]
        predictor._calculate_descriptors_batch = Mock(return_value=[(frame, 200) for frame in frames])
        predictor._build_prediction_error_report = Mock(side_effect=lambda chemical, details, error, **kwargs: {"error": error})
        predictor._build_reports_for_precomputed_predictions_parallel = Mock(
            side_effect=lambda model, frame, chemicals, values, *args, **kwargs: [{"value": float(v)} for v in values])
        chemicals = [{"canonicalSmiles": s} for s in ["CC", "CCC", "CO"]]
        with patch("model_ws_db_utilities._standardized_chemical_changes_identity", return_value=False):
            results = predictor._predict_model_smiles_batch_from_standardized_results(
                model, "api", ["CC", "CCC", "CO"], [(chemical, 200) for chemical in chemicals])
        self.assertEqual(results[0], {"value": 4.0})
        self.assertIn("non-finite", results[1]["error"])
        self.assertEqual(results[2], {"value": 8.0})
        self.assertEqual(len(model.do_predictions.call_args.args[0]), 2)


if __name__ == "__main__":
    unittest.main()
