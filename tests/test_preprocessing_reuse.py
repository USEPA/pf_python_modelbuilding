import ast
from pathlib import Path
import logging
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from util.preprocessing_cache import PreprocessingCache


source = Path(__file__).resolve().parents[1] / "model_ws_db_utilities.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "ModelPredictor"
              for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_reuse_preprocessing")
method.decorator_list = []
namespace = {"logging": logging}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
reuse = namespace["_reuse_preprocessing"]
predictor_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelPredictor")
matching_names = {
    "_standardization_match_key", "_get_standardization_response_match_keys",
    "_build_standardization_expected_id_map", "_build_standardization_results_by_index",
    "_normalize_standardization_batch_results",
}
matching_class = ast.ClassDef(name="Matcher", bases=[], keywords=[], decorator_list=[],
                             body=[node for node in predictor_node.body if isinstance(node, ast.FunctionDef)
                                   and node.name in matching_names])
exec(compile(ast.fix_missing_locations(ast.Module(body=[matching_class], type_ignores=[])),
             str(source), "exec"), namespace)


class StandardizationMatchingTests(unittest.TestCase):
    def setUp(self):
        self.matcher = namespace["Matcher"]()
        self.matcher._preview_log_value = repr
        self.model = SimpleNamespace(omitSalts=True, qsarReadyRuleSet="workflow")

    def test_changed_connectivity_uses_explicit_id(self):
        output = [{"id": "0", "canonicalSmiles": "CC(=O)O"}]
        # No InChI identity guess is needed when the service preserves the input ID.
        normalized, _, _, missing = self.matcher._normalize_standardization_batch_results(
            output, ["CC(=O)[O-].[Na+]"], self.model)
        self.assertEqual(normalized, [(output[0], 200)])
        self.assertEqual(missing, [])

    def test_reordered_rows_match_ids(self):
        output = [{"id": "1", "canonicalSmiles": "CCC"}, {"id": "0", "canonicalSmiles": "CC"}]
        normalized, _, _, missing = self.matcher._normalize_standardization_batch_results(
            output, ["CC", "CCC"], self.model)
        self.assertEqual([row[0]["canonicalSmiles"] for row in normalized], ["CC", "CCC"])
        self.assertEqual(missing, [])

    def test_split_mixture_is_rejected_without_remapping_neighbor(self):
        output = [{"id": "0", "canonicalSmiles": "CC"}, {"id": "0", "canonicalSmiles": "CCC"},
                  {"id": "1", "canonicalSmiles": "CO"}]
        normalized, _, _, missing = self.matcher._normalize_standardization_batch_results(
            output, ["CC.CCC", "CO"], self.model)
        self.assertEqual(normalized[0][1], 400)
        self.assertEqual(normalized[1], (output[2], 200))
        self.assertEqual(missing, [])


class PreprocessingReuseTests(unittest.TestCase):
    def test_shared_models_reuse_success_but_retry_failures(self):
        cache = PreprocessingCache()
        calls = []
        def calculate(items):
            calls.append(items)
            return [({"canonicalSmiles": item, "id": str(i)}, 200) if item != "bad"
                    else ("temporary failure", 503) for i, item in enumerate(items)]
        first = reuse(cache, ("endpoint", "workflow", False), ["CC", "bad"], calculate)
        first[0][0]["canonicalSmiles"] = "mutated"
        second = reuse(cache, ("endpoint", "workflow", False), ["bad", "CC"], calculate)
        self.assertEqual(calls, [["CC", "bad"], ["bad"]])
        self.assertEqual(second[1][0], {"canonicalSmiles": "CC"})
        reuse(cache, ("endpoint", "other-workflow", False), ["CC"], calculate)
        self.assertEqual(calls[-1], ["CC"])

    def test_capacity_expiry_and_copy(self):
        cache = PreprocessingCache(capacity=1, ttl=10)
        with patch("util.preprocessing_cache.time.monotonic", return_value=0):
            cache.put("a", ({"x": [1]}, 200))
            value = cache.get("a")
            value[0]["x"].append(2)
            self.assertEqual(cache.get("a")[0]["x"], [1])
            cache.put("b", ("b", 200))
            self.assertIsNone(cache.get("a"))
        with patch("util.preprocessing_cache.time.monotonic", return_value=11):
            self.assertIsNone(cache.get("b"))

    def test_different_descriptor_services_do_not_share(self):
        cache = PreprocessingCache()
        calls = []
        def calculate(items):
            calls.append(items)
            return [({"v": 1}, 200) for _ in items]
        reuse(cache, ("api", "webtest"), ["CC"], calculate)
        reuse(cache, ("api", "mordred"), ["CC"], calculate)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
