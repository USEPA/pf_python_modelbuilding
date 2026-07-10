import pathlib
import pickle
import sys
import unittest


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
MODEL_FIXTURES = (
    "bcf_rf.pickle",
    "bcf_svm.pickle",
    "bcf_xgb.pickle",
    "llna_rf.pickle",
    "llna_svm.pickle",
    "llna_xgb.pickle",
)


class PredictorModelsRuntimeNoCacheTests(unittest.TestCase):
    def test_local_model_fixtures_run_predictions_without_prediction_cache(self):
        missing_fixtures = [
            file_name
            for file_name in MODEL_FIXTURES
            if not (TEST_DATA_DIR / file_name).exists()
        ]
        if missing_fixtures:
            self.skipTest(
                "Predictor model runtime fixtures are unavailable: "
                + ", ".join(missing_fixtures)
            )

        missing = object()
        saved_models_modules = {
            name: sys.modules.get(name, missing)
            for name in ("models", "models.df_utilities")
        }
        for name in saved_models_modules:
            sys.modules.pop(name, None)

        original_sys_path = list(sys.path)
        sys.path.insert(0, str(PROJECT_ROOT))
        try:
            try:
                from models.df_utilities import load_df_from_file
            except (ImportError, ModuleNotFoundError) as exc:
                raise unittest.SkipTest(
                    f"Predictor model runtime dependencies are unavailable: {exc}"
                ) from exc

            prediction_frame = load_df_from_file(str(TEST_DATA_DIR / "ext_pred_set.tsv"))
            for file_name in MODEL_FIXTURES:
                with self.subTest(model_fixture=file_name):
                    with (TEST_DATA_DIR / file_name).open("rb") as model_file:
                        model = pickle.load(model_file)
                    predictions = model.do_predictions(prediction_frame)
                    self.assertEqual(len(predictions), prediction_frame.shape[0])
                    self.assertTrue(
                        all(isinstance(prediction, float) for prediction in predictions)
                    )
        finally:
            sys.path[:] = original_sys_path
            for name, module in saved_models_modules.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


if __name__ == "__main__":
    unittest.main()
