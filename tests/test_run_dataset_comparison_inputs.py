from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
import unittest
import zipfile
from unittest.mock import patch

import pandas as pd

from models.RunDataset import ModelBuilder, RunDataset, RunParams
from models.case_studies.ModelComparison import ModelComparison, test_model_comparison as run_test_model_comparison
from util import predict_constants as pc


class RunDatasetComparisonInputTests(unittest.TestCase):
    def setUp(self):
        self.training = pd.DataFrame({
            "ID": ["a", "b", "c", "d"],
            "Property": [0.0, 1.0, 0.0, 1.0],
            "descriptor": [0.1, 0.2, 0.3, 0.4],
        })

    def test_generate_params_accepts_data_folds_and_output_controls(self):
        folds = {
            1: {"train": self.training.iloc[2:].copy(), "pred": self.training.iloc[:2].copy()},
            2: {"train": self.training.iloc[:2].copy(), "pred": self.training.iloc[2:].copy()},
        }
        runner = RunDataset()

        with patch.object(runner, "post_init"):
            params = runner.generate_params(
                dataset_name="test",
                qsar_method="svm",
                ad_measure=[],
                engine=object(),
                session=object(),
                df_training=self.training,
                df_prediction=self.training.copy(),
                df_cv_dict=folds,
                write_excel_files=False,
                write_results_dict=False,
            )

        self.assertIs(params.df_cv_dict, folds)
        self.assertTrue(params.df_training.equals(self.training))
        self.assertFalse(params.write_excel_files)
        self.assertFalse(params.write_results_dict)

    def test_query_dataframes_uses_supplied_frames_without_database(self):
        params = SimpleNamespace(
            df_training=self.training,
            df_prediction=self.training.copy(),
            df_external=None,
            dataset_name_ext=None,
            hyperparameters=SimpleNamespace(
                dataset_name="test",
                descriptor_set_name="descriptors",
                splitting_name="split",
            ),
        )
        runner = RunDataset()

        with patch.object(runner, "ensure_session", side_effect=AssertionError("database accessed")):
            runner.query_dataframes(params)

        self.assertTrue(params.df_training.equals(self.training))
        self.assertTrue(params.df_prediction.equals(self.training))

    def test_setup_cross_validation_preserves_supplied_folds(self):
        folds = {
            0: {"train": self.training.iloc[2:].copy(), "pred": self.training.iloc[:2].copy()},
            1: {"train": self.training.iloc[:2].copy(), "pred": self.training.iloc[2:].copy()},
        }
        params = SimpleNamespace(
            cross_validate=True,
            df_training=self.training,
            df_cv_dict=folds,
            hyperparameters=SimpleNamespace(dataset_name="test", descriptor_set_name="descriptors"),
        )

        with patch("models.RunDataset.du.get_training_cv_instances", side_effect=AssertionError("database accessed")):
            RunDataset().setup_cross_validation(params)

        self.assertIs(params.df_cv_dict, folds)
        self.assertEqual(params.cv.get_n_splits(), 2)

    def test_crossvalidate_uses_all_supplied_folds_and_zero_based_labels(self):
        folds = {
            0: {"train": self.training.iloc[2:].copy(), "pred": self.training.iloc[:2].copy()},
            1: {"train": self.training.iloc[:2].copy(), "pred": self.training.iloc[2:].copy()},
        }

        def build_predictions(training, prediction, params, embedding):
            return pd.DataFrame({
                "ID": prediction["ID"],
                "exp": prediction["Property"],
                "pred": prediction["Property"],
            })

        with patch.object(ModelBuilder, "build_and_test_model2", side_effect=build_predictions) as build:
            with patch("models.RunDataset.calculate_binary_statistics", return_value={"metric": 1.0}):
                predictions, stats = ModelBuilder.crossvalidate(
                    folds, SimpleNamespace(), None, is_binary=True
                )

        self.assertEqual(build.call_count, 2)
        self.assertEqual(predictions["cv_fold"].tolist(), [0, 0, 1, 1])
        self.assertEqual(stats, {"metric": 1.0})

    def test_comparison_fold_uses_run_dataset_stages_and_isolates_embedding(self):
        calls = []
        ad_indices = []
        external_state = []

        class FakeRunner:
            def apply_outlier_filtering(self, params):
                calls.append("outliers")

            def setup_logp_martin(self, params):
                calls.append("logp")

            def run_feature_selection(self, params):
                calls.append("selection")
                params.embedding = ["selected"]

            def run_model_building(self, params):
                calls.append("build")
                params.test_stats = {"score": 1.0}

            def run_post_build_feature_selection(self, params):
                calls.append("post-selection")

            def run_applicability_domain(self, params):
                calls.append("ad")
                ad_indices.append((params.df_training.index, params.df_prediction.index))
                external_state.append((params.dataset_name_ext, params.df_prediction_ext, params.df_external))
                params.ad_stats = {"AD_metric": 0.75}

            def extract_ad_stats(self, params):
                calls.append("extract-ad")

        params = RunParams(
            hyperparameters=SimpleNamespace(),
            embedding=["initial"],
            dataset_name_ext="auto-selected-external",
            df_prediction_ext=self.training.copy(),
            df_external=self.training.copy(),
        )
        comparison = ModelComparison()
        score = comparison._evaluate_fold(
            FakeRunner(),
            params,
            self.training.iloc[[2, 0]],
            self.training.iloc[[3, 1]],
            is_binary=True,
            n_splits=2,
        )

        self.assertEqual(score, {"score": 1.0, "AD_metric": 0.75})
        self.assertEqual(
            calls,
            ["outliers", "logp", "selection", "build", "post-selection", "ad", "extract-ad"],
        )
        self.assertEqual(list(ad_indices[0][0]), [0, 1])
        self.assertEqual(list(ad_indices[0][1]), [0, 1])
        self.assertEqual(external_state, [(None, None, None)])
        self.assertEqual(params.embedding, ["initial"])

    def test_prepare_params_clears_run_dataset_auto_external_selection(self):
        generated_params = RunParams(
            hyperparameters=SimpleNamespace(),
            dataset_name_ext="KOC v2 external",
            df_prediction_ext=self.training.copy(),
            df_external=self.training.copy(),
        )

        class FakeRunner:
            def generate_params(self, **configuration):
                return generated_params

        params = ModelComparison()._prepare_params(
            FakeRunner(), {"dataset_name": "KOC v1 modeling"}
        )

        self.assertIsNone(params.dataset_name_ext)
        self.assertIsNone(params.df_prediction_ext)
        self.assertIsNone(params.df_external)

    def test_paired_summary_names_models_and_reports_corrected_significance(self):
        results = pd.DataFrame([
            {"model": model, "repeat": repeat, "fold": 1, "training_size": 8, "test_size": 2, "MAE_Test": score}
            for repeat, scores in enumerate(((3.0, 1.0), (3.2, 1.2), (2.8, 0.8)), start=1)
            for model, score in zip(("Model A RF", "Model B GCM"), scores)
        ])
        comparison = ModelComparison()
        paired_differences = comparison._build_paired_differences(
            results, ["MAE_Test"], ("Model A RF", "Model B GCM")
        )
        paired_summary = comparison._build_paired_summary(
            results, paired_differences, ["MAE_Test"], ("Model A RF", "Model B GCM"), n_splits=5
        )

        self.assertIn("Score - Model A RF", paired_differences.columns)
        self.assertIn("Score - Model B GCM", paired_differences.columns)
        self.assertIn("Difference (Model B GCM - Model A RF)", paired_differences.columns)
        self.assertEqual(
            paired_differences["Difference (Model B GCM - Model A RF)"].round(6).tolist(),
            [-2.0, -2.0, -2.0],
        )
        self.assertEqual(paired_summary.at[0, "Conclusion"], "Model B GCM performs better")
        self.assertTrue(paired_summary.at[0, "Significant at alpha=0.05"])

    def test_save_results_writes_one_workbook_with_cover_first(self):
        results = pd.DataFrame([
            {"model": model, "repeat": repeat, "fold": fold, "random_state": repeat,
             "training_size": 8, "test_size": 2, "MAE_Test": mae, "PearsonRSQ_Test": score}
            for repeat in (1,)
            for fold, scores in enumerate(((3.0, 0.8, 0.4, 0.6), (2.5, 0.9, 0.5, 0.7)), start=1)
            for model, mae, score in zip(("RF", "GCM"), scores[:2], scores[2:])
        ])

        with TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            ModelComparison().save_results(
                results,
                output_dir,
                ("RF", "GCM"),
                "Test dataset",
                ("qsar_method='rf'", "qsar_method='gcm'"),
                n_repeats=1,
                n_splits=2,
            )
            workbook_path = output_dir / "model_comparison.xlsx"
            with zipfile.ZipFile(workbook_path) as workbook:
                workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
            namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            sheet_names = [sheet.attrib["name"] for sheet in workbook_xml.findall("main:sheets/main:sheet", namespace)]

            self.assertEqual(
                sheet_names,
                ["Comparison Summary", "Paired Summary", "Paired Differences", "Model Summary", "Fold Results"],
            )
            self.assertFalse(any(path.suffix == ".csv" for path in output_dir.iterdir()))

    def test_test_model_comparison_uses_requested_configs_and_output_root(self):
        with patch.object(ModelComparison, "compare", autospec=True, return_value=pd.DataFrame()) as compare:
            run_test_model_comparison()

        comparison, configuration_rf, configuration_gcm = compare.call_args.args[:3]
        self.assertEqual(configuration_rf["dataset_name"], "KOC v1 modeling")
        self.assertEqual(configuration_rf["qsar_method"], "rf")
        self.assertEqual(configuration_gcm["qsar_method"], "gcm")
        self.assertFalse(configuration_rf["feature_selection"])
        self.assertTrue(configuration_rf["run_AD"])
        self.assertFalse(configuration_rf["write_to_db"])
        self.assertEqual(
            configuration_rf["ad_measure"],
            [
                pc.Applicability_Domain_TEST_Embedding_Euclidean,
                pc.Applicability_Domain_TEST_Fragment_Counts,
            ],
        )
        self.assertEqual(compare.call_args.kwargs["n_repeats"], 1)
        self.assertEqual(
            comparison._make_output_dir("KOC v1 modeling", "KOC_v1_RF_vs_GCM"),
            Path(__file__).resolve().parents[1]
            / "data"
            / "model_comparisons"
            / "KOC v1 modeling"
            / "comparisons"
            / "KOC_v1_RF_vs_GCM",
        )


if __name__ == "__main__":
    unittest.main()