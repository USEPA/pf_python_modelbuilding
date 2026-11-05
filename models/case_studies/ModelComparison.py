"""Compare two RunDataset model configurations with repeated local CV."""

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.model_selection import KFold, StratifiedKFold
from utils import to_json_safe

from models.RunDataset import (
	RunDataset,
	RunParams,
)
from util import predict_constants as pc

import logging
from logging import INFO, DEBUG, ERROR
import coloredlogs

custom_level_styles = {
    'debug': {'color': 'cyan'},
    'info': {'color': 'yellow'},
    'warning': {'color': 'red', 'bold': True},
    'error': {'color': 'white', 'background': 'red'},
}

level = INFO

coloredlogs.install(level=level, milliseconds=True, level_styles=custom_level_styles,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

RunConfiguration = Union[Mapping[str, object], RunParams]


class ModelComparison:
	"""Build and compare two model configurations on identical repeated CV folds.
    
	CV is generated locally from the configured dataset's base training portion,
	matching RunDataset's existing CV boundary while avoiding database split IDs.
	Binary targets use stratified folds; continuous targets use quantile-stratified
	folds when feasible and shuffled K-folds otherwise.
	"""

	def __init__(
		self,
		output_root: Optional[Path] = None,
		result_processors: Optional[list[Callable[[pd.DataFrame], pd.DataFrame]]] = None,
	) -> None:
		self.output_root = output_root
		self.result_processors = result_processors or []

	def compare(
		self,
		configuration_a: RunConfiguration,
		configuration_b: RunConfiguration,
		n_repeats: int = 10,
		n_splits: int = 5,
		random_state: int = 42,
		model_names: Optional[tuple[str, str]] = None,
		comparison_name: Optional[str] = None,
	) -> pd.DataFrame:
		"""Run repeated CV and write an explanatory Excel report and plots.

		Configuration mappings contain keyword arguments accepted by
		``RunDataset.generate_params``. Alternatively, callers may pass its
		``RunParams`` result. Both configurations must use the same dataset,
		descriptor set, and source split so every score is paired fairly.
		"""
		if n_repeats < 1:
			raise ValueError("n_repeats must be at least 1")
		if n_splits < 2:
			raise ValueError("n_splits must be at least 2")

		runner = RunDataset()
		params_a = self._prepare_params(runner, configuration_a)
		params_b = self._prepare_params(runner, configuration_b, params_a)
		self._validate_configurations(params_a, params_b)
		logging.info(
			"Starting comparison for %s: %s vs %s (%d repeats, %d folds)",
			params_a.hyperparameters.dataset_name,
			params_a.hyperparameters.qsar_method,
			params_b.hyperparameters.qsar_method,
			n_repeats,
			n_splits,
		)

		logging.info("Loading comparison training data")
		runner.query_dataframes(params_a)
		if params_a.df_training is None or params_a.df_training.empty:
			raise ValueError("RunDataset did not provide a non-empty training dataframe")
		if "Property" not in params_a.df_training.columns:
			raise ValueError("Training dataframe must contain a 'Property' target column")

		training_data = params_a.df_training.copy()
		logging.info("Loaded %d training rows and %d columns", *training_data.shape)
		target = pd.to_numeric(training_data["Property"], errors="coerce")
		if target.isna().any():
			raise ValueError("Training target contains missing or non-numeric values")
		is_binary = target.isin([0, 1]).all()
		if is_binary:
			class_counts = target.value_counts()
			if class_counts.min() < n_splits:
				raise ValueError(
					f"Cannot create {n_splits} balanced folds: each binary class must "
					"contain at least n_splits records"
				)

		default_name_a = params_a.hyperparameters.qsar_method
		default_name_b = params_b.hyperparameters.qsar_method
		if default_name_a == default_name_b:
			default_name_a = f"{default_name_a}_A"
			default_name_b = f"{default_name_b}_B"
		labels = model_names or (default_name_a, default_name_b)
		if labels[0] == labels[1]:
			raise ValueError("model_names must contain two distinct names")
		results: list[dict[str, object]] = []
		cache_path = self._make_cache_path(params_a.hyperparameters.dataset_name)
		cache = self._load_run_cache(cache_path)
		cache_configurations = [self._configuration_cache_data(params) for params in (params_a, params_b)]
		training_fingerprint = self._training_fingerprint(training_data)
		for repeat in range(n_repeats):
			seed = random_state + repeat
			if is_binary:
				splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
				splits = splitter.split(training_data, target)
			else:
				target_bins = self._balanced_regression_bins(target, n_splits)
				if target_bins is None:
					splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
					splits = splitter.split(training_data)
				else:
					splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
					splits = splitter.split(training_data, target_bins)

			for fold, (train_indices, test_indices) in enumerate(splits, start=1):
				fold_training = training_data.iloc[train_indices].copy()
				fold_prediction = training_data.iloc[test_indices].copy()
				logging.info(
					"Evaluating repeat %d/%d, fold %d/%d (%d train, %d test rows)",
					repeat + 1,
					n_repeats,
					fold,
					n_splits,
					len(fold_training),
					len(fold_prediction),
				)
				for label, params, (configuration_key, configuration_data) in zip(
					labels, (params_a, params_b), cache_configurations
				):
					logging.debug("Building %s for repeat %d, fold %d", label, repeat + 1, fold)
					row: dict[str, object] = {
						"model": label,
						"repeat": repeat + 1,
						"fold": fold,
						"random_state": seed,
						"training_size": len(fold_training),
						"test_size": len(fold_prediction),
					}
					cached_statistics = self._get_cached_fold(
						cache,
						configuration_key,
						seed,
						fold,
						n_splits,
						training_fingerprint,
						len(fold_training),
						len(fold_prediction),
					)
					if cached_statistics is not None:
						logging.info("Reusing cached %s results for seed %d, fold %d", label, seed, fold)
						row.update(cached_statistics)
					else:
						statistics = self._evaluate_fold(
							runner,
							params,
							fold_training,
							fold_prediction,
							is_binary,
							n_splits,
						)
						row.update(statistics)
						self._store_cached_fold(
							cache,
							cache_path,
							configuration_key,
							configuration_data,
							seed,
							fold,
							n_splits,
							training_fingerprint,
							len(fold_training),
							len(fold_prediction),
							statistics,
						)
					results.append(row)

		results_df = self.process_results(pd.DataFrame(results))
		output_dir = self._make_output_dir(params_a.hyperparameters.dataset_name, comparison_name)
		logging.info("Saving comparison outputs to %s", output_dir.resolve())
		configuration_details = (
			self._format_configuration(configuration_a, params_a),
			self._format_configuration(configuration_b, params_b),
		)
		self.save_results(
			results_df,
			output_dir,
			labels,
			params_a.hyperparameters.dataset_name,
			configuration_details,
			n_repeats,
			n_splits,
		)
		logging.info("Comparison complete: %d result rows saved to %s", len(results_df), output_dir.resolve())
		return results_df

	def process_results(self, results: pd.DataFrame) -> pd.DataFrame:
		"""Apply configured result transformations; override to add analyses."""
		processed = results.copy()
		for processor in self.result_processors:
			processed = processor(processed)
		return processed

	def save_results(
		self,
		results: pd.DataFrame,
		output_dir: Path,
		model_names: tuple[str, str],
		dataset_name: str,
		configuration_details: tuple[str, str],
		n_repeats: int,
		n_splits: int,
	) -> None:
		"""Write an Excel report and metric distribution plots to the output folder."""
		output_dir.mkdir(parents=True, exist_ok=True)
		metric_columns = [
			column for column in results.select_dtypes(include=[np.number]).columns
			if column not in {"repeat", "fold", "random_state", "training_size", "test_size"}
		]
		model_summary = self._build_summary(results, metric_columns)
		paired_differences = self._build_paired_differences(results, metric_columns, model_names)
		paired_summary = self._build_paired_summary(
			results, paired_differences, metric_columns, model_names, n_splits
		)
		self._write_excel_report(
			results,
			model_summary,
			paired_differences,
			paired_summary,
			output_dir / "model_comparison.xlsx",
			dataset_name,
			model_names,
			configuration_details,
			n_repeats,
			n_splits,
		)
		self._write_metric_plots(results, metric_columns, model_names, output_dir)

	@staticmethod
	def _format_configuration(configuration: RunConfiguration, params: RunParams) -> str:
		if isinstance(configuration, Mapping):
			values = dict(configuration)
		else:
			values = vars(params.hyperparameters)
		excluded = {"engine", "session", "df_training", "df_prediction", "df_external"}
		display_values = []
		for key, value in values.items():
			if key in excluded or isinstance(value, pd.DataFrame):
				continue
			if value is None or isinstance(value, (str, bool, int, float, list, tuple)):
				display_values.append(f"{key}={value!r}")
		return "; ".join(display_values) or params.hyperparameters.qsar_method

	def _prepare_params(
		self,
		runner: RunDataset,
		configuration: RunConfiguration,
		shared_params: Optional[RunParams] = None,
	) -> RunParams:
		if isinstance(configuration, RunParams):
			return replace(configuration, cross_validate=False, write_to_db=False,
				write_excel_files=False, write_results_dict=False,
				dataset_name_ext=None, df_external=None, df_prediction_ext=None)

		config = dict(configuration)
		if shared_params is not None:
			config.setdefault("engine", shared_params.engine)
			config.setdefault("session", shared_params.session)
		config["cross_validate"] = False
		config["write_to_db"] = False
		config["write_excel_files"] = False
		config["write_results_dict"] = False
		config["dataset_name_ext"] = None
		config["df_external"] = None
		params = runner.generate_params(**config)
		params.dataset_name_ext = None
		params.df_external = None
		params.df_prediction_ext = None
		return params

	@staticmethod
	def _validate_configurations(params_a: RunParams, params_b: RunParams) -> None:
		required_equal = (
			"dataset_name",
			"descriptor_set_name",
			"splitting_name",
		)
		for field_name in required_equal:
			value_a = getattr(params_a.hyperparameters, field_name)
			value_b = getattr(params_b.hyperparameters, field_name)
			if value_a != value_b:
				raise ValueError(f"Both configurations must use the same {field_name}")

	def _evaluate_fold(
		self,
		runner: RunDataset,
		params: RunParams,
		fold_training: pd.DataFrame,
		fold_prediction: pd.DataFrame,
		is_binary: bool,
		n_splits: int,
	) -> dict[str, object]:
		fold_params = replace(
			params,
			embedding=list(params.embedding) if params.embedding is not None else None,
		)
		fold_params.df_training = fold_training.reset_index(drop=True)
		fold_params.df_prediction = fold_prediction.reset_index(drop=True)
		fold_params.df_cv_dict = None
		fold_params.dataset_name_ext = None
		fold_params.df_external = None
		fold_params.df_prediction_ext = None
		fold_params.cv = n_splits
		fold_params.is_binary = is_binary
		fold_params.cross_validate = False

		runner.apply_outlier_filtering(fold_params)
		runner.setup_logp_martin(fold_params)
		runner.run_feature_selection(fold_params)
		runner.run_model_building(fold_params)
		runner.run_post_build_feature_selection(fold_params)
		runner.run_applicability_domain(fold_params)
		runner.extract_ad_stats(fold_params)
		return fold_params.test_stats.copy() | fold_params.ad_stats

	@staticmethod
	def _balanced_regression_bins(target: pd.Series, n_splits: int) -> Optional[pd.Series]:
		max_bins = min(10, len(target) // n_splits)
		for bin_count in range(max_bins, 1, -1):
			bins = pd.qcut(target, q=bin_count, labels=False, duplicates="drop")
			counts = bins.value_counts()
			if len(counts) > 1 and counts.min() >= n_splits:
				return bins
		return None

	@staticmethod
	def _build_summary(results: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
		summary_rows: list[dict[str, object]] = []
		for model_name, group in results.groupby("model", sort=False):
			for metric in metrics:
				values = pd.to_numeric(group[metric], errors="coerce").dropna()
				summary_rows.append({
					"model": model_name,
					"metric": metric,
					"count": int(values.count()),
					"mean": values.mean(),
					"std": values.std(),
					"median": values.median(),
					"min": values.min(),
					"max": values.max(),
				})
		return pd.DataFrame(summary_rows)

	@staticmethod
	def _build_paired_differences(
		results: pd.DataFrame,
		metrics: list[str],
		model_names: tuple[str, str],
	) -> pd.DataFrame:
		keys = ["repeat", "fold"]
		paired = results.pivot(index=keys, columns="model", values=metrics)
		difference_rows: list[dict[str, object]] = []
		for metric in metrics:
			if (metric, model_names[0]) not in paired or (metric, model_names[1]) not in paired:
				continue
			score_a = paired[(metric, model_names[0])]
			score_b = paired[(metric, model_names[1])]
			differences = score_b - score_a
			for (repeat, fold), difference in differences.items():
				difference_rows.append({
					"repeat": repeat,
					"fold": fold,
					"metric": metric,
					f"Score - {model_names[0]}": score_a.loc[(repeat, fold)],
					f"Score - {model_names[1]}": score_b.loc[(repeat, fold)],
					f"Difference ({model_names[1]} - {model_names[0]})": difference,
				})
		return pd.DataFrame(difference_rows)

	@classmethod
	def _build_paired_summary(
		cls,
		results: pd.DataFrame,
		paired_differences: pd.DataFrame,
		metrics: list[str],
		model_names: tuple[str, str],
		n_splits: int,
		alpha: float = 0.05,
	) -> pd.DataFrame:
		rows: list[dict[str, object]] = []
		difference_column = f"Difference ({model_names[1]} - {model_names[0]})"
		fold_ratio = (results[["repeat", "fold", "test_size", "training_size"]]
			.drop_duplicates(["repeat", "fold"])
			.assign(ratio=lambda frame: frame["test_size"] / frame["training_size"])
			["ratio"].mean())
		for metric in metrics:
			metric_differences = paired_differences.loc[
				paired_differences["metric"] == metric, difference_column
			].dropna()
			if metric_differences.empty:
				continue
			paired_metric = paired_differences.loc[
				paired_differences["metric"] == metric
			].dropna(subset=[difference_column])
			score_a = pd.to_numeric(paired_metric[f"Score - {model_names[0]}"], errors="coerce").dropna()
			score_b = pd.to_numeric(paired_metric[f"Score - {model_names[1]}"], errors="coerce").dropna()
			differences = pd.to_numeric(metric_differences, errors="coerce").dropna()
			count = len(differences)
			mean_difference = float(differences.mean())
			std_difference = float(differences.std(ddof=1)) if count > 1 else float("nan")
			correction = (1 / count) + fold_ratio if count else float("nan")
			standard_error = float(np.sqrt(correction * std_difference ** 2)) if count > 1 else float("nan")
			if count < 2:
				t_statistic, p_value = float("nan"), float("nan")
				ci_lower, ci_upper = float("nan"), float("nan")
			elif standard_error > 0:
				t_statistic = mean_difference / standard_error
				p_value = float(2 * student_t.sf(abs(t_statistic), df=count - 1))
				critical_value = float(student_t.ppf(1 - alpha / 2, df=count - 1))
				ci_lower = mean_difference - critical_value * standard_error
				ci_upper = mean_difference + critical_value * standard_error
			elif mean_difference == 0:
				t_statistic, p_value = 0.0, 1.0
				ci_lower, ci_upper = 0.0, 0.0
			else:
				t_statistic, p_value = float(np.sign(mean_difference) * np.inf), 0.0
				ci_lower, ci_upper = mean_difference, mean_difference
			significant = p_value < alpha
			direction = cls._metric_direction(metric)
			if count < 2:
				conclusion = "Insufficient paired folds to test significance"
			elif not significant:
				conclusion = "No statistically significant difference"
			elif direction == "higher":
				conclusion = f"{model_names[1] if mean_difference > 0 else model_names[0]} performs better"
			elif direction == "lower":
				conclusion = f"{model_names[1] if mean_difference < 0 else model_names[0]} performs better"
			else:
				conclusion = "Statistically significant difference; metric direction is not ranked"
			rows.append({
				"Metric": metric,
				"Model A": model_names[0],
				"Mean Score - Model A": float(score_a.mean()) if not score_a.empty else float("nan"),
				"Model B": model_names[1],
				"Mean Score - Model B": float(score_b.mean()) if not score_b.empty else float("nan"),
				"Mean Difference (B - A)": mean_difference,
				"Std. Dev. of Differences": std_difference,
				"Paired Folds": count,
				"Fold/Test-to-Train Ratio": fold_ratio if pd.notna(fold_ratio) else 1 / (n_splits - 1),
				"Corrected Standard Error": standard_error,
				"Corrected 95% CI Lower": ci_lower,
				"Corrected 95% CI Upper": ci_upper,
				"t Statistic": t_statistic,
				"Degrees of Freedom": count - 1,
				"p-value (corrected paired t-test)": p_value,
				f"Significant at alpha={alpha:.2f}": significant,
				"Metric Direction": f"{direction}-is-better" if direction else "not ranked",
				"Conclusion": conclusion,
			})
		return pd.DataFrame(rows)

	@staticmethod
	def _metric_direction(metric: str) -> Optional[str]:
		name = metric.lower()
		if any(token in name for token in ("mae", "rmse", "mse", "error")):
			return "lower"
		if any(token in name for token in (
			"r2", "q2", "pearson", "concordance", "balanced_accuracy", "accuracy",
			"recall", "sensitivity", "specificity",
		)):
			return "higher"
		if name.startswith(("sn_", "sp_", "ba_")):
			return "higher"
		return None

	@staticmethod
	def _write_excel_report(
		fold_results: pd.DataFrame,
		model_summary: pd.DataFrame,
		paired_differences: pd.DataFrame,
		paired_summary: pd.DataFrame,
		output_path: Path,
		dataset_name: str,
		model_names: tuple[str, str],
		configuration_details: tuple[str, str],
		n_repeats: int,
		n_splits: int,
	) -> None:
		with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
			workbook = writer.book
			cover = workbook.add_worksheet("Comparison Summary")
			writer.sheets["Comparison Summary"] = cover
			cover_formats = {
				"title": workbook.add_format({"bold": True, "font_size": 16, "font_color": "#FFFFFF", "bg_color": "#1F4E78", "valign": "vcenter"}),
				"section": workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#4472C4"}),
				"label": workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "valign": "top", "text_wrap": True}),
				"value": workbook.add_format({"valign": "top", "text_wrap": True}),
				"header": workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "text_wrap": True, "valign": "vcenter"}),
				"number": workbook.add_format({"num_format": "0.000", "border": 1}),
				"text": workbook.add_format({"border": 1, "text_wrap": True, "valign": "top"}),
			}
			cover.merge_range("A1:F1", "Model Comparison Summary", cover_formats["title"])
			cover.set_row(0, 28)
			cover.write("A3", "Comparison", cover_formats["label"])
			cover.write("B3", f"{model_names[0]} vs {model_names[1]}", cover_formats["value"])
			cover.write("A4", "Dataset", cover_formats["label"])
			cover.write("B4", dataset_name, cover_formats["value"])
			cover.write("A5", "Validation design", cover_formats["label"])
			cover.write("B5", f"{n_repeats} repeats of {n_splits}-fold CV; same folds for both models", cover_formats["value"])
			cover.write("A6", "Model A configuration", cover_formats["label"])
			cover.merge_range("B6:F6", configuration_details[0], cover_formats["value"])
			cover.write("A7", "Model B configuration", cover_formats["label"])
			cover.merge_range("B7:F7", configuration_details[1], cover_formats["value"])
			cover.set_row(5, 42)
			cover.set_row(6, 42)
			cover.merge_range("A9:F9", "Statistical comparison", cover_formats["section"])
			cover.merge_range(
				"A10:F10",
				"Differences are Model B minus Model A. Significance uses a two-sided corrected repeated-k-fold paired t-test (alpha = 0.05), with the fold test-to-train ratio correction for overlap between training sets. Metric direction is used only where it is recognizable from the metric name.",
				cover_formats["value"],
			)
			cover.set_row(9, 36)
			# cover.set_row(10, 48)
			cover_headers = [
				"Metric", "Mean Score - Model A", "Mean Score - Model B",
				"Mean Difference (B - A)", "Corrected p-value", "Conclusion",
			]
			start_row = 11
			for column, header in enumerate(cover_headers):
				cover.write(start_row, column, header, cover_formats["header"])
			for row_index, row in paired_summary.iterrows():
				values = [
					row["Metric"], row["Mean Score - Model A"], row["Mean Score - Model B"],
					row["Mean Difference (B - A)"], row["p-value (corrected paired t-test)"], row["Conclusion"],
				]
				for column, value in enumerate(values):
					cell_format = cover_formats["number"] if column in (1, 2, 3, 4) else cover_formats["text"]
					cover.write(start_row + row_index + 1, column, value, cell_format)
			cover.set_column("A:A", 24)
			cover.set_column("B:C", 21)
			cover.set_column("D:E", 22)
			cover.set_column("F:F", 48)
			cover.set_column("A:A", 24, cover_formats["label"])
			cover.autofilter(start_row, 0, start_row + max(len(paired_summary), 1), len(cover_headers) - 1)
			# cover.freeze_panes(start_row + 1, 0)

			model_summary = model_summary.rename(columns={
				"model": "Model",
				"metric": "Metric",
				"count": "Available Fold Scores",
				"mean": "Mean Score",
				"std": "Standard Deviation",
				"median": "Median Score",
				"min": "Minimum Score",
				"max": "Maximum Score",
			})
			paired_differences = paired_differences.rename(columns={
				"repeat": "Repeat",
				"fold": "Fold",
				"metric": "Metric",
			})
			fold_results = fold_results.rename(columns={
				"model": "Model",
				"repeat": "Repeat",
				"fold": "Fold",
				"random_state": "Random State",
				"training_size": "Training Size",
				"test_size": "Test Size",
			})
			sheets = (
				("Paired Summary", paired_summary),
				("Paired Differences", paired_differences),
				("Model Summary", model_summary),
				("Fold Results", fold_results),
			)
			for sheet_name, dataframe in sheets:
				dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
				worksheet = writer.sheets[sheet_name]
				worksheet.freeze_panes(1, 0)
				if len(dataframe.columns):
					worksheet.autofilter(0, 0, max(len(dataframe), 1), len(dataframe.columns) - 1)
					for column_index, column in enumerate(dataframe.columns):
						if column in {"Conclusion"}:
							width = 48
						elif column in {"Metric"}:
							width = 24
						elif column in {"ad_measure"}:
							width = min(max(len(str(dataframe[column].iloc[0])) + 2, 12), 42)
						else:
							width = min(max(len(str(column)) + 2, 12), 42)
						worksheet.set_column(column_index, column_index, width)
					worksheet.set_row(0, 32, cover_formats["header"])

	@staticmethod
	def _write_metric_plots(
		results: pd.DataFrame,
		metrics: list[str],
		model_names: tuple[str, str],
		output_dir: Path,
	) -> None:
		plot_dir = output_dir / "plots"
		plot_dir.mkdir(exist_ok=True)
		for metric in metrics:
			series = [
				pd.to_numeric(results.loc[results["model"] == name, metric], errors="coerce").dropna()
				for name in model_names
			]
			fig, axis = plt.subplots(figsize=(7, 5))
			axis.boxplot(series, tick_labels=list(model_names), showmeans=True)
			axis.set_title(f"{metric} across repeated CV folds")
			axis.set_ylabel(metric)
			axis.grid(axis="y", alpha=0.25)
			fig.tight_layout()
			slug = re.sub(r"[^A-Za-z0-9._-]+", "_", metric).strip("_")
			fig.savefig(plot_dir / f"{slug}.png", dpi=160)
			plt.close(fig)

	def _make_output_dir(self, dataset_name: str, comparison_name: Optional[str]) -> Path:
		root = self._model_comparisons_root()
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		name = comparison_name or f"comparison_{timestamp}"
		safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
		return root / dataset_name / "comparisons" / safe_name

	def _make_cache_path(self, dataset_name: str) -> Path:
		return self._model_comparisons_root() / dataset_name / "model_comparison_runs.json"

	def _model_comparisons_root(self) -> Path:
		return self.output_root or Path(__file__).resolve().parents[2] / "data" / "model_comparisons"

	@classmethod
	def _configuration_cache_data(cls, params: RunParams) -> tuple[str, dict[str, object]]:
		configuration = {
			"hyperparameters": cls._normalize_cache_value(vars(params.hyperparameters)),
			"run_settings": cls._normalize_cache_value({
				"embedding": params.embedding,
				"folder_embedding": params.folder_embedding,
				"run_AD": params.run_AD,
				"fs_previous_embedding": params.fs_previous_embedding,
				"add_LOGP_Martin": params.add_LOGP_Martin,
			}),
		}
		canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
		return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), configuration

	@classmethod
	def _normalize_cache_value(cls, value: object) -> object:
		if isinstance(value, Mapping):
			return {str(key): cls._normalize_cache_value(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [cls._normalize_cache_value(item) for item in value]
		if isinstance(value, set):
			return sorted((cls._normalize_cache_value(item) for item in value), key=repr)
		if isinstance(value, Path):
			return str(value)
		if callable(value):
			return f"{value.__module__}.{value.__qualname__}"
		if isinstance(value, np.generic):
			return cls._normalize_cache_value(value.item())
		if value is None or isinstance(value, (str, int, float, bool)):
			return value
		return f"{type(value).__module__}.{type(value).__qualname__}"

	@staticmethod
	def _training_fingerprint(training_data: pd.DataFrame) -> str:
		digest = hashlib.sha256()
		digest.update(json.dumps(
			[(str(column), str(dtype)) for column, dtype in zip(training_data.columns, training_data.dtypes)],
			separators=(",", ":"),
		).encode("utf-8"))
		digest.update(pd.util.hash_pandas_object(training_data, index=False).to_numpy().tobytes())
		return digest.hexdigest()

	@staticmethod
	def _load_run_cache(cache_path: Path) -> dict[str, object]:
		if not cache_path.exists():
			return {"schema_version": 1, "configurations": {}}
		try:
			with cache_path.open(encoding="utf-8") as cache_file:
				cache = json.load(cache_file)
			if cache.get("schema_version") != 1 or not isinstance(cache.get("configurations"), dict):
				raise ValueError("unsupported or malformed cache structure")
			return cache
		except (OSError, json.JSONDecodeError, AttributeError, ValueError) as error:
			logging.warning("Unable to read model comparison cache %s: %s", cache_path, error)
			return {"schema_version": 1, "configurations": {}}

	@staticmethod
	def _get_cached_fold(
		cache: dict[str, object],
		configuration_key: str,
		seed: int,
		fold: int,
		n_splits: int,
		training_fingerprint: str,
		training_size: int,
		test_size: int,
	) -> Optional[dict[str, object]]:
		configurations = cache.get("configurations")
		if not isinstance(configurations, dict):
			return None
		configuration = configurations.get(configuration_key)
		if not isinstance(configuration, dict):
			return None
		runs = configuration.get("runs")
		if not isinstance(runs, dict):
			return None
		run = runs.get(str(seed))
		if not isinstance(run, dict) or run.get("n_splits") != n_splits:
			return None
		if run.get("training_fingerprint") != training_fingerprint:
			return None
		folds = run.get("folds")
		if not isinstance(folds, dict):
			return None
		fold_result = folds.get(str(fold))
		if not isinstance(fold_result, dict):
			return None
		if fold_result.get("training_size") != training_size or fold_result.get("test_size") != test_size:
			return None
		statistics = fold_result.get("statistics")
		return statistics if isinstance(statistics, dict) else None

	@staticmethod
	def _store_cached_fold(
		cache: dict[str, object],
		cache_path: Path,
		configuration_key: str,
		configuration_data: dict[str, object],
		seed: int,
		fold: int,
		n_splits: int,
		training_fingerprint: str,
		training_size: int,
		test_size: int,
		statistics: dict[str, object],
	) -> None:
		configurations = cache.setdefault("configurations", {})
		if not isinstance(configurations, dict):
			configurations = {}
			cache["configurations"] = configurations
		configuration = configurations.setdefault(configuration_key, {
			"configuration": configuration_data,
			"runs": {},
		})
		if not isinstance(configuration, dict):
			configuration = {"configuration": configuration_data, "runs": {}}
			configurations[configuration_key] = configuration
		runs = configuration.setdefault("runs", {})
		if not isinstance(runs, dict):
			runs = {}
			configuration["runs"] = runs
		run = runs.setdefault(str(seed), {
			"n_splits": n_splits,
			"training_fingerprint": training_fingerprint,
			"folds": {},
		})
		if not isinstance(run, dict) or run.get("n_splits") != n_splits or run.get("training_fingerprint") != training_fingerprint:
			run = {
				"n_splits": n_splits,
				"training_fingerprint": training_fingerprint,
				"folds": {},
			}
			runs[str(seed)] = run
		folds = run.setdefault("folds", {})
		if not isinstance(folds, dict):
			folds = {}
			run["folds"] = folds
		folds[str(fold)] = {
			"training_size": training_size,
			"test_size": test_size,
			"statistics": to_json_safe(statistics),
		}
		cache_path.parent.mkdir(parents=True, exist_ok=True)
		temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
		with temporary_path.open("w", encoding="utf-8") as cache_file:
			json.dump(cache, cache_file, indent=2, allow_nan=False)
		temporary_path.replace(cache_path)


def test_model_comparison() -> pd.DataFrame:
	"""Compare random forest and group contribution models on the KOC v1 dataset."""
	ad_measures = [
		pc.Applicability_Domain_TEST_Embedding_Euclidean,
		pc.Applicability_Domain_TEST_Fragment_Counts,
	]
	
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": ad_measures,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_gcm = shared_configuration | {"qsar_method": "gcm"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_gcm,
		n_repeats=1,
		comparison_name="KOC_v1_RF_vs_GCM_test",
	)


def koc_rf_vs_gcm() -> pd.DataFrame:
	"""Compare random forest and group contribution models on the KOC v1 dataset."""
	ad_measures = [
		pc.Applicability_Domain_TEST_Embedding_Euclidean,
		pc.Applicability_Domain_TEST_Fragment_Counts,
	]
	
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": ad_measures,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_gcm = shared_configuration | {"qsar_method": "gcm"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_gcm,
		n_repeats=100,
		comparison_name="KOC_v1_RF_vs_GCM",
	)


def main():
	# test_model_comparison()
	koc_rf_vs_gcm()


if __name__ == "__main__":
	main()