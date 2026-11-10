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

# Constants for testing functions
AD_MEASURES = [
	pc.Applicability_Domain_TEST_Embedding_Euclidean,
	pc.Applicability_Domain_TEST_Fragment_Counts,
]

PHYSCHEM_PROPERTIES = [
	"HLC",
	"WS",
	"VP",
	"BP",
	"LogP",
	"MP"
]

DESCRIPTOR_SETS = {
	"WebTEST": "WebTEST-default",
	"Mordred": "Mordred-default",
	"ToxPrints": "ToxPrints-default",
	"RDKit": "RDKit-default",
	"PaDEL": "PaDEL-default"
}


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
		``RunParams`` result. Both configurations must use the same dataset and
		source split. Different descriptor sets are restricted to their shared
		training chemicals before common folds are generated.
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
		runner.query_dataframes(params_b)
		if params_a.df_training is None or params_b.df_training is None:
			raise ValueError("RunDataset did not provide both training dataframes")
		training_data_a, training_data_b = self._align_training_dataframes(
			[params_a.df_training, params_b.df_training]
		)
		training_data = training_data_a
		logging.info("Loaded %d training rows and %d columns", *training_data.shape)
		target = pd.to_numeric(training_data["Property"], errors="coerce")
		if len(training_data) < n_splits:
			raise ValueError(
				f"Cannot create {n_splits} folds from {len(training_data)} shared training chemicals"
			)
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
		training_fingerprints = [
			self._training_fingerprint(frame) for frame in (training_data_a, training_data_b)
		]
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
				fold_training_data = [
					frame.iloc[train_indices].copy()
					for frame in (training_data_a, training_data_b)
				]
				fold_prediction_data = [
					frame.iloc[test_indices].copy()
					for frame in (training_data_a, training_data_b)
				]
				fold_training = fold_training_data[0]
				fold_prediction = fold_prediction_data[0]
				logging.info(
					"Evaluating repeat %d/%d, fold %d/%d (%d train, %d test rows)",
					repeat + 1,
					n_repeats,
					fold,
					n_splits,
					len(fold_training),
					len(fold_prediction),
				)
				for (
					label,
					params,
					cache_configuration,
					training_fingerprint,
					model_training,
					model_prediction,
				) in zip(
					labels,
					(params_a, params_b),
					cache_configurations,
					training_fingerprints,
					fold_training_data,
					fold_prediction_data,
				):
					configuration_key, configuration_data = cache_configuration
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
							model_training,
							model_prediction,
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

	def build_configurations(
		self,
		configurations: Sequence[RunConfiguration],
		n_repeats: int = 10,
		n_splits: int = 5,
		random_state: int = 42,
	) -> pd.DataFrame:
		"""Build configurations on repeated local CV and save fold outputs to JSON.

		Each configuration uses the same folds and the same cache format as
		``compare``. No pairwise reports or plots are generated.
		"""
		if not configurations:
			raise ValueError("At least one configuration is required")
		if n_repeats < 1:
			raise ValueError("n_repeats must be at least 1")
		if n_splits < 2:
			raise ValueError("n_splits must be at least 2")

		runner = RunDataset()
		params = [self._prepare_params(runner, configurations[0])]
		params.extend(
			self._prepare_params(runner, configuration, params[0])
			for configuration in configurations[1:]
		)
		for candidate in params[1:]:
			self._validate_configurations(params[0], candidate)

		logging.info(
			"Building %d configurations for %s (%d repeats, %d folds)",
			len(params),
			params[0].hyperparameters.dataset_name,
			n_repeats,
			n_splits,
		)
		logging.info("Loading comparison training data")
		for candidate in params:
			runner.query_dataframes(candidate)
		if any(candidate.df_training is None for candidate in params):
			raise ValueError("RunDataset did not provide every configuration's training dataframe")
		training_dataframes = self._align_training_dataframes(
			[candidate.df_training for candidate in params if candidate.df_training is not None]
		)
		training_data = training_dataframes[0]
		logging.info("Loaded %d training rows and %d columns", *training_data.shape)
		target = pd.to_numeric(training_data["Property"], errors="coerce")
		if len(training_data) < n_splits:
			raise ValueError(
				f"Cannot create {n_splits} folds from {len(training_data)} shared training chemicals"
			)
		is_binary = target.isin([0, 1]).all()
		if is_binary:
			class_counts = target.value_counts()
			if class_counts.min() < n_splits:
				raise ValueError(
					f"Cannot create {n_splits} balanced folds: each binary class must "
					"contain at least n_splits records"
				)

		labels: list[str] = []
		label_counts: dict[str, int] = {}
		for candidate in params:
			base_label = candidate.hyperparameters.qsar_method
			label_counts[base_label] = label_counts.get(base_label, 0) + 1
			labels.append(
				base_label if label_counts[base_label] == 1
				else f"{base_label}_{label_counts[base_label]}"
			)

		cache_path = self._make_cache_path(params[0].hyperparameters.dataset_name)
		cache = self._load_run_cache(cache_path)
		cache_configurations = [self._configuration_cache_data(candidate) for candidate in params]
		training_fingerprints = [
			self._training_fingerprint(frame) for frame in training_dataframes
		]
		results: list[dict[str, object]] = []		
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
				fold_training_data = [
					frame.iloc[train_indices].copy() for frame in training_dataframes
				]
				fold_prediction_data = [
					frame.iloc[test_indices].copy() for frame in training_dataframes
				]
				fold_training = fold_training_data[0]
				fold_prediction = fold_prediction_data[0]
				logging.info(
					"Evaluating repeat %d/%d, fold %d/%d (%d train, %d test rows)",
					repeat + 1,
					n_repeats,
					fold,
					n_splits,
					len(fold_training),
					len(fold_prediction),
				)
				for (
					label,
					candidate,
					cache_configuration,
					training_fingerprint,
					fold_training,
					fold_prediction,
				) in zip(
					labels,
					params,
					cache_configurations,
					training_fingerprints,
					fold_training_data,
					fold_prediction_data,
				):
					configuration_key, configuration_data = cache_configuration
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
						cached_statistics = self._evaluate_fold(
							runner,
							candidate,
							fold_training,
							fold_prediction,
							is_binary,
							n_splits,
						)
						row.update(cached_statistics)
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
							cached_statistics,
						)
						results.append(row)

		return self.process_results(pd.DataFrame(results))

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
		required_equal = ("dataset_name", "splitting_name")
		for field_name in required_equal:
			value_a = getattr(params_a.hyperparameters, field_name)
			value_b = getattr(params_b.hyperparameters, field_name)
			if value_a != value_b:
				raise ValueError(f"Both configurations must use the same {field_name}")

	@staticmethod
	def _align_training_dataframes(
		training_dataframes: Sequence[pd.DataFrame],
	) -> list[pd.DataFrame]:
		"""Align descriptor-specific dataframes to the first frame's shared IDs."""
		if not training_dataframes:
			raise ValueError("At least one training dataframe is required")
		for training_data in training_dataframes:
			if training_data.empty:
				raise ValueError("RunDataset did not provide a non-empty training dataframe")
			if "ID" not in training_data.columns:
				raise ValueError("Training dataframe must contain an 'ID' column")
			if training_data["ID"].isna().any():
				raise ValueError("Training dataframe IDs must not be missing")
			if training_data["ID"].duplicated().any():
				raise ValueError("Training dataframe IDs must be unique")
			if "Property" not in training_data.columns:
				raise ValueError("Training dataframe must contain a 'Property' target column")

		shared_ids = pd.Index(training_dataframes[0]["ID"])
		for training_data in training_dataframes[1:]:
			shared_ids = shared_ids[shared_ids.isin(training_data["ID"])]
		if shared_ids.empty:
			raise ValueError("Training dataframes do not share any chemical IDs")

		aligned_dataframes = [
			training_data.set_index("ID", drop=False).loc[shared_ids].reset_index(drop=True)
			for training_data in training_dataframes
		]
		targets = [
			pd.to_numeric(training_data["Property"], errors="coerce")
			for training_data in aligned_dataframes
		]
		if any(target.isna().any() for target in targets):
			raise ValueError("Training target contains missing or non-numeric values")
		if any(not np.array_equal(targets[0].to_numpy(), target.to_numpy()) for target in targets[1:]):
			raise ValueError("Training target values differ for shared chemical IDs")
		return aligned_dataframes

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

# --------------------------------------------------------------------------------------------------------

def test_model_comparison() -> pd.DataFrame:
	"""Compare random forest and group contribution models on the KOC v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
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
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_gcm = shared_configuration | {"qsar_method": "gcm"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_gcm,
		n_repeats=10,
		comparison_name="KOC_v1_RF_vs_GCM",
	)


def koc_rf_vs_xgboost() -> pd.DataFrame:
	"""Compare random forest and XGBoost models on the KOC v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_xgb = shared_configuration | {"qsar_method": "xgb"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_xgb,
		n_repeats=10,
		comparison_name="KOC_v1_RF_vs_XGB",
	)


def koc_rf_vs_knn() -> pd.DataFrame:
	"""Compare random forest and KNN models on the KOC v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_knn = shared_configuration | {"qsar_method": "knn"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_knn,
		n_repeats=10,
		comparison_name="KOC_v1_RF_vs_KNN",
	)


def koc_rf_feature_selection_comparison() -> pd.DataFrame:
	"""Compare random forest models with and without feature selection on the KOC v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"qsar_method": "rf",
	}
	
	configuration_fs = shared_configuration | {"feature_selection": True}
	configuration_no_fs = shared_configuration | {"feature_selection": False}
	
	comparison = ModelComparison()
	
	result = comparison.compare(
		configuration_fs,
		configuration_no_fs,
		n_repeats=10,
		comparison_name="KOC_v1_RF_FS_vs_No_FS"
	)

	return result


def koc_rf_feature_set_comparisons() -> pd.DataFrame:
	"""Compare random forest models with different feature sets on the KOC v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "KOC v1 modeling",
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"feature_selection": False,
		"write_to_db": False,
		"qsar_method": "rf"
	}

	configurations = []
	for descriptor_set in DESCRIPTOR_SETS:
		configurations.append(shared_configuration | {"descriptor_set_name": DESCRIPTOR_SETS[descriptor_set]})

	comparison = ModelComparison()

	# comparisons = comparison.build_configurations(
	# 	configurations,
	# 	n_repeats=10,
	# )

	comparisons = pd.DataFrame()
	for i in range(len(configurations)):
		for j in range(i + 1, len(configurations)):
			temp = comparison.compare(
				configurations[i],
				configurations[j],
				n_repeats=10,
				comparison_name=f"KOC_v1_RF_{list(DESCRIPTOR_SETS.keys())[i]}_vs_{list(DESCRIPTOR_SETS.keys())[j]}",
			)
			comparisons = pd.concat([comparisons, temp], ignore_index=True)

	return comparisons


def bcf_rf_vs_gcm() -> pd.DataFrame:
	"""Compare random forest and group contribution models on the BCF v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "exp_prop_BCF_v1_modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_gcm = shared_configuration | {"qsar_method": "gcm"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_gcm,
		n_repeats=10,
		comparison_name="BCF_v1_RF_vs_GCM",
	)


def bcf_rf_vs_xgboost() -> pd.DataFrame:
	"""Compare random forest and XGBoost models on the BCF v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "exp_prop_BCF_v1_modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_xgb = shared_configuration | {"qsar_method": "xgb"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_xgb,
		n_repeats=10,
		comparison_name="BCF_v1_RF_vs_XGB",
	)


def bcf_rf_vs_knn() -> pd.DataFrame:
	"""Compare random forest and KNN models on the BCF v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "exp_prop_BCF_v1_modeling",
		"feature_selection": False,
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_knn = shared_configuration | {"qsar_method": "knn"}
	
	comparison = ModelComparison()
	
	return comparison.compare(
		configuration_rf,
		configuration_knn,
		n_repeats=10,
		comparison_name="BCF_v1_RF_vs_KNN",
	)


def bcf_rf_feature_selection_comparison() -> pd.DataFrame:
	"""Compare random forest models with and without feature selection on the BCF v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "exp_prop_BCF_v1_modeling",
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"qsar_method": "rf",
	}
	
	configuration_fs = shared_configuration | {"feature_selection": True}
	configuration_no_fs = shared_configuration | {"feature_selection": False}
	
	comparison = ModelComparison()
	
	result = comparison.compare(
		configuration_fs,
		configuration_no_fs,
		n_repeats=10,
		comparison_name="BCF_v1_RF_FS_vs_No_FS"
	)

	return result


def bcf_feature_set_comparisons() -> pd.DataFrame:
	"""Compare random forest models with and without feature selection on the BCF v1 dataset."""
	shared_configuration: dict[str, object] = {
		"dataset_name": "exp_prop_BCF_v1_modeling",
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"feature_selection": False,
		"write_to_db": False,
		"qsar_method": "rf"
	}

	configurations = []
	for descriptor_set in DESCRIPTOR_SETS:
		configurations.append(shared_configuration | {"descriptor_set_name": DESCRIPTOR_SETS[descriptor_set]})

	comparison = ModelComparison()

	comparisons = pd.DataFrame()
	for i in range(len(configurations)):
		for j in range(i + 1, len(configurations)):
			temp = comparison.compare(
				configurations[i],
				configurations[j],
				n_repeats=10,
				comparison_name=f"BCF_v1_RF_{list(DESCRIPTOR_SETS.keys())[i]}_vs_{list(DESCRIPTOR_SETS.keys())[j]}",
			)
			comparisons = pd.concat([comparisons, temp], ignore_index=True)

	return comparisons


def physchem_rf_vs_gcm() -> pd.DataFrame:
	"""Compare random forest and GCM models on the Physchem v1 datasets."""
	shared_configuration: dict[str, object] = {
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"feature_selection": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_gcm = shared_configuration | {"qsar_method": "gcm"}
	
	comparison = ModelComparison()
	
	results = []
	for prop in PHYSCHEM_PROPERTIES:
		configuration_rf["dataset_name"] = f"{prop} v1 modeling"
		configuration_gcm["dataset_name"] = f"{prop} v1 modeling"

		result = comparison.compare(
			configuration_rf,
			configuration_gcm,
			n_repeats=10,
			comparison_name=f"{prop}_v1_RF_vs_GCM"
		)
		results.append(result)

	return results


def physchem_rf_vs_xgboost() -> pd.DataFrame:
	"""Compare random forest and XGBoost models on the Physchem v1 datasets."""
	shared_configuration: dict[str, object] = {
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"feature_selection": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_xgb = shared_configuration | {"qsar_method": "xgb"}
	
	comparison = ModelComparison()
	
	results = []
	for prop in PHYSCHEM_PROPERTIES:
		configuration_rf["dataset_name"] = f"{prop} v1 modeling"
		configuration_xgb["dataset_name"] = f"{prop} v1 modeling"

		result = comparison.compare(
			configuration_rf,
			configuration_xgb,
			n_repeats=10,
			comparison_name=f"{prop}_v1_RF_vs_XGB"
		)
		results.append(result)

	return results


def physchem_rf_vs_knn() -> pd.DataFrame:
	"""Compare random forest and KNN models on the Physchem v1 datasets."""
	shared_configuration: dict[str, object] = {
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"feature_selection": False,
	}
	
	configuration_rf = shared_configuration | {"qsar_method": "rf"}
	configuration_knn = shared_configuration | {"qsar_method": "knn"}
	
	comparison = ModelComparison()
	
	results = []
	for prop in PHYSCHEM_PROPERTIES:
		configuration_rf["dataset_name"] = f"{prop} v1 modeling"
		configuration_knn["dataset_name"] = f"{prop} v1 modeling"

		result = comparison.compare(
			configuration_rf,
			configuration_knn,
			n_repeats=10,
			comparison_name=f"{prop}_v1_RF_vs_KNN"
		)
		results.append(result)

	return results


def physchem_rf_feature_selection_comparison() -> pd.DataFrame:
	"""Compare random forest models with and without feature selection on the Physchem v1 datasets."""
	shared_configuration: dict[str, object] = {
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"qsar_method": "rf",
	}
	
	configuration_fs = shared_configuration | {"feature_selection": True}
	configuration_no_fs = shared_configuration | {"feature_selection": False}
	
	comparison = ModelComparison()
	
	results = []
	for prop in PHYSCHEM_PROPERTIES:
		configuration_fs["dataset_name"] = f"{prop} v1 modeling"
		configuration_no_fs["dataset_name"] = f"{prop} v1 modeling"

		result = comparison.compare(
			configuration_fs,
			configuration_no_fs,
			n_repeats=5,
			comparison_name=f"{prop}_v1_RF_FS_vs_No_FS"
		)
		results.append(result)

	return results


def physchem_feature_set_comparison() -> pd.DataFrame:
	"""Compare random forest models with and without feature selection on the Physchem v1 datasets."""
	shared_configuration: dict[str, object] = {
		"ad_measure": AD_MEASURES,
		"run_AD": True,
		"write_to_db": False,
		"feature_selection": False,
		"qsar_method": "rf",
	}
	
	comparison = ModelComparison()
	
	comparisons = pd.DataFrame()
	for prop in PHYSCHEM_PROPERTIES:
		configuration = shared_configuration | {"dataset_name": f"{prop} v1 modeling"}
		for i in range(len(DESCRIPTOR_SETS)):
			for j in range(i + 1, len(DESCRIPTOR_SETS)):
				configuration_a = configuration | {"descriptor_set_name": DESCRIPTOR_SETS[list(DESCRIPTOR_SETS.keys())[i]]}
				configuration_b = configuration | {"descriptor_set_name": DESCRIPTOR_SETS[list(DESCRIPTOR_SETS.keys())[j]]}

			result = comparison.compare(
				configuration_a,
				configuration_b,
				n_repeats=10,
				comparison_name=f"{prop}_v1_RF_{list(DESCRIPTOR_SETS.keys())[i]}_vs_{list(DESCRIPTOR_SETS.keys())[j]}"
			)
			comparisons = pd.concat([comparisons, result], ignore_index=True)

	return comparisons


def main():
	# test_model_comparison()

	# koc_rf_vs_gcm()
	# koc_rf_vs_xgboost()
	# koc_rf_vs_knn()
	# koc_rf_feature_selection_comparison()
	# koc_rf_feature_set_comparisons()
	
	# bcf_rf_vs_gcm()
	# bcf_rf_vs_xgboost()
	# bcf_rf_vs_knn()
	# bcf_rf_feature_selection_comparison()
	bcf_feature_set_comparisons()
	
	physchem_rf_vs_gcm()
	physchem_rf_vs_xgboost()
	physchem_rf_vs_knn()
	# physchem_rf_feature_selection_comparison()
	physchem_feature_set_comparison()

if __name__ == "__main__":
	main()