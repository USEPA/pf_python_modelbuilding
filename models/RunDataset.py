"""Run database-backed QSAR model building and persist its outputs."""

import json
import logging
import os
from typing import Optional, Union, List, Dict, Any, Iterator as _Iterator
from io import StringIO
from pathlib import Path
from dataclasses import asdict, dataclass, field, fields

from sqlalchemy import URL, Engine, text, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError, SQLAlchemyError

import pandas as pd
import numpy as np
import math
import time
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

from model_ws_utilities import \
    apply_outlier_filter, \
    call_build_embedding_ga_db, \
    call_build_embedding_importance_from_df, \
    call_do_predictions_from_df, \
    call_build_model_with_preselected_descriptors_from_df
from models.ModelBuilder import Model as _QSARModel

from util.database_utilities import DatabaseUtilities
from util.serialization_compat import serialize_model
from util import predict_constants as pc
import models.db_utilities.dataset_utilities_db as du
from models.db_utilities.raw_exp_data_db import ExpDataGetter
from models.ModelToExcel import ModelToExcel, ModelDataObjects
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics
from models.EmbeddingFromImportance import perform_iterative_recursive_feature_elimination as run_rfe
from models.EmbeddingFromImportance import perform_iterative_sequential_feature_selection as run_sfs
from applicability_domain import applicability_domain_utilities as adu
from models import make_test_plots as mtp

PROJECT_ROOT: Optional[str] = os.getenv("PROJECT_ROOT")

FEATURE_SELECTION_METHOD_GENETIC_ALGORITHM: str = "Genetic algorithm"
FEATURE_SELECTION_METHOD_GROUP_CONTRIBUTION: str = "Group contribution"
FEATURE_SELECTION_METHOD_IMPORTANCE: str = "Importance"

SNAPSHOT_ID: int = 4
DUPLICATE_STRATEGY: str = "id_suffix"


class StaticHelpers:
    """Stateless helpers shared by RunDataset stages."""

    @staticmethod
    def write_prediction_csvs(
        df_pred_test: pd.DataFrame,
        df_pred_ext: Optional[pd.DataFrame],
        folder_path: str,
    ) -> None:
        """Write test and optional external predictions to CSV files."""
        prediction_csv_path = os.path.join(folder_path, f"test set predictions.csv")
        df_pred_test.to_csv(prediction_csv_path, index=False)
        if df_pred_ext is not None:
            prediction_csv_path = os.path.join(folder_path, f"external set predictions.csv")
            df_pred_ext.to_csv(prediction_csv_path, index=False)

    @staticmethod
    def write_plots(
        df_pred_test: pd.DataFrame,
        model: _QSARModel,
        df_pred_cv: Optional[pd.DataFrame],
        folder_path: str,
    ) -> None:
        """Write prediction plots using cross-validation and test predictions."""
        mpsTraining = []
        mpsTest = df_pred_test.to_dict(orient='records')
        if df_pred_cv is not None:
            mpsTraining = df_pred_cv.to_dict(orient='records') # use CV values so the predictions are fair estimates (exp values wont change from training set)
            filePathOutHistogram = os.path.join(folder_path, "histogram.png")
            mtp.generateHistogram2(filePathOutHistogram, model.propertyName, model.unitsModel, mpsTraining, mpsTest, seriesNameTrain="Training Set", seriesNameTest="Test Set")
        
        if not model.is_binary:
            filePathOutScatter = os.path.join(folder_path, "scatter_plot.png")
            title = "Prediction results for " + model.propertyName
            mtp.generateScatterPlot2(filePathOut=filePathOutScatter, title=title, unitName=model.unitsModel, 
                mpsTraining=mpsTraining, mpsTest=mpsTest, 
                seriesNameTrain="Training Set (CV)", seriesNameTest="Test Set")
    
    @staticmethod
    def getEngine(connect_args: Optional[Dict[str, object]] = {}) -> Engine:
        """Create SQLAlchemy engine for PostgreSQL connection.

        Uses environment variables for database credentials:
            DEV_QSAR_USER, DEV_QSAR_PASS, DEV_QSAR_HOST, DEV_QSAR_PORT, DEV_QSAR_DATABASE

        Args:
            connect_args: Additional arguments to pass to URL.create(). Defaults to empty dict.

        Returns:
            Engine: SQLAlchemy engine configured for PostgreSQL connection.
        """
        connect_url = URL.create(
            # drivername='postgresql+psycopg2',
            # username=os.getenv('DEV_QSAR_USER'),
            # password=os.getenv('DEV_QSAR_PASS'),
            # host=os.getenv('DEV_QSAR_HOST', 'localhost'),
            # port=os.getenv('DEV_QSAR_PORT', 5432),
            # database=os.getenv('DEV_QSAR_DATABASE'),
            drivername='postgresql+psycopg2',
            username=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD'),
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB'),
            **connect_args
        )
        engine = create_engine(connect_url, echo=False)

        return engine

    @staticmethod
    def getSession(engine: Optional[Engine] = None) -> Session:
        """Create SQLAlchemy session for database queries.

        Args:
            engine: SQLAlchemy engine instance. If None, creates new engine.

        Returns:
            Session: SQLAlchemy session object for executing queries.
        """
        if engine is None:
            engine = RunDataset.getEngine()
        Session = sessionmaker(bind=engine)
        session = Session()
        return session

    @staticmethod
    def get_current_user_id() -> str:
        """Return the configured user ID or the legacy fallback ID."""
        user_id = os.getenv("USER_ID")
        if user_id is None:
            logging.warning("USER_ID environment variable is not set, falling back to default (tmarti02)")
            user_id = "tmarti02"
        return user_id

    @staticmethod
    def ensure_static_session(session: Session) -> Session:
        """Check a session and replace it after an operational failure."""
        try:
            # cheap no-op to force a DB round trip
            session.execute(text("SELECT 1"))
            return session
        except OperationalError:
            try:
                session.close()
            except Exception:
                pass
            return RunDataset.getSession()
    
    @staticmethod
    def build_output_subfolder(
        params: "Parameters",
        folder_embedding: Optional[Union[str, List[str]]] = None,
        logp_columns: Optional[Union[str, List[str]]] = None,
        include_feature_count: bool = False,
    ) -> str:
        """Build an output folder name from model and run settings."""
        if isinstance(logp_columns, str):
            logp_columns = [logp_columns]

        selected_columns = [c for c in (logp_columns or []) if c]

        if selected_columns:
            subfolder = f"{params.qsar_method}_{'_'.join(selected_columns)}"
        else:
            subfolder = f"{params.qsar_method}_{params.descriptor_set_name}_fs={str(params.feature_selection)}"

        if include_feature_count and hasattr(params, "max_descriptor_count") and params.max_descriptor_count is not None:
            subfolder = f"{subfolder}_max_descriptor_count={str(params.max_descriptor_count)}"
        elif include_feature_count and hasattr(params, "max_features") and params.max_features is not None:
            subfolder = f"{subfolder}_max_features={str(params.max_features)}"

        if getattr(params, "outlier_filter_methods", False) and params.outlier_filter_methods is not None and len(params.outlier_filter_methods) > 0:
            subfolder = f"{subfolder}_outlier_filter_methods={'_'.join(params.outlier_filter_methods)}"

        if folder_embedding is not None:
            subfolder = f"{subfolder}_{folder_embedding}"

        return subfolder

    @staticmethod
    def safe_div(n: Union[int, float], d: Union[int, float, None]) -> float:
        """Divide safely, returning NaN for a zero or missing denominator."""
        try:
            return n / d if (d is not None and d != 0 and not (isinstance(d, float) and math.isnan(d))) else float('nan')
        except ZeroDivisionError:
            return float('nan')

    @staticmethod
    def add_source_chemical_info(df_pred: pd.DataFrame, df_dps: pd.DataFrame) -> pd.DataFrame:
        """Join mapped chemical and experimental-property identifiers."""
        df_pred.rename(columns={'id':'canon_qsar_smiles'}, inplace=True)
        df_pred = df_pred.merge(df_dps, on='canon_qsar_smiles', how='left')
        df_pred.rename(columns={'qsar_exp_prop_property_values_id_first':'exp_prop_id'}, inplace=True)
        return df_pred

    @staticmethod
    def add_log_p_martin_columns(
        df_training: Optional[pd.DataFrame],
        df_prediction: Optional[pd.DataFrame],
        cross_validate: bool,
        df_cv_dict: Optional[Dict[int, Dict[str, pd.DataFrame]]] = None,
        df_external: Optional[pd.DataFrame] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Add pretrained LOGP Martin predictions to each supplied dataframe."""
        import warnings
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            warnings.filterwarnings("ignore", message=".*serialized model.*XGBoost.*")
            warnings.filterwarnings("ignore", message=".*old JSON model.*XGBoost.*")

            model_id = "1069"
            prediction_name = "LOGP_Martin"
            from models.db_utilities.dataset_utilities_db import add_model_prediction_to_df

            def add_columns(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                """Add one model prediction column when a frame is present."""
                if frame is None:
                    return None
                return add_model_prediction_to_df(frame, model_id, prediction_name)

            df_training = add_columns(df_training)
            df_prediction = add_columns(df_prediction)
            df_external = add_columns(df_external)

            if cross_validate:
                for fold in df_cv_dict.values():
                    fold["train"] = add_columns(fold["train"])
                    fold["pred"] = add_columns(fold["pred"])

        return df_training, df_prediction, df_external

    @staticmethod
    def get_identifier(
        unique_identifier: Optional[str],
        test_stats: Dict[str, object],
        ext_stats: Optional[Dict[str, object]],
    ) -> Optional[str]:
        """Return the requested run identifier, optionally based on statistics."""
        if unique_identifier is not None:
            # Fuzzy matching on excel_identifier
            if "stat" in unique_identifier:
                if "ext" in unique_identifier:
                    unique_identifier = "external_stat"
                elif "test" in unique_identifier:
                    unique_identifier = "test_stat"

        # Prepare the unique identifier based on the method specified
        if unique_identifier == "time":
            identifier = int(time.time() * 1000) # time in ms as identifier
            identifier = f"_{identifier}"
        elif unique_identifier == "external_stat" and ext_stats:
            identifier = f"{'RMSE' if ext_stats.get('RMSE_External') else 'BA'}_external_{ext_stats.get('RMSE_External', ext_stats.get('BA_External', '')):.3f}"
        elif unique_identifier == "test_stat" or (unique_identifier == "external_stat" and not ext_stats):
            identifier = f"{'RMSE' if test_stats.get('RMSE_Test') else 'BA'}_test_{test_stats.get('RMSE_Test', test_stats.get('BA_Test', '')):.3f}"
        else:
            identifier = None
        
        return identifier

    @staticmethod
    def feature_selection(
        df_training: pd.DataFrame,
        df_prediction: pd.DataFrame,
        params: "Parameters",
        cv: object = 5,
    ) -> Optional[List[str]]:
        """Select descriptors with the configured method."""
        if params.feature_selection_method == FEATURE_SELECTION_METHOD_GENETIC_ALGORITHM:
            embedding, _ = call_build_embedding_ga_db(
                df_training,
                df_prediction,
                params,
                cv
            )

        elif params.feature_selection_method == FEATURE_SELECTION_METHOD_IMPORTANCE:
            ip = params
            embedding, _ = call_build_embedding_importance_from_df(
                params.qsar_method,
                df_training,
                df_prediction,
                ip.remove_log_p_descriptors,
                ip.remove_fragment_descriptors,
                ip.remove_acnt_descriptors,
                ip.n_threads,
                ip.num_generations,
                ip.use_permutative,
                ip.run_rfe,
                ip.fraction_of_max_importance,
                ip.min_descriptor_count,
                ip.max_descriptor_count,
                ip.use_wards,
                hyperparameter_grid=ip.hyperparameter_grid,
                run_sfs=ip.run_sfs,
                cv=cv,
                descriptor_coefficient=ip.descriptor_coefficient,
                alpha=ip.alpha,
                n_min=ip.n_min,
                n_max=ip.n_max
            )

        elif params.feature_selection_method == FEATURE_SELECTION_METHOD_GROUP_CONTRIBUTION:
            embedding = None  # determine during model building

        else:
            print(f"Invalid qsar_method for feature selection: {params.qsar_method}")
            return None

        return embedding

    def create_cv_dict(
            df: pd.DataFrame,
            n: int = 5,
            mode: str = "kfold",
            seed: Optional[int] = None,
            splitter_kwargs: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[int, Dict[str, pd.DataFrame]]]:
        """Create a cross-validation dictionary that can be saved to params.df_cv_dict to specify a new splitting."""
        if splitter_kwargs is None:
            splitter_kwargs = dict()

        if mode == "kfold":
            split_method = KFold
        elif mode == "stratified":
            split_method = StratifiedKFold
        elif mode == "group":
            split_method = GroupKFold
        elif mode == "time":
            split_method = TimeSeriesSplit
        else:
            logging.error("Invalid mode provided to create_cv_dict, must be one of {'kfold', 'stratified', 'group', 'time'}.")
            return None

        if seed is None:
            cv_splitter = split_method(n_splits=n, **splitter_kwargs)
        else:
            cv_splitter = split_method(n_splits=n, random_state=seed, **splitter_kwargs)

        if "Property" not in df.columns.to_list():
            logging.error("'Property' must be a column in df provided to create_cv_dict.")
            return None
        
        X = df.drop(columns=["Property"], inplace=False)
        y = df.Property

        df_cv_dict = dict()
        for fold, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y)):
            df_train = df[train_idx]
            df_pred = df[test_idx]

            df_cv_dict[fold] = {
                "train": df_train,
                "pred": df_pred
            }

        return df_cv_dict


@dataclass
class Parameters:
    """Shared settings for a single QSAR model configuration."""

    dataset_name: str
    qsar_method: str
    descriptor_set_name: str
    ad_measure: list[str]
    splitting_name: str = "RND_REPRESENTATIVE"
    feature_selection_method: str = FEATURE_SELECTION_METHOD_GROUP_CONTRIBUTION
    hyperparameter_grid: Optional[Dict[str, object]] = None
    feature_selection: bool = False
    remove_log_p_descriptors: bool = False
    n_threads: Optional[int] = 10
    scale_features: bool = False
    use_pmml_pipeline: bool = False
    logp_columns: Optional[List[str]] = None
    outlier_filter_methods: Optional[List[str]] = None
    outlier_filter_columns: Optional[List[str]] = None
    outlier_filter_trim_mode: Optional[str] = None
    outlier_filter_iqr_k: float = 1.5
    outlier_filter_hampel_k: float = 3.0
    outlier_filter_robust_z_thresh: float = 3.5
    outlier_filter_esd_alpha: float = 0.05
    outlier_filter_esd_max_outliers: int = 10

    def to_dict(self) -> Dict[str, object]:
        """Serialize the model configuration to a dictionary."""
        values = asdict(self)
        for name in (
            "outlier_filter_methods",
            "outlier_filter_columns",
            "outlier_filter_trim_mode",
            "outlier_filter_iqr_k",
            "outlier_filter_hampel_k",
            "outlier_filter_robust_z_thresh",
            "outlier_filter_esd_alpha",
            "outlier_filter_esd_max_outliers",
        ):
            values.pop(name, None)
        return values

    @staticmethod
    def set_hyper_parameters(
        qsar_method: str,
        feature_selection: bool,
        descriptor_set_name: str,
        splitting_name: str,
        dataset_name: str,
        ad_measure: List[str],
    ) -> "Parameters":
        """Create method-specific settings with established default grids."""
        common = {
            "dataset_name": dataset_name,
            "qsar_method": qsar_method,
            "descriptor_set_name": descriptor_set_name,
            "splitting_name": splitting_name,
            "feature_selection": feature_selection,
            "ad_measure": ad_measure,
        }
        method = qsar_method.lower()

        if method in {"xgb", "lgb", "rf"}:
            grids = {
                "xgb": {"estimator__booster": ["gbtree"]},
                "lgb": {},
                "rf": {"estimator__max_features": ["sqrt", "log2"]},
            }
            return ParametersImportance(**common, hyperparameter_grid=grids[method])
        if method == "knn":
            return ParametersGeneticAlgorithm(
                **common,
                hyperparameter_grid={"estimator__n_neighbors": [3], "estimator__weights": ["distance"]},
            )
        if method in {"reg", "las"}:
            return ParametersGeneticAlgorithm(**common, hyperparameter_grid={})
        if method == "gcm":
            return ParametersGroupContribution(**common, hyperparameter_grid={})
        if method == "svm":
            return ParametersGeneric(**common, hyperparameter_grid={})
        if method == "huber":
            return ParametersHuber(**common, hyperparameter_grid={})
        if method == "ransac":
            return ParametersRansac(**common, hyperparameter_grid={})
        if method == "theil_sen":
            return ParametersTheilSen(**common, hyperparameter_grid={})
        raise ValueError(f"Unsupported qsar_method: {qsar_method}")


@dataclass
class ParametersImportance(Parameters):
    """Settings for importance-based descriptor selection methods."""

    feature_selection_method: str = FEATURE_SELECTION_METHOD_IMPORTANCE
    n_threads: Optional[int] = 4
    num_generations: int = 1
    use_permutative: bool = True
    use_wards: bool = False
    run_rfe: bool = True
    run_sfs: bool = True
    n_features_to_select: Union[int, str] = "auto"
    min_descriptor_count: int = 30
    max_descriptor_count: int = 40
    n_min: int = 2
    n_max: int = 20
    descriptor_coefficient: float = 0.006
    alpha: float = 0.7
    fraction_of_max_importance: float = field(init=False)
    remove_fragment_descriptors: bool = True
    remove_acnt_descriptors: bool = True

    def __post_init__(self) -> None:
        importance_fractions = {"rf": 0.25, "xgb": 0.03, "lgb": 0.03}
        try:
            self.fraction_of_max_importance = importance_fractions[self.qsar_method.lower()]
        except KeyError as exc:
            raise ValueError(f"Invalid importance method: {self.qsar_method}") from exc


@dataclass
class ParametersGroupContribution(Parameters):
    """Settings for the group-contribution QSAR method."""

    feature_selection: bool = True
    min_count: int = 3


@dataclass
class ParametersGeneticAlgorithm(Parameters):
    """Shared settings for genetic-algorithm QSAR methods."""

    feature_selection_method: str = FEATURE_SELECTION_METHOD_GENETIC_ALGORITHM
    num_generations: int = 100
    num_optimizers: int = 100
    num_jobs: int = 4
    n_threads: Optional[int] = None
    max_length: int = 24
    max_features: int = 25
    descriptor_coefficient: float = 0.006
    alpha: float = 0.7
    threshold: int = 1
    elitism: bool = True
    crossover_probability: float = 0.9
    mutation_probability: float = 0.05
    use_wards: bool = False
    run_rfe: bool = True
    run_sfs: bool = True
    n_features_to_select: Union[int, str] = "auto"
    remove_fragment_descriptors: bool = True
    remove_acnt_descriptors: bool = True


@dataclass
class ParametersHuber(ParametersGeneticAlgorithm):
    """Huber-regression settings."""

    n_threads: Optional[int] = 10
    scale_features: bool = True
    min_descriptor_count: int = 5
    min_count: int = 3


@dataclass
class ParametersTheilSen(ParametersGeneticAlgorithm):
    """Theil-Sen-regression settings."""

    n_threads: Optional[int] = 10
    scale_features: bool = True
    min_descriptor_count: int = 5
    min_count: int = 3


@dataclass
class ParametersRansac(ParametersGeneticAlgorithm):
    """RANSAC-regression settings."""

    n_threads: Optional[int] = 10
    scale_features: bool = True
    min_descriptor_count: int = 5
    min_count: int = 3


@dataclass
class ParametersGeneric(Parameters):
    """Settings for generic QSAR methods without feature selection."""

    feature_selection: bool = False


@dataclass
class RunParams:
    """Execution controls and outputs for one RunDataset invocation."""

    hyperparameters: Parameters
    embedding: Optional[List[str]] = None
    folder_embedding: Optional[str] = None
    cross_validate: bool = True
    run_AD: bool = True
    fs_previous_embedding: bool = True
    add_LOGP_Martin: bool = False
    write_to_db: bool = False
    user: str = ""
    unique_identifier: Optional[str] = None
    append_to_models_folder: str = ""
    subfolder: Optional[str] = None
    save_initial_dfs: bool = False
    include_feature_count_in_subfolder: bool = False
    engine: Optional[Engine] = None
    session: Optional[Session] = None
    current_user_id: str = ""
    cv: object = 5
    is_binary: bool = False
    model: Optional[_QSARModel] = None
    dataset_name_ext: Optional[str] = None
    dataset_description_ext: Optional[str] = None
    cv_stats: Optional[Dict[str, object]] = None
    ext_stats: Optional[Dict[str, object]] = None
    ad_stats: Dict[str, object] = field(default_factory=dict)
    external_ad_stats: Dict[str, object] = field(default_factory=dict)
    stats_dict: Dict[str, Dict[str, object]] = field(default_factory=dict)
    ext_stats_dict: Dict[str, Dict[str, object]] = field(default_factory=dict)
    df_training: Optional[pd.DataFrame] = None
    df_prediction: Optional[pd.DataFrame] = None
    df_prediction_ext: Optional[pd.DataFrame] = None
    df_external: Optional[pd.DataFrame] = None
    df_cv_dict: Optional[Dict[int, Dict[str, pd.DataFrame]]] = None
    df_pred_training: Optional[pd.DataFrame] = None
    df_pred_test: Optional[pd.DataFrame] = None
    df_pred_cv: Optional[pd.DataFrame] = None
    df_pred_ext: Optional[pd.DataFrame] = None
    df_pv: Optional[pd.DataFrame] = None
    df_dps: Optional[pd.DataFrame] = None
    df_dps_ext: Optional[pd.DataFrame] = None
    training_stats: Dict[str, object] = field(default_factory=dict)
    test_stats: Dict[str, object] = field(default_factory=dict)
    folder_path: str = ""
    folder_path_path: Optional[Path] = None
    json_path: str = ""
    detailed_summary_path: str = ""
    results_dict: Dict[str, object] = field(default_factory=dict)


class ModelBuilder:
    """Build QSAR models and calculate their prediction statistics."""

    @staticmethod 
    def predict(
        model: _QSARModel,
        df: pd.DataFrame,
        tag: str,
        is_binary: bool,
    ) -> tuple[pd.DataFrame, Dict[str, object]]:
        """Predict one dataset and calculate its binary or continuous statistics."""
        json_predictions = call_do_predictions_from_df(df, model)
        
        df_predictions = pd.read_json(StringIO(json_predictions), orient="records")
                
        if is_binary:
            stats = calculate_binary_statistics(df_predictions, 0.5, tag)
        else:
            mean_exp_training = df_predictions["exp"].mean()
            stats = calculate_continuous_statistics(df_predictions, mean_exp_training, tag)
    
        return df_predictions, stats
    
    @staticmethod
    def crossvalidate(
        df_cv_dict: Dict[int, Dict[str, pd.DataFrame]],
        params: Parameters,
        embedding: Optional[List[str]],
        is_binary: bool,
    ) -> tuple[pd.DataFrame, Dict[str, object]]:
        """Build predictions and aggregate statistics across the five CV folds."""
        logging.info("Start running CV calculations ...")
    
        def _fold_predictions() -> _Iterator[pd.DataFrame]:
            for i in range(1, 6):
                splittingName = f"RND_REPRESENTATIVE_CV{i}"
                logging.debug(f"Running CV split = {splittingName}")
    
                df_training = df_cv_dict[i]["train"]
                df_prediction = df_cv_dict[i]["pred"]
    
                df_predictions = ModelBuilder.build_and_test_model2(
                    df_training.copy(),
                    df_prediction.copy(),
                    params,
                    embedding
                )
    
                # Add the fold index as a column
                yield df_predictions.assign(cv_fold=i)
    
        df_predictions_all = pd.concat(_fold_predictions(), ignore_index=True)
    
        if is_binary:
            cv_stats = calculate_binary_statistics(
                df_predictions_all, 0.5, "_CV_Training"
            )
            
        else:
            mean_exp_training = df_predictions_all["exp"].mean()
            cv_stats = calculate_continuous_statistics(
                df_predictions_all, mean_exp_training, "_CV_Training"
            )
    
        logging.info("Done running CV calculations")
        return df_predictions_all, cv_stats

    @staticmethod
    def build_and_test_model2(
        df_training: pd.DataFrame,
        df_prediction: pd.DataFrame,
        params: Parameters,
        embedding: Optional[List[str]],
    ) -> pd.DataFrame:
        """Fit a fold model and return predictions for that fold's holdout set."""
        model = call_build_model_with_preselected_descriptors_from_df(
            params,
            df_training,
            df_prediction,
            cv=None,
            descriptor_names_tsv=embedding,
            filterColumnsInBothSets=True
        )
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        return pd.read_json(StringIO(json_predictions), orient="records")

    @staticmethod
    def build_and_test_model(
        df_training: pd.DataFrame,
        df_prediction: pd.DataFrame,
        cv: object,
        params: Parameters,
        embedding: Optional[List[str]],
        is_binary: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object], _QSARModel]:
        """Fit a model and return training/test predictions and their statistics."""
        model = call_build_model_with_preselected_descriptors_from_df(
            params,
            df_training.copy(),
            df_prediction.copy(),
            cv,
            descriptor_names_tsv=embedding,
            filterColumnsInBothSets=True
        )
        
        # generate predictions for test set:
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        json_predictions_training = call_do_predictions_from_df(df_training, model)
        
        df_predictions_test = pd.read_json(StringIO(json_predictions), orient="records")
        df_predictions_training = pd.read_json(StringIO(json_predictions_training), orient="records")
            
        # calculate stats for test set
        
        if is_binary:
            test_stats = calculate_binary_statistics(df_predictions_test, 0.5, "_Test")
            training_stats = calculate_binary_statistics(df_predictions_training, 0.5, "_Training")

        else:
            mean_exp_training = df_training["Property"].mean()
            test_stats = calculate_continuous_statistics(df_predictions_test, mean_exp_training, "_Test")        
            training_stats = calculate_continuous_statistics(df_predictions_training, mean_exp_training, "_Training")
                
        return df_predictions_training, df_predictions_test, training_stats, test_stats, model


class ModelLoader():
    """Persist trained models and their related records."""

    def __init__(self, session: Session) -> None:
        """Bind model-loading operations to an active database session."""
        self.session = session
        self.dbl = DatabaseUtilities(schema="qsar_models", session=self.session)

    def load_model_file(
        self,
        file_path: Union[str, Path],
        username: str,
        fk_model_id: int,
        fk_file_type_id: int,
    ) -> Optional[int]:
        """Read a file and store it as a model-file database record."""
        p = Path(file_path)
        if not p.is_file():
            print(f"File does not exist: {p}")
            return None  # or: raise FileNotFoundError(f"File does not exist: {p}")
    
        try:
            with p.open('rb') as f:
                binary_data = f.read()
                
            if len(binary_data)==0:
                print(f"File length is zero: {p}")
                return None  # or: raise FileNotFoundError(f"File does not exist: {p}")

            record = {
                'created_at': datetime.now(),
                'created_by': username,
                'file': binary_data,
                'updated_at': datetime.now(),
                'updated_by': username,
                'fk_file_type_id': fk_file_type_id,
                'fk_model_id': fk_model_id,
            }
    
            new_id = self.dbl.create_row(table="model_files", record=record)
            return new_id
    
        except Exception as e:
            self.dbl.session.rollback()
            print(f"An error occurred: {e}")
            return None  # or re-raise: rais
    
    def update_model_file(
        self,
        file_path: Union[str, Path],
        username: str,
        fk_model_id: int,
        fk_file_type_id: int,
    ) -> Optional[int]:
        """Replace the stored bytes for a model file."""
        try:
            # Read the image file as binary
            with open(file_path, 'rb') as file:
                binary_data = file.read()
        
            # Data to update
            values = {
                'file': binary_data,
                'updated_at': datetime.now(),
                'updated_by': username,
            }
        
            return self.dbl.update_row(table="model_files", values=values, fk_model_id=fk_model_id, fk_file_type_id=fk_file_type_id)

        except Exception as e:
            self.dbl.session.rollback()
            print(f"An error occurred: {e}")
            return None

    def add_model_statistics(
        self,
        user: str,
        fk_model_id: int,
        stats_dict_name: str,
        model_statistics_dict: Dict[str, object],
        stats_lookup: Dict[str, int],
        created_at: datetime,
        model_statistics_rows: List[Dict[str, object]],
    ) -> None:
        """Append recognized statistics as database-ready rows."""
        if stats_dict_name not in model_statistics_dict:
            print(stats_dict_name, "Skipping loading stats")
            return
    
        stats = model_statistics_dict[stats_dict_name]
    
        for stat_name in stats:
            if "AD" not in stats_dict_name and "Coverage" in stat_name:
                continue
    
            if stat_name in ("ad_measure", "mae_ratio"):
                continue
    
            if stat_name in stats_lookup:
                fk_statistic_id = stats_lookup[stat_name]
                stat_value = stats[stat_name]
    
                # Convert NumPy scalars to native Python types for DB insertion
                if isinstance(stat_value, np.generic):
                    stat_value = stat_value.item()
    
                # Skip missing values                
                if stat_value is None or pd.isna(stat_value):
                    stat_value = None
    
                model_statistics_row = {
                    "statistic_value": stat_value,
                    "fk_model_id": fk_model_id,
                    "fk_statistic_id": fk_statistic_id,
                    "created_by": user,
                    "updated_by": user,
                    "created_at": created_at,
                    "updated_at": created_at
                }
    
                model_statistics_rows.append(model_statistics_row)
    
            else:
                if stat_name not in ("ba_ratio", "Concordance_CV_Training"):
                    print(stat_name, "Skipping loading stat")
    
    def load_stats(self, results: Dict[str, object], user: str, fk_model_id: int) -> None:
        """Load model statistics from a completed run's results payload."""
        stats_rows = self.dbl.get_rows("statistics")
        stats_lookup = {}  # lookup for the fk_statistic_id
        for stat_row in stats_rows:
            stats_lookup[stat_row.name] = stat_row.id
        
        created_at = datetime.now()
        model_statistics_rows = []
        
        dict_list = ["training_stats", "test_stats", "cv_stats", "test_stats_AD", "ext_stats", "ext_stats_AD"]
        
        for dict_name in dict_list: 
            self.add_model_statistics(
                user=user,
                fk_model_id=fk_model_id,
                stats_dict_name=dict_name,
                model_statistics_dict=results["model_statistics"],
                stats_lookup=stats_lookup,
                created_at=created_at,
                model_statistics_rows=model_statistics_rows,
            )
                
        stats_row_ids = self.dbl.create_many("model_statistics", model_statistics_rows)
        
    def divide_array(
        self,
        source: Union[bytes, bytearray],
        chunksize: int = 26214400,
    ) -> Optional[List[Union[bytes, bytearray]]]:
        """Split model bytes into fixed-size chunks, returning None on mismatch."""
        parts = [source[i:i + chunksize] for i in range(0, len(source), chunksize)]
    
        # Optional: mimic the Java prints
        logging.info(f"Size of model bytes={len(source)}")
        logging.info(f"# Parts = {len(parts)}")
    
        total = sum(len(p) for p in parts)
        if total != len(source):
            print(f"byte length mismatch:{total}\t{len(source)}")
            return None
    
        return parts

    def load_model_from_object(
        self,
        user: str,
        model: _QSARModel,
        params: Dict[str, object],
        fk_descriptor_embedding_id: int,
        fk_method_id: int,
        fk_ad_method: int,
    ) -> int:
        """Insert a model record and assign its generated database name."""
        created_at = datetime.now()
                
        model_row = {
            "name":model.modelName,
            "dataset_name":params["dataset_name"],
            "descriptor_set_name":params["descriptor_set_name"],
            "splitting_name":params["splitting_name"],
            "fk_method_id":fk_method_id,  # todo lookup from qsar_method, use general method instead of versioned
            "fk_descriptor_embedding_id":fk_descriptor_embedding_id,
            "fk_source_id":3,  # cheminformatics modules, TODO lookup from sources table
            "fk_ad_method":fk_ad_method,  # lookup from ad_methods table using ad_method_name currently cant have multiple AD methods the way the db is configured
            "hyperparameter_grid":json.dumps(model.hyperparameter_grid),
            "hyperparameters":json.dumps(model.hyperparameters),  # JSON/JSONB column
            "details":model.get_model_description().encode("utf-8"),  # this column is bytes in the database (deprecated) 
            "details_text":model.get_model_description(),  # this will be column to use in future
            "is_public":False,
            "name_ccd":model.modelName,
            "has_qmrf":False,
            "created_by":user,
            "updated_by":user,
            "created_at":created_at,
            "updated_at":created_at,
            "external_dataset_name": model.external_dataset_name
        }

        new_id = self.dbl.create_row(table="models", record=model_row)
        new_name = f"{user}_{new_id}"
        
        # Filters as kwargs (id=new_id)
        updated = self.dbl.update_row(
            table="models",
            values={"name": new_name, "name_ccd":new_name, "updated_by": user, "updated_at": datetime.now()},
            id=new_id
        )
        return new_id        

    def create_model_bytes(self, bytes_list: List[Dict[str, object]]) -> List[int]:
        """Insert serialized model chunks and return their row identifiers."""
        try:
            # Insert records
            result = self.dbl.create_many(table="model_bytes", records=bytes_list)
            # Commit the transaction
            # At this point, success if we reached here without exception
            logging.info("Model bytes loaded")
            return result
        except SQLAlchemyError:
            # Roll back on any DB/SQLAlchemy error
            self.session.rollback()
            # Re-raise or log the error
            raise        

    def load_model_bytes(self, user: str, model: _QSARModel, fk_model_id: int) -> List[int]:
        """Serialize a model and store it in the model-bytes table."""
        model_bytes = serialize_model(model)
        bytes_list = self.divide_array(model_bytes)

        created_at = datetime.now()
        bytes_rows = [
            {
                "fk_model_id": fk_model_id,
                "bytes": chunk,
                "created_by": user,
                "updated_by": user,
                "created_at": created_at,
                "updated_at": created_at,
            }
            for chunk in bytes_list
        ]

        # Insert
        return self.create_model_bytes(bytes_rows)
    
    def load_predictions(
        self,
        user: str,
        set_name: str,
        df: pd.DataFrame,
        fk_model_id: int,
        fk_splitting_id: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> int:
        """Insert prediction rows and any associated AD estimates in batches."""
        if df is None or df.empty:
            logging.info(f"For {set_name} set, no predictions to load.")
            return 0

        required_cols = ["canon_qsar_smiles", "pred"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if not user:
            raise ValueError("A non-empty user must be provided.")
        if fk_model_id is None:
            raise ValueError("fk_model_id must be provided.")

        total_count = 0

        for start in range(0, len(df), chunk_size):
            batch_df = df.iloc[start:start + chunk_size].copy()
            created_at = datetime.now()

            rows_df = pd.DataFrame({
                "canon_qsar_smiles": batch_df["canon_qsar_smiles"].astype(str),
                "qsar_predicted_value": pd.to_numeric(batch_df["pred"], errors="coerce"),
            })

            if "cv_fold" in batch_df.columns:
                cv = pd.to_numeric(batch_df["cv_fold"], errors="coerce")
                if cv.isna().any():
                    raise ValueError("cv_fold contains null or non-numeric values.")
                rows_df["fk_splitting_id"] = cv.astype(int) + 1
            else:
                if fk_splitting_id is None:
                    raise ValueError("fk_splitting_id must be provided when cv_fold is not present.")
                rows_df["fk_splitting_id"] = fk_splitting_id

            rows_df["fk_model_id"] = fk_model_id
            rows_df["created_by"] = user
            rows_df["updated_by"] = user
            rows_df["created_at"] = created_at
            rows_df["updated_at"] = created_at

            prediction_records = rows_df.replace({np.nan: None}).to_dict(orient="records")

            prediction_ids = self.dbl.create_many(
                table="predictions",
                records=prediction_records,
                commit=False,
            )

            total_count += len(prediction_ids)
            logging.info(
                f"For {set_name} set, loaded prediction batch {start}..{start + len(batch_df) - 1} "
                f"({len(prediction_ids)} rows)"
            )

            ad_cols = [c for c in batch_df.columns if c.startswith("AD_")]
            if ad_cols:
                ad_df = self.build_predictions_ad_estimates_df(
                    df=batch_df,
                    prediction_ids=prediction_ids,
                    user=user,
                )

                if ad_df is not None and not ad_df.empty:
                    ad_records = ad_df.replace({np.nan: None}).to_dict(orient="records")
                    self.dbl.create_many(
                        table="predictions_ad_estimates",
                        records=ad_records,
                        commit=False,
                    )

            self.dbl.session.commit()

        logging.info(f"For {set_name} set, # predictions loaded: {total_count}")
        return total_count

    def build_predictions_ad_estimates_df(
        self,
        df: pd.DataFrame,
        prediction_ids: list[int],
        user: str,
    ) -> pd.DataFrame:
        """Build one AD-estimate row per prediction and AD method."""
        if df is None or df.empty:
            return pd.DataFrame()

        if len(df) != len(prediction_ids):
            raise ValueError(
                f"Row mismatch: df has {len(df)} rows but prediction_ids has {len(prediction_ids)} ids"
            )

        created_at = datetime.now()
        rows = []

        # Only treat AD flags as AD methods
        ad_cols = [c for c in df.columns if c.startswith("AD_")]
        if not ad_cols:
            return pd.DataFrame()

        for idx, (_, row) in enumerate(df.iterrows()):
            fk_predictions_id = prediction_ids[idx]

            for ad_col in ad_cols:
                # Convert column name back to DB method name
                db_ad_name = self.normalize_ad_method_name(ad_col)
                fk_ad_method_id = self.get_ad_method_id_by_name(ad_col)
                if fk_ad_method_id is None:
                    continue

                # Raw applicability value, if available
                raw_value = row.get(f"{db_ad_name}_value", None)
                if raw_value is None or pd.isna(raw_value):
                    raw_value = row.get(ad_col, None)

                # Conclusion
                conclusion_val = row.get(f"{db_ad_name}_conclusion", None)
                if conclusion_val is None or pd.isna(conclusion_val):
                    conclusion_val = row.get(ad_col, None)

                if isinstance(conclusion_val, (bool, np.bool_)):
                    conclusion_text = "TRUE" if conclusion_val else "FALSE"
                elif conclusion_val is None or pd.isna(conclusion_val):
                    conclusion_text = None
                else:
                    conclusion_text = str(conclusion_val)

                # Reasoning
                reasoning = row.get(f"{db_ad_name}_reasoning", None)
                if reasoning is None or pd.isna(reasoning):
                    cutoff = row.get(f"{db_ad_name}_cutoff", None)
                    if raw_value is not None and not pd.isna(raw_value) and cutoff is not None and not pd.isna(cutoff):
                        reasoning = f"Applicability value {raw_value} compared to cutoff {cutoff} for AD '{db_ad_name}'."
                    else:
                        reasoning = f"Applicability domain evaluation for AD '{db_ad_name}'."

                rows.append({
                    "created_at": created_at,
                    "created_by": user,
                    "updated_at": created_at,
                    "updated_by": user,
                    "applicability_value": None if raw_value is None or pd.isna(raw_value) else float(raw_value),
                    "conclusion": conclusion_text,
                    "reasoning": reasoning,
                    "fk_ad_method_id": fk_ad_method_id,
                    "fk_predictions_id": fk_predictions_id,
                })

        return pd.DataFrame(rows)

    def normalize_ad_method_name(self, ad_name: str) -> str:
        """Convert an AD dataframe column name to its database display name."""
        if ad_name.startswith("AD_"):
            ad_name = ad_name[3:]
        return ad_name.replace("_", " ").strip()

    def get_ad_method_id_by_name(self, ad_name: str) -> Optional[int]:
        """Look up an applicability-domain method identifier by name."""
        db_name = self.normalize_ad_method_name(ad_name)
        row_ad_method = self.dbl.get_row("ad_methods", name=db_name)
        if row_ad_method is None:
            logging.warning(f"Could not find ad_methods row for name='{db_name}' (from '{ad_name}')")
            return None
        return row_ad_method.id

    def create_method(self, user: str, isBinary: bool, fullMethodName: str) -> int:
        """Create a methods-table entry for the current QSAR model."""
        logging.info(f"Creating method")
        created_at = datetime.now()
        record = {
            "created_by":user,
            "updated_by":user,
            "created_at":created_at,
            "updated_at":created_at,
            "description":"todo",
            "is_binary":isBinary,
            "name":fullMethodName,
            "description_url":"todo"}
        new_id = self.dbl.create_row("methods", record)
        return new_id

    def load_model(
        self,
        user: str,
        model: _QSARModel,
        results: Dict[str, object],
        df_pred_training: pd.DataFrame,
        df_pred_test: pd.DataFrame,
        df_pred_cv: Optional[pd.DataFrame],
        folder_path: str,
        df_pred_external: Optional[pd.DataFrame] = None,
    ) -> None:
        """Persist model data, predictions, statistics, and plot files."""
        params = results["params"]

        # ---- get fk_descriptor_embedding_id ----
        embedding_tsv = "\t".join(results["model_details"]["embedding"])
        
        # print(embedding_tsv)
        embedding_row = self.dbl.get_row("descriptor_embeddings", embedding_tsv=embedding_tsv, dataset_name=params["dataset_name"])
        if embedding_row is None:
            fk_descriptor_embedding_id = self.create_descriptor_embedding_from_params(user, results)
            logging.info(f"descriptor_embedding created: {fk_descriptor_embedding_id}")
        else:
            fk_descriptor_embedding_id = embedding_row.id
            logging.info(f"descriptor_embedding in database: {fk_descriptor_embedding_id}")
        
        # fk_descriptor_embedding_id=281
        if fk_descriptor_embedding_id is None:
            logging.error("Cant create embedding")
            return
        
        # ---- get fk_method_id ----
        isBinary = results["model_details"]["is_binary"]
        methodName = results["model_details"]["qsar_method"]
        if isBinary:
            classOrRegr = "classifier"
        else:
            classOrRegr = "regressor"
        fullMethodName = methodName + "_" + classOrRegr
                                
        row_method = self.dbl.get_row("methods", name=fullMethodName)
        if row_method is not None:
            fk_method_id = row_method.id
        else:
            fk_method_id = self.create_method(user, isBinary, fullMethodName)
            logging.info(f"Created new method with id: {fk_method_id}")
            
        logging.info(f"fk_method_id:{fk_method_id}")
        
        # ---- get fk_ad_method_id ----
        ad_measure = " and ".join(params["ad_measure"])
        row_ad_method = self.dbl.get_row("ad_methods", name=ad_measure)
        if row_ad_method is not None:
            fk_ad_method = row_ad_method.id
            logging.info(f"fk_ad_method:{fk_ad_method}")
        else:
            logging.error(f"Cant find fk for ad_measure={ad_measure}")
            return
        
        # ---- store model into the models table ----
        fk_model_id = self.load_model_from_object(
            user,
            model,
            params,
            fk_descriptor_embedding_id,
            fk_method_id,
            fk_ad_method
        )
        
        if fk_model_id is None:
            logging.error(f"Cant create model")
            return
        logging.info(f"fk_model_id:{fk_model_id}")
        
        model.modelId = fk_model_id
        
        results["model_details"]["modelId"] = fk_model_id
        
        # ---- store model_bytes into the model_bytes table:----
        self.load_model_bytes(user, model, fk_model_id)
        
        # ---- store model_statistics into the model_statistics table:----        
        self.load_stats(results, user, fk_model_id)
        
        # ---- store predictions into prediction table ----        
        self.load_predictions(
            user,
            "training",
            df_pred_training,
            fk_model_id,
            fk_splitting_id=1,
        )
        self.load_predictions(
            user,
            "test",
            df_pred_test,
            fk_model_id,
            fk_splitting_id=1,
        )
        self.load_predictions(
            user,
            "training cv",
            df_pred_cv,
            fk_model_id,
        )
        if df_pred_external is not None:
            self.load_predictions(
                user,
                "external",
                df_pred_external,
                fk_model_id,
                fk_splitting_id=43,
            )
        
        # ---- store plots in model_files table ----
        if not isBinary:             
            filePathOutScatter = os.path.join(folder_path, "scatter_plot.png")
            image_id = self.load_model_file(filePathOutScatter, user, fk_model_id, 3)
            logging.info(f"Scatter plot loaded to db with id: {image_id}")
        
        filePathOutHistogram = os.path.join(folder_path, "histogram.png")
        image_id = self.load_model_file(filePathOutHistogram, user, fk_model_id, 4)
        logging.info(f"Histogram plot loaded to db with id: {image_id}")
        
    def create_descriptor_embedding_from_params(self, user: str, results: Dict[str, object]) -> int:
        """Create a descriptor-embedding record from serialized run results."""
        params = results["params"]

        epoch_ms = time.time_ns() // 1_000_000
        created_at = datetime.now()
        
        name_cols = [params["dataset_name"], params["descriptor_set_name"], str(epoch_ms)]
        embedding_name = "_".join(name_cols)  # give it a unique name
        
        descriptor_embedding = {
            "name": embedding_name,
            "description": "Genetic algorithm with RFE and SFS",
            "dataset_name": params["dataset_name"],
            "descriptor_set_name": params["descriptor_set_name"],
            "qsar_method": params["qsar_method"],
            "splitting_name": params["splitting_name"],
            "embedding_tsv": "\t".join(results["model_details"]["embedding"]),
            "importance_tsv": "Not used",
            "created_by": user,
            "updated_by": user,
            "created_at": created_at,
            "updated_at": created_at,
        }
        
        new_id = self.dbl.create_row(table="descriptor_embeddings", record=descriptor_embedding)
        return new_id


class RunDataset:
    """Coordinate dataset retrieval, model building, and result persistence."""

    def generate_params(
        self,
        dataset_name: Optional[str] = None,
        qsar_method: Optional[str] = None,
        embedding: Optional[List[str]] = None,
        folder_embedding: Optional[str] = None,
        cross_validate: bool = True,
        run_AD: bool = True,
        feature_selection: Optional[bool] = None,
        fs_previous_embedding: bool = True,
        hyperparameters: Optional[Union[Parameters, Dict[str, object]]] = None,
        descriptor_set_name: Optional[str] = None,
        splitting_name: Optional[str] = None,
        ad_measure: Optional[List[str]] = None,
        add_LOGP_Martin: bool = False,
        logp_columns: Optional[Union[str, List[str]]] = None,
        write_to_db: bool = False,
        user: str = "",
        unique_identifier: Optional[str] = None,
        append_to_models_folder: str = "",
        subfolder: Optional[str] = None,
        save_initial_dfs: bool = False,
        outlier_filter_methods: Optional[List[str]] = None,
        include_feature_count_in_subfolder: bool = False,
        engine: Optional[Engine] = None,
        session: Optional[Session] = None,
        df_training: Optional[pd.DataFrame] = None,
        df_prediction: Optional[pd.DataFrame] = None,
        df_external: Optional[pd.DataFrame] = None,
        dataset_name_ext: Optional[str] = None,
    ) -> RunParams:
        """Create run state and resolve model configuration in one object.

        Args:
            dataset_name: Dataset to build a model for.
            qsar_method: QSAR method name used to choose default parameters.
            embedding: Optional preselected descriptor names.
            folder_embedding: Existing results folder supplying a descriptor embedding.
            cross_validate: Whether to calculate predictions across CV folds.
            run_AD: Whether to calculate applicability-domain estimates.
            feature_selection: Optional override for the model's feature-selection setting.
            fs_previous_embedding: Whether to refine a supplied embedding after model fitting.
            hyperparameters: A method-specific ``Parameters`` instance or a
                dictionary of model tuning fields to override on generated defaults.
            descriptor_set_name: Descriptor set used by the dataset.
            splitting_name: Database split used to retrieve training and test records.
            ad_measure: Applicability-domain measures to calculate.
            add_LOGP_Martin: Whether to add pretrained LOGP Martin predictions.
            logp_columns: Optional LOGP columns included in output naming and model settings.
            write_to_db: Whether to persist the completed model to the database.
            user: User identifier used for output and database records.
            unique_identifier: Optional unique or statistic-derived output suffix.
            append_to_models_folder: Optional suffix added to the models directory.
            subfolder: Explicit output subfolder, overriding generated naming.
            save_initial_dfs: Whether to save initial intermediate dataframes.
            outlier_filter_methods: Optional methods to set on the hyperparameter object.
            include_feature_count_in_subfolder: Include feature limits in output naming.
            engine: Optional existing SQLAlchemy engine.
            session: Optional existing SQLAlchemy session.
            df_training: Optional existing DataFrame for the training set.
            df_prediction: Optional existing DataFrame for the prediction set.
            df_external: Optional existing DataFrame for the external prediction set.
            dataset_name_ext: Optional existing name for the external prediction set.

        Returns:
            RunParams containing the hyperparameters, run controls, and runtime state.

        A dictionary may override method-specific tuning fields. Pass dataset,
        method, split, AD, and other run configuration through their dedicated
        arguments. A supplied ``Parameters`` instance remains authoritative;
        conflicting configuration arguments raise instead of silently overriding it.
        """
        configured_values = {
            "dataset_name": dataset_name,
            "qsar_method": qsar_method,
            "descriptor_set_name": descriptor_set_name,
            "splitting_name": splitting_name,
            "feature_selection": feature_selection,
            "ad_measure": ad_measure,
        }
        hyperparameter_overrides: Dict[str, object] = {}
        if isinstance(hyperparameters, dict):
            hyperparameter_overrides = hyperparameters
            hyperparameters = None

        if hyperparameters is None:
            if dataset_name is None or qsar_method is None:
                raise ValueError("dataset_name and qsar_method are required without hyperparameters")

            descriptor_set_name = descriptor_set_name or "WebTEST-default"
            splitting_name = splitting_name or "RND_REPRESENTATIVE"
            if feature_selection is None:
                feature_selection = qsar_method not in {"gcm", "svm"}
            if qsar_method in {"gcm", "svm"}:
                feature_selection = False

            if ad_measure is None:
                ad_measure = [
                    pc.Applicability_Domain_TEST_Embedding_Euclidean,
                    pc.Applicability_Domain_TEST_All_Descriptors_Euclidean,
                ]
            else:
                ad_measure = list(ad_measure)
            if descriptor_set_name == "WebTEST-default" and pc.Applicability_Domain_TEST_Fragment_Counts not in ad_measure:
                ad_measure.append(pc.Applicability_Domain_TEST_Fragment_Counts)
            elif descriptor_set_name != "WebTEST-default" and pc.Applicability_Domain_TEST_Fragment_Counts in ad_measure:
                ad_measure.remove(pc.Applicability_Domain_TEST_Fragment_Counts)

            hyperparameters = Parameters.set_hyper_parameters(
                qsar_method=qsar_method,
                feature_selection=feature_selection,
                descriptor_set_name=descriptor_set_name,
                splitting_name=splitting_name,
                dataset_name=dataset_name,
                ad_measure=ad_measure,
            )
        else:
            for name, supplied_value in configured_values.items():
                if supplied_value is not None and supplied_value != getattr(hyperparameters, name):
                    raise ValueError(f"{name} conflicts with hyperparameters.{name}")

            if hyperparameters.qsar_method in {"gcm", "svm"}:
                hyperparameters.feature_selection = False

        if hyperparameter_overrides:
            parameter_fields = {
                parameter_field.name: parameter_field
                for parameter_field in fields(hyperparameters)
                if parameter_field.init
            }
            configuration_fields = {
                "dataset_name",
                "qsar_method",
                "descriptor_set_name",
                "splitting_name",
                "feature_selection",
                "ad_measure",
                "logp_columns",
                "outlier_filter_methods",
            }
            unsupported_overrides = set(hyperparameter_overrides) - set(parameter_fields)
            configuration_overrides = set(hyperparameter_overrides) & configuration_fields
            if unsupported_overrides or configuration_overrides:
                invalid_names = sorted(unsupported_overrides | configuration_overrides)
                invalid_message = ", ".join(invalid_names)
                raise ValueError(
                    f"Unsupported hyperparameter override(s): {invalid_message}. "
                    "Pass run configuration through generate_params arguments."
                )

            for name, value in hyperparameter_overrides.items():
                setattr(hyperparameters, name, value)

        if hyperparameters.qsar_method in {"gcm", "svm"}:
            hyperparameters.feature_selection = False

        if logp_columns is not None:
            normalized_logp_columns = [logp_columns] if isinstance(logp_columns, str) else logp_columns
            if hyperparameters.logp_columns is not None and hyperparameters.logp_columns != normalized_logp_columns:
                raise ValueError("logp_columns conflicts with hyperparameters.logp_columns")
            hyperparameters.logp_columns = normalized_logp_columns

        if outlier_filter_methods is not None:
            if hyperparameters.outlier_filter_methods is not None and hyperparameters.outlier_filter_methods != outlier_filter_methods:
                raise ValueError("outlier_filter_methods conflicts with hyperparameters.outlier_filter_methods")
            hyperparameters.outlier_filter_methods = outlier_filter_methods

        if df_external is not None:
            df_prediction_ext = df_external.copy()
        else:
            df_prediction_ext = df_external

        params = RunParams(
            embedding=embedding,
            folder_embedding=folder_embedding,
            cross_validate=cross_validate,
            run_AD=run_AD,
            fs_previous_embedding=fs_previous_embedding,
            hyperparameters=hyperparameters,
            add_LOGP_Martin=add_LOGP_Martin,
            write_to_db=write_to_db,
            user=user,
            unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder,
            subfolder=subfolder,
            save_initial_dfs=save_initial_dfs,
            include_feature_count_in_subfolder=include_feature_count_in_subfolder,
            engine=engine,
            session=session,
            df_training=df_training,
            df_prediction=df_prediction,
            df_prediction_ext=df_prediction_ext,
            df_external=df_external,
            dataset_name_ext=dataset_name_ext
        )
        self.post_init(params)
        return params

    def post_init(self, params: RunParams) -> None:
        """Initialize run infrastructure and derive output paths."""
        # Initialize database engine and session
        params.engine = params.engine or StaticHelpers.getEngine()
        params.session = params.session or StaticHelpers.getSession(params.engine)
        params.current_user_id = StaticHelpers.get_current_user_id()
        if not params.user:
            params.user = params.current_user_id

        # Set embedding and feature selection based on provided parameters
        if params.embedding is not None:
            params.folder_embedding = "custom"
            params.hyperparameters.feature_selection = False
        elif params.folder_embedding is not None:
            params.hyperparameters.feature_selection = False
            file_name_embedding = "results.json"
            file_path_embedding = os.path.join(PROJECT_ROOT, "data/models", params.hyperparameters.dataset_name, params.folder_embedding, file_name_embedding)
            
            with open(file_path_embedding, "r", encoding="utf-8") as f:
                results = json.load(f)
                params.embedding = results["model_details"]["embedding"]
        else:
            params.fs_previous_embedding = False

        if params.hyperparameters.outlier_filter_methods:
            if params.hyperparameters.outlier_filter_columns is None:
                params.hyperparameters.outlier_filter_columns = ["Property"]
            if params.hyperparameters.outlier_filter_trim_mode is None:
                params.hyperparameters.outlier_filter_trim_mode = "remove"

        # Set external dataset
        params.df_prediction_ext = getattr(params, "df_prediction_ext", None)
        params.dataset_name_ext = getattr(params, "dataset_name_ext", None)

        if params.dataset_name_ext is None:
            if params.hyperparameters.dataset_name == 'KOC v1 modeling' or params.hyperparameters.dataset_name == 'KOC v2 modeling':
                params.dataset_name_ext = 'KOC v2 external'
            elif params.hyperparameters.dataset_name == 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling':
                params.dataset_name_ext = 'QSAR_Toolbox_96HR_Fish_LC50_v3 modeling'
            elif params.hyperparameters.dataset_name == 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling':
                params.dataset_name_ext = 'QSAR_Toolbox_96HR_Fish_LC50_v3b modeling'
            elif params.hyperparameters.dataset_name == "exp_prop_BCF_v1_modeling":
                params.dataset_name_ext = "exp_prop_BCF_v1_external"
            elif params.hyperparameters.dataset_name == 'exp_prop_RBIODEG_RIFM_CHEMREG':
                params.dataset_name_ext = 'exp_prop_RBIODEG_301F v1 modeling'

        params.subfolder = params.subfolder if params.subfolder is not None else StaticHelpers.build_output_subfolder(
            params.hyperparameters,
            params.folder_embedding,
            params.hyperparameters.logp_columns,
            include_feature_count=params.include_feature_count_in_subfolder
        )

        path_segments = [PROJECT_ROOT, "data", "models" + params.append_to_models_folder, params.hyperparameters.dataset_name, params.subfolder]
        params.folder_path = os.path.join(*path_segments)
        params.folder_path_path = Path(params.folder_path)
        params.folder_path_path.mkdir(parents=True, exist_ok=True)

    def ensure_session(self, params: RunParams) -> Session:
        """Return a working database session, replacing stale sessions."""
        session = params.session
        try:
            # cheap no-op to force a DB round trip
            session.execute(text("SELECT 1"))
            return session
        except OperationalError:
            try:
                session.close()
            except Exception:
                pass
            session = StaticHelpers.getSession(params.engine)
            params.session = session
            return session
    
    def runAD(
        self,
        df_training: pd.DataFrame,
        df_prediction: pd.DataFrame,
        params: RunParams,
        embedding: Optional[List[str]],
        df_predictions: pd.DataFrame,
        ad_measure: str,
        stats_dict: Dict[str, Dict[str, object]],
        is_binary: bool = False,
        is_external: bool = False,
    ) -> pd.DataFrame:
        """
        Note: This configuration for runAD should be treated as the preferred run-pattern, and anywhere the AD must be recalculated
        should be done in accordance with this method's call pattern and behavior within this file.
        """
        if is_external:
            tag = pc.TAG_EXTERNAL
            stat_insert = "External"
        else:
            tag = pc.TAG_TEST
            stat_insert = "Test"

        df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
            train_df=df_training.copy(), test_df=df_prediction.copy(),
            remove_log_p=params.hyperparameters.remove_log_p_descriptors,
            embedding=embedding, applicability_domain=ad_measure,
            filterColumnsInBothSets=False,
            returnTrainingAD=False
        )
        
        # AD coverage (fraction of test rows inside AD)
        count = df_ad_output.shape[0]
        count_inside = df_ad_output['AD'].eq(True).sum()
        coverage_ad = count_inside / count if count else float('nan')
                
        colAD = "AD_" + ad_measure.replace(" ", "_")
        
        # Append AD flag to predictions
        df_predictions = (
            df_predictions
            .merge(
                df_ad_output.rename(columns={'idTest': 'id'})[['id', 'AD']],
                on='id', how='left'
            )
            .rename(columns={'AD': colAD})
        )

        # Subsets: inside and outside AD, only exp/pred columns are needed by metrics
        inside_mask = df_predictions[colAD].eq(True)
        outside_mask = df_predictions[colAD].eq(False)
        df_inside = df_predictions.loc[inside_mask, ['exp', 'pred']].copy()
        df_outside = df_predictions.loc[outside_mask, ['exp', 'pred']].copy()

        # Compute statistics using the provided functions
        if is_binary:
            cutoff = getattr(params.hyperparameters, 'cutoff', getattr(params.hyperparameters, 'classification_cutoff', 0.5))

            stats_inside = calculate_binary_statistics(df_inside, cutoff=cutoff, tag=tag)
            stats_outside = calculate_binary_statistics(df_outside, cutoff=cutoff, tag=tag)
            
            ba_inside = stats_inside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
            ba_outside = stats_outside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
            ba_ratio = StaticHelpers.safe_div(ba_inside, ba_outside)

            stats = {
                "ad_measure": ad_measure,
                f"BA_{stat_insert}_inside_AD": ba_inside,
                f"BA_{stat_insert}_outside_AD": ba_outside,
                "ba_ratio": ba_ratio,
                f"Coverage_{stat_insert}": coverage_ad
            }

        else:
            # Mean of training exp for Q2/R2 in continuous stats
            mean_exp_training = float('nan')  # not needed for calculating MAE inside and outside

            # calculate_continuous_statistics raises if no valid predictions; handle gracefully
            try:
                stats_inside = calculate_continuous_statistics(df_inside, mean_exp_training=mean_exp_training, tag=tag)
                mae_inside = stats_inside.get(pc.MAE + tag, float('nan'))
            except Exception:
                mae_inside = float('nan')

            try:
                stats_outside = calculate_continuous_statistics(df_outside, mean_exp_training=mean_exp_training, tag=tag)
                mae_outside = stats_outside.get(pc.MAE + tag, float('nan'))
            except Exception:
                mae_outside = float('nan')

            mae_ratio = StaticHelpers.safe_div(mae_outside, mae_inside)

            stats = {
                "ad_measure": ad_measure,
                f"MAE_{stat_insert}_inside_AD": mae_inside,
                f"MAE_{stat_insert}_outside_AD": mae_outside,
                f"mae_ratio": mae_ratio,
                f"Coverage_{stat_insert}": coverage_ad
            }

        stats_dict[f"{ad_measure}{' External' if is_external else ''}"] = stats
        return df_predictions

    def query_dataframes(self, params: RunParams) -> None:
        """Load training, test, and configured external datasets."""
        logging.info("Querying dataframes from database...")

        self.ensure_session(params)
        if params.df_training is None or params.df_prediction is None:
            df_training, df_prediction = du.get_training_prediction_instances(
                params.session,
                params.hyperparameters.dataset_name,
                params.hyperparameters.descriptor_set_name,
                params.hyperparameters.splitting_name,
            )
            df_training = params.df_training or df_training
            df_prediction = params.df_prediction or df_prediction
        
        if df_training is None or df_prediction is None:
            logging.error("Failed to retrieve training or prediction dataframes from the database")
            return

        params.df_training = df_training.copy()
        params.df_prediction = df_prediction.copy()

        # Query external dataset if specified
        if params.df_external is None:
            if params.dataset_name_ext is not None:
                self.ensure_session(params)
                params.dataset_description_ext = du.get_dataset_details(params.session, params.dataset_name_ext).get('dataset_description')

                self.ensure_session(params)
                df_prediction_ext = du.get_instances_excluding(
                    params.session,
                    params.dataset_name_ext,
                    params.hyperparameters.dataset_name,
                    params.hyperparameters.descriptor_set_name,
                )

                params.df_prediction_ext = df_prediction_ext.copy()
                params.df_external = params.df_prediction_ext.copy()
        else:
            self.ensure_session(params)
            params.dataset_description_ext = du.get_dataset_details(params.session, params.dataset_name_ext).get("dataset_description")

        logging.info("Dataframes queried successfully")

    def set_is_binary(self, params: RunParams) -> None:
        """Determine whether the target contains only binary labels."""
        if params.df_training is None:
            logging.error("Training dataframe is not available. Please run query_dataframes() first.")
            return

        s = params.df_training.iloc[:, 1]  # Assuming the second column is the target variable
        params.is_binary = s.isin([0, 1]).all()

    def apply_outlier_filtering(self, params: RunParams) -> None:
        """Apply the configured outlier filters to training data."""
        if params.hyperparameters.outlier_filter_methods:
            logging.info(
                f"Applying outlier filter: methods={params.hyperparameters.outlier_filter_methods}, "
                f"columns={params.hyperparameters.outlier_filter_columns}, mode={params.hyperparameters.outlier_filter_trim_mode}"
            )
        
            df_training_before = params.df_training.copy()
        
            params.df_training, _ = apply_outlier_filter(
                params.df_training,
                columns=params.hyperparameters.outlier_filter_columns,
                methods=params.hyperparameters.outlier_filter_methods,
                target_col="Property",
                trim_mode=params.hyperparameters.outlier_filter_trim_mode,
                iqr_k=getattr(params.hyperparameters, "outlier_filter_iqr_k", 1.5),
                hampel_k=getattr(params.hyperparameters, "outlier_filter_hampel_k", 3.0),
                robust_z_thresh=getattr(params.hyperparameters, "outlier_filter_robust_z_thresh", 3.5),
                esd_alpha=getattr(params.hyperparameters, "outlier_filter_esd_alpha", 0.05),
                esd_max_outliers=getattr(params.hyperparameters, "outlier_filter_esd_max_outliers", 10),
            )
        
            logging.info(
                f"Outlier filter removed {len(df_training_before) - len(params.df_training)} rows "
                f"out of {len(df_training_before)}"
            )

    def setup_cross_validation(self, params: RunParams) -> None:
        """Load CV folds and prepare the cross-validation splitter."""
        if getattr(params, "df_cv_dict", False):
            # If a cv dict is provided, use that
            # Format of a df_cv_dict is as follows:
            #   {
            #       fold_num (int): {
            #                           "train": df_training   (DataFrame),
            #                           "pred":  df_prediction (DataFrame)
            #                       }
            #   }
            _, _, params.cv, _ = du.make_cv_for_base_training(params.df_training, params.df_cv_dict, "ID", "Property")

        if params.cross_validate:
            self.ensure_session(params)
            params.df_cv_dict = du.get_training_cv_instances(
                params.session,
                params.hyperparameters.dataset_name,
                params.hyperparameters.descriptor_set_name,
            )

            _, _, params.cv, _ = du.make_cv_for_base_training(params.df_training, params.df_cv_dict, "ID", "Property")
        else:
            params.cv = 5

    def setup_logp_martin(self, params: RunParams) -> None:
        """Add pretrained LOGP Martin predictions when requested."""
        # Add LOGP Martin columns if specified
        if params.add_LOGP_Martin:
            params.df_training, params.df_prediction, df_prediction_ext = StaticHelpers.add_log_p_martin_columns(
                params.df_training,
                params.df_prediction,
                params.cross_validate,
                params.df_cv_dict,
                df_external=params.df_prediction_ext
            )
            params.df_prediction_ext = df_prediction_ext.copy() if df_prediction_ext is not None else None
            params.df_external = params.df_prediction_ext
    
    def run_feature_selection(self, params: RunParams) -> None:
        """Select descriptors before fitting when feature selection is enabled."""
        if params.hyperparameters.feature_selection:
            params.embedding = StaticHelpers.feature_selection(
                params.df_training,
                params.df_prediction,
                params.hyperparameters,
                params.cv
            )

            if params.hyperparameters.qsar_method == "huber":
                original_feature_count = params.df_training.shape[1] - 2  # adjust if needed for ID/Property columns
                selected_feature_count = len(params.embedding) if params.embedding is not None else 0
                
                # print(f"Huber feature selection: {selected_feature_count} selected out of {original_feature_count}")
                
                if params.embedding is None or selected_feature_count >= original_feature_count:
                    raise ValueError(
                        f"Huber feature selection did not reduce the feature set. "
                        f"Selected {selected_feature_count} features from {original_feature_count}."
                    )

    def run_model_building(self, params: RunParams) -> None:
        """Fit the configured model and calculate training/test statistics."""
        params.df_pred_training, params.df_pred_test, params.training_stats, params.test_stats, params.model = ModelBuilder.build_and_test_model(
            params.df_training,
            params.df_prediction,
            params.cv,
            params.hyperparameters,
            params.embedding,
            params.is_binary
        )

    def run_post_build_feature_selection(self, params: RunParams) -> None:
        """Refine a supplied embedding and rebuild the model if requested."""
        if not params.hyperparameters.feature_selection and params.embedding is not None and params.hyperparameters.qsar_method != 'gcm':
        
            logging.info(f"Before FS, previous embedding has {len(params.model.embedding)} descriptors: {params.model.embedding}")
        
            if params.hyperparameters.run_rfe:
                run_rfe(params.model, params.df_training, 1, 1)
                logging.info(f"After RFE, {len(params.model.embedding)} descriptors: {params.model.embedding}")
        
            if params.hyperparameters.run_sfs:
                run_sfs(params.model, params.df_training, cv=params.cv)  # iterative
                logging.info(f"After SFS, {len(params.model.embedding)} descriptors: {params.model.embedding}")
        
            # redo model and predictions:
            params.df_pred_training, params.df_pred_test, params.training_stats, params.test_stats, params.model = ModelBuilder.build_and_test_model(
                params.df_training,
                params.df_prediction,
                params.cv,
                params.hyperparameters,
                params.model.embedding,
                params.is_binary
            )
            logging.info(f"After FS, embedding has {len(params.model.embedding)} descriptors: {params.model.embedding}")

    def run_cross_validation(self, params: RunParams) -> None:
        """Build held-out predictions and statistics for all CV folds."""
        if params.cross_validate:
            params.df_pred_cv, params.cv_stats = ModelBuilder.crossvalidate(
                params.df_cv_dict,
                params.hyperparameters,
                params.model.embedding,params.is_binary
            )

    def generate_external_stats(self, params: RunParams) -> None:
        """Predict an optional external dataset and calculate its statistics."""
        params.ext_stats = None
        params.df_pred_ext = None
        
        if params.df_prediction_ext is not None:
            params.model.external_dataset_name = params.dataset_name_ext
            params.df_pred_ext, params.ext_stats = ModelBuilder.predict(params.model, params.df_prediction_ext, '_External', params.is_binary)

    def run_applicability_domain(self, params: RunParams) -> None:
        """Calculate requested AD flags, consensus flags, and statistics."""
        params.stats_dict = {}
        if params.run_AD:
            for ad_measure in params.hyperparameters.ad_measure:
                params.df_pred_test = self.runAD(
                    params.df_training,
                    params.df_prediction,
                    params,
                    params.model.embedding,
                    params.df_pred_test,
                    ad_measure,
                    params.stats_dict,
                    is_binary=params.is_binary,
                    is_external=False
                )
        
            if len(params.hyperparameters.ad_measure) > 1:
                adu.generate_consensus_ad(
                    params.df_pred_test,
                    params.stats_dict,
                    params.hyperparameters.ad_measure,
                    is_binary=params.is_binary,
                    is_external=False
                )

            if params.dataset_name_ext is not None:
                params.ext_stats_dict = {}
                for ad_measure in params.hyperparameters.ad_measure:
                    params.df_pred_ext = self.runAD(
                            params.df_training,
                            params.df_prediction_ext,
                            params,
                            params.model.embedding,
                            params.df_pred_ext,
                            ad_measure,
                            params.ext_stats_dict,
                            is_binary=params.is_binary,
                            is_external=True
                        )

                if len(params.hyperparameters.ad_measure) > 1:
                    adu.generate_consensus_ad(
                        params.df_pred_ext,
                        params.ext_stats_dict,
                        params.hyperparameters.ad_measure,
                        is_binary=params.is_binary,
                        is_external=True
                    )

    def add_cv_info_to_df_training(self, params: RunParams) -> None:
        """Attach each training record's CV fold to the training dataframe."""
        if params.cross_validate:
            # Add CV fold information to df_training
            cv_fold_data = []
            for fold_num, fold_dict in sorted(params.df_cv_dict.items()):
                pred_df = fold_dict.get("pred")
                if pred_df is not None and not pred_df.empty:
                    for idx, row in pred_df.iterrows():
                        cv_fold_data.append({"ID": row["ID"], "cv_fold": fold_num})
        
            if cv_fold_data:
                df_cv_folds = pd.DataFrame(cv_fold_data)
                params.df_training = params.df_training.merge(df_cv_folds, on="ID", how="left")

    def extract_ad_stats(self, params: RunParams) -> None:
        """Extract consensus and external applicability-domain statistics."""
        # Safely extract AD stats (could be missing or None)
        ad_measure_final = " and ".join(params.hyperparameters.ad_measure) if params.hyperparameters.ad_measure else None
        ad_measure_final_external = ad_measure_final + " External" if ad_measure_final else None

        params.ad_stats = params.stats_dict.get(ad_measure_final) or {}

        external_ad_stats = getattr(params, "ext_stats_dict", {}).get(ad_measure_final_external) or {}
        external_ad_stat_key_mapping = {
            "ad_measure": "ad_measure_external",
            "MAE_External_inside_AD": "MAE_External_inside_AD",
            "MAE_External_outside_AD": "MAE_External_outside_AD",
            "mae_ratio": "mae_ratio_external",
            "Coverage_External": "Coverage_External",
            "BA_External_inside_AD": "BA_External_inside_AD",
            "BA_External_outside_AD": "BA_External_outside_AD",
            "ba_ratio": "ba_ratio_external"
        }
        params.external_ad_stats = {external_ad_stat_key_mapping.get(key, key): value for key, value in external_ad_stats.items()}

    def get_mapped_property_values_from_db(self, params: RunParams) -> None:
        """Load mapped property values and datapoint identifiers."""
        logging.info("Retrieving mapped property values from database...")

        session = self.ensure_session(params)
        edg = ExpDataGetter()
        params.df_pv, unique_params = edg.get_mapped_property_values(
            session,
            params.hyperparameters.dataset_name,
            SNAPSHOT_ID,
            duplicate_strategy=DUPLICATE_STRATEGY,
        )

        session = self.ensure_session(params)
        params.df_dps = du.getMappedDatapoints(session, params.hyperparameters.dataset_name)

        params.df_dps_ext = None
        if params.df_prediction_ext is not None:
            params.df_dps_ext = du.getMappedDatapoints(session, params.dataset_name_ext)
        
        logging.info("Mapped property values retrieved successfully")

    def add_source_chemical_info_to_dfs(self, params: RunParams) -> None:
        """Add source chemical and experimental-property IDs to predictions."""
        if getattr(params, "df_dps", None) is None:
            logging.error("Data points dataframe (df_dps) is not available. Please run get_mapped_property_values_from_db() first.")
            return
        
        params.df_pred_training = StaticHelpers.add_source_chemical_info(params.df_pred_training, params.df_dps)
        params.df_pred_test = StaticHelpers.add_source_chemical_info(params.df_pred_test, params.df_dps)
        if params.df_pred_cv is not None:
            params.df_pred_cv = StaticHelpers.add_source_chemical_info(params.df_pred_cv, params.df_dps)

        if params.df_prediction_ext is not None and getattr(params, "df_dps_ext", None) is not None:
            params.df_pred_ext = StaticHelpers.add_source_chemical_info(params.df_pred_ext, params.df_dps_ext)
        elif params.df_prediction_ext is not None:
            logging.warning("External data points dataframe (df_dps_ext) is not available. Please run get_mapped_property_values_from_db() first.")

    def log_stats_to_console(self, params: RunParams) -> None:
        """Log aggregate model and applicability-domain statistics."""
        logging.info(f"Cross-validation Stats = {json.dumps(getattr(params, 'cv_stats', None), indent=4)}")
        logging.info(f"Test Set Stats = {json.dumps(params.test_stats, indent=4)}")
        logging.info(f"External Set Stats = {json.dumps(params.ext_stats, indent=4)}")

        if params.run_AD:
            logging.info(f"Test Set AD Stats = {json.dumps(params.ad_stats, indent=4)}")
            if params.dataset_name_ext is not None:
                logging.info(f"External Set AD Stats = {json.dumps(params.external_ad_stats, indent=4)}")

    def create_dirs(self, params: RunParams) -> None:
        """Select output filenames for the current run."""
        identifier = StaticHelpers.get_identifier(params.unique_identifier, params.test_stats, params.ext_stats)
        
        if identifier is None:
            params.json_path = os.path.join(params.folder_path, "results.json")
            params.detailed_summary_path = os.path.join(params.folder_path, "detailed_summary.xlsx")
        else:
            params.json_path = os.path.join(params.folder_path, f"results_{identifier}.json")
            params.detailed_summary_path = os.path.join(params.folder_path, f"detailed_summary_{identifier}.xlsx")

    def append_info_to_model_object(self, params: RunParams) -> None:
        """Attach dataset metadata, predictions, and statistics to the model."""
        session = self.ensure_session(params)
        dataset_info = du.get_dataset_details(session, params.hyperparameters.dataset_name)
        dsstox_mapping_strategy = json.loads(dataset_info["dsstox_mapping_strategy"])

        params.model.propertyName = dataset_info["property_name"]
        if params.model.propertyName is None:
            logging.error("propertyName is None, need to set name_ccd column for property in properties table")
            return

        params.model.qsarReadyRuleSet = dsstox_mapping_strategy.get("qsarReadyRuleSet", "qsar-ready")
        params.model.propertyDescription = dataset_info["property_description"]
        params.model.unitsModel = dataset_info["units_model"]
        params.model.unitsDisplay = dataset_info["units_display"]
        epoch_ms = time.time_ns() // 1_000_000
        params.model.modelName = params.user + "_" + str(epoch_ms)
        params.model.datasetName = params.hyperparameters.dataset_name
        params.model.datasetDescription = dataset_info["dataset_description"]
        params.model.omitSalts = dsstox_mapping_strategy["omitSalts"]
        params.model.applicabilityDomainName = " and ".join(params.hyperparameters.ad_measure) if params.run_AD else None
        params.model.num_training = params.df_training.shape[0]
        params.model.num_prediction = params.df_prediction.shape[0]
        params.model.descriptorSetName = params.hyperparameters.descriptor_set_name
        params.model.df_preds_training_cv = params.df_pred_cv
        params.model.df_preds_test = params.df_pred_test
        if params.dataset_name_ext is not None:
            params.model.external_dataset_name = params.dataset_name_ext
            params.model.external_dataset_description = params.dataset_description_ext
            params.model.df_dsstoxRecords_external = params.df_prediction_ext
            params.model.df_preds_external = params.df_pred_ext
            params.model.df_external = params.df_external

        params.model.modelStatistics = (
            (params.training_stats or {})
            | (getattr(params, "cv_stats", None) or {})
            | (params.test_stats or {})
            | (params.ad_stats or {})
            | (params.ext_stats or {})
            | (params.external_ad_stats or {})
        )

        params.model.df_training = params.df_training
        params.model.df_prediction = params.df_prediction
        params.model.modelMethod = getattr(params.model, "regressor_name", None) or getattr(params.model, "qsar_method", None)
        params.model.modelMethodDescription = getattr(params.model, "description", None)

        params.model.subfolder = params.subfolder

    def create_results_dict(self, params: RunParams) -> Dict[str, object]:
        """Build the JSON-serializable result payload for this run."""
        # TODO: Clean up this function considerably, if needed, particulary wrt params object
        hyperparameters = params.hyperparameters
        training_stats = params.training_stats
        test_stats = params.test_stats
        cv_stats = params.cv_stats
        ext_stats = params.ext_stats
        stats_dict = params.stats_dict
        ext_stats_dict = params.ext_stats_dict

        results_dict = {"params":hyperparameters.to_dict()}

        md = params.model.get_model_description_dict()
        results_dict["model_details"] = md

        # Store the embedding length safely (handles None)
        embedding = getattr(params.model, "embedding", None)
        md["embedding_len"] = len(embedding) if embedding is not None else 0

        md["qsar_method_description"] = md.pop("description")
        md["qsar_method_description_url"] = md.pop("description_url")
        md.pop("embedding")
        md["embedding"] = embedding

        md.pop("descriptorService")

        md.pop("training_descriptor_std_devs")
        md.pop("training_descriptor_means")

        md["splittingName"] = hyperparameters.splitting_name

        md["descriptor_set_name"] = hyperparameters.descriptor_set_name

        qsar_method = hyperparameters.qsar_method

        if qsar_method == 'reg' or qsar_method == 'las' or qsar_method == 'gcm':
            y = params.df_training[params.df_training.columns[1]]
            X = params.df_training[params.model.embedding]
            results_dict["model_details"]['model_coefficients'] = json.loads(params.model.getOriginalRegressionCoefficients2(X, y))

        ms = {}
        results_dict["model_statistics"] = ms

        if len(stats_dict) > 0:
            ms["test_stats_all_AD"] = stats_dict

        training_stats.pop("Coverage_Training", None)
        ms["training_stats"] = training_stats

        if cv_stats:
            cv_stats.pop("Coverage_CV_Training", None)
            ms["cv_stats"] = cv_stats

        if ext_stats:
            ext_stats.pop("Coverage_External", None)
            ms["ext_stats"] = ext_stats

        test_stats.pop("Coverage_Test", None)
        ms["test_stats"] = test_stats

        if len(stats_dict) > 0:
            str_ad_measure_final = " and ".join(params.hyperparameters.ad_measure)
            ms["test_stats_AD"] = stats_dict[str_ad_measure_final]
            ms["test_stats_all_AD"] = stats_dict

        if len(stats_dict) > 0 and ext_stats:
            external_ad = str_ad_measure_final + " External"
            ms["ext_stats_AD"] = ext_stats_dict[external_ad]
            ms["ext_stats_all_AD"] = ext_stats_dict

        params.results_dict = results_dict
        return results_dict

    def write_files(self, params: RunParams) -> None:
        """Write prediction files, plots, workbook, and results JSON."""
        StaticHelpers.write_prediction_csvs(params.df_pred_test, params.df_pred_ext, params.folder_path)
        StaticHelpers.write_plots(params.df_pred_test, params.model, params.df_pred_cv, params.folder_path)

        if params.write_to_db:
            self.ensure_session(params)
            ml = ModelLoader(params.session)
            ml.load_model(params.user, params.model, params.results_dict, params.df_pred_training, params.df_pred_test, params.df_pred_cv, params.folder_path, df_pred_external=params.df_pred_ext)
        
            # build excel after loading model so have model id number set:
            mdo = ModelDataObjects(model=params.model, df_pv=params.df_pv, df_gmd=params.df_dps, df_gmd_external=params.df_dps_ext)
            mte = ModelToExcel(mdo, params.detailed_summary_path)
            mte.create_excel()
        
            # need to load model separately from rest of model objects since need to load model into db to get id then create the excel file that displays that id in the summary:
            filePathOutExcelSummary = os.path.join(params.folder_path, "detailed_summary.xlsx")
            image_id = ml.load_model_file(filePathOutExcelSummary, params.user, params.model.modelId, 2)
            logging.info(f"Excel summary loaded to db with id: {image_id}")
        else:
            # build excel without loading model, so no id number will be set:
            mdo = ModelDataObjects(model=params.model, df_pv=params.df_pv, df_gmd=params.df_dps, df_gmd_external=params.df_dps_ext)
            mte = ModelToExcel(mdo, params.detailed_summary_path)
            mte.create_excel()
        
        
        with open(params.json_path, 'w') as json_file:
            json.dump(params.results_dict, json_file, indent=4)

    # -------------------------------------------------------------------------------------------------------------

    def run(self, params: RunParams) -> _QSARModel:
        """Run every dataset-processing, model-building, and output stage."""
        print("\n")
        logging.info(f"Start RunDataset:")
        logging.info(f"\tdataset_name = {params.hyperparameters.dataset_name}")
        logging.info(f"\tqsar_method = {params.hyperparameters.qsar_method}")
        logging.info(f"\tdescriptor_set_name = {params.hyperparameters.descriptor_set_name}")
        logging.info(f"\tfeature_selection = {params.hyperparameters.feature_selection}")
        logging.info(f"\tcross_validate = {params.cross_validate}")

        self.query_dataframes(params)

        logging.info("Performing additional dataframe processing...")
        self.set_is_binary(params)
        self.apply_outlier_filtering(params)
        self.setup_cross_validation(params)
        self.setup_logp_martin(params)
        logging.info("Dataframe processing completed")

        self.run_feature_selection(params)
        self.run_model_building(params)
        self.run_post_build_feature_selection(params)
        self.run_cross_validation(params)
        self.generate_external_stats(params)
        self.run_applicability_domain(params)
        self.add_cv_info_to_df_training(params)
        self.extract_ad_stats(params)
        self.get_mapped_property_values_from_db(params)
        self.add_source_chemical_info_to_dfs(params)
        self.log_stats_to_console(params)
        self.create_dirs(params)
        self.append_info_to_model_object(params)
        self.create_results_dict(params)
        self.write_files(params)

        return params.model

# -------------------------------------------------------------------------------------------------------------

def test_run_dataset() -> None:
    """Run the database-backed KOC example configuration."""
    write_to_db = False
    dataset_name = "KOC v1 modeling"
    # descriptor_set_name = "WebTEST-default"
    qsar_method = "gcm"
    # feature_selection = False
    # splitting_name = "RND_REPRESENTATIVE"
    # unique_identifier = None
    append_to_models_folder = "_RunDataset_test"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    runner = RunDataset()
    params = runner.generate_params(
        dataset_name=dataset_name,
        qsar_method=qsar_method,
        # feature_selection=feature_selection,
        # descriptor_set_name=descriptor_set_name,
        # splitting_name=splitting_name,
        ad_measure=ad_measure_model,
        write_to_db=write_to_db,
        # unique_identifier=unique_identifier,
        append_to_models_folder=append_to_models_folder
    )

    runner.run(params)


def test_customized_run() -> None:
    """Run a test of RunDataset using a copied run configuration from run_case_studies1.py with customization options"""
    dataset_name = "exp_prop_BCF_v1_modeling"
    qsar_method = "knn"
    write_to_db = False
    append_to_models_folder = "_RunDataset_test"

    ad_measure = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    hyperparameters = {
        "max_features": 25,
        "descriptor_coefficient": 0.006
    }

    runner = RunDataset()
    params = runner.generate_params(
        dataset_name = dataset_name,
        qsar_method = qsar_method,
        write_to_db = write_to_db,
        append_to_models_folder = append_to_models_folder,
        ad_measure = ad_measure,
        hyperparameters = hyperparameters
    )
    runner.run(params)


def main() -> None:
    test_run_dataset()
    # test_customized_run()


if __name__ == "__main__":
    main()
