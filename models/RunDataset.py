import json
import logging
import os
from typing import Optional, Union, List
from io import StringIO
from pathlib import Path

from sqlalchemy import URL, Engine, text, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError, SQLAlchemyError

import pandas as pd
import numpy as np
import math
import time
from datetime import datetime

from model_ws_utilities import \
    apply_outlier_filter, \
    call_build_embedding_ga_db, \
    call_build_embedding_importance_from_df, \
    call_do_predictions_from_df, \
    call_build_model_with_preselected_descriptors_from_df

from util.database_utilities import DatabaseUtilities
from util.serialization_compat import serialize_model
from util import predict_constants as pc
from models.case_studies import run_model_building_db as rmdb
import models.db_utilities.dataset_utilities_db as du
from models.db_utilities.raw_exp_data_db import ExpDataGetter
from models.ModelToExcel import ModelToExcel, ModelDataObjects
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics
from models.EmbeddingFromImportance import perform_iterative_recursive_feature_elimination as run_rfe
from models.EmbeddingFromImportance import perform_iterative_sequential_feature_selection as run_sfs
from applicability_domain import applicability_domain_utilities as adu
from models import make_test_plots as mtp

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

FEATURE_SELECTION_METHOD_GENETIC_ALGORITHM = "Genetic algorithm"
FEATURE_SELECTION_METHOD_GROUP_CONTRIBUTION = "Group contribution" 
FEATURE_SELECTION_METHOD_IMPORTANCE = "Importance"

SNAPSHOT_ID = 4
DUPLICATE_STRATEGY = "id_suffix"


class StaticHelpers:
    @staticmethod
    def write_prediction_csvs(df_pred_test, df_pred_ext, folder_path):
        prediction_csv_path = os.path.join(folder_path, f"test set predictions.csv")
        df_pred_test.to_csv(prediction_csv_path, index=False)
        if df_pred_ext is not None:
            prediction_csv_path = os.path.join(folder_path, f"external set predictions.csv")
            df_pred_ext.to_csv(prediction_csv_path, index=False)

    @staticmethod
    def write_plots(df_pred_test, model, df_pred_cv, folder_path):
        if df_pred_cv is not None:
            mpsTraining = df_pred_cv.to_dict(orient='records') # use CV values so the predictions are fair estimates (exp values wont change from training set)
            mpsTest = df_pred_test.to_dict(orient='records')
            filePathOutHistogram = os.path.join(folder_path, "histogram.png")
            mtp.generateHistogram2(filePathOutHistogram, model.propertyName, model.unitsModel, mpsTraining, mpsTest, seriesNameTrain="Training Set", seriesNameTest="Test Set")
        
        if not model.is_binary:
            filePathOutScatter = os.path.join(folder_path, "scatter_plot.png")
            title = "Prediction results for " + model.propertyName
            mtp.generateScatterPlot2(filePathOut=filePathOutScatter, title=title, unitName=model.unitsModel, 
                mpsTraining=mpsTraining, mpsTest=mpsTest, 
                seriesNameTrain="Training Set (CV)", seriesNameTest="Test Set")
    
    @staticmethod
    def getEngine(connect_args: Optional[dict] = {}) -> Engine:
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
    def getSession(engine: Engine = None) -> Session:
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
    def get_current_user_id():
        user_id = os.getenv("USER_ID")
        if user_id is None:
            logging.warning("USER_ID environment variable is not set, falling back to default (tmarti02)")
            user_id = "tmarti02"
        return user_id

    @staticmethod
    def ensure_static_session(session):
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
    def build_output_subfolder(params, folder_embedding=None, logp_columns=None, include_feature_count=False):
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
    def safe_div(n, d):
        try:
            return n / d if (d is not None and d != 0 and not (isinstance(d, float) and math.isnan(d))) else float('nan')
        except ZeroDivisionError:
            return float('nan')

    @staticmethod
    def add_source_chemical_info(df_pred, df_dps):
        df_pred.rename(columns={'id':'canon_qsar_smiles'}, inplace=True)
        df_pred = df_pred.merge(df_dps, on='canon_qsar_smiles', how='left')
        df_pred.rename(columns={'qsar_exp_prop_property_values_id_first':'exp_prop_id'}, inplace=True)
        return df_pred

    @staticmethod
    def get_identifier(unique_identifier, test_stats, ext_stats):
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
        elif unique_identifier == "external_stat" and ext_stats is not None and not ext_stats.empty:
            identifier = f"{'RMSE' if ext_stats.get('RMSE_External') else 'BA'}_external_{ext_stats.get('RMSE_External', ext_stats.get('BA_External', '')):.3f}"
        elif unique_identifier == "test_stat" or (unique_identifier == "external_stat" and (ext_stats is None or ext_stats.empty)):
            identifier = f"{'RMSE' if test_stats.get('RMSE_Test') else 'BA'}_test_{test_stats.get('RMSE_Test', test_stats.get('BA_Test', '')):.3f}"
        else:
            identifier = None
        
        return identifier

    @staticmethod
    def feature_selection(df_training, df_prediction, params, cv=5):
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


class ModelBuilder:
    @staticmethod 
    def predict(model, df, tag, is_binary):
        json_predictions = call_do_predictions_from_df(df, model)
        
        df_predictions = pd.read_json(StringIO(json_predictions), orient="records")
                
        if is_binary:
            stats = calculate_binary_statistics(df_predictions, 0.5, tag)
        else:
            mean_exp_training = df_predictions["exp"].mean()
            stats = calculate_continuous_statistics(df_predictions, mean_exp_training, tag)
    
        return df_predictions, stats
    
    @staticmethod
    def crossvalidate(df_cv_dict, params, embedding, is_binary):
        logging.info("Start running CV calculations ...")
    
        def _fold_predictions():
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
    def build_and_test_model(df_training, df_prediction, cv, params, embedding, is_binary):
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
    def __init__(self, session):
        self.session = session
        self.dbl = DatabaseUtilities(schema="qsar_models", session=self.session)

    def load_model_file(self, file_path, username, fk_model_id, fk_file_type_id):
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
    
    def update_model_file(self, file_path, username, fk_model_id, fk_file_type_id):
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

    def add_model_statistics(self, user, fk_model_id, stats_dict_name, model_statistics_dict, stats_lookup, created_at, model_statistics_rows):
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
    
    def load_stats(self, results, user, fk_model_id):
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
        
    def divide_array(self, source: Union[bytes, bytearray], chunksize: int=26214400) -> List[Union[bytes, bytearray]] | None:
        """
        Split a bytes-like object into chunks of size `chunksize`.
        Returns a list of slices (copies). Last chunk may be shorter.
        """
        parts = [source[i:i + chunksize] for i in range(0, len(source), chunksize)]
    
        # Optional: mimic the Java prints
        logging.info(f"Size of model bytes={len(source)}")
        logging.info(f"# Parts = {len(parts)}")
    
        total = sum(len(p) for p in parts)
        if total != len(source):
            print(f"byte length mismatch:{total}\t{len(source)}")
            return None
    
        return parts

    def load_model_from_object(self, user, model, params, fk_descriptor_embedding_id, fk_method_id, fk_ad_method):
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

    def create_model_bytes(self, bytes_list):
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

    def load_model_bytes(self, user, model, fk_model_id):
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
        """
        Build one row per prediction per AD method, using each AD column's
        corresponding ad_methods.id.
        """
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
        """
        Convert dataframe AD column naming into the database ad_methods.name format.

        Examples:
        AD_TEST_Euclidean_Distance_All_Descriptors -> TEST Euclidean Distance All Descriptors
        TEST_Euclidean_Distance_All_Descriptors -> TEST Euclidean Distance All Descriptors
        OPERA_Global_Index -> OPERA Global Index
        """
        if ad_name.startswith("AD_"):
            ad_name = ad_name[3:]
        return ad_name.replace("_", " ").strip()

    def get_ad_method_id_by_name(self, ad_name: str):
        db_name = self.normalize_ad_method_name(ad_name)
        row_ad_method = self.dbl.get_row("ad_methods", name=db_name)
        if row_ad_method is None:
            logging.warning(f"Could not find ad_methods row for name='{db_name}' (from '{ad_name}')")
            return None
        return row_ad_method.id

    def create_method(self, user, isBinary, fullMethodName):
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

    def load_model(self, user, model, results, df_pred_training, df_pred_test, df_pred_cv, folder_path, df_pred_external=None):
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
        
    def create_descriptor_embedding_from_params(self, user, results):
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
    def __init__(
            self,
            dataset_name,
            qsar_method,
            embedding=None,
            folder_embedding=None,
            cross_validate=True,
            run_AD=True,
            feature_selection=True,
            fs_previous_embedding=True,
            params=None,
            descriptor_set_name="WebTEST-default",
            splitting_name="RND_REPRESENTATIVE",
            ad_measure=None,
            add_LOGP_Martin=False,
            logp_columns=None,
            write_to_db=False,
            user="",
            unique_identifier=None,
            append_to_models_folder="",
            subfolder=None,
            save_initial_dfs=False,
            outlier_filter_methods=None,
            include_feature_count_in_subfolder=False
        ):
        self.dataset_name = dataset_name
        self.qsar_method = qsar_method
        self.embedding = embedding
        self.folder_embedding = folder_embedding
        self.cross_validate = cross_validate
        self.run_AD = run_AD
        self.feature_selection = feature_selection
        self.fs_previous_embedding = fs_previous_embedding
        self.params = params
        self.descriptor_set_name = descriptor_set_name
        self.splitting_name = splitting_name
        self.ad_measure = ad_measure
        self.add_LOGP_Martin = add_LOGP_Martin
        self.logp_columns = logp_columns
        self.write_to_db = write_to_db
        self.user = user
        self.unique_identifier = unique_identifier
        self.append_to_models_folder = append_to_models_folder
        self.subfolder = subfolder
        self.save_initial_dfs = save_initial_dfs
        self.outlier_filter_methods = outlier_filter_methods or []
        self.include_feature_count_in_subfolder = include_feature_count_in_subfolder

        self.post_init()

    def post_init(self):
        # Initialize database engine and session
        self.engine = self.engine or self.getEngine()
        self.session = self.session or self.getSession(self.engine)
        self.current_user_id = self.get_current_user_id()
        if not self.user:
            self.user = self.current_user_id

        # Set default splitting name if not provided
        self.splitting_name = self.splitting_name or "RND_REPRESENTATIVE"

        # Set feature selection to False for certain QSAR methods
        if self.qsar_method in {"gcm", "svm"}:
            self.feature_selection = False

        # Set embedding and feature selection based on provided parameters
        if self.embedding is not None:
            self.folder_embedding = "custom"
            self.feature_selection = False
        elif self.folder_embedding is not None:
            self.feature_selection = False
            file_name_embedding = "results.json"
            file_path_embedding = os.path.join(PROJECT_ROOT, "data/models", self.dataset_name, self.folder_embedding, file_name_embedding)
            
            with open(file_path_embedding, "r", encoding="utf-8") as f:
                results = json.load(f)
                self.folder_embedding = results["model_details"]["embedding"]
                # print(f"from {folder_embedding}:{embedding}")
        else:
            self.fs_previous_embedding = False

        # Set default applicability domain measures if not provided
        self.ad_measure = self.ad_measure or [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_All_Descriptors_Euclidean]
        if self.descriptor_set_name == "WebTEST-default":
            self.ad_measure.append(pc.Applicability_Domain_TEST_Fragment_Counts)

        # Set hyperparameters if not provided
        if self.params is None:
            self.params = rmdb.set_hyper_parameters(
                qsar_method=self.qsar_method,
                feature_selection=self.feature_selection,
                descriptor_set_name=self.descriptor_set_name,
                splitting_name=self.splitting_name,
                dataset_name=self.dataset_name,
                ad_measure=self.ad_measure
            )
        self.params.feature_selection = self.feature_selection

        # Set parameters related to outlier handling
        if not hasattr(self.params, "outlier_filter_methods") and self.outlier_filter_methods is not None and self.outlier_filter_methods != False:
            # outlier_filter_methods = ["iqr", "hampel", "robust_z", "esd"]
            self.params.outlier_filter_methods = self.outlier_filter_methods

        if getattr(self.params, "outlier_filter_methods", False) and not hasattr(self.params, "outlier_filter_columns"):
            self.params.outlier_filter_columns = ["Property"]

        if getattr(self.params, "outlier_filter_methods", False) and not hasattr(self.params, "outlier_filter_trim_mode"):
            self.params.outlier_filter_trim_mode = "remove"

        # Set external dataset
        self.df_prediction_ext = None
        self.dataset_name_ext = None
    
        if self.dataset_name == 'KOC v1 modeling' or self.dataset_name == 'KOC v2 modeling':
            self.dataset_name_ext = 'KOC v2 external'
        elif self.dataset_name == 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling':
            self.dataset_name_ext = 'QSAR_Toolbox_96HR_Fish_LC50_v3 modeling'    
        elif self.dataset_name == 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling':
            self.dataset_name_ext = 'QSAR_Toolbox_96HR_Fish_LC50_v3b modeling'    
        elif self.dataset_name == "exp_prop_BCF_v1_modeling":
            self.dataset_name_ext = "exp_prop_BCF_v1_external"
        elif self.dataset_name == 'exp_prop_RBIODEG_RIFM_CHEMREG':
            self.dataset_name_ext = 'exp_prop_RBIODEG_301F v1 modeling' # use ECHA data as external set to see if works ok

        # Ensure logp_columns is a list
        if self.logp_columns is not None:
            if isinstance(self.logp_columns, str):
                self.logp_columns = [self.logp_columns]
            self.params.logp_columns = self.logp_columns

        self.subfolder = self.subfolder if self.subfolder is not None else self.build_output_subfolder(
            self.params,
            self.folder_embedding,
            self.logp_columns,
            include_feature_count=self.include_feature_count_in_subfolder
        )

        self.path_segments = [PROJECT_ROOT, "data", "models" + self.append_to_models_folder, self.params.dataset_name, self.subfolder]
        
        self.folder_path = os.path.join(*self.path_segments)
        self.folder_path_path = Path(self.folder_path)
        self.folder_path_path.mkdir(parents=True, exist_ok=True)

    def ensure_session(self) -> Session:
        session = self.session
        try:
            # cheap no-op to force a DB round trip
            session.execute(text("SELECT 1"))
            return session
        except OperationalError:
            try:
                session.close()
            except Exception:
                pass
            session = RunDataset.getSession(self.engine)
            self.session = session
            return session
    
    def runAD(self, df_training, df_prediction, params, embedding, df_predictions, ad_measure, stats_dict, is_binary=False, is_external=False):
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
            remove_log_p=params.remove_log_p_descriptors,
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
            cutoff = getattr(params, 'cutoff', getattr(params, 'classification_cutoff', 0.5))

            stats_inside = calculate_binary_statistics(df_inside, cutoff=cutoff, tag=tag)
            stats_outside = calculate_binary_statistics(df_outside, cutoff=cutoff, tag=tag)
            
            ba_inside = stats_inside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
            ba_outside = stats_outside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
            ba_ratio = RunDataset.safe_div(ba_inside, ba_outside)

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

            mae_ratio = RunDataset.safe_div(mae_outside, mae_inside)

            stats = {
                "ad_measure": ad_measure,
                f"MAE_{stat_insert}_inside_AD": mae_inside,
                f"MAE_{stat_insert}_outside_AD": mae_outside,
                f"mae_ratio": mae_ratio,
                f"Coverage_{stat_insert}": coverage_ad
            }

        stats_dict[f"{ad_measure}{' External' if is_external else ''}"] = stats
        return df_predictions

    def query_dataframes(self):
        logging.info("Querying dataframes from database...")

        self.ensure_session()
        df_training, df_prediction = du.get_training_prediction_instances(self.session, self.dataset_name, self.descriptor_set_name, self.params.splitting_name)
        
        if df_training is None or df_prediction is None:
            logging.error("Failed to retrieve training or prediction dataframes from the database")
            return

        self.df_training = df_training.copy()
        self.df_prediction = df_prediction.copy()

        # Query external dataset if specified
        self.df_external = None
        if self.dataset_name_ext is not None:
            self.ensure_session()
            dataset_description_ext = du.get_dataset_details(self.session, self.dataset_name_ext).get('dataset_description')

            self.ensure_session()
            df_prediction_ext = du.get_instances_excluding(self.session, self.dataset_name_ext, self.dataset_name, self.descriptor_set_name)

            self.df_external = df_prediction_ext.copy()

        logging.info("Dataframes queried successfully")

    def set_is_binary(self):
        if self.df_training is None:
            logging.error("Training dataframe is not available. Please run query_dataframes() first.")
            return

        s = self.df_training.iloc[:, 1]  # Assuming the second column is the target variable
        self.is_binary = s.isin([0, 1]).all()

    def apply_outlier_filtering(self):
        if getattr(self.params, "outlier_filter_methods", False) and self.params.outlier_filter_methods is not None and len(self.params.outlier_filter_methods) > 0:
            logging.info(
                f"Applying outlier filter: methods={self.params.outlier_filter_methods}, "
                f"columns={self.params.outlier_filter_columns}, mode={self.params.outlier_filter_trim_mode}"
            )
        
            df_training_before = self.df_training.copy()
        
            self.df_training, outlier_report = apply_outlier_filter(
                self.df_training,
                columns=self.params.outlier_filter_columns,
                methods=self.params.outlier_filter_methods,
                target_col="Property",
                trim_mode=self.params.outlier_filter_trim_mode,
                iqr_k=getattr(self.params, "outlier_filter_iqr_k", 1.5),
                hampel_k=getattr(self.params, "outlier_filter_hampel_k", 3.0),
                robust_z_thresh=getattr(self.params, "outlier_filter_robust_z_thresh", 3.5),
                esd_alpha=getattr(self.params, "outlier_filter_esd_alpha", 0.05),
                esd_max_outliers=getattr(self.params, "outlier_filter_esd_max_outliers", 10),
            )
        
            logging.info(
                f"Outlier filter removed {len(df_training_before) - len(self.df_training)} rows "
                f"out of {len(df_training_before)}"
            )

    def setup_cross_validation(self):
        self.df_cv_dict = None

        if self.cross_validate:
            self.ensure_session()
            self.df_cv_dict = du.get_cross_validation_instances(self.session, self.dataset_name, self.descriptor_set_name)

            X, y, self.cv, feature_cols = du.make_cv_for_base_training(self.df_training, self.df_cv_dict, "ID", "Property")

    def setup_logp_martin(self):
        # Add LOGP Martin columns if specified
        if self.add_LOGP_Martin:
            self.df_training, self.df_prediction, df_prediction_ext = rmdb.add_log_p_martin_columns(
                self.df_training,
                self.df_prediction,
                self.cross_validate,
                self.df_cv_dict,
                df_external=self.df_external
            )
            self.df_external = df_prediction_ext.copy() if df_prediction_ext is not None else None
    
    def run_feature_selection(self):
        if self.feature_selection:
            self.embedding = self.feature_selection(
                self.df_training,
                self.df_prediction,
                self.params,
                self.cv
            )

            if self.qsar_method == "huber":
                original_feature_count = self.df_training.shape[1] - 2  # adjust if needed for ID/Property columns
                selected_feature_count = len(self.embedding) if self.embedding is not None else 0
                
                # print(f"Huber feature selection: {selected_feature_count} selected out of {original_feature_count}")
                
                if self.embedding is None or selected_feature_count >= original_feature_count:
                    raise ValueError(
                        f"Huber feature selection did not reduce the feature set. "
                        f"Selected {selected_feature_count} features from {original_feature_count}."
                    )

    def run_model_building(self):
        self.df_pred_training, self.df_pred_test, self.training_stats, self.test_stats, self.model = ModelBuilder.build_and_test_model(
            self.df_training,
            self.df_prediction,
            self.cv,
            self.params,
            self.embedding,
            self.is_binary
        )

    def run_post_build_feature_selection(self):
        if not self.feature_selection and self.embedding is not None and self.qsar_method != 'gcm':
        
            logging.info(f"Before FS, previous embedding has {len(self.model.embedding)} descriptors: {self.model.embedding}")
        
            if self.params.run_rfe:
                run_rfe(self.model, self.df_training, 1, 1)
                logging.info(f"After RFE, {len(self.model.embedding)} descriptors: {self.model.embedding}")
        
            if self.params.run_sfs:
                # run_sfs(self.model, self.df_training, self.params.n_features_to_select)
                run_sfs(self.model, self.df_training, cv=self.cv)  # iterative
                logging.info(f"After SFS, {len(self.model.embedding)} descriptors: {self.model.embedding}")
        
            # redo model and predictions:
            df_pred_training, df_pred_test, training_stats, test_stats, model = ModelBuilder.build_and_test_model(
                self.df_training,
                self.df_prediction,
                self.params,
                self.model.embedding,
                self.is_binary
            )
            logging.info(f"After FS, embedding has {len(model.embedding)} descriptors: {model.embedding}")

    def run_cross_validation(self):
        if self.cross_validate:
            self.df_pred_cv, self.cv_stats = ModelBuilder.crossvalidate(
                self.df_cv_dict,
                self.params,
                self.model.embedding,self.is_binary
            )

    def generate_external_stats(self):
        self.ext_stats = None
        self.df_pred_ext = None
        
        if self.df_prediction_ext is not None:
            self.model.external_dataset_name = self.dataset_name_ext
            self.df_pred_ext, self.ext_stats = ModelBuilder.predict(self.model, self.df_prediction_ext, '_External', self.is_binary)

    def run_applicability_domain(self):
        self.stats_dict = {}
        if self.run_AD:
            for ad_measure in self.ad_measures:
                self.df_pred_test = self.runAD(
                    self.df_training,
                    self.df_prediction,
                    self.params,
                    self.model.embedding,
                    self.df_pred_test,
                    ad_measure,
                    self.stats_dict,
                    is_binary=self.is_binary,
                    is_external=False
                )
        
            if len(self.ad_measures) > 1:
                adu.generate_consensus_ad(
                    self.df_pred_test,
                    self.stats_dict,
                    self.ad_measure_model,
                    is_binary=self.is_binary,
                    is_external=False
                )

            if self.dataset_name_ext is not None:
                self.ext_stats_dict = {}
                if self.dataset_name_ext is not None:
                    for ad_measure in self.ad_measures:
                        self.df_pred_ext = self.runAD(
                            self.df_training,
                            self.df_prediction_ext,
                            self.params,
                            self.model.embedding,
                            self.df_pred_ext,
                            ad_measure,
                            self.ext_stats_dict,
                            is_binary=self.is_binary,
                            is_external=True
                        )

                    if len(self.ad_measures) > 1:
                        adu.generate_consensus_ad(
                            self.df_pred_ext,
                            self.ext_stats_dict,
                            self.ad_measure_model,
                            is_binary=self.is_binary,
                            is_external=True
                        )

    def add_cv_info_to_df_training(self):
        if self.cross_validate:
            # Add CV fold information to df_training
            cv_fold_data = []
            for fold_num, fold_dict in sorted(self.df_cv_dict.items()):
                pred_df = fold_dict.get("pred")
                if pred_df is not None and not pred_df.empty:
                    for idx, row in pred_df.iterrows():
                        cv_fold_data.append({"ID": row["ID"], "cv_fold": fold_num})
        
            if cv_fold_data:
                df_cv_folds = pd.DataFrame(cv_fold_data)
                self.df_training = self.df_training.merge(df_cv_folds, on="ID", how="left")

    def extract_ad_stats(self):
        # Safely extract AD stats (could be missing or None)
        ad_measure_final = " and ".join(self.ad_measure_model) if self.ad_measure_model else None
        ad_measure_final_external = ad_measure_final + " External" if ad_measure_final else None

        self.ad_stats = self.stats_dict.get(ad_measure_final) or {}

        external_ad_stats = self.stats_dict.get(ad_measure_final_external) or {}
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
        self.external_ad_stats = {external_ad_stat_key_mapping.get(key, key): value for key, value in external_ad_stats.items()}

    def get_mapped_property_values_from_db(self):
        logging.info("Retrieving mapped property values from database...")

        session = self.ensure_session()
        edg = ExpDataGetter()
        self.df_pv, unique_params = edg.get_mapped_property_values(session, self.dataset_name, SNAPSHOT_ID, duplicate_strategy=DUPLICATE_STRATEGY)

        session = self.ensure_session()
        self.df_dps = du.getMappedDatapoints(session, self.dataset_name)

        self.df_dps_ext = None
        if self.df_prediction_ext is not None:
            self.df_dps_ext = du.getMappedDatapoints(session, self.dataset_name_ext)
        
        logging.info("Mapped property values retrieved successfully")

    def add_source_chemical_info_to_dfs(self):
        if getattr(self, "df_dps", None) is None:
            logging.error("Data points dataframe (df_dps) is not available. Please run get_mapped_property_values_from_db() first.")
            return
        
        self.df_pred_training = self.add_source_chemical_info(self.df_pred_training, self.df_dps)
        self.df_pred_test = self.add_source_chemical_info(self.df_pred_test, self.df_dps)
        self.df_pred_cv = self.add_source_chemical_info(self.df_pred_cv, self.df_dps)

        if self.df_prediction_ext is not None and getattr(self, "df_dps_ext", None) is not None:
            self.df_pred_ext = self.add_source_chemical_info(self.df_pred_ext, self.df_dps_ext)
        elif self.df_prediction_ext is not None:
            logging.warning("External data points dataframe (df_dps_ext) is not available. Please run get_mapped_property_values_from_db() first.")

    def log_stats_to_console(self):
        logging.info(f"Cross-validation Stats = {json.dumps(self.cv_stats, indent=4)}")
        logging.info(f"Test Set Stats = {json.dumps(self.test_stats, indent=4)}")
        logging.info(f"External Set Stats = {json.dumps(self.ext_stats, indent=4)}")

        if self.run_AD:
            logging.info(f"Test Set AD Stats = {json.dumps(self.ad_stats, indent=4)}")
            if self.dataset_name_ext is not None:
                logging.info(f"External Set AD Stats = {json.dumps(self.external_ad_stats, indent=4)}")

    def create_dirs(self):
        identifier = self.get_identifier(self.unique_identifier, self.test_stats, self.ext_stats)
        
        if identifier is None:
            self.json_path = os.path.join(self.folder_path, "results.json")
            self.detailed_summary_path = os.path.join(self.folder_path, f"detailed_summary.xlsx")
        else:
            self.json_path = os.path.join(self.folder_path, f"results_{identifier}.json")
            self.detailed_summary_path = os.path.join(self.folder_path, f"detailed_summary_{identifier}.xlsx")

    def append_info_to_model_object(self):
        session = self.ensure_session()
        dataset_info = du.get_dataset_details(session, self.dataset_name)
        dsstox_mapping_strategy = json.loads(dataset_info["dsstox_mapping_strategy"])

        if getattr(self.model, "propertyName", None) is None:
            print("propertyName is None, need to set name_ccd column for property in properties table")
            return

        self.model.qsarReadyRuleSet = dsstox_mapping_strategy.get("qsarReadyRuleSet", "qsar-ready")
        self.model.propertyName = dataset_info["property_name"]
        self.model.propertyDescription = dataset_info["property_description"]
        self.model.unitsModel = dataset_info["units_model"] 
        self.model.unitsDisplay = dataset_info["units_display"]
        epoch_ms = time.time_ns() // 1_000_000
        self.model.modelName = self.user + "_" + str(epoch_ms)
        self.model.datasetName = self.dataset_name
        self.model.datasetDescription = dataset_info["dataset_description"]
        self.model.omitSalts = dsstox_mapping_strategy["omitSalts"]
        self.model.applicabilityDomainName = " and ".join(self.params.ad_measure) if self.run_AD else None
        self.model.num_training = self.df_training.shape[0]
        self.model.num_prediction = self.df_prediction.shape[0]
        self.model.descriptorSetName = self.descriptor_set_name
        self.model.df_preds_training_cv = self.df_preds_training_cv
        self.model.df_preds_test = self.df_preds_test
        if self.dataset_name_ext is not None:
            self.model.external_dataset_name = self.dataset_name_ext if self.dataset_name_ext else None
            self.model.external_dataset_description = self.dataset_description_ext
            self.model.df_dsstoxRecords_external = self.df_prediction_ext
            self.model.df_preds_external = self.df_pred_ext
            self.model.df_external = self.df_external
            # self.model.num_external = self.df_external.shape[0] if self.df_external is not None else 0

        self.model.modelStatistics = (
            (self.training_stats or {})
            | (self.cv_stats or {})
            | (self.test_stats or {})
            | (self.ad_stats or {})
            | (self.ext_stats or {})
            | (self.external_ad_stats or {})
        )

        self.model.df_training = self.df_training  # df_training_model?
        self.model.df_prediction = self.df_prediction  # df_test_model?
        self.model.modelMethod = getattr(self.model, "regressor_name", None) or getattr(self.model, "qsar_method", None)
        self.model.modelMethodDescription = getattr(self.model, "description", None)

        self.model.subfolder = self.subfolder

    def create_results_dict(self):
        # TODO: Clean up this function considerably, if needed, particulary wrt params object
        # Map variables from self to names used in the original function from run_model_building_db
        ad_measure_model = self.ad_measure_model
        df_training = self.df_training
        params = self.params
        model = self.model
        training_stats = self.training_stats
        test_stats = self.test_stats
        cv_stats = self.cv_stats
        ext_stats = self.ext_stats
        stats_dict = self.stats_dict
        ext_stats_dict = self.ext_stats_dict

        results_dict = {"params":params.to_dict()}

        md = model.get_model_description_dict()
        results_dict["model_details"] = md

        # Store the embedding length safely (handles None)
        embedding = getattr(model, "embedding", None)
        md["embedding_len"] = len(embedding) if embedding is not None else 0

        md["qsar_method_description"] = md.pop("description")
        md["qsar_method_description_url"] = md.pop("description_url")
        md.pop("embedding")
        md["embedding"] = embedding

        md.pop("descriptorService")

        md.pop("training_descriptor_std_devs")
        md.pop("training_descriptor_means")

        md["splittingName"] = params.splitting_name

        md["descriptor_set_name"] = params.descriptor_set_name

        qsar_method = params.qsar_method

        if qsar_method == 'reg' or qsar_method == 'las' or qsar_method == 'gcm':
            y = df_training[df_training.columns[1]]
            X = df_training[model.embedding]
            results_dict["model_details"]['model_coefficients'] = json.loads(model.getOriginalRegressionCoefficients2(X, y))            

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
            str_ad_measure_final = " and ".join(ad_measure_model)                        
            ms["test_stats_AD"] = stats_dict[str_ad_measure_final]
            ms["test_stats_all_AD"] = stats_dict

        if len(stats_dict) > 0 and ext_stats:
            external_ad = str_ad_measure_final + " External"
            ms["ext_stats_AD"] = ext_stats_dict[external_ad]
            ms["ext_stats_all_AD"] = ext_stats_dict

        self.results_dict = results_dict

        return results_dict

    def write_files(self):
        StaticHelpers.write_prediction_csvs(self.df_pred_test, self.df_pred_ext, self.folder_path)
        StaticHelpers.write_plots(self.df_pred_test, self.model, self.df_pred_cv, self.folder_path)

        if self.write_to_db:
            self.ensure_session()
            ml = ModelLoader(self.session)
            ml.load_model(self.user, self.model, self.results_dict, self.df_pred_training, self.df_pred_test, self.df_pred_cv, self.folder_path, df_pred_external=self.df_pred_ext)
        
            # build excel after loading model so have model id number set:
            mdo = ModelDataObjects(model=self.model, df_pv=self.df_pv, df_gmd=self.df_dps, df_gmd_external=self.df_dps_ext)
            mte = ModelToExcel(mdo, self.detailed_summary_path)
            mte.create_excel()
        
            # need to load model separately from rest of model objects since need to load model into db to get id then create the excel file that displays that id in the summary:
            filePathOutExcelSummary = os.path.join(self.folder_path, "detailed_summary.xlsx")
            image_id = ml.load_model_file(filePathOutExcelSummary, self.user, self.model.modelId, 2)
            logging.info(f"Excel summary loaded to db with id: {image_id}")
        else:
            # build excel without loading model, so no id number will be set:
            mdo = ModelDataObjects(model=self.model, df_pv=self.df_pv, df_gmd=self.df_dps, df_gmd_external=self.df_dps_ext)
            mte = ModelToExcel(mdo, self.detailed_summary_path)
            mte.create_excel()
        
        
        with open(self.json_path, 'w') as json_file:
            json.dump(self.results_dict, json_file, indent=4)

    # -------------------------------------------------------------------------------------------------------------

    def run(self):
        print("\n")
        logging.info(f"Start RunDataset:")
        logging.info(f"\tdataset_name = {self.dataset_name}")
        logging.info(f"\tqsar_method = {self.qsar_method}")
        logging.info(f"\tdescriptor_set_name = {self.descriptor_set_name}")
        logging.info(f"\tfeature_selection = {self.feature_selection}")
        logging.info(f"\tcross_validate = {self.cross_validate}")

        self.query_dataframes()

        logging.info("Performing additional dataframe processing...")
        self.set_is_binary()
        self.apply_outlier_filtering()
        self.setup_cross_validation()
        self.setup_logp_martin()
        logging.info("Dataframe processing completed")

        self.run_feature_selection()
        self.run_model_building()
        self.run_post_build_feature_selection()
        self.run_cross_vallidation()
        self.generate_external_stats()
        self.run_applicability_domain()
        self.add_cv_info_to_df_training()
        self.extract_ad_stats()
        self.get_mapped_property_values_from_db()
        self.add_source_chemical_info_to_dfs()
        self.log_stats_to_console()
        self.create_dirs()
        self.append_info_to_model_object()
        self.create_results_dict()
        self.write_files()

        return self.model

# -------------------------------------------------------------------------------------------------------------

def test_run_dataset():
    write_to_db = False
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    qsar_method = "gcm"
    feature_selection = False
    splitting_name = "RND_REPRESENTATIVE"
    unique_identifier = None
    append_to_models_folder = "_RunDataset_test"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    test_dataset = RunDataset(
        dataset_name=dataset_name,
        qsar_method=qsar_method,
        feature_selection=feature_selection,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        ad_measure=ad_measure_model,
        write_to_db=write_to_db,
        unique_identifier=unique_identifier,
        append_to_models_folder=append_to_models_folder
    )

    test_dataset.run()

# EXAMPLE OF DESIRED USAGE
# def example():
#     run_dataset = RunDataset()

#     # OPTION A (Control only needed params like qsar_method)
#     params = run_dataset.create_params(qsar_method) # Everything handled automatically

#     # OPTION B (Fine control of params/hyperparams) (Not used by default)
#     params = run_dataset.create_params(
#         a=a,
#         b=b,
#         ...
#     )

#     run_dataset.run(params)
