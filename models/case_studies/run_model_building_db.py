
'''
Created on Dec 30, 2025
@author: TMARTI02
'''
from dns.flags import AD

"""
from __future__ import annotations lets you:
-Use forward references without quotes (refer to classes not yet defined)
-Avoid importing heavy typing/related modules at import time
-Speed up imports and reduce circular-import issues
"""

from datetime import datetime
import os, json

# from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from xlsxwriter.utility import xl_rowcol_to_cell
from sqlalchemy.exc import SQLAlchemyError

import numpy as np
import pandas as pd
from io import StringIO
import math
from pathlib import Path
import time

import pickle 
import re
import traceback

import util.database_utilities as dbl

from model_ws_utilities import call_build_embedding_ga_db, call_build_model_with_preselected_descriptors_from_df, \
    call_do_predictions_from_df, call_build_embedding_importance_from_df

from models.EmbeddingFromImportance import perform_iterative_recursive_feature_elimination as run_rfe
from models.EmbeddingFromImportance import perform_sequential_feature_selection as run_sfs

import StatsCalculator as sc

from models.dataset_utilities_db import get_training_prediction_instances, get_training_cv_instances

from utils import print_first_row, row_to_json, to_json_safe

from applicability_domain import applicability_domain_utilities as  adu

custom_level_styles = {
    'debug': {'color': 'cyan'},
    'info': {'color': 'yellow'},
    'warning': {'color': 'red', 'bold': True},
    'error': {'color': 'white', 'background': 'red'},
}

import logging
from logging import INFO, DEBUG, ERROR
import coloredlogs

level = INFO

coloredlogs.install(level=level, milliseconds=True, level_styles=custom_level_styles,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

from typing import Optional, Dict, Any, Iterable, Tuple
from dataclasses import dataclass, field, asdict

feature_selection_method_genetic_algorithm = "Genetic algorithm"
feature_selection_method_group_contribution = "Group contribution" 
feature_selection_method_importance = "Importance"

from predict_constants import PredictConstants as pc

# PROJECT_ROOT=r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 python\modeling services\pf_python_modelbuilding"    
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

 
@dataclass
class ParametersImportance:
    
    dataset_name: str
    qsar_method: str    
    descriptor_set_name: str
    ad_measure: list[str]

    splitting_name: str = "RND_REPRESENTATIVE"
    feature_selection_method: str = feature_selection_method_importance
    hyperparameter_grid: Optional[Dict[str, Any]] = None
    feature_selection: bool = False
    remove_log_p_descriptors: bool = False
    
    num_generations: int = 1

    use_permutative: bool = True
    use_wards: bool = False
    run_rfe: bool = True
    run_sfs: bool = True

    min_descriptor_count: int = 20
    max_descriptor_count: int = 30

    include_standardization_in_pmml: bool = False
    use_pmml_pipeline: bool = False
    n_threads: Optional[int] = 4  # Set to n/2 where n is the number of logical processors on your computer

    # Derived value (set in __post_init__)
    fraction_of_max_importance: float = field(init=False)

    def __post_init__(self):
        method = self.qsar_method.lower()
                
        if method == "rf":
            self.fraction_of_max_importance = 0.25
        elif method == "xgb":
            self.fraction_of_max_importance = 0.03
        elif method == "lgb":
            self.fraction_of_max_importance = 0.03  # TODO this needs checking
        else:
            raise ValueError(f"invalid method: {self.qsar_method}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

@dataclass
class ParametersGroupContribution:

    dataset_name: str
    qsar_method: str    
    descriptor_set_name: str
    ad_measure: list[str]
    splitting_name: str = "RND_REPRESENTATIVE"
    
    feature_selection_method: str = feature_selection_method_group_contribution 
    hyperparameter_grid: Optional[Dict[str, Any]] = None
    feature_selection: bool = True
    min_count = 3  # minimum number of nonzero fragment values to keep a fragment column and its associated rows
    remove_log_p_descriptors: bool = False
    n_threads: int = 10
    include_standardization_in_pmml: bool = False
    use_pmml_pipeline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParametersGeneric:

    dataset_name: str
    qsar_method: str    
    descriptor_set_name: str
    ad_measure: list[str]

    splitting_name: str = "RND_REPRESENTATIVE"
    
    hyperparameter_grid: Optional[Dict[str, Any]] = None
    feature_selection: bool = False
    remove_log_p_descriptors: bool = False
    n_threads: int = 10
    include_standardization_in_pmml: bool = False
    use_pmml_pipeline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParametersGeneticAlgorithm:

    dataset_name: str
    qsar_method: str    
    descriptor_set_name: str
    ad_measure: list[str]

    splitting_name: str = "RND_REPRESENTATIVE"

    feature_selection_method: str = feature_selection_method_genetic_algorithm 
    hyperparameter_grid: Optional[Dict[str, Any]] = None
    feature_selection: bool = False
    remove_log_p_descriptors: bool = False

    num_generations: int = 100
    num_optimizers: int = 100
    num_jobs: int = 4
    n_threads: Optional[int] = None  # set to an int (e.g., 4) if you want to pin threads
    max_length: int = 24  # max number of variables
    descriptor_coefficient: float = 0.002
    threshold: int = 1
    elitism = True
    crossover_probability = 0.9
    mutation_probability = 0.05
    max_features = 20

    use_wards: bool = False
    run_rfe: bool = True
    run_sfs: bool = True
    remove_fragment_descriptors: bool = True
    remove_acnt_descriptors: bool = True
    use_wards: bool = False

    include_standardization_in_pmml: bool = False
    use_pmml_pipeline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


from typing import Mapping
from sqlalchemy.orm import Session
from util.database_utilities import DatabaseLoader

dbl = DatabaseLoader(default_schema="qsar_models")


class ModelLoader():

    def add_model_statistics(self, user, fk_model_id, stats_dict_name, results, stats_lookup, created_at,  model_statistics_rows):        
        
        stats = results[stats_dict_name]
        
        for stat_name in stats:
        
            if "AD" not in stats_dict_name and "Coverage" in stat_name:
                continue
        
            if stat_name =="ad_measure" or stat_name=="mae_ratio":
                continue
        
            if stat_name in stats_lookup:
                fk_statistic_id = stats_lookup[stat_name]
                # print(stat_name,fk_statistic_id)
                model_statistics_row = {
                    "statistic_value":stats[stat_name], 
                    "fk_model_id":fk_model_id, 
                    "fk_statistic_id":fk_statistic_id, 
                    "created_by":user, 
                    "updated_by":user, 
                    "created_at":created_at, 
                    "updated_at":created_at}
                model_statistics_rows.append(model_statistics_row)                
                # print(stat_name,"\n",json.dumps(to_json_safe(model_statistics_row),indent=4))
                
                # print(stat_name,stats[stat_name],fk_statistic_id)
                
            else:
                print(stat_name, "Skipping loading stat")

    def load_stats(self,session, results, user, fk_model_id):
        
        stats_rows = dbl.get_rows(session,"statistics")
        stats_lookup={} #lookup for the fk_statistic_id
        for stat_row in stats_rows:
            stats_lookup[stat_row.name]=stat_row.id

        # dbl.print_rows_as_json(stats_rows)
        # print(json.dumps(stats_dict, indent=4))
        # dbl.print_rows_as_json(stats_rows)
        
        created_at = datetime.now()
        model_statistics_rows=[]
        
        dict_list = ["training_stats","test_stats","cv_stats","test_stats_AD"]
        
        for dict_name in dict_list:                
            self.add_model_statistics(
                user=user,
                fk_model_id=fk_model_id,
                stats_dict_name=dict_name,
                results=results, 
                stats_lookup=stats_lookup,
                created_at = created_at,
                model_statistics_rows=model_statistics_rows,
            )
        # print(json.dumps(model_statistics_rows,indent=4))
        
        stats_row_ids = dbl.create_many(session, "model_statistics", model_statistics_rows)
        session.commit()
        
        # for stats_row_id in stats_row_ids:
        #     print(stats_row_id)

    from typing import List, Union
    
    def divide_array(self, source: Union[bytes, bytearray], chunksize: int = 26214400) -> List[Union[bytes, bytearray]]:
        """
        Split a bytes-like object into chunks of size `chunksize`.
        Returns a list of slices (copies). Last chunk may be shorter.
        """
        parts = [source[i:i + chunksize] for i in range(0, len(source), chunksize)]
    
        # Optional: mimic the Java prints
        print(f"Size of model bytes={len(source)}")
        print(f"# Parts = {len(parts)}")
    
        total = sum(len(p) for p in parts)
        if total != len(source):
            print(f"byte length mismatch:{total}\t{len(source)}")
            return None
    
        return parts

    def load_model_from_object(self, session, user, model, params, fk_descriptor_embedding_id, fk_method_id, fk_ad_method):

        epoch_ms = time.time_ns() // 1_000_000
        created_at = datetime.now()

        model_name = user + "_" + str(epoch_ms)
        model_row = {
            "name":model_name,
            "dataset_name":params["dataset_name"],
            "descriptor_set_name":params["descriptor_set_name"],
            "splitting_name":params["splitting_name"],
            "fk_method_id":fk_method_id,  # todo lookup from qsar_method, use general method instead of versioned
            "fk_descriptor_embedding_id":fk_descriptor_embedding_id,
            "fk_source_id":3,  # cheminformatics modules, TODO lookup from sources table
            "fk_ad_method":fk_ad_method,  # lookup from ad_methods table using ad_method_name currently cant have multiple AD methods the way the db is configured
            "hyperparameter_grid":json.dumps(model.hyperparameter_grid),
            "hyperparameters":json.dumps(model.hyperparameters),  # JSON/JSONB column
            "details":model.get_model_description().encode("utf-8"),  # this column is bytes in the database. In the future that column should be converted to text field
            "is_public":False,
            "name_ccd":model_name,
            "has_qmrf":False,
            "created_by":user,
            "updated_by":user,
            "created_at":created_at,
            "updated_at":created_at}
        # print(json.dumps(to_json_safe(model_row)))
        # print(json.dumps(model_row,indent=4))
        fk_model_id = self.create_model(session, model_row)
        return fk_model_id



    def create_model_bytes(self, session: Session, bytes_list):
    
        try:
            # Insert records
            result = dbl.create_many(session, table="model_bytes", records=bytes_list)
            # Ensure SQL is sent and constraints checked before commit
            session.flush()
            # Commit the transaction
            session.commit()
            # At this point, success if we reached here without exception
        
            print("Model bytes loaded")
            
            return result
        except SQLAlchemyError:
            # Roll back on any DB/SQLAlchemy error
            session.rollback()
            # Re-raise or log the error
            raise        


    def load_model_bytes(self, session, user, model, fk_model_id):
        model_bytes = pickle.dumps(model)
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
        return self.create_model_bytes(session, bytes_rows)
        

    def load_model(self, session, user, model, results, df_predictions, test_stats):

        params = results["params"]

        # ---- get fk_descriptor_embedding_id ----
        embedding_tsv = "\t".join(results["embedding"])
        # print(embedding_tsv)
        embedding_row = dbl.get_row(session, "descriptor_embeddings", embedding_tsv=embedding_tsv, dataset_name=params["dataset_name"])
        if embedding_row is None:
            fk_descriptor_embedding_id = self.create_descriptor_embedding_from_params(session, user, results)
            logging.info(f"descriptor_embedding created: {fk_descriptor_embedding_id}")
        else:
            fk_descriptor_embedding_id = embedding_row.id
            logging.info(f"descriptor_embedding in database: {fk_descriptor_embedding_id}")
        
        # fk_descriptor_embedding_id=281
        if fk_descriptor_embedding_id is None:
            logging.error("Cant create embedding")
            return
        
        # ---- get fk_method_id ----
        row_method = dbl.get_row(session, "methods", name=params["qsar_method"])
        if row_method is not None:
            fk_method_id = row_method.id
        else:    
            logging.error(f"Cant find fk for qsar_method={params['qsar_method']}")
            return
        logging.info(f"fk_method_id:{fk_method_id}")
        
        # ---- get fk_ad_method_id ----
        ad_measure = " and ".join(params["ad_measure"])
        row_ad_method = dbl.get_row(session, "ad_methods", name=ad_measure)
        if row_ad_method is not None:
            fk_ad_method = row_ad_method.id
            logging.info(f"fk_ad_method:{fk_ad_method}")
        else:
            logging.error(f"Cant find fk for ad_measure={ad_measure}")
            return
        
        # ---- store model into the models table:----
        # fk_model_id = self.load_model_from_object(session, user, model, params, fk_descriptor_embedding_id, fk_method_id, fk_ad_method)

        fk_model_id = 1638
        
        if fk_model_id is None:
            logging.error(f"Cant create model")
            return
        logging.info(f"fk_model_id:{fk_model_id}")
        
        # ---- store model_bytes into the model_bytes table:----
        # self.load_model_bytes(session, user, model, fk_model_id)


        # ---- store model_statistics into the model_statistics table:----        
        # self.load_stats(session, results, user, fk_model_id)
        
        
        
        """
        TODO store predictions for:
        1. Test set
        2. Prediction set
        3. Cross-validation of training set
        4. Training set for completeness?
        """

        
    
    def create_descriptor_embedding_from_params(self, session, user, results):
        
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
            "embedding_tsv": "\t".join(results["embedding"]),
            "importance_tsv": "Not used",
            "created_by": user,
            "updated_by": user,
            "created_at": created_at,
            "updated_at": created_at,
        }

        new_id = dbl.create_row(session, table="descriptor_embeddings", record=descriptor_embedding)
        session.commit()
        return new_id
    
    def create_descriptor_embedding(self, session: Session, descriptor_embedding_dict: Mapping[str, Any]):
        new_id = dbl.create_row(session, table="descriptor_embeddings", record=descriptor_embedding_dict)
        session.commit()
        # print(new_id)  # typically an integer or UUID
        return new_id



    
    def create_model(self, session: Session, model_dict: Mapping[str, Any]):
        new_id = dbl.create_row(session, table="models", record=model_dict)
        session.commit()
        # print(new_id)  # typically an integer or UUID
        return new_id
    
   
    def test_create_model(self):
    
        from predict_constants import PredictConstants
        pc=PredictConstants    
        from model_ws_db_utilities import getSession    
        session = getSession()
                
        user = 'tmarti02'
        qsar_method = 'knn'
        embedding=["col1","col2"]
        dataset_name = "KOC v1 modeling"
        descriptor_set_name = pc.DESCRIPTOR_SET_WEBTEST
        splitting_name = pc.SPLITTING_RND_REPRESENTATIVE
        
        epoch_ms = time.time_ns() // 1_000_000
        name_cols = [dataset_name, descriptor_set_name, str(epoch_ms)]
        embedding_name = "_".join(name_cols) # give it a unique name
        from datetime import datetime
        
        created_at = datetime.now()
        
        descriptor_embedding = {
            "name": embedding_name,
            "description": "Genetic algorithm with RFE and SFS",
            "dataset_name": dataset_name,
            "descriptor_set_name": descriptor_set_name,
            "qsar_method": qsar_method,
            "splitting_name": splitting_name,
            "embedding_tsv": "\t".join(embedding),
            "importance_tsv": "Not used",
            "created_by": user,
            "updated_by": user,
            "created_at": created_at,
            "updated_at": created_at,
        }
        
        
        from utils import to_json_safe
        # print(to_json_safe(descriptor_embedding))
            
        fk_descriptor_embedding_id = self.create_descriptor_embedding(session, descriptor_embedding)
        
        # fk_descriptor_embedding_id = 276
    
        from models.ModelBuilder import Model
        
        model = Model()
        model.hyperparameter_grid = {"n_estimators": 500, "max_depth": 20}
        model.hyperparameters= {"n_estimators": 500, "max_depth": 20}
    
        model_name = user+"_"+str(epoch_ms)
    
        fk_method_id = self.get_method_id(session, qsar_method) #use the generic method_id that doesnt have a version so can set the hyperparameter_grid
        
    
        model_row = {
            "name": model_name,
            "dataset_name": dataset_name,
            "descriptor_set_name": descriptor_set_name,
            "splitting_name": splitting_name,
            "fk_method_id": fk_method_id, #todo lookup from qsar_method, use general method instead of versioned
            "fk_descriptor_embedding_id": fk_descriptor_embedding_id,
            "fk_source_id": 3, #cheminformatics modules, TODO lookup from sources table
            "fk_ad_method": 7, #lookup from ad_methods table using ad_method_name currently cant have multiple AD methods the way the db is configured        
            "hyperparameter_grid": json.dumps(model.hyperparameter_grid),
            "hyperparameters": json.dumps(model.hyperparameters),  # JSON/JSONB column
            "details": model.get_model_description().encode("utf-8"),  #this column is bytes in the database. In the future that column should be converted to text field
            "is_public": False,
            "name_ccd": model_name,
            "has_qmrf": False,
            "created_by": user,
            "updated_by": user,
            "created_at": created_at,
            "updated_at": created_at,
        }
        
        print(json.dumps(to_json_safe(model_row)))
        
        # print(json.dumps(model_row,indent=4))
        fk_model_id = self.create_model(session, model_row)
        

class ModelBuilder:
    
    def crossvalidate(self, df_cv_dict, params, embedding):
        
        logging.info(f"Start running CV calculations ...")
        
        all_df_predictions = []
        
        folds = {}
        for i in range(1, 6):
            splittingName = 'RND_REPRESENTATIVE_CV' + str(i)
            
            logging.debug(f"Running CV split = {splittingName}")
            df_training = df_cv_dict[i]["train"]
            df_prediction = df_cv_dict[i]["pred"]
            folds[i] = {"train": df_training, "pred": df_prediction}
            
            df_predictions = self.build_and_test_model2(df_training.copy(), df_prediction.copy(), params, embedding)
            all_df_predictions.append(df_predictions)
        
        df_predictions_all = pd.concat(all_df_predictions, ignore_index=True)

        # print(all_df_predictions.shape)
        # print(df_predictions_all.shape)
        
        mean_exp_training = df_predictions_all["exp"].mean()
        cv_stats = sc.calculate_continuous_statistics(df_predictions_all, mean_exp_training, "_CV_Training")
        # print('cross validation stats:\n', json.dumps(cv_stats, indent=4))        

        logging.info(f"Done running CV calculations")
        
        return df_predictions_all, cv_stats  
        
        # logging.info(f"MAE_TRAINING_CV = {cv_stats['MAE_Test']:.3f}")

    # def buildModelNoEmbedding(self, session, df_training, df_prediction, dataset_name, descriptorSetName,
    #                           qsar_method, hyperparameter_grid=None):
    #     '''
    #     Builds model with no extra feature selection, doesnt let you set hyperparameters for scikit-learn        
    #     :param session:
    #     :param df_training:
    #     :param df_prediction:
    #     :param dataset_name:
    #     :param descriptorSetName:
    #     :param qsar_method:
    #     '''
    #
    #     # df_training, df_prediction = self.get_training_prediction_instances(session, dataset_name, descriptorSetName, splittingName)
    #     # self.build_and_test_model(qsar_method, include_standardization_in_pmml, use_pmml_pipeline, df_training, df_prediction, params, embedding_name, embedding)
    #     # self.crossvalidate(session, dataset_name, descriptorSetName, qsar_method, include_standardization_in_pmml, use_pmml_pipeline, params, embedding_name, embedding)
    #
    #     params = ParametersNoEmbedding(qsar_method, hyperparameter_grid)
    #
    #     embedding = None
    #
    #     self.build_and_test_model(df_training, df_prediction, params, embedding)
        
        # self.crossvalidate(session, dataset_name, descriptorSetName, params, embedding)
    
    def build_and_test_model(self, df_training, df_prediction, params, embedding):
        
        model = call_build_model_with_preselected_descriptors_from_df(params, df_training.copy(), df_prediction.copy(),
                                                                      descriptor_names_tsv=embedding,
                                                                      filterColumnsInBothSets=True)
        
        # generate predictions for test set:
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        json_predictions_training = call_do_predictions_from_df(df_training, model)
        
        df_predictions = pd.read_json(StringIO(json_predictions), orient="records")
        df_predictions_training = pd.read_json(StringIO(json_predictions_training), orient="records")
        
        # first_row_dict = df_predictions.loc[0].to_dict()
        # print(json.dumps(first_row_dict, indent=4))
    
        # calculate stats for test set
        mean_exp_training = df_training["Property"].mean()
        test_stats = sc.calculate_continuous_statistics(df_predictions, mean_exp_training, "_Test")
        training_stats = sc.calculate_continuous_statistics(df_predictions_training, mean_exp_training, "_Training")
                
        return df_predictions, training_stats, test_stats, model
                
        # logging.info(f"MAE_TEST = {test_stats['MAE_Test']:.3f}")

    def build_and_test_model2(self, df_training, df_prediction, params, embedding):

        model = call_build_model_with_preselected_descriptors_from_df(params, df_training, df_prediction,
                                                  descriptor_names_tsv=embedding, filterColumnsInBothSets=True)
        # generate predictions for test set:
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        df_predictions = pd.read_json(StringIO(json_predictions), orient="records")
        
        # mean_exp_training = df_training["Property"].mean()
        # test_stats = sc.calculate_continuous_statistics(df_predictions, mean_exp_training, "_Test")
        # print(test_stats["MAE_Test"])
        return df_predictions


class EmbeddingGenerator:
    
    def feature_selection(self, df_training, df_prediction, params):
        
        # ga_methods = ['knn', 'reg','las']
        # imp_methods = ['rf', 'xgb']
        
        if params.feature_selection_method == feature_selection_method_genetic_algorithm:
            embedding, _ = call_build_embedding_ga_db(df_training, df_prediction, params)
        
        elif params.feature_selection_method == feature_selection_method_importance:
            ip = params
            embedding, _ = call_build_embedding_importance_from_df(params.qsar_method, df_training, df_prediction, ip.remove_log_p_descriptors,
                                                       ip.n_threads, ip.num_generations, ip.use_permutative, ip.run_rfe,
                                                       ip.fraction_of_max_importance, ip.min_descriptor_count, ip.max_descriptor_count,
                                                       ip.use_wards, hyperparameter_grid=ip.hyperparameter_grid, run_sfs=ip.run_sfs)

        elif params.feature_selection_method == feature_selection_method_group_contribution:
            # embedding = call_build_embedding_group_contribution_from_df(df_training, params.min_count)
            embedding = None  # determine during model building
        
        else:
            print("cant do feature selection for " + params.qsar_method)
            return None

        # logging.info(f"qsar_method={params.qsar_method}, embedding generated with {len(embedding)} descriptors: {embedding}")

        return embedding


def getSession():
    connect_url = URL.create(
        drivername='postgresql+psycopg2',
        username=os.getenv('DEV_QSAR_USER'),
        password=os.getenv('DEV_QSAR_PASS'),
        host=os.getenv('DEV_QSAR_HOST', 'localhost'),
        port=os.getenv('DEV_QSAR_PORT', 5432),
        database=os.getenv('DEV_QSAR_DATABASE')
    )
    # print(connect_url)
    engine = create_engine(connect_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def prepare_df(df):
    # Validate required columns
    required_cols = {"id", "exp", "pred"}
    if not required_cols.issubset(df.columns):
        missing = list(required_cols - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")
# Ensure numeric, compute abs_diff, and sort descending by abs_diff
    df = df.copy()
    df["exp"] = pd.to_numeric(df["exp"], errors="coerce")
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df = df.dropna(subset=["exp", "pred"])
    if df.empty:
        raise ValueError("No valid rows to plot after cleaning.")
    df["abs_diff"] = abs(df["exp"] - df["pred"])
    df = df.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return df
    
    # statistics_AD = None
    # if doAD:
    #     df_results, statistics_AD = runAD(mp.is_binary, adMeasure, df_results, model, training_tsv, prediction_tsv)


class ExcelCreator:

    def add_subtotal_count_fraction_and_visible(self, ws, df, column_name='abs_diff'):
        """
        Add rows below the data with formulas in the target column and place labels
        in the adjacent column (next to the formula cells):
    
          - Subtotal (numeric visible count via SUBTOTAL(102, ...))
          - fraction_inside (visible_count / numeric_count for that column)
          - visible_count (visible non-empty cells via SUBTOTAL(103, ...))
    
        Labels are written in the column adjacent to the target column:
          - left if possible; otherwise right.
        
        Returns the list of formulas written, or zeros if no data rows, for reusability
        """
        if column_name not in df.columns:
            return  # target column not present
    
        nrows, _ = df.shape
        col_idx = df.columns.get_loc(column_name)
    
        # Rows (0-based): header at 0, data 1..nrows
        subtotal_row = nrows + 2
        fraction_row = nrows + 3
        visible_row = nrows + 4
    
        # Label column: place next to the formula cells
        label_col = col_idx - 1 if col_idx > 0 else col_idx + 1
    
        # Write labels next to the formula cells
        ws.write(subtotal_row, label_col, 'MAE')  # numeric visible count
        ws.write(fraction_row, label_col, 'fraction_inside')  # visible_count / numeric_count
        ws.write(visible_row, label_col, 'count_inside')  # visible non-empty count
    
        if nrows > 0:
            # Range for the target column (data rows only)
            first_cell = xl_rowcol_to_cell(1, col_idx)
            last_cell = xl_rowcol_to_cell(nrows, col_idx)
            range_ref = f"{first_cell}:{last_cell}"
    
            # Numeric visible count (ignores filtered-out rows)
            count_numeric_formula = f"=SUBTOTAL(101,{range_ref})"
            ws.write_formula(subtotal_row, col_idx, count_numeric_formula)
    
            # Visible non-empty count (ignores filtered-out rows)
            visible_count_formula = f"=SUBTOTAL(103,{range_ref})"
            ws.write_formula(visible_row, col_idx, visible_count_formula)
    
            # fraction_inside = visible_count / numeric_count for that column (guard against divide-by-zero)
            fraction_formula = f"=IF(COUNT({range_ref})>0,SUBTOTAL(103,{range_ref})/COUNT({range_ref}),0)"
            ws.write_formula(fraction_row, col_idx, fraction_formula)
            return [count_numeric_formula, fraction_formula, visible_count_formula]
        else:
            # No data rows; write zeros
            ws.write(subtotal_row, col_idx, 0)
            ws.write(fraction_row, col_idx, 0)
            ws.write(visible_row, col_idx, 0)
            return [0, 0, 0]
    
    def add_filter(self, writer, sheet_name, df):
        
        # Get the worksheet object
        ws = writer.sheets[sheet_name]

        # Determine the range (0-indexed for xlsxwriter)
        nrows, ncols = df.shape

        # Add an auto-filter to the full range (header row is 0; data ends at row nrows)
        ws.autofilter(0, 0, nrows, ncols - 1)

        # Optional: freeze the header row
        ws.freeze_panes(1, 0)
    
    def nice_integer_major_unit(self, span: int, target_ticks: int=5) -> int:
        raw = max(1, int(math.ceil(span / max(1, target_ticks))))
        exp = int(math.floor(math.log10(raw))) if raw > 0 else 0
        base = raw / (10 ** exp)
        if base <= 1:
            step_base = 1
        elif base <= 2:
            step_base = 2
        elif base <= 5:
            step_base = 5
        else:
            step_base = 10
        return int(step_base * (10 ** exp))

    def compute_equal_axis_bounds(self,
        x_values: Iterable[float],
        y_values: Iterable[float],
        pad_ratio: float=0.02,
        integer_ticks: bool=True,
        target_ticks: int=5,
    ) -> Tuple[float, float, Optional[float]]:
        
        xs = [float(v) for v in x_values if v is not None and v == v]
        ys = [float(v) for v in y_values if v is not None and v == v]
        if not xs or not ys:
            raise ValueError("No numeric values provided for axis scaling.")
    
        mn = min(min(xs), min(ys))
        mx = max(max(xs), max(ys))
    
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        else:
            span = mx - mn
            mn -= span * pad_ratio
            mx += span * pad_ratio
    
        if integer_ticks:
            int_min = math.floor(mn)
            int_max = math.ceil(mx)
            if int_min == int_max:
                int_min -= 1
                int_max += 1
            major_unit = self.nice_integer_major_unit(int_max - int_min, target_ticks=target_ticks)
            return float(int_min), float(int_max), float(major_unit)
    
        return mn, mx, None

    def add_plot(self, df, sheet_name, sheet_name_plot, chart_size_px, pad_ratio, integer_ticks, yx_offset_rows, writer, workbook):
        worksheet = writer.sheets[sheet_name]
        worksheet_plot = writer.sheets[sheet_name_plot]
    # Hide worksheet gridlines (screen + print)
        worksheet.hide_gridlines(2)
        nrows = len(df)
    # Compute unified bounds (also used for y=x line)
        mn, mx, major_unit = self.compute_equal_axis_bounds(
            df["exp"], df["pred"], pad_ratio=pad_ratio, integer_ticks=integer_ticks, target_ticks=5)
    # Create scatter chart with markers
        chart = workbook.add_chart({"type":"scatter", "subtype":"straight_with_markers"})
        chart.set_title({"name":"Predicted vs Experimental"})
        chart.set_style(10)
    # Series 1: data points (markers only)
    # exp is column B (index 1), pred is column C (index 2) even after adding abs_diff (column D)
        chart.add_series({
                "name":"Pred vs Exp",
                "categories":[sheet_name, 1, 1, nrows, 1],  # B2..B(nrows+1)
                "values":[sheet_name, 1, 2, nrows, 2],  # C2..C(nrows+1)
                "marker":{"type":"circle", "size":6},
                "line":{"none":True}})
    # Write y = x helper points a few rows below the data, in columns B/C
        yx_row_start = nrows + 1 + yx_offset_rows  # 0-based row index after header
        worksheet.write_number(yx_row_start, 1, mn)  # B{yx_row_start+1}
        worksheet.write_number(yx_row_start, 2, mn)  # C{yx_row_start+1}
        worksheet.write_number(yx_row_start + 1, 1, mx)  # B{yx_row_start+2}
        worksheet.write_number(yx_row_start + 1, 2, mx)  # C{yx_row_start+2}
    # Series 2: y = x line (dark blue, solid, no markers)
        chart.add_series({
                "name":"y = x",
                "categories":[sheet_name, yx_row_start, 1, yx_row_start + 1, 1],  # B(r)..B(r+1)
                "values":[sheet_name, yx_row_start, 2, yx_row_start + 1, 2],  # C(r)..C(r+1)
                "marker":{"type":"none"},
                "line":{"color":"#1f4e79", "width":2.25}})  # dark blue
    # Axes with same bounds, integer labels, no gridlines
        x_axis_opts = {
            "name":"exp",
            "min":mn, "max":mx,
            "num_format":"0",
            "crossing": "min",
            "major_gridlines":{"visible":False},
            "minor_gridlines":{"visible":False}}
        y_axis_opts = {
            "name":"pred",
            "min":mn, "max":mx,
            "num_format":"0",
            "crossing": "min",
            "major_gridlines":{"visible":False},
            "minor_gridlines":{"visible":False}}
        if major_unit is not None:
            x_axis_opts["major_unit"] = major_unit
            y_axis_opts["major_unit"] = major_unit
        
        chart.set_x_axis(x_axis_opts)
        chart.set_y_axis(y_axis_opts)
                
    # Make the chart square
        chart.set_size({"width":chart_size_px, "height":chart_size_px})
    # Plot area: border + margins so axis titles/labels aren't overlapped
        chart.set_plotarea({
                "fill":{"none":True},
                "border":{"color":"#666666", "width":1.0},
                "layout":{
                    "x":0.12,  # left padding for y-axis title + labels
                    "y":0.10,  # top padding for chart title
                    "width":0.84,  # reduce width so right edge doesn't crowd labels/legend
                    "height":0.80}})  # reduce height to leave room for x-axis title
    # Legend: bottom-right, overlaid, explicit size to avoid warnings
        chart.set_legend({
                "overlay":True,
                "layout":{"x":0.7, "y":0.75, "width":0.25, "height":0.1}})
    # Insert chart
        if sheet_name == sheet_name_plot: 
            worksheet_plot.insert_chart("E2", chart, {"x_offset":20, "y_offset":10})
        else:
            worksheet_plot.insert_chart(0, 0, chart, {'x_scale': 1.0, 'y_scale': 1.0})

    def writeModelCoefficients(self, results_dict, writer, workbook): 
        if results_dict and "model_coefficients" in results_dict:
            sheet_name = "model coefficients"
            df = pd.DataFrame(results_dict["model_coefficients"], columns=["name", "coefficient", "std_error"])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            text_fmt = workbook.add_format({"num_format":"@"})
            name_col_idx = df.columns.get_loc("name")  # usually 0
            worksheet = writer.sheets[sheet_name]
            # Overwrite each cell in the Name column as a string to prevent formula evaluation
            # Header is row 0; data starts at row 1 in XlsxWriter's 0-based indexing.
            for i, val in enumerate(df["name"], start=1):
                worksheet.write_string(i, name_col_idx, val, text_fmt)

    def writeDescriptors(self, sheet_name, df, writer, workbook):
        if df is not None:
            # df.to_excel(writer, sheet_name="training set descriptors", index=False)
            
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            # Optional header format
            header_fmt = workbook.add_format({"bold":True})
            # Pandas writes headers at row=0, col=0 when index=False
            # If index=True, headers start at col=1 (index header at col=0)
            start_row = 0
            start_col = 0
            col_offset = 0
            safe_cols = [str(c).lstrip("'") for c in df.columns]
            # Overwrite header cells as literal strings (prevents Excel from treating "=..." as formulas)
            for col_idx, col_name in enumerate(safe_cols, start=start_col + col_offset):
                worksheet.write_string(start_row, col_idx, col_name, header_fmt)

    def add_summary_stats_to_ws(self, ws, source_ws, summary_stats, start_row=2, start_col=10):
        """
        Adds summary statistics from source_ws to ws at the given position.
        
        :param ws: Worksheet to write summary statistics to
        :param source_ws: Name of the worksheet to take stats from
        :param summary_stats: Formulas/values for MAE, fraction_inside, count_inside
        :param start_row: Starting row in ws to put values
        :param start_col: Starting column in ws to put values
        """
        if summary_stats is None:
            return
        else:
            # Adjust formulas to point to values in source_ws
            for i, stat in enumerate(summary_stats):
                stat = str(stat)
                summary_stats[i] = re.sub(r"([A-Z]+\d+)", rf"'{source_ws}'!\1", stat)
        
        # Write each statistic label and formula/value
        row = start_row
        for label, formula in zip(
            ["MAE", "Coverage_Test", "count_inside"],
            summary_stats
        ):
            ws.write(row, start_col, label)
            ws.write_formula(row, start_col + 1, formula)
            row += 1

    def create_excel(self,
        df_test: pd.DataFrame,
        df_training_cv: pd.DataFrame,
        df_test_model: pd.DataFrame=None,
        results_dict=None,
        excel_path: str="report.xlsx",
        chart_size_px: int=520,  # square chart size
        pad_ratio: float=0.02,
        integer_ticks: bool=True,
        yx_offset_rows: int=3,  # empty rows between data and y=x helper points
    ):
        # TODO: Change file names for output Excel files under data directory
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            # Write data (now includes abs_diff column)
            workbook = writer.book
            
            if df_training_cv is not None:
                sheet_name_cv = "training cv predictions"
                df_training_cv.to_excel(writer, sheet_name=sheet_name_cv, index=False)
                self.add_plot(df_training_cv, sheet_name_cv, sheet_name_cv, chart_size_px, pad_ratio, integer_ticks, yx_offset_rows, writer, workbook)
                self.add_filter(writer, sheet_name_cv, df_training_cv)

            sheet_name_test = "test set predictions"
            df_test.to_excel(writer, sheet_name=sheet_name_test, index=False)
            self.add_filter(writer, sheet_name_test, df_test)
            ws = writer.sheets[sheet_name_test]
            summary_stats = self.add_subtotal_count_fraction_and_visible(ws, df_test, column_name='abs_diff')

            # self.add_plot(df_test, sheet_name_test, chart_size_px, pad_ratio, integer_ticks, yx_offset_rows, writer, workbook)
            
            # Create a dedicated sheet for the test set plot
            chart_sheet_name_test = "test set plot"
            # Add the worksheet explicitly and register it with writer.sheets
            chart_ws = workbook.add_worksheet(chart_sheet_name_test)
            writer.sheets[chart_sheet_name_test] = chart_ws
            self.add_plot(df_test, sheet_name_test, chart_sheet_name_test, chart_size_px, pad_ratio, integer_ticks, yx_offset_rows, writer, workbook)
            self.add_summary_stats_to_ws(chart_ws, sheet_name_test, summary_stats, start_row=2, start_col=10)

            self.writeDescriptors("test set descriptors", df_test_model, writer, workbook)
            self.writeModelCoefficients(results_dict, writer, workbook)
        

def set_hyper_parameters(qsar_method, feature_selection, descriptor_set_name, splitting_name, dataset_name, ad_measure):

    params = None    
        
    if qsar_method == "xgb":
        grid = {'estimator__booster': ['gbtree']}
        params = ParametersImportance(qsar_method=qsar_method, feature_selection=feature_selection, hyperparameter_grid=grid,
                                      descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                      splitting_name=splitting_name, ad_measure=ad_measure)
        
        # print(params.dataset_name)
        
    elif qsar_method == "lgb":
        grid = {}        
        params = ParametersImportance(qsar_method=qsar_method, feature_selection=feature_selection, hyperparameter_grid=grid,
                                      descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                      splitting_name=splitting_name, ad_measure=ad_measure)
    
    elif qsar_method == "rf":
        # grid = {'estimator__max_features': ['sqrt', 'log2'],
        #                              'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                              'estimator__n_estimators': [10, 100, 250, 500]}

        grid = {"estimator__max_features": ["sqrt", "log2"]}

        params = ParametersImportance(qsar_method=qsar_method, feature_selection=feature_selection, hyperparameter_grid=grid,
                                      descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                      splitting_name=splitting_name, ad_measure=ad_measure)

    elif qsar_method == "knn": 
        # grid = {'estimator__n_neighbors': [5], 'estimator__weights': ['distance']}  # default, same as OPERA
        
        grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']}  # matches AD in terms of using 3
        
        params = ParametersGeneticAlgorithm(qsar_method=qsar_method, hyperparameter_grid=grid, feature_selection=feature_selection,
                                            descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                            splitting_name=splitting_name, ad_measure=ad_measure)
        
        # params.num_generations=1
        # params.num_optimizers=1
        # params.run_rfe = False #doesnt work for knn 

    elif qsar_method == "reg": 
        grid = {}  # default, same as OPERA
        params = ParametersGeneticAlgorithm(qsar_method=qsar_method, hyperparameter_grid=grid, feature_selection=feature_selection,
                                            descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                            splitting_name=splitting_name, ad_measure=ad_measure)
        
        params.remove_fragment_descriptors = True
        params.remove_acnt_descriptors = True        

    elif qsar_method == "las": 
        grid = {'estimator__alpha': [np.round(i, 5) for i in np.logspace(-4, 0, num=20)],
                                    'estimator__max_iter': [1000000]}
        params = ParametersGeneticAlgorithm(qsar_method=qsar_method, hyperparameter_grid=grid, feature_selection=feature_selection,
                                            descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                            splitting_name=splitting_name, ad_measure=ad_measure)
    elif qsar_method == "gcm":
        grid = {}        
        params = ParametersGroupContribution(qsar_method=qsar_method, hyperparameter_grid=grid, feature_selection=feature_selection,
                                             descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                             splitting_name=splitting_name, ad_measure=ad_measure)

    elif qsar_method == "svm":
        grid = {}        
        params = ParametersGeneric(qsar_method=qsar_method, hyperparameter_grid=grid, feature_selection=feature_selection,
                                             descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                             splitting_name=splitting_name, ad_measure=ad_measure)
    else:
        print('qsar_method not handled:', qsar_method)
        return None

    return params
        

def runAD(df_training, df_prediction, params, embedding, df_predictions, ad_measure, stats_dict):
    
    df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
        train_df=df_training.copy(), test_df=df_prediction.copy(),
        remove_log_p=params.remove_log_p_descriptors,
        embedding=embedding, applicability_domain=ad_measure,
        filterColumnsInBothSets=False,
        returnTrainingAD=False)
    
    count = df_ad_output.shape[0]
    count_inside = df_ad_output['AD'].eq(True).sum()
    coverage = count_inside / count
    
    # print_first_row(df_ad_output)
    colAD = "AD_" + ad_measure.replace(" ", "_")
    
    # append the AD to the predictions dataframe:
    df_predictions = df_predictions.merge(df_ad_output.rename(columns={'idTest':'id'})[['id', 'AD']], on='id', how='left').rename(columns={'AD':colAD})
 
    mae_inside = (df_predictions.loc[df_predictions[colAD].eq(True), 'exp'] 
              -df_predictions.loc[df_predictions[colAD].eq(True), 'pred']).abs().mean()

    mae_outside = (df_predictions.loc[df_predictions[colAD].eq(False), 'exp'] 
              -df_predictions.loc[df_predictions[colAD].eq(False), 'pred']).abs().mean()

    mae_ratio = mae_outside / mae_inside
       
    stats = {"ad_measure":ad_measure, "MAE_Test_inside_AD": mae_inside, "MAE_Test_outside_AD": mae_outside, "mae_ratio": mae_ratio,
             "Coverage_Test":coverage}    
    stats_dict[ad_measure] = stats

    # product = mae_ratio * coverage
    # print(f"ad_measure={ad_measure}, mae_ratio={mae_ratio:.2f}, Coverage={coverage:.2f}, Product={product:.2f}")
     
    return df_predictions


def generate_consensus_ad(df_predictions, stats_dict, ad_measure_final):
    
    # Build list of AD columns
    colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]

    # Rows where all AD flags are True (inside consensus AD)
    mask_all_true = df_predictions[colsAD].eq(True).all(axis=1)

    # MAE inside the consensus AD
    mae_inside = (df_predictions.loc[mask_all_true, 'exp'] - 
                  df_predictions.loc[mask_all_true, 'pred']).abs().mean()

    # Rows outside consensus AD: at least one AD flag is False
    mask_outside = ~mask_all_true

    # MAE outside the consensus AD
    mae_outside = (df_predictions.loc[mask_outside, 'exp'] - 
                   df_predictions.loc[mask_outside, 'pred']).abs().mean()
                                      
    mae_ratio = mae_outside / mae_inside
    
    ad_measure = " and ".join(ad_measure_final)
    
    total_rows = len(df_predictions)
    coverage = (mask_all_true.sum() / total_rows) if total_rows > 0 else float('nan')
    
    stats = {"ad_measure":ad_measure, "MAE_Test_inside_AD": mae_inside, "MAE_Test_outside_AD": mae_outside, "mae_ratio": mae_ratio,
             "Coverage_Test":coverage}    
    stats_dict[ad_measure] = stats


def add_log_p_martin_columns(df_training, df_prediction, cross_validate, df_cv_dict=None):
    """
    Does adding columns for my LOGP prediction work better than ALOGP and XLOGP?    
    """
    
    model_id = str(1069)
    pred_name = 'LOGP_Martin'
    from model_ws_db_utilities import add_model_prediction_to_df as add_mp
    df_prediction = add_mp(df_prediction, model_id, pred_name)  # will generate some XGB warnings
    df_training = add_mp(df_training, model_id, pred_name)
    
    if cross_validate:
        for fold_num in df_cv_dict:
            fold = df_cv_dict[fold_num]
            fold["train"] = add_mp(fold["train"], model_id, pred_name)
            fold["pred"] = add_mp(fold["pred"], model_id, pred_name)
    
    return df_training, df_prediction


def run_dataset(dataset_name, qsar_method, embedding=None, folder_embedding=None, cross_validate=True,
                run_AD=True, feature_selection=True, fs_previous_embedding=True, params=None,
                descriptor_set_name="WebTEST-default", splitting_name="RND_REPRESENTATIVE",
                ad_measure_model=None, add_LOGP_Martin=False, write_to_db=False, user="tmarti02"):
    
    try:
                   
        if qsar_method == 'gcm' or qsar_method == 'svm':
            feature_selection = False
        
        if embedding is not None:
            folder_embedding = 'custom'
            feature_selection = False
        elif folder_embedding is not None:
            feature_selection = False
            file_name_embedding = "results.json"
            file_path_embedding = os.path.join(PROJECT_ROOT, "data/models", dataset_name, folder_embedding, file_name_embedding)
            
            with open(file_path_embedding, "r", encoding="utf-8") as f:
                results = json.load(f)
                embedding = results["embedding"] 
                # print(f"from {folder_embedding}:{embedding}")
        else:
            fs_previous_embedding = False
                    
        # if True:
        #     return
    
        # **************************************************************************************************************************
        
        if ad_measure_model is None: 
            ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
        
        ad_measures = [] # list of measures for comparison purposes
        ad_measures.append(pc.Applicability_Domain_TEST_Embedding_Euclidean)
        ad_measures.append(pc.Applicability_Domain_TEST_All_Descriptors_Euclidean)
        # ad_measures.append(pc.Applicability_Domain_TEST_Embedding_Cosine)
        # ad_measures.append(pc.Applicability_Domain_TEST_All_Descriptors_Cosine)
        ad_measures.append(pc.Applicability_Domain_TEST_Fragment_Counts)
        ad_measures.append(pc.Applicability_Domain_OPERA_local_index)
        # ad_measures.append(pc.Applicability_Domain_OPERA_global_index) # Turn off when not doing feature selection on knn, fails due to a matrix singularity
        # ******************************************************************************************************
        # Print main info:
        logging.info(f"dataset_name={dataset_name}")
        logging.info(f"qsar_method={qsar_method}")
        logging.info(f"descriptor_set_name={descriptor_set_name}")
        logging.info(f"feature_selection={feature_selection}")
        logging.info(f"cross_validate={cross_validate}")
        # ******************************************************************************************************
        
        if params is None: 
            params = set_hyper_parameters(qsar_method, feature_selection, descriptor_set_name, splitting_name, dataset_name, ad_measure_model)
        # hyperparameter_grid = None # use default
        # ******************************************************************************************************
    
        session = getSession()
        mb = ModelBuilder()
        eg = EmbeddingGenerator()
    
        logging.info("start getting dataframes from db")
    
        df_training, df_prediction = get_training_prediction_instances(session, dataset_name, descriptor_set_name, params.splitting_name)
        # print(df_training.shape)
    
        df_cv_dict = None 
        if cross_validate:
            df_cv_dict = get_training_cv_instances(session, dataset_name, descriptor_set_name)
        
        if add_LOGP_Martin:
            df_training, df_prediction = add_log_p_martin_columns(df_training, df_prediction, cross_validate, df_cv_dict)
        
        logging.info("done getting dataframes from db")
        # ******************************************************************************************************
    
        if feature_selection:
            embedding = eg.feature_selection(df_training, df_prediction, params)
            
        df_predictions, training_stats, test_stats, model = mb.build_and_test_model(df_training, df_prediction, params, embedding)
        
        if not feature_selection and fs_previous_embedding and qsar_method != 'gcm':
    
            logging.info(f"Before FS, previous embedding has {len(model.embedding)} descriptors: {model.embedding}")
            
            if params.run_rfe:
                run_rfe(model, df_training, 1, 1)
                logging.info(f"After RFE, {len(model.embedding)} descriptors: {model.embedding}")
                
            if params.run_sfs:
                run_sfs(model, df_training)
                logging.info(f"After SFS, {len(model.embedding)} descriptors: {model.embedding}")
    
            # redo model and predictions:
            df_predictions, training_stats, test_stats, model = mb.build_and_test_model(df_training, df_prediction, params, model.embedding)
            logging.info(f"After FS, embedding has {len(model.embedding)} descriptors: {model.embedding}")
        
        embedding = None  # use model.embedding from here on 
        
        # ******************************************************************************************************
        # Cross validation calculations
        cv_stats = None
        df_cv_predictions = None
        
        if cross_validate: 
            df_cv_predictions, cv_stats = mb.crossvalidate(df_cv_dict, params, model.embedding)
            
        # ******************************************************************************************************
        # Applicability domain calcs:
        stats_dict = {}
        if run_AD:
            for ad_measure in ad_measures:
                df_predictions = runAD(df_training, df_prediction, params, model.embedding, df_predictions, ad_measure, stats_dict)
    
            if len(ad_measure) > 1:
                generate_consensus_ad(df_predictions, stats_dict, ad_measure_model)
    
            # print(json.dumps(stats_dict,indent=4))
        
        # ******************************************************************************************************
        # look at first prediction to make sure it looks right:
        logging.debug("First row of df_predictions:")
        logging.debug(row_to_json(df_predictions))
        # ******************************************************************************************************
        
        # create results file:
        
        r = Results()
        
        results_dict = r.create_results_dict(
            ad_measure_model=ad_measure_model,
            df_training=df_training,
            params=params,
            model=model,
            training_stats=training_stats,
            test_stats=test_stats,
            cv_stats=cv_stats,
            stats_dict=stats_dict
            )
                
        if model.embedding:
            columns = model.embedding.copy()
            columns.insert(0, "Property")
            columns.insert(0, "ID")
            df_test_model = df_prediction[columns]
            
        r.save_results(results_dict, df_predictions, df_cv_predictions, df_test_model, folder_embedding)
        logging.info(f"test set stats={json.dumps(test_stats, indent=4)}")
        logging.info(f"training cross validation stats={json.dumps(cv_stats, indent=4)}")   
        logging.info(f"test set AD stats={json.dumps( results_dict['test_stats_AD'] , indent=4)}")
        
        if write_to_db:
            ml = ModelLoader()
            ml.load_model(session, user, model, results_dict, df_predictions, test_stats)
        
        return results_dict
    
    except Exception:
        # Print the exception traceback to standard error
        traceback.print_exc()
        
    # coeff_dict = model.getOriginalRegressionCoefficients()
    # logging.debug(f"coeffs{coeff_dict}")

    
class Results:
    
    def save_results(self, results_dict, df_predictions, df_cv_predictions=None, df_test_model=None, folder_embedding=None):
    
        params = results_dict["params"]
        
        # print (json.dumps(params))
    
        df_predictions = prepare_df(df_predictions)
        
        if df_cv_predictions is not None:
            df_cv_predictions = prepare_df(df_cv_predictions)
    
        subfolder = params["qsar_method"] + "_" + params["descriptor_set_name"] + "_fs=" + str(params["feature_selection"])
    
        if folder_embedding is not None:
            subfolder = subfolder + "_" + folder_embedding
        
        path_segments = [PROJECT_ROOT, "data", "models", params["dataset_name"], subfolder]
        
        folder_path = os.path.join(*path_segments)
        
        logging.info(f"Results folder\n: {folder_path}")
        
        os.makedirs(folder_path, exist_ok=True)

        identifier = int(time.time() * 1000)  # time in ms as identifier
        
        prediction_csv_path = os.path.join(folder_path, f"predictions_{identifier}.csv")    
        df_predictions.to_csv(prediction_csv_path, index=False)
        
        prediction_excel_path = os.path.join(folder_path, f"predictions_{identifier}.xlsx")
        ec = ExcelCreator()
        ec.create_excel(df_predictions, df_cv_predictions, df_test_model, results_dict, prediction_excel_path)
        
        json_path = os.path.join(folder_path, "results.json")
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)
    
    def create_results_dict(self, ad_measure_model, df_training, params, model, training_stats, test_stats, cv_stats, stats_dict):
    
        results_dict = {"params":params.to_dict()}
        
        # if feature_selection:
            # results_dict["embedding"] = model.embedding
        
        results_dict["embedding"] = model.embedding
        
        results_dict["len(embedding)"] = len(model.embedding)
            
        qsar_method = params.qsar_method
        
        if qsar_method == 'reg' or qsar_method == 'las' or qsar_method == 'gcm':
            # results_dict['model_coefficients'] = json.loads(model.getOriginalRegressionCoefficients())
            y = df_training[df_training.columns[1]]
            X = df_training[model.embedding]
            results_dict['model_coefficients'] = json.loads(model.getOriginalRegressionCoefficients2(X, y))
        
        
        if len(stats_dict) > 0:
            results_dict["test_stats_all_AD"] = stats_dict

        results_dict["training_stats"] = training_stats
        
        if cv_stats:
            results_dict["cv_stats"] = cv_stats

        results_dict["test_stats"] = test_stats

        if len(stats_dict) > 0:
            str_ad_measure_final = " and ".join(ad_measure_model)                        
            results_dict["test_stats_AD"] = stats_dict[str_ad_measure_final]
            results_dict["test_stats_all_AD"] = stats_dict
        
        
        return results_dict
    
    def summarize_model_stats(self, dataset_name, excel_name="model_stats.xlsx", sheet_name="stats"):
        """
        Iterate subfolders, print model stats, collect them into a DataFrame,
        and save to an Excel file in that folder.
    
        Returns:
            df_stats (pd.DataFrame): Summary table of runs and stats.
            excel_path (str): Path to the saved Excel file.
        """
        
        folder = os.path.join(PROJECT_ROOT, "data", "models", dataset_name)
        os.makedirs(folder, exist_ok=True)
        
        print(folder)
    
        print("\n\nStats for all models for " + dataset_name)
        print("Run\tMAE_Test\tMAE_Training_CV\t#_variables")
    
        rows = []  # collect printed results for the dataframe
    
        for entry in Path(folder).iterdir():
            if entry.is_dir():
                json_path = entry / "results.json"
                if json_path.is_file():
                    try:
                        with json_path.open("r", encoding="utf-8") as f:
                            results = json.load(f)  # this is a dict
    
                            # Extract values safely
                            mae_test_val = results.get("test_stats", {}).get("MAE_Test", None)
                            mae_cv_val = results.get("cv_stats", {}).get("MAE_Test", None)
    
                            # Embedding length
                            if "len(embedding)" in results:
                                lenEmbedding = results["len(embedding)"]
                            elif "embedding" in results and isinstance(results["embedding"], (list, tuple)):
                                lenEmbedding = len(results["embedding"])
                            else:
                                lenEmbedding = None
    
                            # Format for printing (and store as strings to match the print)
                            mae_test_str = f"{mae_test_val:.3f}" if isinstance(mae_test_val, (int, float)) else "N/A"
                            mae_cv_str = f"{mae_cv_val:.3f}"   if isinstance(mae_cv_val, (int, float)) else "N/A"
                            lenEmb_str = str(lenEmbedding) if lenEmbedding is not None else "N/A"
    
                            print(f"{entry.name}\t{mae_test_str}\t{mae_cv_str}\t{lenEmb_str}")
    
                            rows.append({
                                "Run": entry.name,
                                "MAE_Test": mae_test_str,
                                "MAE_Training_CV": mae_cv_str,
                                "#_variables": lenEmb_str
                            })
    
                    except json.JSONDecodeError as e:
                        print(f"Skipping {json_path}: invalid JSON ({e})")
    
        # Save the collected results to Excel in the same folder
        df_stats = pd.DataFrame(rows, columns=["Run", "MAE_Test", "MAE_Training_CV", "#_variables"])
        excel_path = os.path.join(folder, excel_name)
    
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_stats.to_excel(writer, sheet_name=sheet_name, index=False)
    
            # Add autofilter and freeze header row
            ws = writer.sheets[sheet_name]
            nrows, ncols = df_stats.shape
            ws.autofilter(0, 0, nrows, ncols - 1)
            ws.freeze_panes(1, 0)
    
        print(f"Saved summary to: {excel_path}")
        return df_stats, excel_path



if __name__ == '__main__':
    
    ml=ModelLoader()
    ml.test_create_model()
    