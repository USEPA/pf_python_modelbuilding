'''
Created on Dec 30, 2025

@author: TMARTI02
'''
import os, json

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, bindparam

import pandas as pd
from io import StringIO
import logging
import sys

from model_ws_utilities import call_build_embedding_ga_db, call_build_model_with_preselected_descriptors_from_df,\
    call_do_predictions_from_df,call_build_embedding_importance_from_df2

import StatsCalculator as sc
from numba.core.types import none
from models.runGA import qsar_method


# logging.basicConfig(
#     level=logging.DEBUG, 
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     force=True
# )

custom_level_styles = {
    'debug': {'color': 'cyan'},
    'info': {'color': 'yellow'},
    'warning': {'color': 'red', 'bold': True},
    'error': {'color': 'white', 'background': 'red'},
}



from logging import INFO, DEBUG, ERROR
import coloredlogs
coloredlogs.install(level=INFO, milliseconds=True, level_styles=custom_level_styles,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')


include_standardization_in_pmml = True
use_pmml_pipeline = False
n_threads = 20


class ParametersNoEmbedding:

    def __init__(self, qsar_method):

        self.remove_log_p_descriptors = False
        self.n_threads = n_threads
        self.include_standardization_in_pmml = include_standardization_in_pmml
        self.use_pmml_pipeline = use_pmml_pipeline
        self.qsar_method=qsar_method


class ParametersImportance:
    
    def __init__(self, qsar_method):
        
        self.qsar_method = qsar_method
        self.remove_log_p_descriptors = False
        self.num_generations = 1

        self.use_permutative = True
        self.use_wards = False
        self.run_rfe = True
        
        self.min_descriptor_count = 20
        self.max_descriptor_count = 30
        
                
        if qsar_method == 'rf':
            self.fraction_of_max_importance = 0.25;
        elif qsar_method == 'xgb':
            self.fraction_of_max_importance = 0.03;
        else:
            print("invalid method:"+qsar_method)
            sys.exit()

        self.include_standardization_in_pmml = include_standardization_in_pmml
        self.use_pmml_pipeline = use_pmml_pipeline
        self.n_threads = n_threads

class ParametersGeneticAlgorithm:

    def __init__(self, qsar_method):
        
        self.qsar_method = qsar_method
        self.num_generations = 100
        self.num_optimizers = 10
        self.num_jobs = 4
        self.n_threads = n_threads        
        self.max_length = 24 # max number of variables   
        self.descriptor_coefficient = 0.002    
        self.threshold = 1

        self.remove_log_p_descriptors = False
        
        self.use_wards = False
        self.run_rfe = True
        self.remove_fragment_descriptors = False
        self.remove_acnt_descriptors = False
        
        self.include_standardization_in_pmml = include_standardization_in_pmml
        self.use_pmml_pipeline = use_pmml_pipeline



class EmbeddingGenerator:
    
    
    def generate_instance(self, chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\r\n"

    
    def get_training_prediction_instances(self, session, datasetName, descriptorSetName, splittingName):
        
        sql = text("""
            select headers_tsv from qsar_descriptors.descriptor_sets ds
            where ds.name=:descriptorSetName
            """)

        try:
            results = session.execute(sql, {'descriptorSetName': descriptorSetName}).fetchone()
            instance_header = "ID\tProperty\t"+results[0] + "\r\n"
            # print(instance_header)
            
            sql = text("""
                SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv, dpis.split_num
                FROM qsar_datasets.datasets d
                JOIN qsar_datasets.data_points dp on dp.fk_dataset_id = d.id
                JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
                JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
                JOIN qsar_descriptors.descriptor_sets ds on ds.id = dv.fk_descriptor_set_id
                join qsar_datasets.splittings s on s.id = dpis.fk_splitting_id
                WHERE d.name = :datasetName
                AND ds.name = :descriptorSetName
                AND s.name = :splittingName
                ORDER BY dp.canon_qsar_smiles;
                """)
            
            sb_training = [instance_header]
            sb_prediction = [instance_header]
        
            results = session.execute(sql, {'datasetName': datasetName, 'descriptorSetName': descriptorSetName,
                                            'splittingName': splittingName})
        
            for row in results:
                chemical_id, qsar_property_value, descriptors, split_num = row
                instance = self.generate_instance(chemical_id, qsar_property_value, descriptors)
                
                # print(len(instance.split("\t")))
                            
                if instance is None:
                    logging.debug(f"{id}\tnull instance\tdatasetName={datasetName}\tdescriptorSetName={descriptorSetName}")
                    continue
            
                if split_num == 0:
                    sb_training.append(instance)
            
                elif split_num == 1:
                    sb_prediction.append(instance)
                    
            df_training = load_df(''.join(sb_training))
            df_prediction = load_df(''.join(sb_prediction))
            
            # print(df_training.shape)
            # first_row_dict = df_training.loc[0].to_dict()
            # print(json.dumps(first_row_dict, indent=4))
                    
            logging.debug('trainingSet shape'+ str(df_training.shape))
            logging.debug('predictionSet shape'+ str(df_prediction.shape))

            return df_training, df_prediction
    
        except SQLAlchemyError:
            logging.exception("An exception was thrown!")
            return None, None 

    
    def buildEmbeddingImportance(self, datasetName, descriptorSetName, splittingName, qsar_method):

        session = getSession()
        df_training, df_prediction = self.get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName)

        params = ParametersImportance(qsar_method)

        embedding, timeEmbedding = call_build_embedding_importance_from_df2(qsar_method, df_training, df_prediction, params)
        logging.debug(f"embedding: {embedding}, time:{timeEmbedding}")
                
        include_standardization_in_pmml = True
        use_pmml_pipeline = False
        
        embedding_name = "Importance-XGB"
        self.build_and_test_model(df_training, df_prediction, params, embedding_name, embedding)
        self.crossvalidate(session, datasetName, descriptorSetName, params, embedding)


    def build_and_test_model2(self,  df_training, df_prediction, params, embedding):

        model = call_build_model_with_preselected_descriptors_from_df(qsar_method, df_training, df_prediction, params.remove_log_p_descriptors, use_pmml_pipeline,
                                                  include_standardization_in_pmml, descriptor_names_tsv=embedding,
                                                  n_jobs=params.n_threads,filterColumnsInBothSets=True)
        # generate predictions for test set:
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        df_predictions = pd.read_json(StringIO(json_predictions), orient="records")
        
        # mean_exp_training = df_training["Property"].mean()
        # test_stats = sc.calculate_continuous_statistics(df_predictions, mean_exp_training, "_Test")
        # print(test_stats["MAE_Test"])
        return df_predictions
    

    def build_and_test_model(self, df_training, df_prediction, params, embedding_name, embedding):
        
        model = call_build_model_with_preselected_descriptors_from_df(params.qsar_method, df_training, df_prediction, params.remove_log_p_descriptors, use_pmml_pipeline, 
            include_standardization_in_pmml, descriptor_names_tsv=embedding, 
            n_jobs=params.n_threads, filterColumnsInBothSets=True)
        
        # generate predictions for test set:
        json_predictions = call_do_predictions_from_df(df_prediction, model)
        df_predictions = pd.read_json(json_predictions, orient="records")
        # first_row_dict = df_predictions.loc[0].to_dict()
        # print(json.dumps(first_row_dict, indent=4))
    
        # calculate stats for test set
        mean_exp_training = df_training["Property"].mean()
        test_stats = sc.calculate_continuous_statistics(df_predictions, mean_exp_training, "_Test")
        logging.info(f"qsar_method={params.qsar_method}, embedding_name={embedding_name}, embedding={embedding}")
        # logging.info(json.dumps(test_stats, indent=4))
        
        logging.info(f"MAE_TEST = {test_stats['MAE_Test']:.3f}")
        
        
        # TODO: generate cross validation stats or try an external test set (e.g. REACH data)
        
        # coeff_dict = model.getOriginalRegressionCoefficients()
        # logging.debug(f"coeffs{coeff_dict}")

    
    def crossvalidate(self,session, datasetName, descriptorSetName, params, embedding):
    
        
        all_df_predictions = []
        for i in range(1, 6):
            splittingName ='RND_REPRESENTATIVE_CV'+str(i)
            df_training, df_prediction = self.get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName)
            df_predictions = self.build_and_test_model2(df_training, df_prediction, params, embedding)
            all_df_predictions.append(df_predictions)
        
        df_predictions_all = pd.concat(all_df_predictions, ignore_index=True)

        # print(all_df_predictions.shape)
        # print(df_predictions_all.shape)
        
        mean_exp_training = df_predictions_all["exp"].mean()
        cv_stats = sc.calculate_continuous_statistics(df_predictions_all, mean_exp_training, "_Test")
        # print('cross validation stats:\n', json.dumps(cv_stats, indent=4))
        logging.info(f"MAE_TRAINING_CV = {cv_stats['MAE_Test']:.3f}")

    
    def buildEmbeddingGeneticAlgorithm(self, datasetName, descriptorSetName, splittingName, qsar_method):
    
        include_standardization_in_pmml = True
        use_pmml_pipeline = False

        session = getSession()
        df_training, df_prediction = self.get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName)

        params = ParametersGeneticAlgorithm(qsar_method)
        # params.num_generations = 5
        # params.num_optimizers = 1
        params.remove_fragment_descriptors = False
        params.remove_acnt_descriptors = False
                
        # embedding, timeEmbedding = call_build_embedding_ga_db(qsar_method, df_training, df_prediction, params)
        # logging.debug(f"embedding: {embedding}, time:{timeEmbedding}")
        # embedding_name = f"remove_fragment_descriptors = {params.remove_fragment_descriptors}, remove_acnt_descriptors = {params.remove_acnt_descriptors}"

        # embedding_name = 'previous GA'
        # # previously generated using GA:
        # embedding = ['SdssNp','-NH2 [aliphatic attach]','ide','MATS6v','MDEC33','XLOGP','BEHv5','Gmax','SsCl']
        #
        # # TODO: store embedding in the database using the embedding_name (make name more unique)
        # self.build_and_test_model(qsar_method, include_standardization_in_pmml, use_pmml_pipeline, df_training, df_prediction, params, embedding_name, embedding)
        # self.crossvalidate(session, datasetName, descriptorSetName, qsar_method, include_standardization_in_pmml, use_pmml_pipeline, params, embedding_name, embedding)


        embedding_dict = {}
        embedding_dict["remove_fragment_descriptors = False, remove_acnt_descriptors = False"] = ['GATS6v', 'xch8', 'BELv4', 'piPC10', 'Gmax', '-Cl [aromatic attach]', 'XLOGP', 'MDEC13', 'Gmin', 'MDEC33', '-NH2 [aliphatic attach]']
        embedding_dict["remove_fragment_descriptors = True, remove_acnt_descriptors = False"] = ['XLOGP', 'GATS5p', 'MDEC23', 'CID2', 'GATS5v', 'MDEC13', 'MDEC33', 'BEHv2', 'SssssC_acnt', 'Gmax', 'BEHm6']
        embedding_dict["remove_fragment_descriptors = True, remove_acnt_descriptors = True"] = ['MATS5v', 'nC', 'ATS6m', 'XLOGP', 'GATS5p', 'SaaCH', 'MDEC13', 'nBnz', 'BEHv2', 'Gmax', 'ide', 'GATS6v']
        
        pd.set_option('display.max_columns', None)
        
        for embedding_name in embedding_dict:
            embedding = embedding_dict[embedding_name]
            
            # df_training_embedding = df_training[embedding]
            # logging.info(embedding_name)
            # logging.info(f"descriptor info:\n{df_training_embedding.describe()}")            
            
            self.build_and_test_model(df_training, df_prediction, params, embedding_name, embedding)
            self.crossvalidate(session, datasetName, descriptorSetName, params, embedding)
        
# TODO: make it run cross validation of training set to see which embedding works best

        

    def buildModelNoEmbedding(self, datasetName, descriptorSetName, splittingName, qsar_method):
        

        session = getSession()
        df_training, df_prediction = self.get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName)
    
        # self.build_and_test_model(qsar_method, include_standardization_in_pmml, use_pmml_pipeline, df_training, df_prediction, params, embedding_name, embedding)
        # self.crossvalidate(session, datasetName, descriptorSetName, qsar_method, include_standardization_in_pmml, use_pmml_pipeline, params, embedding_name, embedding)

        params = ParametersNoEmbedding(qsar_method)

        n_threads = 20
        remove_log_p_descriptors = False
        
        embedding = None
        embedding_name = None
        
        self.build_and_test_model(df_training, df_prediction, params, embedding_name, embedding)
        self.crossvalidate(session, datasetName, descriptorSetName, params, embedding)


        # model = call_build_model_with_preselected_descriptors_from_df(qsar_method, df_training, df_prediction, remove_log_p_descriptors, use_pmml_pipeline, 
        #     include_standardization_in_pmml, descriptor_names_tsv=None, 
        #     n_jobs=n_threads, filterColumnsInBothSets=True)
        #
        # # generate predictions for test set:
        # json_predictions = call_do_predictions_from_df(df_prediction, model)
        # df_predictions = pd.read_json(json_predictions, orient="records")
        # # first_row_dict = df_predictions.loc[0].to_dict()
        # # print(json.dumps(first_row_dict, indent=4))
        #
        # # calculate stats for test set
        # mean_exp_training = df_training["Property"].mean()
        # test_stats = sc.calculate_continuous_statistics(df_predictions, mean_exp_training, "_Test")
        # logging.info("No embedding")
        # # logging.info(json.dumps(test_stats, indent=4))
        #
        # logging.info(f"MAE_TEST = {test_stats['MAE_Test']:.3f}")



    
             
        


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

def load_df(tsv_string):
    """Loads data from TSV/CSV into a pandas dataframe"""
    if "\t" in tsv_string:
        separator = '\t'
    else:
        separator = ','
        
    # print('separator*',separator,'*')
        
    df = pd.read_csv(StringIO(tsv_string), sep=separator, na_values="null")
    return df


if __name__ == '__main__':
    
    datasetName = "KOC v1 modeling"
    descriptorSetName = "WebTEST-default"
    splittingName = "RND_REPRESENTATIVE"
    
    eg = EmbeddingGenerator()
    eg.buildModelNoEmbedding(datasetName, descriptorSetName, splittingName, 'xgb')
    eg.buildEmbeddingGeneticAlgorithm(datasetName, descriptorSetName, splittingName, 'reg')    
    eg.buildEmbeddingImportance(datasetName, descriptorSetName, splittingName, 'xgb')
    
    # import warnings
    # warnings.simplefilter("always")
    
