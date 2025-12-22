import concurrent
import concurrent.futures
import json
import os
import threading
from io import BytesIO
import pathlib

from indigo import Indigo
from indigo.renderer import IndigoRenderer
import base64

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from API_Utilities import QsarSmilesAPI, DescriptorsAPI
from db.mongo_cache import get_cached_prediction, cache_prediction
from model_ws_utilities import call_do_predictions_from_df, models
from models import df_utilities as dfu
from models.ModelBuilder import Model

import StatsCalculator as stats
import pandas as pd
from datetime import datetime, timezone
import pytz
from StatsCalculator import PredictConstants

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import model_ws_utilities as mwu
import numpy as np

from report_creator_dict import ReportCreator

import webbrowser
from predict_constants import UnitsConverter
from predict_constants import PredictConstants as pc

from utils import timer
from models.df_utilities import remove_log_p_descriptors,\
    do_remove_correlated_descriptors
# from bleach._vendor.html5lib.serializer import serialize

debug = False
import logging
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

fk_dsstox_snapshot_id = 3
USE_TEMPORARY_MODEL_PLOTS = False
urlCtxApi = "https://ctx-api-dev.ccte.epa.gov/chemical/property/model/file/search/"
imgURLCid = "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-files/image/by-dtxcid/";

"""
Not completed:
TODO: add qsarMethod and qsarMethodVersion
TODO: add similarities and colors
TODO: Add plot/ histo for overall sets (Use cross validation for plot for training)
TODO: make a batch mode
TODO: Add experimental tab with raw data

Completed:
X TODO Add display units
X TODO Add exp, pred values for neighbors
X TODO Move ad code to that class
X TODO Add experimental value to to predict tab
X TODO Add propertyDescription, sourceName
X TODO Add URL’s to excel and qmrf
X TODO Add plots for neighbors
X TODO fix error for melting point formatting (quantize error)
X TODO make it still create reports if smiles or descriptors fails
X TODO add green/red color box for AD result in report
X TODO in report mention that the AD checks if distance is less than cutoff...
X TODO run for smiles not in DSSTOX and fix code to still work 
X TODO Fragment table for ocspp
X TODO fix axis limits on neighbor plots
X TODO Add size of training sets
"""

lock = threading.Lock()

# def init_model(model_id):
#     with lock:
#         if model_id in models:
#             logging.debug('have model already initialized')
#             model = models[model_id]
#         else:
#             mi=ModelInitializer()
#             model = mi.initModel(model_id)
#             models[model_id] = model
#
#     return model
#
#
# @timer
# def predictFromDB(model_id, smiles):
#     """
#     Runs whole workflow: standardize, descriptors, prediction, applicability domain
#     :param model_id:
#     :param smiles:
#     :param mwu:
#     :return:
#     """
#
#     # Make sure the model is loaded before the concurrency
#     init_model(model_id)
#
#     if isinstance(smiles, str):
#         key = f"{smiles}-{model_id}"
#         prediction = get_cached_prediction(key)
#         if prediction:
#             return prediction
#         else:
#             prediction = predict_model_smiles(model_id, smiles)
#             cache_prediction(key, prediction)
#             return prediction
#     else:
#         result, missing = [], []
#         for smi in smiles:
#             key = f"{smi}-{model_id}"
#             prediction = get_cached_prediction(key)
#             if prediction:
#                 result.append(prediction)
#             else:
#                 missing.append(smi)
#
#         with concurrent.futures.ThreadPoolExecutor() as pool:
#             results = pool.map(predict_model_smiles, [model_id for _ in missing], missing)
#
#             for (prediction, code, smi) in results:
#                 if code != 200:
#                     prediction = dict(smiles=smi, error=prediction)
#                     result.append(prediction)
#                 else:
#                     result.append(prediction)
#
#                 key = f"{smi}-{model_id}"
#                 cache_prediction(key, prediction)
#
#         return result
#

# @timer
# def predict_model_smiles(model_id, smiles):
#     # serverAPIs = "https://hcd.rtpnc.epa.gov" # TODO this should come from environment variable
#     serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
#
#     # initialize model bytes and all details from db:
#     model = init_model(model_id)    
#
#     model_details=ModelDetails(model)
#
#     # dont need following for simple report (delete these if you want them): 
#     model_details.modelStatistics = None
#     model_details.embedding = None
#
#
#     mp = ModelPredictor()
#
#     # Standardize smiles:
#     chemical, code = mp.standardizeStructure(serverAPIs, smiles, model)
#     if code != 200:
#         return chemical, code, smiles
#
#     qsarSmiles = chemical["canonicalSmiles"]
#
#     # Descriptor calcs:
#     descriptorAPI = DescriptorsAPI()
#     df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, model.descriptorService)
#     if code != 200:
#         return df_prediction, 400, smiles
#
#     # Run model prediction:
#     # df_prediction = model.model_details.predictionSet #all chemicals in the model's prediction set, for testing
#     # print("for qsarSmiles="+qsarSmiles+", descriptors="+json.dumps(descriptorsResults,indent=4))
#     pred_results = json.loads(call_do_predictions_from_df(df_prediction, model))
#     pred_value = pred_results[0]['pred']
#
#     # applicability domain calcs:
#     ad_results = None
#     if model.applicabilityDomainName:
#         ad_results =mp.determineApplicabilityDomain(model, df_prediction)
#     else:
#         logging.debug('AD method for model was not set:', model_id)
#
#     # store everything in results:
#     model_results = ModelResults(chemical=chemical, modelDetails=model_details,adResults=ad_results)
#     model_results.smiles = smiles
#
#     # model_results.qsarSmiles = qsarSmiles
#
#     model_results.unitsDisplay = model.unitsDisplay
#     model_results.unitsModel = model.unitsModel  # duplicated so displayed near prediction value
#
#     uc = UnitsConverter()
#     model_results.predictionValueUnitsModel = pred_value
#     model_results.predictionValueUnitsDisplay = uc.convert_units(model.propertyName, pred_value, model.unitsModel, 
#                                                                  model.unitsDisplay, smiles, chemical["averageMass"])
#
#     mp.setExpValue(model, model_results)
#
#     if model_results.experimentalValueUnitsModel:
#         model_results.experimentalValueUnitsDisplay=uc.convert_units(model.propertyName, model_results.experimentalValueUnitsModel, 
#                                                                      model.unitsModel, model.unitsDisplay, 
#                                                                      smiles, chemical["averageMass"])
#
#
#     model_results.adResults = ad_results
#
#     results_json = model_results.to_json()
#
#     return results_json, 200, smiles

#
#
# def predictSetFromDB(model_id, excel_file_path):
#     """
#     Runs whole workflow: standardize, descriptors, prediction, applicability domain
#     :param model_id:
#     :param smiles:
#     :param mwu:
#     :return:
#     """
#
#     descriptorAPI = DescriptorsAPI()
#
#     # serverAPIs = "https://hcd.rtpnc.epa.gov" # TODO this should come from environment variable
#     serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
#
#     # initialize model bytes and all details from db:
#     model = init_model(model_id)
#
#     import pandas as pd
#     df = pd.read_excel(excel_file_path, sheet_name='Test set')
#     smiles_list = df['Smiles'].tolist()  # Extract the 'Smiles' column into a list
#
#     directory = os.path.dirname(excel_file_path)
#
#     # Create a text file path in the same directory
#     text_file_path = os.path.join(directory, "output.txt")
#     logging.debug(text_file_path)
#
#     with open(text_file_path, 'w') as file:
#         file.write("smiles\tqsarSmiles\tpred_value\tpred_AD\n")
#
#         # for smiles, predOld in zip(smiles_list, pred_list):
#         for smiles in smiles_list:
#             qsarSmiles, code = standardizeStructure(serverAPIs, smiles, model)
#             if code != 200:
#                 logging.warn(smiles, qsarSmiles)
#                 file.write(smiles + "\terror smiles")
#                 continue
#
#             df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, model.descriptorService)
#             if code != 200:
#                 logging.warn(smiles, 'error descriptors')
#                 file.write(smiles + "\terror descriptors")
#
#                 continue
#
#             pred_results = json.loads(call_do_predictions_from_df(df_prediction, model))
#             pred_value = pred_results[0]['pred']
#
#             str_ad_results = determineApplicabilityDomain(model, df_prediction)
#             ad_results = json.loads(str_ad_results)
#             pred_AD = ad_results[0]["AD"]
#
#             line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\t" + str(pred_AD) + "\n"
#             print(line)
#             file.write(line)
#             file.flush()
#
#     if True:
#         return
#
#     # Standardize smiles:
#
#     # Descriptor calcs:
#
#     # Run model prediction:
#     # df_prediction = model.model_details.predictionSet #all chemicals in the model's prediction set, for testing
#     # print("for qsarSmiles="+qsarSmiles+", descriptors="+json.dumps(descriptorsResults,indent=4))
#
#     logging.debug(pred_results)
#
#     # # applicability domain calcs:
#     # ad_results = None
#     # if model.applicabilityDomainName:
#     #     str_ad_results = determineApplicabilityDomain(model, df_prediction)
#     #     # str_ad_results = determineApplicabilityDomain(model, model.df_prediction) #testing AD method using multiple chemicals in df
#     #
#     #     ad_results = json.loads(str_ad_results)[0]  # TODO check len first?
#     #     print(ad_results)
#     # else:
#     #     print('AD method for model was not set:', model_id)
#
#     return "OK", 200
#
#
# @timer
# def determineApplicabilityDomain(model: Model, test_tsv):
#     """
#     Calculate the applicability domain using the model's training set and the AD measure assigned to the model in the DB
#     TODO make sure this works when a model doesnt have a set embedding object
#     :param model:
#     :param test_tsv:
#     :return:
#     """
#     json_model_description = model.get_model_description()
#     model_description = json.loads(json_model_description)
#     remove_log_p = model_description["remove_log_p_descriptors"]  # just set to False instead?
#     # print("remove_log_p", remove_log_p)
#
#     from applicability_domain import applicability_domain_utilities as adu
#     # model.applicabilityDomainName = adu.strOPERA_local_index  # for testing diff number of neighbors
#
#     output = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
#         train_df=model.df_training,
#         test_df=test_tsv,
#         remove_log_p=remove_log_p,
#         embedding=model.embedding,
#         applicability_domain=model.applicabilityDomainName,
#         filterColumnsInBothSets=True)
#
#     # return output.to_json(orient='records', lines=True) # gives each object on separate line
#     return output.to_json(orient='records', lines=False)  # gives an array instead of each object on separate line


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
    engine = create_engine(connect_url, echo=debug)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


class ModelInitializer:

    def init_model(self, model_id):

        with lock:
            if model_id in models:
                logging.debug('have model already initialized')
                model = models[model_id]
            else:
                model = self.initModel(model_id)
                models[model_id] = model

        return model

    # @timer
    # def initModel(self, model_id):
    #     session = getSession()
    #     model_bytes = self.get_model_bytes(model_id, session)
    #
    #     import pickle
    #     model = pickle.loads(model_bytes)
    #
    #     if debug:
    #         print('model_description from pickled model:', model.get_model_description())
    #
    #     if not hasattr(model, "is_binary"):
    #         print('model.is_binary is none, setting to false')
    #         model.is_binary = False
    #
    #     # Stores model under provided number
    #
    #     self.get_model_details(model_id, model, session)
    #
    #     logging.debug(model.get_model_description_pretty())
    #
    #     # this wont be necessary if the training/test sets are in the pickled model:
    #     self.get_training_prediction_instances(session, model)
    #
    #     # TODO: for the training/prediction instances, could also query the descriptor api but it would take longer and
    #     #  sometimes the descriptors will come out different due to the fact that the descriptors will be pulled from the
    #     #  cache by inchi key (TEST descriptors come out differently sometimes for two different structures with the same inchi key but different smiles
    #
    #     return model

    def get_model_bytes(self, model_id, session):
        """
        This method allows for the fact that model bytes might be stored as separate rows for very large models
        :rtype: bytearray
        """
        # Database connection parameters
        try:
            # Get a connection from the session
            connection = session.connection()

            # SQL query to retrieve bytes
            sql = text("SELECT bytes FROM qsar_models.model_bytes WHERE fk_model_id = :model_id ORDER BY id")

            # Execute the query with the parameter
            result = connection.execute(sql, {"model_id": model_id})

            # Use BytesIO to collect the byte data
            output_stream = BytesIO()

            # Fetch and write byte data to the output stream
            for record in result:
                output_stream.write(record.bytes)  # Assuming the column name is 'bytes'

            # Return the combined byte array
            return output_stream.getvalue()

        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def get_model_statistics(self, model: Model, session):

        sql = text("""
            select s.name, ms.statistic_value from qsar_models.models m
            join qsar_models.model_statistics ms on m.id = ms.fk_model_id
            join qsar_models.statistics s on ms.fk_statistic_id = s.id
            where m.id=:modelId
            """)

        try:
            results = session.execute(sql, {'modelId': model.modelId})

            stats = {}
            for row in results:
                stat_name, stat_value = row

                if stat_name != "Coverage_Training":  # dont have right value in database (have dummy value)
                    stats[stat_name] = stat_value

            model.modelStatistics = stats

            # print(model.modelStatistics)

        except SQLAlchemyError as ex:
            print("error getting stats for modelId=" + str(model.modelId))

    def get_predictions(self, session, model: Model, split_num, fk_splitting_id):

        if debug:
            print("Getting model training/prediction set TSVs")

        sql = text("""
            select dp.canon_qsar_smiles, dp.qsar_property_value,p.qsar_predicted_value
            from qsar_datasets.datasets d
            join qsar_datasets.data_points dp on d.id = dp.fk_dataset_id
            join qsar_datasets.data_points_in_splittings dpis on dp.id = dpis.fk_data_point_id
            join qsar_models.models m on m.dataset_name = d.name
            join qsar_models.predictions p on p.canon_qsar_smiles=dp.canon_qsar_smiles and p.fk_model_id=m.id
            where m.id = :model_id and split_num = :split_num  and dpis.fk_splitting_id=:fk_splitting_id and p.fk_splitting_id=:fk_splitting_id;
            """)
        # print(sql)
        try:
            results = session.execute(sql, {'model_id': model.modelId,
                                            'fk_splitting_id': fk_splitting_id,
                                            'split_num': split_num})
            import pandas as pd
            df = pd.DataFrame(results, columns=["id", "exp", "pred"])
            # print(split_num, df.shape)
            return df

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
            # print('done getting tsvs')
            session.close()

    def get_cv_predictions(self, session, model: Model):
        sql = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, p.qsar_predicted_value
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON d.id = dp.fk_dataset_id
            JOIN qsar_datasets.data_points_in_splittings dpis ON dp.id = dpis.fk_data_point_id
            join qsar_datasets.splittings s on dpis.fk_splitting_id = s.id
            JOIN qsar_models.models m ON m.dataset_name = d.name
            JOIN qsar_models.predictions p ON p.canon_qsar_smiles = dp.canon_qsar_smiles AND p.fk_model_id = m.id
            WHERE m.id = :model_id  AND dpis.fk_splitting_id = p.fk_splitting_id  AND s.name like 'RND_REPRESENTATIVE_CV%'
        """)

        try:
            results = session.execute(sql, {'model_id': model.modelId})
            df = pd.DataFrame(results, columns=["id", "exp", "pred"])
            # print('cv_shape', df.shape)
            return df

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
            # print('done getting tsvs')
            session.close()

    def get_model_details(self, m: Model, session):
        """
        Gets m meta data (except training and test set tsvs).
        TODO Should this info be stored directly in m object and then for new models we won't need to query the db since will be already in the pickled object?
        """
        try:
            # SQL query to retrieve m details
            sql = text(self.getModelMetaDataQuery() + "\nWHERE m.id = :model_id")

            # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for m)
            # print(sql)

            # Execute the query
            row = session.execute(sql, {'model_id': m.modelId}).fetchone()

            # Process the result
            if row:
                self.row_to_model_details(m, row)

        except Exception as ex:
            ex.with_traceback()
            print(f"Exception occurred: {ex}")
        # finally:
            # Close the session - close it later after get training/test sets
            # print('done with details')
            # session.close()

    def replace_id_with_dsstox_record(self, df_set, df_dsstoxRecords):

        # Create a dictionary for fast lookup by canonicalSmiles
        dsstox_dict = df_dsstoxRecords.set_index('canonicalSmiles').to_dict(orient='index')

        # Iterate over each row in df_set

        for index, row in df_set.iterrows():

            qsar_smiles = row['ID']
            # Find the matching row in df_dsstoxRecords

            if qsar_smiles in dsstox_dict:
                # Replace the ID with the matching row as a dictionary

                dictionary = dsstox_dict[qsar_smiles]
                dictionary["qsarSmiles"] = qsar_smiles  # also store qsar_smiles
                df_set.at[index, 'ID'] = dictionary

        return df_set

    @timer
    def initModel(self, model_id):

        session = getSession()

        model_bytes = self.get_model_bytes(model_id, session)

        if not model_bytes:
            print("Couldnt load " + model_id + " from model bytes")
            return

        import pickle
        model = pickle.loads(model_bytes)

        if not model:
            print("Couldnt load " + model_id + " from model bytes")
            return

        if debug:
            print('model_description from pickled model:', model.get_model_description())

        if not hasattr(model, "is_binary"):
            print('model.is_binary is none, setting to false')
            model.is_binary = False

        # Stores model under provided number

        model.modelId = model_id

        self.get_model_details(model, session)

        # print(model.get_model_description())

        self.get_model_statistics(model, session)

        self.get_training_prediction_instances(session, model)
        self.get_dsstox_records_for_dataset(model, session)

        # get following for pred values for neighbors:
        model.df_preds_test = self.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
        model.df_preds_training_cv = self.get_cv_predictions(session, model)

        # self.replace_id_with_dsstox_record(model.df_prediction, model.df_dsstoxRecords)
        # self.replace_id_with_dsstox_record(model.df_training, model.df_dsstoxRecords)

        # print('model_description with added metadata', model.get_model_description_pretty())

        if debug:
            print('model_description with added metadata', model.get_model_description_pretty())

        # TODO: for the training/prediction instances, could also query the descriptor api but it would take longer and
        #  sometimes the descriptors will come out different due to the fact that the descriptors will be pulled from the
        #  cache by inchi key (TEST descriptors come out differently sometimes for two different structures with the same inchi key but different smiles

        session.close()

        return model

    def get_dsstox_records_for_dataset(self, model: Model, session):
        """
        Gets the dsstox records for the dataset from res_qsar postgreSQL db (could also get from dsstox or a snapshot of dsstox)
        """
        try:
            # Get a connection from the session
            connection = session.connection()

            # SQL query to retrieve bytes

            # Note: in the data_points table, sometimes the qsar_dtxcid is pipe delimited pair of cids
            sql = """
                SELECT dp.canon_qsar_smiles as "canonicalSmiles", dr.dtxsid as sid, dr.dtxcid as cid, dr.casrn, dr.preferred_name as "name" , dr.smiles, dr.indigo_inchi_key as "inchiKey"
                    FROM qsar_datasets.datasets d
                    JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
                    JOIN qsar_models.models m ON m.dataset_name = d.name
                    LEFT JOIN qsar_models.dsstox_records dr ON dr.dtxcid = SUBSTRING(dp.qsar_dtxcid FROM 1 FOR POSITION('|' IN dp.qsar_dtxcid || '|') - 1)
                """

            sql = text(sql + "\nWHERE m.id = :model_id and dr.fk_dsstox_snapshot_id = :fk_dsstox_snapshot_id;")

            # print(sql)

            # Execute the query with the parameter
            result = connection.execute(sql, {"model_id": model.modelId, "fk_dsstox_snapshot_id": fk_dsstox_snapshot_id})

            # Convert result to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            model.df_dsstoxRecords = df;

            # print(df.head())

            none_sid_smiles = df[df['sid'].isnull()]['canonicalSmiles']

            if len(none_sid_smiles) > 0:
                print(model.modelId, "Have canonicalSmiles in dataset that isn't in dsstox records:", none_sid_smiles)

            # if not df.empty:
            #     first_row_dict = df.iloc[0].to_dict()
            #     pretty_json = json.dumps(first_row_dict, indent=4)
            #     print("First row as pretty JSON:")
            #     print(pretty_json)

            return df

        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def get_training_prediction_instances(self, session, model:Model):
        if debug:
            print("Getting model training/prediction set TSVs")

        instance_header = f"ID\tProperty\t{model.headersTsv}\r\n"
        sql = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv, dpis.split_num
            FROM qsar_datasets.data_points dp
            JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
            JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
            WHERE dp.fk_dataset_id = :datasetId
            AND dv.fk_descriptor_set_id = :descriptorSetId
            AND dpis.fk_splitting_id = :splittingId
            ORDER BY dp.canon_qsar_smiles;
            """)

        sb_training = [instance_header]
        sb_prediction = [instance_header]

        counter_train = 0
        counter_prediction = 0

        try:
            results = session.execute(sql, {'datasetId': model.datasetId, 'descriptorSetId': model.descriptorSetId,
                                            'splittingId': model.splittingId})

            for row in results:
                chemical_id, qsar_property_value, descriptors, split_num = row
                instance = self.generate_instance(chemical_id, qsar_property_value, descriptors)

                if instance is None and debug:
                    print(
                        f"{id}\tnull instance\tdatasetName={model.datasetName}\tdescriptorSetName={model.descriptorSetName}")
                    continue

                if split_num == 0:
                    sb_training.append(instance)
                    counter_train += 1

                elif split_num == 1:
                    sb_prediction.append(instance)
                    counter_prediction += 1

            model.df_training = dfu.load_df(''.join(sb_training))
            model.df_prediction = dfu.load_df(''.join(sb_prediction))

            model.num_training = model.df_training.shape[0]
            model.num_prediction = model.df_prediction.shape[0]

                    # Replace IDs in df_set

            if debug:
                print('trainingSet shape', model.df_training.shape)
                print('predictionSet shape', model.df_prediction.shape)

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
            pass
            # print('done getting tsvs')
            # session.close()

    def generate_instance(self, chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\n"

    def getModelMetaDataQuery(self):
        return """
        SELECT 
                m.id,
                m.name_ccd,
                m.details,
                d.id,
                d.name,
                u.abbreviation_ccd,
                u2.abbreviation_ccd,
                d.dsstox_mapping_strategy,
                p.name_ccd,
                p.description,
                ds.id,
                ds.name,
                ds.descriptor_service,
                ds.headers_tsv,
                s.id,
                s.name,
                adm.name,
                adm.description,
                s2.name,
                m2.name,
                m2.description,
                m2.description_url
            FROM qsar_models.models m
            LEFT JOIN qsar_datasets.datasets d ON d.name = m.dataset_name
            LEFT JOIN qsar_datasets.units u ON d.fk_unit_id = u.id
            LEFT JOIN qsar_datasets.units u2 ON d.fk_unit_id_contributor = u2.id
            LEFT JOIN qsar_datasets.properties p ON d.fk_property_id = p.id
            LEFT JOIN qsar_descriptors.descriptor_sets ds ON m.descriptor_set_name = ds.name
            LEFT JOIN qsar_datasets.splittings s ON m.splitting_name = s.name
            LEFT JOIN qsar_models.ad_methods adm ON m.fk_ad_method = adm.id
            LEFT JOIN qsar_models.sources s2 ON m.fk_source_id = s2.id
            LEFT JOIN qsar_models.methods m2 ON m.fk_method_id = m2.id
        """

    def get_available_models(self):
        """
        Gets  list of available models with meta data
        """
        try:
            session = getSession()

            # SQL query to retrieve model details
            sql = text(self.getModelMetaDataQuery() + "\nWHERE m.fk_source_id = 3 and m.is_public=true;")

            # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
            # print(sql)

            # Execute the query
            results = session.execute(sql).fetchall()

            models = []
            # Process the result
            for row in results:
                model = Model()
                self.row_to_model_details(model, row)
                models.append(model.get_model_description_dict())

            return models

        except Exception as ex:
            print(f"Exception occurred: {ex}")
        finally:
            # Close the session - close it later after get training/test sets
            session.close()

        return None

    def row_to_model_details(self, m: Model, row):

        (m.modelId,
         m.modelName,
        m.detailsFile,
        m.datasetId,
        m.datasetName,
        m.unitsModel,
        m.unitsDisplay,
        m.dsstox_mapping_strategy,
        m.propertyName,
        m.propertyDescription,
        m.descriptorSetId,
        m.descriptorSetName,
        m.descriptorService,
        m.headersTsv,
        m.splittingId,
        m.splittingName,
        m.applicabilityDomainName,
        m.applicabilityDomainDescription,
        m.modelSource,
        m.modelMethod,
        m.modelMethodDescription,
        m.modelMethodDescriptionURL
        ) = row

        # model_details.descriptorEmbeddingTsv = row[13]  #dont need already have in pickled m
        # print (type(details))
        # print(details)
        # print(m.modelId)
        
        m.modelId = str(m.modelId)
        details = json.loads(m.detailsFile.tobytes().decode('utf-8'))  # it's stored as a file object in database for now
        m.is_binary = details['is_binary']
        m.remove_log_p_descriptors = details['remove_log_p_descriptors']
        m.embedding = details['embedding']
        m.description = details['description']
        m.description_url = details['description_url']

        m.qsar_method = details['qsar_method']

        m.hyperparameters = details['hyperparameters']
        m.hyperparameter_grid = details['hyperparameter_grid']
        m.qsar_method = details['qsar_method']
        m.use_pmml = details['use_pmml']        
        
        m.detailsFile = None

        if "version" in details:  # misnomer it's the method version not the m version
            m.qsar_method_version = details['version']

        if "qsar_method_version" in details:
            m.qsar_method_version = details['qsar_method_version']

        m.include_standardization_in_pmml = details['include_standardization_in_pmml']

        # omit for now:
        #  'training_stats'
        #  'training_descriptor_std_devs'
        #  'training_descriptor_means'

        # Parse JSON for dsstox_mapping_strategy
        dsstox_mapping = json.loads(m.dsstox_mapping_strategy)
        if 'omitSalts' in dsstox_mapping:
            m.omitSalts = dsstox_mapping.get('omitSalts', False)
        if 'qsarReadyRuleSet' in dsstox_mapping:
            m.qsarReadyRuleSet = dsstox_mapping.get('qsarReadyRuleSet', "qsar-ready")
        else:
            m.qsarReadyRuleSet = "qsar-ready"


class ModelDetails:

    def __init__(self, model: Model):
        self.modelId = model.modelId
        self.modelName = model.modelName
        self.modelSource = model.modelSource
        self.modelStatistics = model.modelStatistics
        
        self.modelMethod = model.modelMethod
        self.modelMethodDescription = model.modelMethodDescription
        self.modelMethodDescriptionURL = model.modelMethodDescriptionURL
        
        if hasattr(model, 'modelSource'):  # TODO: add to model object
            self.modelSource = model.modelSource
        else:
            self.modelSource = None

        self.propertyName = model.propertyName
        self.propertyDescription = model.propertyDescription

        if hasattr(model, 'propertyDescription'):  # TODO add to model object
            self.propertyDescription = model.propertyDescription
        else:
            self.propertyDescription = None

        # self.version = '0.0.1'
        self.is_binary = model.is_binary
        # self.description = model.description # TODO: in database
        # self.description_url = model.description_url #TODO in database
        self.datasetName = model.datasetName
        
        self.unitsModel = model.unitsModel
        self.unitsDisplay = model.unitsDisplay

        self.urlQMRF = None
        self.urlExcelSummary = None
        self.imgSrcPlotScatter = None
        self.imgSrcPlotHistogram = None

        self.descriptorService = model.descriptorService
        self.applicabilityDomainName = model.applicabilityDomainName
        self.applicabilityDomainDescription = model.applicabilityDomainDescription
        self.qsarReadyRuleSet = model.qsarReadyRuleSet
        self.embedding = model.embedding

# from pydantic import BaseModel
# class ModelResults(BaseModel):
    
    
class ModelResults:

    def __init__(self, chemical=None, modelDetails: ModelDetails=None, adResults=None):
        
        self.chemical = chemical
        
        self.experimentalValueUnitsModel = None
        self.experimentalValueUnitsDisplay = None
        self.experimentalValueSet = None

        self.predictionValueUnitsModel = None
        self.unitsModel = None

        self.predictionValueUnitsDisplay = None
        self.unitsDisplay = None

        self.predictionError = None

        self.adResults = None  # old way just store directly
        self.applicabilityDomains = []
        self.modelDetails = modelDetails
                
        self.neighborsForSets = []

    def to_dict(self):
        # Convert the object to a dictionary, including nested objects
        return {
            "chemical": self.chemical,
            "experimentalValueUnitsModel": self.experimentalValueUnitsModel,
            "experimentalValueUnitsDisplay": self.experimentalValueUnitsDisplay,
            "experimentalValueSet": self.experimentalValueSet,
            "predictionValueUnitsModel": self.predictionValueUnitsModel,
            "unitsModel": self.unitsModel,
            "predictionValueUnitsDisplay": self.predictionValueUnitsDisplay,
            "unitsDisplay": self.unitsDisplay,
            "predictionError": self.predictionError,
            "modelDetails": self.modelDetails.__dict__ if self.modelDetails else None,
            "adResults": self.adResults if self.adResults is not None else None,
            "applicabilityDomains": self.applicabilityDomains if self.applicabilityDomains is not None else None,
            "neighborsForSets": self.neighborsForSets if self.neighborsForSets is not None else None,

        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def get_model_list(self, session):
        """Gets model meta data (except training and test set tsvs).
        TODO Should this info be stored directly in model object and then for new models we won't need to query the db since will be already in the pickled object?
        """
        try:
            # SQL query to retrieve model details
            sql = text("""
                       SELECT m.name_ccd,
                              d.id,
                              d.name,
                              u.abbreviation_ccd,
                              d.dsstox_mapping_strategy,
                              p.name_ccd,
                              ds.id,
                              ds.name,
                              ds.descriptor_service,
                              ds.headers_tsv,
                              s.id,
                              s.name,
                              adm.name,
                              de.embedding_tsv
                       FROM qsar_models.models m
                                LEFT JOIN qsar_datasets.datasets d ON d.name = m.dataset_name
                                LEFT JOIN qsar_datasets.units u ON d.fk_unit_id = u.id
                                LEFT JOIN qsar_datasets.properties p ON d.fk_property_id = p.id
                                LEFT JOIN qsar_descriptors.descriptor_sets ds ON m.descriptor_set_name = ds.name
                                LEFT JOIN qsar_datasets.splittings s ON m.splitting_name = s.name
                                LEFT JOIN qsar_models.ad_methods adm ON m.fk_ad_method = adm.id
                                LEFT JOIN qsar_models.descriptor_embeddings de ON m.fk_descriptor_embedding_id = de.id
                       WHERE fk_source_id = 3
                         and is_public = true
                       """)

            # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
            # print(sql)

            # Execute the query
            rows = session.execute(sql).fetch()

            models = []

            for row in rows:
                model_details = self.row_to_model_details(row)
                models.append(model_details)

            return models

        except Exception as ex:
            print(f"Exception occurred: {ex}")
        finally:
            # Close the session
            print('done with details')
            # session.close()

        return None


class ModelStatistics:

    def redo_cv_stats(self, session, model: Model):

        mi = ModelInitializer()
        df_preds_training_cv = mi.get_cv_predictions(session, model)

        stats_training_cv = stats.calculate_continuous_statistics(df_preds_training_cv, 0,
            stats.PredictConstants.TAG_CV + stats.PredictConstants.TAG_TRAINING)

        statistic_name = "RMSE_CV_Training"
        statistic_value = stats_training_cv[statistic_name]
        print(model.propertyName, stats_training_cv["RMSE_CV_Training"])
        self.update_statistic_value(session, model.modelId, statistic_name, statistic_value, "tmarti02")

        self.compare_stats(model, stats_training_cv)

    def recalculate_test_set_stats(self, session, model):
        """Recalculate stats using predictions stored in the db"""

        mi = ModelInitializer()
        mi.get_model_statistics(model, session)

        df_preds_training = mi.get_predictions(session, model=model, split_num=0, fk_splitting_id=1)
        df_preds_test = mi.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
        mean_exp_training = stats.calculate_mean_exp_training(df_preds_training)
        stats_test_set = stats.calculate_continuous_statistics(df_preds_test, mean_exp_training,
            stats.PredictConstants.TAG_TEST)

        self.compare_stats(model, stats_test_set)

    def calculate_ad_stats(self, session, model:Model):

        mi = ModelInitializer()
        mp = ModelPredictor()

        mi.get_training_prediction_instances(session, model)

        df_preds_test = mi.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
        str_ad_results = mp.determineApplicabilityDomain(model, model.df_prediction)
        ad_results = json.loads(str_ad_results)  # convert json to list of AD results
        df_ad = pd.DataFrame(ad_results)  # convert to dataframe

        self.calculate_AD_stats(df_ad, df_preds_test, PredictConstants.TAG_TEST, model.modelId, session)

    def updateStatsPredictModuleModels(self):

        try:

            mi = ModelInitializer()

            session = getSession()
            sql = text(mi.getModelMetaDataQuery() + "\nWHERE m.fk_source_id = 3 and m.is_public=true order by m.id;")  # fk_source_id=3 => cheminformatics modules

            # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
            # print(sql)

            # Execute the query
            results = session.execute(sql).fetchall()

            # Process the result
            for row in results:
                model = Model()
                mi.row_to_model_details(model, row)
                # self.recalculate_test_set_stats(session, model)
                # self.redo_cv_stats(session, model)
                self.calculate_ad_stats(session, model)

                # if True:
                #     break # stop after first model for testing

        except Exception as ex:
            ex.with_traceback()
            print(f"Exception occurred: {ex}")
        finally:
            # Close the session - close it later after get training/test sets
            session.close()

    def calculate_AD_stats(self, df_ad, df_preds, tag, model_id, session):

        merged_df = pd.merge(df_ad, df_preds, left_on='idTest', right_on='id')
        # print(merged_df.columns)

        df_inside_ad = merged_df[merged_df['AD']].loc[:, ['id', 'exp', 'pred']]
        stats_test_inside_AD = stats.calculate_continuous_statistics(df_inside_ad, 0, tag + "_inside_AD")
        # print(stats_test_inside_AD)
        MAE_inside_AD = stats_test_inside_AD ["MAE" + tag + "_inside_AD"]
        # print(MAE_Test_inside_AD)

        df_outside_ad = merged_df[~merged_df['AD']].loc[:, ['id', 'exp', 'pred']]
        stats_test_outside_AD = stats.calculate_continuous_statistics(df_outside_ad, 0, tag + "_outside_AD")
        # print(stats_test_inside_AD)
        MAE_outside_AD = stats_test_outside_AD ["MAE" + tag + "_outside_AD"]

        count_inside = df_inside_ad.shape[0]
        count_outside = df_outside_ad.shape[0]
        coverage = count_inside / (count_inside + count_outside)

        dict_stats = {}
        dict_stats["MAE" + tag + "_inside_AD"] = MAE_inside_AD
        dict_stats["MAE" + tag + "_outside_AD"] = MAE_outside_AD
        dict_stats["Coverage" + tag] = coverage

        for statistic_name in dict_stats:
            new_statistic_value = dict_stats[statistic_name]
            print(model_id, statistic_name, new_statistic_value)
            self.update_statistic_value(session, model_id, statistic_name, new_statistic_value, "tmarti02")

        # print(dict_stats)

        return MAE_inside_AD, MAE_outside_AD, coverage

    def update_statistic_value(self, session, model_id: int, statistic_name: str, new_statistic_value: float, user_id: str):
        try:
            # Query to find the statistic ID
            statistic_id_result = session.execute(
                text("SELECT id FROM qsar_models.\"statistics\" WHERE name = :name"),
                {"name": statistic_name}
            ).fetchone()

            if statistic_id_result is None:
                raise ValueError(f"Statistic '{statistic_name}' does not exist.")

            statistic_id = statistic_id_result[0]

            # Query to check if the statistic already exists for the model
            existing_statistic_result = session.execute(
                text("""
                    SELECT 1 FROM qsar_models.model_statistics
                    WHERE fk_model_id = :model_id AND fk_statistic_id = :statistic_id
                """),
                {"model_id": model_id, "statistic_id": statistic_id}
            ).fetchone()

            est = pytz.timezone('US/Eastern')
            current_time_utc = datetime.now(timezone.utc)  # Use timezone-aware UTC now
            current_time_est = current_time_utc.astimezone(est)

            # print(existing_statistic_result, statistic_id)
            #
            # if True:
            #     return

            if existing_statistic_result is None:
                # Insert if not exists
                session.execute(
                    text("""
                        INSERT INTO qsar_models.model_statistics (
                            fk_model_id, fk_statistic_id, statistic_value, created_at, updated_at, created_by, updated_by
                        ) VALUES (
                            :model_id, :statistic_id, :statistic_value, :created_at, :updated_at, :created_by, :updated_by
                        )
                    """),
                    {
                        "model_id": model_id,
                        "statistic_id": statistic_id,
                        "statistic_value": new_statistic_value,
                        "created_at": current_time_est,
                        "updated_at": current_time_est,
                        "created_by": user_id,
                        "updated_by": user_id
                    }
                )
            else:
                # Update if exists
                session.execute(
                    text("""
                        UPDATE qsar_models.model_statistics
                        SET statistic_value = :statistic_value,
                            updated_at = :updated_at,
                            updated_by = :updated_by
                        WHERE fk_model_id = :model_id AND fk_statistic_id = :statistic_id
                    """),
                    {
                        "model_id": model_id,
                        "statistic_id": statistic_id,
                        "statistic_value": new_statistic_value,
                        "updated_at": current_time_est,
                        "updated_by": user_id
                    }
                )

            # Commit the transaction
            session.commit()
        except Exception as e:
            e.with_traceback()
            session.rollback()
            raise e

    def compare_stats(self, model, stats_new):

        data = []

        for stat_name in stats_new:

            if stat_name == "Coverage_CV_Training":
                continue

            stat_value_old = model.modelStatistics.get(stat_name, "N/A")
            stat_value_new = stats_new[stat_name]
            data.append({
                'Statistic': stat_name,
                'Old Value': stat_value_old,
                'New Value': stat_value_new
            })

        # Create a DataFrame
        df = pd.DataFrame(data, columns=['Statistic', 'Old Value', 'New Value'])

        # Print the DataFrame in a readable format
        print("\n" + model.propertyName + "\n")
        print(df.to_string(index=False, float_format='{:.3f}'.format))


class NeighborGetter:

    def get_neighbors(self, col_name_id, id, test_indices, n_neighbors, df_set):
        """Get AD dataframe for k neighbors. TODO can be done in clearer way?
        """

        # Recoding so that can have arbitrary number of neighbors (i.e. store as list), TODO check Java AD code for places this breaks things
        # Extract neighbors
        neighbors = [test_indices[:, i] for i in range(n_neighbors)]

        # Retrieve IDs for each neighbor and combine them into a list
        ids_combined = [df_set[col_name_id].loc[neighbor].tolist() for neighbor in neighbors]

        # print(json.dumps(ids_combined,indent=4))

        # Transpose ids_combined to align with test_indices rows
        ids_combined_transposed = np.array(ids_combined).T.tolist()

        neighbors = ids_combined_transposed[0]
        return neighbors

    def find_neighbors_in_set(self, model, df_set, df_test_chemicals):

        n_neighbors = 10
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        
        useEmbedding = True # probably should be consistent with what is use to find the Applicability domain analogs
        
        if useEmbedding: #Using embedding descriptors picks weird neighbors sometimes:
            TrainSet = df_set[model.embedding]
            TestSet = df_test_chemicals[model.embedding]
            scaler = StandardScaler().fit(TrainSet)
            train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
        
        else: # Following just uses all TEST descriptors (removes constant ones)
            TrainSet = df_set
            TestSet = df_test_chemicals
            ids_train, labels_train, features_train, column_names_train, is_binary= dfu.prepare_instances(TrainSet, "Training", model.remove_log_p_descriptors, remove_corr=False, remove_constant=True)        
            features_test = TestSet[features_train.columns]
            scaler = StandardScaler().fit(features_train)
            train_x, test_x = scaler.transform(features_train), scaler.transform(features_test)

        nbrs.fit(train_x)
        # train_distances, train_indices = nbrs.kneighbors(train_x)
        test_distances, test_indices = nbrs.kneighbors(test_x)

        col_name_id = df_set.columns[0]
        id_name = df_test_chemicals[col_name_id]

        neighbors = self.get_neighbors(col_name_id, id_name, test_indices, n_neighbors, df_set)

        # print(test_distances)
        # print(neighbors)

        distances = list(test_distances[0])

        return neighbors, distances


class PlotCreator:
    
    def insert_image(self, file_path, username, fk_model_id, fk_file_type_id, session):

        try:
        # Read the image file as binary
            with open(file_path, 'rb') as file:
                binary_data = file.read()
        
            # Prepare the SQL query
            insert_query = text("""
            INSERT INTO qsar_models.model_files (created_at, created_by, file, updated_at, updated_by, fk_file_type_id, fk_model_id)
            VALUES (:created_at, :created_by, :file, :updated_at, :updated_by, :fk_file_type_id, :fk_model_id)
            """)
        
            # Data to insert
            data = {
                'created_at': datetime.now(),
                'created_by': username,
                'file': binary_data,
                'updated_at': datetime.now(),
                'updated_by': username,
                'fk_file_type_id': fk_file_type_id,  # Example foreign key value
                'fk_model_id': fk_model_id  # Example foreign key value
            }
        
            session.execute(insert_query, data)
            session.commit()
            
        except Exception as e:
            session.rollback()
            print(f"An error occurred: {e}")
    
    def display_image(self, fk_model_id, fk_file_type_id, session):
        try:
            # Prepare the SQL query to retrieve the image using foreign keys
            select_query = text("""
            SELECT file FROM qsar_models.model_files 
            WHERE fk_model_id = :fk_model_id AND fk_file_type_id = :fk_file_type_id
            """)
    
            # Execute the query
            result = session.execute(select_query, {'fk_model_id': fk_model_id, 'fk_file_type_id': fk_file_type_id})
            row = result.fetchone()
            
            if row and row[0]:  # Access the first element of the tuple
                # Get the binary data from the result
                binary_data = row[0]
                
                # Define a temporary file path
                temp_file_path = "data/plots/db/" + str(fk_model_id) + "_" + str(fk_file_type_id) + '.png'
                
                # Write the binary data to a temporary file
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(binary_data)
                
                # Open the image in the default web browser
                webbrowser.open('file://' + os.path.realpath(temp_file_path))
            else:
                print("No image found with the specified foreign keys.")
        except Exception as e:
            print(f"An error occurred: {e}")
            session.rollback()
    
    def createTrainingTestPlotsForReports(self):
    
            username = "tmarti02"
    
            try:
                    
                from models import make_test_plots as mtp 
                            
                script_dir = os.path.dirname(os.path.abspath(__file__))
                folder_path = os.path.join(script_dir, "data/plots/")
                os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    
                mi = ModelInitializer()
                models = mi.get_available_models()
                
                session = getSession()
                
                for model_dict in models:
                    model = mi.init_model(model_dict["modelId"])
                    print(model.modelId)              
    
                    mpsTraining = model.df_preds_training_cv.to_dict(orient='records')
                    mpsTest = model.df_preds_test.to_dict(orient='records')
                    
                    filePathOutScatter = os.path.join(folder_path, "scatter_plot_" + str(model.modelId) + ".png")
                    title = model.modelName + " results for " + model.propertyName
                    mtp.generateScatterPlot2(filePathOut=filePathOutScatter, title=title, unitName=model.unitsModel,
                                             mpsTraining=mpsTraining, mpsTest=mpsTest,
                                             seriesNameTrain="Training set (CV)", seriesNameTest="Test set")
    
                    self.insert_image(filePathOutScatter, username, model.modelId, 3, session)
                    self.display_image(model.modelId, 3, session)
                    
                    filePathOutHistogram = os.path.join(folder_path, "histogram_" + str(model.modelId) + ".png")
                    
                    mtp.generateHistogram2(fileOutHistogram=filePathOutHistogram, property_name=model.propertyName, unit_name=model.unitsModel,
                                           mpsTraining=mpsTraining, mpsTest=mpsTest,
                                           seriesNameTrain="Training set", seriesNameTest="TestSet")
                    self.insert_image(filePathOutHistogram, username, model.modelId, 4, session)
                    self.display_image(model.modelId, 4, session)
                    
            except Exception as ex:
                ex.with_traceback()
                print(f"Exception occurred: {ex}")
            
            finally:
                session.close()


cache = {}


class ModelPredictor:

    @timer
    def predictFromDB(self, model_id, smiles, generate_report=True):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        """

        # Make sure the model is loaded before the concurrency
        mi = ModelInitializer()
        mi.init_model(model_id)

        if isinstance(smiles, str):

            key = f"{model_id}-{smiles}"
            
            if key in cache:
                print("in cache: " + key)
                return cache[key]
            else:
                cache[key], code = self.predict_model_smiles(model_id, smiles, generate_report=generate_report)
                return cache[key]
        else:
            result, missing = [], []

            for smi in smiles:
                key = f"{model_id}-{smi}"
                if key in cache:
                    result.append(cache[key])
                else:
                    missing.append(smi)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.map(self.predict_model_smiles, [model_id for _ in missing], missing)

                for (r, code, smi) in results:
                    if code != 200:
                        r = dict(smiles=smi, error=r)
                        result.append(r)
                    else:
                        result.append(r)

                    key = f"{model_id}-{smi}"
                    cache[key] = r

            return result

    def smiles_to_base64(self, smiles_string):
        '''
        TODO: move to utility class
        :param smiles_string:
        '''
        
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        
        try:
            mol = indigo.loadMolecule(smiles_string)
            indigo.setOption("render-output-format", "png") 
            indigo.setOption("render-image-width", 400)
            indigo.setOption("render-image-height", 400)
            img_bytes = renderer.renderToBuffer(mol)
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return base64_string
    
        except Exception as e:
            # print(f"An error occurred while loading the molecule: {e}")
            return None
    
    def setExpValue(self, model, modelResults:ModelResults):

        qsarSmiles = modelResults.chemical["canonicalSmiles"]

        # print(qsarSmiles)

        matching_row_training = model.df_training[model.df_training['ID'] == qsarSmiles]
        matching_row_test = model.df_prediction[model.df_prediction['ID'] == qsarSmiles]

        # print(model.df_training.shape)
        # print(matching_row_training['Property'].values[0])

        if not matching_row_training.empty:
            modelResults.experimentalValueUnitsModel = matching_row_training['Property'].values[0]
            modelResults.experimentalValueSet = "Training"

        if not matching_row_test.empty:
            modelResults.experimentalValueUnitsModel = matching_row_test['Property'].values[0]
            modelResults.experimentalValueSet = "Test"

        # print(modelResults.experimentalValue,modelResults.experimentalValueSet )

    def setExpPredValuesForADAnalogs(self, model, analogs):

        analogs2 = []

        df_preds = model.df_preds_training_cv

        for analog in analogs:  # these are just the qsarSmiles

            matching_row_dsstox = model.df_dsstoxRecords[model.df_dsstoxRecords['canonicalSmiles'] == analog]

            if not matching_row_dsstox.empty:
                row_as_dict = matching_row_dsstox.iloc[0].to_dict()
                analogs2.append(row_as_dict)

            # Find the matching row in model.df_preds_test
            matching_row_pred = df_preds[df_preds['id'] == analog]
            if not matching_row_pred.empty:
                # Add exp and pred values to the record
                row_as_dict['exp'] = matching_row_pred['exp'].values[0]
                row_as_dict['pred'] = matching_row_pred['pred'].values[0]

        return analogs2

    def addDistances(self, analogs, distances):
        # print(len(analogs), len(distances))
        for index, analog in enumerate(analogs):  # these are just the qsarSmiles
            analog["distance"] = distances[index] 


    def setExpPredValuesForNeighbors(self, df_preds, neighbors, df_dsstoxRecords):

        neighbors2 = []

        for neighbor in neighbors:  # these are just the qsarSmiles

            matching_row_dsstox = df_dsstoxRecords[df_dsstoxRecords['canonicalSmiles'] == neighbor]

            if not matching_row_dsstox.empty:
                row_as_dict = matching_row_dsstox.iloc[0].to_dict()
                neighbors2.append(row_as_dict)

            # Find the matching row in model.df_preds_test
            matching_row_pred = df_preds[df_preds['id'] == neighbor]
            if not matching_row_pred.empty:
                # Add exp and pred values to the record
                row_as_dict['exp'] = matching_row_pred['exp'].values[0]
                row_as_dict['pred'] = matching_row_pred['pred'].values[0]

        return neighbors2

    @timer
    def addNeighborsFromSets(self, model:Model, modelResults: ModelResults, df_test_chemicals):

        ng = NeighborGetter()
        # import time 
        # t1 = time.time()
        
        neighborsTest, distances_test = ng.find_neighbors_in_set(model=model, df_set=model.df_prediction, df_test_chemicals=df_test_chemicals)
        neighborsTraining, distances_training = ng.find_neighbors_in_set(model=model, df_set=model.df_training, df_test_chemicals=df_test_chemicals)

        # print(distances_test)


        neighborsTraining = self.setExpPredValuesForNeighbors(model.df_preds_training_cv, neighborsTraining, model.df_dsstoxRecords)
        neighborsTest = self.setExpPredValuesForNeighbors(model.df_preds_test, neighborsTest, model.df_dsstoxRecords)
                
        self.addDistances(neighborsTraining, distances_training)    
        self.addDistances(neighborsTest, distances_test)
                            
                
        df_neighborsTest = pd.DataFrame(neighborsTest)
        stats_test = stats.calculate_continuous_statistics(df_neighborsTest, 0, PredictConstants.TAG_TEST)
        neighborsTestMAE = stats_test[pc.MAE + pc.TAG_TEST]
                    
        df_neighborsTraining = pd.DataFrame(neighborsTraining)
        stats_training = stats.calculate_continuous_statistics(df_neighborsTraining, 0, PredictConstants.TAG_TRAINING)
        neighborsTrainingMAE = stats_training[pc.MAE + pc.TAG_TRAINING]
    
        modelResults.neighborsForSets.append({"set": "Test", "neighbors":neighborsTest, "MAE":neighborsTestMAE})
        modelResults.neighborsForSets.append({"set": "Training", "neighbors":neighborsTraining, "MAE":neighborsTrainingMAE})
        
    @timer
    def getFragmentAD(self, df_prediction, df_training, modelResults:ModelResults):
        start_column = "As [+5 valence, one double bond]"
        stop_column = "-N=S=O"
        df_new = df_prediction.loc[:, start_column:stop_column]
        columns_greater_than_one = df_new.iloc[0] > 0
        df_new = df_new.loc[:, columns_greater_than_one]
        common_columns = df_new.columns.intersection(df_training.columns)
        
        # # Calculate min and max for each common column in df_training
        min_values = df_training[common_columns].apply(lambda col: col[col > 0].min())
        max_values = df_training[common_columns].max()
        
        results = {
            "test_chemical": df_new.loc[0, common_columns].to_dict(),
            "training_min": min_values.to_dict(),
            "training_max": max_values.to_dict()
        }
        
        modelResults.adResultsFrag = results
        
        outside_ad = False
        
        for col_name in modelResults.adResultsFrag["test_chemical"].keys():

            test_value = int(modelResults.adResultsFrag["test_chemical"][col_name])
            training_min = int(modelResults.adResultsFrag["training_min"][col_name])
            training_max = int(modelResults.adResultsFrag["training_max"][col_name])

                    # Determine if the row should be highlighted
            if test_value < training_min or test_value > training_max:
                outside_ad = True
                        
        adResultsFrag = {}
        adResultsFrag["AD"] = not outside_ad
        adResultsFrag["fragmentTable"] = results     
        adResultsFrag["method"] = pc.TEST_FRAGMENTS
        modelResults.applicabilityDomains.append(adResultsFrag)
    
    @timer
    def predict_model_smiles(self, model_id, smiles, generate_report=True, useFileAPI=True):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")

        # initialize model bytes and all details from db:
        
        mi = ModelInitializer()
        model = mi.init_model(model_id)
        
        modelDetails = ModelDetails(model)
    
        # Standardize smiles:
        chemical, code = self.standardizeStructure(serverAPIs, smiles, model)
        # print(json.dumps(chemical, indent=4))
        
        # print(chemical)
    
        if code != 200:
            
            error = chemical
            
            chemical = {}
            chemical["chemId"] = smiles  # TODO add inchiKey
            chemical["smiles"] = smiles
            img_base64 = self.smiles_to_base64(chemical["smiles"])
            
            if img_base64:                            
                chemical["imageSrc"] = f'data:image/png;base64,{img_base64}'
            else:
                chemical["imageSrc"] = "N/A"
            
            modelResults = ModelResults(chemical=chemical, modelDetails=modelDetails)
            # modelResults.predictionError = "Could not generate QSAR Ready SMILES for "+smiles
            modelResults.predictionError = error
            self.addLinks(modelResults, useFileAPI)
            
            return modelResults.to_json(), 200

        if "smiles" in chemical and "cid" not in chemical:
            img_base64 = self.smiles_to_base64(chemical["smiles"])
            chemical["imageSrc"] = f'data:image/png;base64,{img_base64}'
        else:
            chemical["imageSrc"] = imgURLCid + chemical["cid"]


        #TODO: right now it's letting salts through if the qsarReadySmiles isn't salt should we return error here?            
        if model.omitSalts and "." in smiles:
            pass
                            
        qsarSmiles = chemical['canonicalSmiles']
    
        # print("Running descriptors")
        # Descriptor calcs:
        descriptorAPI = DescriptorsAPI()
        df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles,
                                                                  model.descriptorService)
        # print(self.printFirstRowDF(df_prediction))
        
        # print(df_prediction, code)
        
        if code != 200:
            modelResults = ModelResults(chemical=chemical, modelDetails=modelDetails)
            self.addLinks(modelResults, useFileAPI)
            modelResults.predictionError = df_prediction
            return modelResults.to_json(), 200
    
        # Run model prediction:
        # df_prediction = model.model_details.predictionSet #all chemicals in the model's prediction set, for testing
        # print("for qsarSmiles="+qsarSmiles+", descriptors="+json.dumps(descriptorsResults,indent=4))
                
        json_predictions = call_do_predictions_from_df(df_prediction, model)        
        # print(json_predictions)
        
        pred_results = json.loads(json_predictions)
        
        pred_value = pred_results[0]['pred']
        
        # applicability domain calcs:
        ad_results = None
        if model.applicabilityDomainName:
            ad_results = self.determineApplicabilityDomain(model, df_prediction)
        else:
            print('AD method for model was not set:', model_id)
    
        # store everything in results:
        modelResults = ModelResults(chemical, modelDetails)
        modelResults.chemical = chemical
        
        # TODO add values in display units here: 
        modelResults.predictionValueUnitsModel = pred_value
        modelResults.unitsModel = model.unitsModel  # duplicated so displayed near prediction value

        # set exp value
        self.setExpValue(model, modelResults)

        uc = UnitsConverter()
        
        if "sid" not in chemical:
            chemical["sid"] = "N/A"
            chemical["cid"] = "N/A"
            chemical["name"] = "N/A"
                
        if modelResults.experimentalValueUnitsModel:
            modelResults.experimentalValueUnitsDisplay = uc.convert_units(model.propertyName, modelResults.experimentalValueUnitsModel, model.unitsModel, model.unitsDisplay,
                                                                    chemical["sid"], chemical["averageMass"])
        
        # TODO:  convert to unitsDisplay and add here
        modelResults.predictionValueUnitsDisplay = uc.convert_units(model.propertyName, pred_value, model.unitsModel, model.unitsDisplay,
                                                                    chemical["sid"], chemical["averageMass"])
        modelResults.unitsDisplay = model.unitsDisplay
                
        # print("modelResults.predictionValueUnitsDisplay", modelResults.predictionValueUnitsDisplay)
        
        ad_results["method"] = modelDetails.applicabilityDomainName    
        ad_results["analogs"] = self.setExpPredValuesForADAnalogs(model, ad_results["analogs"])
        self.addDistances(ad_results["analogs"], ad_results["distances"])        
        del ad_results['distances']
        
        
        modelResults.applicabilityDomains.append(ad_results)
        
        # print("modelResults", modelResults.to_json)
        
        if generate_report:
            self.addNeighborsFromSets(model, modelResults, df_prediction)
            self.getFragmentAD(df_prediction, model.df_training, modelResults)
            self.addLinks(modelResults, useFileAPI)
            # print(useFileAPI)
            
        # print(results_json)
        return modelResults.to_json(), 200
        # return modelResults
    
    def addLinks(self, modelResults, useFileAPI=True):
        
        modelId = modelResults.modelDetails.modelId
        
        if USE_TEMPORARY_MODEL_PLOTS:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path_scatter = os.path.join(script_dir, "data/plots", "scatter_plot_" + modelId + ".png")
            modelResults.modelDetails.imgSrcPlotScatter = pathlib.Path(file_path_scatter).as_uri()
            file_path_histogram = os.path.join(script_dir, "data/plots", "histogram_" + modelId + ".png")
            modelResults.modelDetails.imgSrcPlotHistogram = pathlib.Path(file_path_histogram).as_uri()
            
        elif useFileAPI:
            # these need to be added to ctx api:
            modelResults.modelDetails.imgSrcPlotScatter = urlCtxApi + "?typeId=3&modelId=" + modelResults.modelDetails.modelId
            modelResults.modelDetails.imgSrcPlotHistogram = urlCtxApi + "?typeId=4&modelId=" + modelResults.modelDetails.modelId
            
        else:
            # TODO generate plot and hardcode in the html
            # plot_base64 = self.generateScatterPlot(neighbors, md.unitsModel, "Results for "+modelResults.modelDetails.modelName, "Exp. vs Pred.")                                    
            # modelResults.modelDetails.imgSrcPlotScatter="data:image/png;base64,"+plot_base64
            pass
        
        modelResults.modelDetails.urlQMRF = urlCtxApi + "?typeId=1&modelId=" + modelResults.modelDetails.modelId
        modelResults.modelDetails.urlExcelSummary = urlCtxApi + "?typeId=2&modelId=" + modelResults.modelDetails.modelId

    @timer
    def standardizeStructure(self, serverAPIs, smiles, model: Model):
        useFullStandardize = False
        qsAPI = QsarSmilesAPI()
        chemicals, code = qsAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                           workflow=model.qsarReadyRuleSet)
        logging.debug(chemicals)

        if code == 500:
            return smiles +": could not generate QSAR Ready SMILES", code 
                
        if len(chemicals) == 0:
            # logging.debug('Standardization failed')
            return f"{smiles} failed standardization" if smiles else 'No Structure', 400

        if len(chemicals) > 1 and model.omitSalts:
            # print('qsar smiles indicates mixture')
            return f"{smiles}: model can't run mixtures", 400

        chemical = chemicals[0]
        qsarSmiles = chemical["canonicalSmiles"]
        logging.debug(f"qsarSmiles: {qsarSmiles}")
        return chemical, 200

    def predictSetFromDB_SmilesFromExcel(self, model_id, excel_file_path, sheetName):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain using smiles in an excel file
        Stores results in tsv file in same folder as excel file
        Runs one at a time since standardizer and descriptors are slow if not cached in mongo (qsar predictions are fast- could aggregate dataframe to run at the end though)
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        descriptorAPI = DescriptorsAPI()

        # serverAPIs = "https://hcd.rtpnc.epa.gov" #TODO: this should come from environment variable
        serverAPIs = "https://cim-dev.sciencedataexperts.com/"

        mi = ModelInitializer()

        # initialize model bytes and all details from db:
        model = mi.init_model(model_id)

        df = pd.read_excel(excel_file_path, sheet_name=sheetName)
        smiles_list = df['Smiles'].tolist()  # Extract the 'Smiles' column into a list

        directory = os.path.dirname(excel_file_path)

        # Create a text file path in the same directory
        text_file_path = os.path.join(directory, "output.txt")
        print(text_file_path)

        with open(text_file_path, 'w') as file:
            file.write("smiles\tqsarSmiles\tpred_value\tpred_AD\n")

            # for smiles, predOld in zip(smiles_list, pred_list):
            for smiles in smiles_list:
                chemical, code = self.standardizeStructure(serverAPIs, smiles, model)

                qsarSmiles = chemical["canonicalSmiles"]

                if code != 200:
                    print(smiles, qsarSmiles)
                    file.write(smiles + "\terror smiles")
                    continue

                df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, model.descriptorService)
                if code != 200:
                    print(smiles, 'error descriptors')
                    file.write(smiles + "\terror descriptors")

                    continue

                pred_results = json.loads(mwu.call_do_predictions_from_df(df_prediction, model))
                pred_value = pred_results[0]['pred']

                ad_results = self.determineApplicabilityDomain(model, df_prediction)
                # ad_results = json.loads(str_ad_results)
                # pred_AD = ad_results[0]["AD"]
                pred_AD = ad_results["AD"]

                line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\t" + str(pred_AD) + "\n"
                print(line)
                file.write(line)
                file.flush()

        if True:
            return

        # Standardize smiles:

        # Descriptor calcs:

        # Run model prediction:
        # df_prediction = model.model_details.predictionSet #all chemicals in the model's prediction set, for testing
        # print("for qsarSmiles="+qsarSmiles+", descriptors="+json.dumps(descriptorsResults,indent=4))

        logging.debug(pred_results)

        # # applicability domain calcs:
        # ad_results = None
        # if model.applicabilityDomainName:
        #     str_ad_results = determineApplicabilityDomain(model, df_prediction)
        #     # str_ad_results = determineApplicabilityDomain(model, model.df_prediction) #testing AD method using multiple chemicals in df
        #
        #     ad_results = json.loads(str_ad_results)[0]  # TODO check len first?
        #     print(ad_results)
        # else:
        #     print('AD method for model was not set:', model_id)

        return "OK", 200

    def printFirstRowDF(self, df):
        first_row_dict = df.loc[0].to_dict()
        print(json.dumps(first_row_dict, indent=4))
        return first_row_dict

    @timer
    def determineApplicabilityDomain(self, model: Model, df_prediction):
        """
        Calculate the applicability domain using the model's training set and the AD measure assigned to the model in the DB
        TODO make sure this works when a model doesnt have a set embedding object
        :param model:
        :param df_prediction:
        :return:
        """
        # json_model_description = model.get_model_description()
        # model_description = json.loads(json_model_description)
        # model.remove_log_p_descriptors = model_description["remove_log_p_descriptors"]  # just set to False instead?

        # print("model.remove_log_p_descriptors", model.remove_log_p_descriptors)

        # print("remove_log_p", remove_log_p)

        from applicability_domain import applicability_domain_utilities as adu
        # model.applicabilityDomainName = adu.strOPERA_local_index  # for testing diff number of neighbors

        output,ad_cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
            train_df=model.df_training,
            test_df=df_prediction,
            # test_df=model.df_prediction,  #for testing running batch type ad calc
            remove_log_p=model.remove_log_p_descriptors,
            embedding=model.embedding,
            applicability_domain=model.applicabilityDomainName,
            filterColumnsInBothSets=True)


        # print("AD_CUTOFF",ad_cutoff)
        # self.printFirstRowDF(output)
                
        # TODO: following code will have to be revised for batch model calculations (have more than one row in df_predictio
        analogsAD = output['ids'].tolist()[0]  # only use first one for singleton prediction
        
        # print(json.dumps(analogsAD,indent=4))
        # dictsAnalogs = [ast.literal_eval(item) for item in analogsAD]
        
        AD = output['AD'].tolist()[0]
        
        distances = list(output["distances"][0])

        results = {"AD":AD, "analogs": analogsAD, "distances": distances, "AD_Cutoff": ad_cutoff}
                    
        # print(results)

        # print(json.dumps(dicts,indent=4))
        return results  # gives an array instead of each object on separate line

# def createHmtlReportFromJson():
#
#     # model_id = str(1065)  # HLC, smallest dataset
#     # model_id = str(1066)  # WS
#     # model_id = str(1067)  # VP
#     # model_id = str(1068)  # BP
#     # model_id = str(1069)  # LogKow
#     model_id = str(1070) # MP, biggest dataset
#     smiles = "c1ccccc1"
#
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(script_dir, model_id + "_report.json")
#     with open(file_path, 'r') as file:
#         modelResults = json.load(file)
#
#     modelResults=ModelResults(**modelResults)
#
#     print(json.dumps(modelResults.modelDetails,indent=4))
#
#     from report_creator import ReportCreator
#     rc = ReportCreator()
#     html=rc.create_html_report(modelResults)
#
#     file_path = os.path.join(script_dir, model_id + "_report.html")
#
#     # Write the HTML to the specified file path
#     with open(file_path, 'w') as f:
#         f.write(html)
#
#     # mi=ModelInitializer()
#     # models=mi.get_available_models()
#     # session=getSession()
#     # for model in models:
#     #     print(model['modelId'])
#     #     mi.get_dsstox_records_for_dataset(model['modelId'], session)
#
#     webbrowser.open(f'file://{file_path}')

# def safe_smiles(smiles):
#     """
#     Convert a SMILES string into a safe version for use in file paths.
#     """
#     import re
#     # Define a regex pattern to match unsafe characters
#     unsafe_chars = r'[<>:"/\\|?*()=]'
#     # Replace unsafe characters with underscores
#     safe_version = re.sub(unsafe_chars, '_', smiles)
#     # Optionally, replace spaces with underscores
#     safe_version = safe_version.replace(' ', '_')
#     return safe_version
#
#
# def runExample2(model_id, smiles, generate_report, file_format, useValeryCode):
#
#     if useValeryCode:
#         output, code, smiles = predict_model_smiles(model_id, smiles) #doesnt create reports
#         file_name = model_id + "_report_valery_" + safe_smiles(smiles) + ".json"
#     else:
#         mp = ModelPredictor()
#         file_name = model_id + "_report_todd_" + safe_smiles(smiles) + ".json"
#         output, code = mp.predict_model_smiles(model_id, smiles, generate_report=generate_report, useFileAPI=True)
#         # print(output)
#
#
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path = os.path.join(script_dir, "data/reports", file_name)
#
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#
#     with open(file_path, 'w', encoding='utf-8') as f:
#         f.write(output)
#
#     # webbrowser.open(f'file://{file_path}')
#
#     if file_format == "html":
#
#         rc=ReportCreator()
#
#         with open(file_path, 'r') as file:
#             modelResults = json.load(file)
#             # print(type(modelResults))
#             # print(modelResults)
#             html=rc.create_html_report(mr=modelResults)
#             file_path_html = file_path.replace(".json", ".html")
#             with open(file_path_html, 'w', encoding='utf-8') as f:
#                 f.write(html)
#             htmlPath = pathlib.Path(file_path_html)
#             webbrowser.open(htmlPath)


def runExample():

    global USE_TEMPORARY_MODEL_PLOTS
    USE_TEMPORARY_MODEL_PLOTS = False

    # model_id = str(1065)  # HLC, smallest dataset
    # model_id = str(1066)  # WS
    # model_id = str(1067)  # VP
    model_id = str(1068)  # BP
    # model_id = str(1069)  # LogKow
    # model_id = str(1070) # MP, biggest dataset
    # model_id = str(1615) # Koc, MLR model

    smiles_list = []
    smiles_list.append("c1ccccc1") # benzene
    # smiles_list.append("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F") # PFOA
    # smiles_list.append("COCOCOCOCCCCCCOCCCCOCOCOCCC") # not in DssTox
    # smiles_list.append("CCCCCCCc1ccccc1") # not in DssTox
    # smiles_list.append("C[Sb]") # passes standardizer, fails test descriptors
    # smiles_list.append("C[As]C[As]C") # violates frag AD
    # smiles_list.append("XX")  # fails standardizer
    # smiles_list.append("CCC.Cl") # not mixture according to qsarReadySmiles
    # smiles_list.append("CCCCC.CCCC") # mixture according to qsarReadySmiles
     
    
    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, "data", "reports")
    os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    
    mp = ModelPredictor()
    
    for smiles in smiles_list:
        print("\nRunning " + smiles)        
    
        runChemical(mp, model_id, smiles, folder_path)
    
def runChemical(mp, model_id, smiles, folder_path):
    
    output, code = mp.predict_model_smiles(model_id, smiles)

    modelResults = json.loads(output)
    chemical = modelResults["chemical"]
    
    chemId = chemical.get("chemId", "N/A")
    file_name = model_id + "_" + chemId + ".json"
    file_path = os.path.join(folder_path, file_name)
    
    # Ensure the directory exists
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(output)
        
    # webbrowser.open(f'file://{file_path}')
    
    rc = ReportCreator()
    
    with open(file_path, 'r') as file:
        modelResults = json.load(file)
        # print(type(modelResults))
        # print(modelResults)
        html = rc.create_html_report(mr=modelResults)

        file_path_html = file_path.replace(".json", ".html")
        with open(file_path_html, 'w', encoding='utf-8') as f:
            f.write(html)
        htmlPath = pathlib.Path(file_path_html)
        webbrowser.open(htmlPath)
            

def runExampleFromService():

    import requests

    # Define the parameters
    smiles = "c1ccccc1"
    model_id = "1065"
    report_format = "json"

    # Define the base URL
    base_url = "http://localhost:5004/models/predictDB"

    # Set up the parameters as a dictionary
    params = {
        'smiles': smiles,
        'model_id': model_id,
        'generate_report': 'true',
        'report_format': report_format
    }

    # Define headers if necessary
    headers = {
        'Accept-Encoding': 'json'
    }

    # Make the GET request with parameters
    response = requests.get(base_url, headers=headers, params=params)

    # Check the response
    if not response.ok:
        print(f"Request failed with status code: {response.status_code}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, model_id + "_report." + report_format)

    # Write the HTML to the specified file path
    with open(file_path, 'w') as f:
        f.write(response.text)

    # mi=ModelInitializer()
    # models=mi.get_available_models()
    # session=getSession()
    # for model in models:
    #     print(model['modelId'])
    #     mi.get_dsstox_records_for_dataset(model['modelId'], session)

    webbrowser.open(f'file://{file_path}')


def test_say_hello():
    import requests
    # Define the name to test
    test_name = "World"
    # Send a GET request to the /hello/<name> endpoint on localhost
    response = requests.get(f'http://localhost:5004/hello/{test_name}')
    print(response.text)


if __name__ == '__main__':
    
    runExample()
    
    # pc = PlotCreator()
    # pc.createTrainingTestPlotsForReports()
    # pc.display_image(1065, 3, getSession())
    # pc.display_image(1065, 4, getSession())
    
    # excel_file_path = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\hibernate_qsar_model_building\data\reports\prediction reports upload\WebTEST2.1\HLC v1 modeling_RND_REPRESENTATIVE.xlsx"    
    # mp=ModelPredictor()
    # mp.predictSetFromDB_SmilesFromExcel(1065,excel_file_path,'Test set')

    # modelStatistics = ModelStatistics()
    # modelStatistics.updateStatsPredictModuleModels()
