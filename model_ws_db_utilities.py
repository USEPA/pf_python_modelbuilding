# import concurrent
import concurrent.futures
import json
import os
import threading
from io import BytesIO
import pathlib
import traceback
from indigo import Indigo
from indigo.renderer import IndigoRenderer
import base64

from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, bindparam

from API_Utilities import QsarSmilesAPI, DescriptorsAPI

# from db.mongo_cache import get_cached_prediction, cache_prediction
from db.mongo_cache import get_cached_prediction, cache_prediction

from util import predict_constants as pc

from model_ws_utilities import call_do_predictions_from_df, models
from models import df_utilities as dfu
from models.ModelBuilder import Model

import StatsCalculator as stats
import pandas as pd
# import pytz

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import model_ws_utilities as mwu
import numpy as np

from util.units_converter import UnitsConverter

from utils import timer, print_first_row
from applicability_domain import applicability_domain_utilities as adu

# debug = False
import logging

logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

fk_dsstox_snapshot_id = 1  # DSSTOX Snapshot 04/23 (Physchem models were created 2024-02-29), if use fk = 2 or 3 will have more missing records

# Following records are cids in the physchem models that didnt make it into the dsstox_records table for fk_dsstox_snapshot_id = 1 (dsstox changed slightly) 
# Following was created by ModelInitializer.findMissingDsstoxRecordsInPhyschemModelDatasets()
dict_missing_dsstox_records = {    
               # "DTXCID001783033": {"smiles": "[H][C@]12[C@@H](Cl)[C@H](Cl)[C@](C(Cl)Cl)([C@@H](Cl)[C@@H]1Cl)C2(C)C(Cl)Cl"},#has no matching sid in dsstox
               "DTXCID201784601": {"smiles": "CC[C@@H]1CCC[C@H]1C", "sid": "DTXSID2075055", "casrn": "930-90-5", "name": "trans-1-Methyl-2-ethylcyclopentane"},
               "DTXCID401783809": {"smiles": "[H][C@]12CO[S@](=O)OC[C@@]1([H])[C@@]1(Cl)C(Cl)=C(Cl)[C@]2(Cl)C1(Cl)Cl", "sid": "DTXSID8037540", "casrn": "33213-65-9", "name": "Endosulfan II"},
               "DTXCID501782911": {"smiles": "CN1C[C@@]2(C=C)[C@@H]3C[C@H]4OC[C@@H]3[C@@H]1[C@@H]2[C@@]41C(=O)NC2=CC=CC=C12", "sid": "DTXSID40878487", "casrn": "509-15-9", "name": "Gelsemine"},
               "DTXCID501782985": {"smiles": "[H][C@]12CC(Cl)(Cl)[C@](CCl)([C@@H](Cl)[C@@H]1Cl)C2(CCl)CCl", "sid": "DTXSID80874069", "casrn": "51775-36-1", "name": "2,2,5-endo,6-exo,8,9,10-Heptachlorobornane"},
               "DTXCID501783733": {"smiles": "[H][C@]12O[C@@]1([H])[C@@]1([H])[C@@]([H])([C@H]2Cl)[C@@]2(Cl)C(Cl)=C(Cl)[C@]1(Cl)C2(Cl)Cl", "sid": "DTXSID1024126", "casrn": "1024-57-3", "name": "Heptachlor epoxide B"},
               "DTXCID601783831": {"smiles": "[H][C@]12CO[S@@](=O)OC[C@@]1([H])[C@@]1(Cl)C(Cl)=C(Cl)[C@]2(Cl)C1(Cl)Cl", "sid": "DTXSID9037539", "casrn": "959-98-8", "name": "Endosulfan I"},
               "DTXCID701521422": {"smiles": "[H][C@@]12CCCC[C@@]1([H])CCCC2", "sid": "DTXSID90883405", "casrn": "493-02-7", "name": "trans-Decahydronaphthalene"}}

USE_TEMPORARY_MODEL_PLOTS = False

imgURLCid = "https://comptox.epa.gov/dashboard-api/ccdapp1/chemical-files/image/by-dtxcid/";

"""
Not completed:
TODO: make a batch mode
TODO: Add experimental tab with raw data
TODO: Add ability to export report as excel
"""

lock = threading.Lock()


def getEngine():
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
    return engine


def getSession():
    engine = getEngine()
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

        except SQLAlchemyError as ex:
            print("error getting stats for modelId=" + str(model.modelId))
    
    def get_predictions(self, session, model: Model, split_num, fk_splitting_id):
        
        logging.debug("Getting model training/prediction set TSVs")

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
            df = pd.DataFrame(results, columns=["id", "exp", "pred"])
            return df

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
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
            return df

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
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

            # Execute the query
            row = session.execute(sql, {'model_id': m.modelId}).fetchone()

            # Process the result
            if row:
                self.row_to_model_details(m, row)

        except Exception as ex:
            ex.with_traceback()
            print(f"Exception occurred: {ex}")

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

    def updateUnits(self, model):
        
        if model.propertyName in [
            pc.WATER_SOLUBILITY,
            pc.ACUTE_AQUATIC_TOXICITY,
            pc.NINETY_SIX_HOUR_FATHEAD_MINNOW_LC50,
            pc.NINETY_SIX_HOUR_SCUD_LC50,
            pc.NINETY_SIX_HOUR_RAINBOW_TROUT_LC50,
            pc.NINETY_SIX_HOUR_BLUEGILL_LC50,
            pc.FORTY_EIGHT_HR_TETRAHYMENA_PYRIFORMIS_IGC50,
            pc.FORTY_EIGHT_HR_DAPHNIA_MAGNA_LC50
        ]: 
            model.unitsDisplay = pc.MG_L  # units that program office wants (Dashboard uses mol/L)

    @timer
    def initModel(self, model_id):

        session = getSession()

        model_bytes = self.get_model_bytes(model_id, session)

        if not model_bytes:
            logging.error(f"Couldnt load {model_id} from model bytes")
            return

        import pickle
        model = pickle.loads(model_bytes)

        if not model:
            logging.error(f"Couldnt load {model_id} from model bytes")
            return

        model.modelId = model_id

        if not hasattr(model, "is_binary"):
            logging.info('model.is_binary is none, setting to false')
            model.is_binary = False

        # Stores model under provided number

        self.get_model_details(model, session)
        
        self.updateUnits(model)
        
        self.get_model_statistics(model, session)
        self.get_training_prediction_instances(session, model)
        self.get_dsstox_records_for_dataset(model, session)

        # get following for pred values for neighbors:
        model.df_preds_test = self.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
        model.df_preds_training_cv = self.get_cv_predictions(session, model)

        logging.debug(f"model_description with added metadata:{model.get_model_description_pretty()}")

        session.close()

        return model
    
    def getQsarDtxcid(self, qsarSmiles, datasetName, session):
        
        sql = text("""
            select dp.qsar_dtxcid  from qsar_datasets.datasets d
            join qsar_datasets.data_points dp on dp.fk_dataset_id = d.id
            where d.name = :datasetName and dp.canon_qsar_smiles = :qsarSmiles;
            """)
        
        # print(sql)
        
        try:
            connection = session.connection()
            row = connection.execute(sql, {"datasetName": datasetName, "qsarSmiles": qsarSmiles}).fetchone()
            if row is not None:
                # Row supports positional access; use row[0] for the first column
                val = row[0]
                return str(val).split("|", 1)[0]            

        except Exception as e:
            print(e)
            return None
    
    def getDtxsid(self, dtxcid, session):
        """
        Some of the dsstox records are missing because in dsstox there is no longer a matching dtxsid for given dtxcid
        """
        
        session = getSession()
        
        sql = text("""
            select dr.dtxsid, dr.preferred_name from qsar_models.dsstox_records dr
            where dr.dtxcid = :dtxcid and dr.fk_dsstox_snapshot_id=:fk_dsstox_snapshot_id;
            """)
        
        try:
            connection = session.connection()
            row = connection.execute(sql, {"dtxcid": dtxcid, "fk_dsstox_snapshot_id": fk_dsstox_snapshot_id }).fetchone()
        
            if row is None:
                # no hit — handle appropriately
                # e.g., return None, or raise, or use defaults
                return None, None
            else:
                a, b = row  # or row[0], row[1]
                return a, b
            
        except Exception as e:
            traceback.print_exc()
            return None

    def get_dsstox_records_for_dataset(self, model: Model, session):
        """
        Gets the dsstox records for the dataset from res_qsar postgreSQL db (could also get from dsstox or a snapshot of dsstox)
        Some of the dp.qsar_dtxcid values may not have matching value in dsstox_records because dsstox has changed and the cid no longer has matching sid
        """
        try:
            # Get a connection from the session
            connection = session.connection()

            # SQL query to retrieve bytes
            
            # TODO: need to fix because the dtxcid may have changed so that the dsstox record will be retrieved

            # Note: in the data_points table, sometimes the qsar_dtxcid is pipe delimited pair of cids
            sql = """
                SELECT dp.canon_qsar_smiles as "canonicalSmiles", dr.dtxsid as sid, dr.dtxcid as cid, dr.casrn, dr.preferred_name as "name" , dr.smiles, dr.indigo_inchi_key as "inchiKey"
                    FROM qsar_datasets.datasets d
                    JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
                    LEFT JOIN qsar_models.dsstox_records dr ON dr.dtxcid = split_part(dp.qsar_dtxcid, '|', 1)
                """                

            sql = text(sql + "\nWHERE d.name = :datasetName and dr.fk_dsstox_snapshot_id = :fk_dsstox_snapshot_id;")

            # print(sql)

            # Execute the query with the parameter
            result = connection.execute(sql, {"datasetName": model.datasetName, "fk_dsstox_snapshot_id": fk_dsstox_snapshot_id})

            # Convert result to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            model.df_dsstoxRecords = df;
            
            none_sid_smiles = df[df['sid'].isnull()]['canonicalSmiles']

            if len(none_sid_smiles) > 0:
                print(model.modelId, "Have canonicalSmiles in dataset that isn't in dsstox records:", none_sid_smiles)

            return df

        except Exception as e:
            print(f"Exception occurred: {e}")
            return None

    def get_training_prediction_instances(self, session, model:Model):
        logging.debug("Getting model training/prediction set TSVs")

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

                if instance is None:
                    logging.debug(f"{id}\tnull instance\tdatasetName={model.datasetName}\tdescriptorSetName={model.descriptorSetName}")
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
            
            logging.debug(f"trainingSet shape:{model.df_training.shape}")
            logging.debug(f"predictionSet shape:{model.df_prediction.shape}")

        except SQLAlchemyError as ex:
            print(f"An error occurred: {ex}")
        finally:
            pass

    def generate_instance(self, chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\n"
    
    def getModelMetaDataQuery(self):
        """
        returns the query to get the model metadata
         left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
        """
        return """
        SELECT 
                m.id,
                m.name_ccd,
                m.details,
                d.id,
                d.name,
                d.description, 
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
        m.datasetDescription,
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
        self.modelSourceName = model.modelSource
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
            
        self.propertyIsBinary = model.is_binary

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
        self.modelCoefficients = None
        
    
class ModelResults:

    def __init__(self, adResults=None):
        
        self.experimentalValueUnitsModel = None
        self.experimentalValueUnitsDisplay = None
        self.experimentalValueSet = None
        self.predictionValueUnitsModel = None
        self.unitsModel = None
        self.predictionValueUnitsDisplay = None
        self.unitsDisplay = None
        self.predictionError = None
        self.adEstimates = []

    def to_dict(self):
        # Convert the object to a dictionary, including nested objects
        return {
            "experimentalValueUnitsModel": self.experimentalValueUnitsModel,
            "experimentalValueUnitsDisplay": self.experimentalValueUnitsDisplay,
            "experimentalValueSet": self.experimentalValueSet,
            "predictionValueUnitsModel": self.predictionValueUnitsModel,
            "unitsModel": self.unitsModel,
            "predictionValueUnitsDisplay": self.predictionValueUnitsDisplay,
            "unitsDisplay": self.unitsDisplay,
            "predictionError": self.predictionError,
            "adEstimates": self.adEstimates
        }

        
class Report:

    def __init__(self, chemical, modelDetails:ModelDetails, modelResults:ModelResults):
        
        self.chemicalIdentifiers = chemical
        self.modelDetails = modelDetails
        self.modelResults = modelResults
        self.neighborResultsPrediction = None
        self.neighborResultsTraining = None 
    
    def to_dict(self):
        # Convert the object to a dictionary, including nested objects
        return {
            "chemicalIdentifiers": self.chemicalIdentifiers,
            "modelDetails": self.modelDetails.__dict__ if self.modelDetails else None,
            "modelResults": self.modelResults.__dict__ if self.modelResults else None,
            "neighborResultsTraining": self.neighborResultsTraining,
            "neighborResultsPrediction": self.neighborResultsPrediction,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


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
        
        useEmbedding = True  # probably should be consistent with what is use to find the Applicability domain analogs
        
        if useEmbedding:  # Using embedding descriptors picks weird neighbors sometimes:
            TrainSet = df_set[model.embedding]
            TestSet = df_test_chemicals[model.embedding]
            scaler = StandardScaler().fit(TrainSet)
            train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
        
        else:  # Following just uses all TEST descriptors (removes constant ones)
            TrainSet = df_set
            TestSet = df_test_chemicals
            ids_train, labels_train, features_train, column_names_train, is_binary = dfu.prepare_instances(df=TrainSet,
                                                                                                           which_set="Training",
                                                                                                           remove_logp=model.remove_log_p_descriptors,
                                                                                                           remove_corr=False, remove_constant=True)        
            features_test = TestSet[features_train.columns]
            scaler = StandardScaler().fit(features_train)
            train_x, test_x = scaler.transform(features_train), scaler.transform(features_test)

        nbrs.fit(train_x)
        # train_distances, train_indices = nbrs.kneighbors(train_x)
        test_distances, test_indices = nbrs.kneighbors(test_x)

        col_name_id = df_set.columns[0]
        id_name = df_test_chemicals[col_name_id]

        neighbors = self.get_neighbors(col_name_id, id_name, test_indices, n_neighbors, df_set)

        # print(len(test_distances))
        # print(len(neighbors))

        distances = list(test_distances[0])

        return neighbors, distances

# cache = {}


class ModelPredictor:

    @timer
    def predictFromDB(self, model_id, smiles):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """
    
        if isinstance(smiles, str):
            key = f"{smiles}-{model_id}"
            prediction = get_cached_prediction(key)
            if prediction:
                return prediction
            else:
                prediction, code = self.predict_model_smiles(model_id, smiles)
                cache_prediction(key, prediction)
                return prediction
        else:
            smiles_list = list(smiles)
            if not smiles_list:
                return []

            result = [None] * len(smiles_list)
            missing = []

            for idx, smi in enumerate(smiles_list):
                key = f"{smi}-{model_id}"
                prediction = get_cached_prediction(key)
                if prediction is not None:
                    result[idx] = prediction
                else:
                    missing.append((idx, smi))

            if missing:
                max_workers = int(os.getenv("PREDICT_THREAD_WORKERS", min(32, (os.cpu_count() or 1) * 5)))
                max_workers = max(1, min(max_workers, len(missing)))

                def _predict_one(item):
                    idx, smi = item
                    prediction, code = self.predict_model_smiles(model_id, smi)
                    if code != 200:
                        prediction = dict(smiles=smi, error=prediction)
                    cache_prediction(f"{smi}-{model_id}", prediction)
                    return idx, prediction

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                    for idx, prediction in pool.map(_predict_one, missing):
                        result[idx] = prediction

            return result

    def addPerformance(self, md: ModelDetails):
        ms = md.modelStatistics or {}

        def set_metric(block: dict, out_key: str, *candidates: str):
            """Put first existing metric from candidates into block[out_key]."""
            for k in candidates:
                if k in ms and ms[k] is not None:
                    block[out_key] = ms[k]
                    return
            block[out_key] = None

        md.performance = {}

        md.performance["train"] = {}
        set_metric(md.performance["train"], "R2", "PearsonRSQ_Training", "R2_Training")
        set_metric(md.performance["train"], "RMSE", "RMSE_Training")
        set_metric(md.performance["train"], "MAE", "MAE_Training")

        md.performance["fiveFoldICV"] = {}
        set_metric(md.performance["fiveFoldICV"], "R2", "PearsonRSQ_CV_Training", "PearsonRSQ_CV")
        set_metric(md.performance["fiveFoldICV"], "RMSE", "RMSE_CV_Training", "RMSE_CV", "RMSE_CV_Train")
        set_metric(md.performance["fiveFoldICV"], "MAE", "MAE_CV_Training", "MAE_CV")

        md.performance["external"] = {}
        set_metric(md.performance["external"], "R2", "PearsonRSQ_Test", "R2_Test")
        set_metric(md.performance["external"], "RMSE", "RMSE_Test")
        set_metric(md.performance["external"], "MAE", "MAE_Test")

        md.performance["externalAD"] = {}
        set_metric(md.performance["externalAD"], "MAE_inside_AD", "MAE_Test_inside_AD")
        set_metric(md.performance["externalAD"], "MAE_outside_AD", "MAE_Test_outside_AD")
        set_metric(md.performance["externalAD"], "Fraction_inside_AD", "Coverage_Test")

        md.modelStatistics = None
    
    def smiles_to_base64(self, smiles_string, width=400, height=400):
        '''
        TODO: move to utility class
        :param smiles_string:
        '''
        
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        
        try:
            mol = indigo.loadMolecule(smiles_string)
            indigo.setOption("render-output-format", "png") 
            indigo.setOption("render-image-width", width)
            indigo.setOption("render-image-height", height)
            img_bytes = renderer.renderToBuffer(mol)
            base64_string = base64.b64encode(img_bytes).decode('utf-8')
            return base64_string
    
        except Exception as e:
            # print(f"An error occurred while loading the molecule: {e}")
            return None
    
    def setExpValue(self, chemical, model, modelResults:ModelResults):

        qsarSmiles = chemical["canonicalSmiles"]

        matching_row_training = model.df_training[model.df_training['ID'] == qsarSmiles]
        matching_row_test = model.df_prediction[model.df_prediction['ID'] == qsarSmiles]

        if not matching_row_training.empty:
            modelResults.experimentalValueUnitsModel = matching_row_training['Property'].values[0]
            modelResults.experimentalValueSet = "Training"

        if not matching_row_test.empty:
            modelResults.experimentalValueUnitsModel = matching_row_test['Property'].values[0]
            modelResults.experimentalValueSet = "Test"

    def setExpPredValuesForADAnalogs(self, model, analogs):

        analogs2 = []

        df_preds = model.df_preds_training_cv

        for analog in analogs:  # these are just the qsarSmiles

            matching_row_dsstox = model.df_dsstoxRecords[model.df_dsstoxRecords['canonicalSmiles'] == analog]

            if not matching_row_dsstox.empty:
                row_as_dict = matching_row_dsstox.iloc[0].to_dict()
            else:
                row_as_dict = self.fixMissingNeighborDsstoxRecord(model.datasetName, analog)

            analogs2.append(row_as_dict)

            # Find the matching row in model.df_preds_test
            matching_row_pred = df_preds[df_preds['id'] == analog]
            if not matching_row_pred.empty:
                # Add exp and pred values to the record
                row_as_dict['exp'] = matching_row_pred['exp'].values[0]
                row_as_dict['pred'] = matching_row_pred['pred'].values[0]
        
        # print(json.dumps(analogs2,indent=4))
        
        return analogs2

    def addDistances(self, analogs, distances):
        # print(len(analogs), len(distances))
        for index, analog in enumerate(analogs):  # these are just the qsarSmiles
            analog["distance"] = distances[index] 

    def fixMissingNeighborDsstoxRecord(self, datasetName, qsarSmiles):
        row_as_dict = {
            "canonicalSmiles":qsarSmiles}
        mi = ModelInitializer()
        
        try:
            session = getSession()

            dtxcid = mi.getQsarDtxcid(qsarSmiles, datasetName, session)
            
            if dtxcid:
                row_as_dict["cid"] = dtxcid
                
                if dtxcid in dict_missing_dsstox_records:  # dict_missing_dsstox_records is global variable
                    rec = dict_missing_dsstox_records[dtxcid]
                    row_as_dict["sid"] = rec["sid"]
                    row_as_dict["name"] = rec["name"]
                    row_as_dict["smiles"] = rec["smiles"]
                    row_as_dict["casrn"] = rec["casrn"]
                    
                else:
                    row_as_dict["name"] = dtxcid    
                
            else:
                row_as_dict["name"] = qsarSmiles

            session.close()
            
            # print(dtxcid, dtxsid)

        except Exception as e:
            print(e)

        return row_as_dict

    def setExpPredValuesForNeighbors(self, model:Model, df_preds, neighbors, df_dsstoxRecords):

        neighbors2 = []

        for neighbor in neighbors:  # these are just the qsarSmiles

            matching_row_dsstox = df_dsstoxRecords[df_dsstoxRecords['canonicalSmiles'] == neighbor]

            if not matching_row_dsstox.empty:
                row_as_dict = matching_row_dsstox.iloc[0].to_dict()
                neighbors2.append(row_as_dict)
            else:
                logging.debug("Finding missing dsstox info for " + neighbor)  # only happens for 8 dtxcids
                row_as_dict = self.fixMissingNeighborDsstoxRecord(model.datasetName, neighbor)
                logging.debug(row_as_dict)
                
                # print(neighbor +" not in dsstox records")
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
        
        neighborsTest, distances_test = ng.find_neighbors_in_set(model=model, df_set=model.df_prediction, df_test_chemicals=df_test_chemicals)
        neighborsTraining, distances_training = ng.find_neighbors_in_set(model=model, df_set=model.df_training, df_test_chemicals=df_test_chemicals)

        neighborsTraining = self.setExpPredValuesForNeighbors(model, model.df_preds_training_cv, neighborsTraining, model.df_dsstoxRecords)
        neighborsTest = self.setExpPredValuesForNeighbors(model, model.df_preds_test, neighborsTest, model.df_dsstoxRecords)
                
        self.addDistances(neighborsTraining, distances_training)    
        self.addDistances(neighborsTest, distances_test)
                
        df_neighborsTest = pd.DataFrame(neighborsTest)
        stats_test = stats.calculate_continuous_statistics(df_neighborsTest, 0, pc.TAG_TEST)
        neighborsTestMAE = stats_test[pc.MAE + pc.TAG_TEST]
                    
        df_neighborsTraining = pd.DataFrame(neighborsTraining)
        stats_training = stats.calculate_continuous_statistics(df_neighborsTraining, 0, pc.TAG_TRAINING)
        neighborsTrainingMAE = stats_training[pc.MAE + pc.TAG_TRAINING]
        
        neighborResultsPrediction = {"set": "Test", "neighbors":neighborsTest, "MAE":neighborsTestMAE,
                                                "unitNeighbor":modelResults.unitsModel,
                                                "title": "Nearest Neighbors from Test Set (External Predictions)"}
        neighborResultsTraining = {"set": "Training", "neighbors":neighborsTraining, "MAE":neighborsTrainingMAE,
                                              "unitNeighbor":modelResults.unitsModel,
                                              "title": "Nearest Neighbors from Training Set (Cross Validation Predictions)"}
        
        return neighborResultsTraining, neighborResultsPrediction
        
    @timer
    def getFragmentAD(self, df_prediction, df_training, modelResults:ModelResults):

        start_column = "As [+5 valence, one double bond]"
        stop_column = "-N=S=O"
        df_new = df_prediction.loc[:, start_column:stop_column]
        columns_greater_than_one = df_new.iloc[0] > 0
        df_new = df_new.loc[:, columns_greater_than_one]
        common_columns = df_new.columns.intersection(df_training.columns)
        
        # # Calculate min and max for each common column in df_training
        min_values = (df_training[common_columns].apply(lambda col: col[col > 0].min()).fillna(0))        

        max_values = df_training[common_columns].max()
        
        results = {
            "test_chemical": df_new.loc[0, common_columns].to_dict(),
            "training_min": min_values.to_dict(),
            "training_max": max_values.to_dict()
        }
        
        outside_ad = False
        
        for col_name in results["test_chemical"].keys():
            test_value = int(results["test_chemical"][col_name])
            training_min = int(results["training_min"][col_name])    
            training_max = int(results["training_max"][col_name])

                    # Determine if the row should be highlighted
            if test_value < training_min or test_value > training_max:
                outside_ad = True
                        
        adResultsFrag = {}
        adResultsFrag["adMethod"] = {}
        adResultsFrag["adMethod"]["name"] = pc.TEST_FRAGMENTS
        adResultsFrag["adMethod"]["description"] = "Whether or not the fragment counts are within the range for chemicals in the training set"

        adResultsFrag["AD"] = not outside_ad
        adResultsFrag["fragmentTable"] = results     
        
        if outside_ad: 
            adResultsFrag["reasoning"] = "fragment counts were not within the training set range"
            adResultsFrag["conclusion"] = "Outside"
        else:
            adResultsFrag["reasoning"] = "fragment counts were within the training set range"
            adResultsFrag["conclusion"] = "Inside"
                
        modelResults.adEstimates.append(adResultsFrag)
    
    @timer
    def predict_model_smiles(self, model_id, smiles, generate_report=True):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
        fileAPI = os.getenv("FILE_API_SERVER", pc.URL_CTX_API)

        logging.info("serverAPIS:{serverAPIs}")
        logging.info("model.qsarReadyRuleSet:{model.qsarReadyRuleSet}")

        # initialize model bytes and all details from db:
        
        mi = ModelInitializer()
        model = mi.init_model(model_id)
        
        if serverAPIs == "https://hcd.rtpnc.epa.gov/" and model.qsarReadyRuleSet == 'qsar-ready_04242025_0':
            model.qsarReadyRuleSet = 'qsar-ready_04242025'  # latest rules arent on there yet
        
        if hasattr(model, 'modelId') == False:
            return f"Invalid model_id: {model_id}", 400
        
        modelDetails = ModelDetails(model)

        if 'reg_' in model.modelMethod or 'las_' in model.modelMethod or 'gcm_' in model.modelMethod:
            y = model.df_training[model.df_training.columns[1]]
            X = model.df_training[model.embedding]
            modelDetails.modelCoefficients = json.loads(model.getOriginalRegressionCoefficients2(X, y))
        
        self.addLinks(modelDetails, fileAPI)
        self.addPerformance(modelDetails)
        
        modelResults = ModelResults()
    
        # Standardize smiles:
        chemical, code = self.standardizeStructure(serverAPIs, smiles, model)
    
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
            
            modelResults.predictionError = error
            report = Report(chemical, modelDetails, modelResults)
            return report.to_json(), 200

        if "smiles" in chemical and "cid" not in chemical:
            img_base64 = self.smiles_to_base64(chemical["smiles"])
            chemical["imageSrc"] = f'data:image/png;base64,{img_base64}'
        else:
            chemical["imageSrc"] = imgURLCid + chemical["cid"]

        # TODO: right now it's letting salts through if the qsarReadySmiles isn't salt should we return error here?            
        if model.omitSalts and "." in smiles:
            pass
                            
        qsarSmiles = chemical['canonicalSmiles']
    
        # print("Running descriptors")
        # Descriptor calcs:
        descriptorAPI = DescriptorsAPI()
        df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles,
                                                                  model.descriptorService)
        
        if code != 200:
            report = Report(chemical, modelDetails, modelResults)
            modelResults.predictionError = df_prediction
            return report.to_json(), 200
                
        json_predictions = call_do_predictions_from_df(df_prediction, model)        
        # print(json_predictions)
        pred_results = json.loads(json_predictions)
        pred_value = pred_results[0]['pred']
        
        # applicability domain calcs:
        if model.applicabilityDomainName:
            applicabilityDomains = modelDetails.applicabilityDomainName.split(" and ")
            if pc.Applicability_Domain_TEST_Fragment_Counts not in applicabilityDomains:
                applicabilityDomains.append(pc.Applicability_Domain_TEST_Fragment_Counts)

            for applicabilityDomain in applicabilityDomains:
                ad_results = self.determineApplicabilityDomain(model, applicabilityDomain, df_prediction)
                modelResults.adEstimates.append(ad_results)
        else:
            print('AD method for model was not set:', model_id)

        modelResults.predictionValueUnitsModel = pred_value
        modelResults.unitsModel = model.unitsModel  # duplicated so displayed near prediction value

        # set exp value
        self.setExpValue(chemical, model, modelResults)

        uc = UnitsConverter()
        
        if "sid" not in chemical:
            chemical["sid"] = "N/A"
            chemical["cid"] = "N/A"
            chemical["name"] = "N/A"
                
        if modelResults.experimentalValueUnitsModel:
            modelResults.experimentalValueUnitsDisplay = uc.convert_units(model.propertyName, modelResults.experimentalValueUnitsModel, model.unitsModel, model.unitsDisplay,
                                                                    chemical["sid"], chemical["averageMass"])
        
        modelResults.predictionValueUnitsDisplay = uc.convert_units(model.propertyName, pred_value, model.unitsModel, model.unitsDisplay,
                                                                    chemical["sid"], chemical["averageMass"])
        modelResults.unitsDisplay = model.unitsDisplay
                
        report = Report(chemical, modelDetails, modelResults)

        if generate_report:
            report.neighborResultsTraining, report.neighborResultsPrediction = self.addNeighborsFromSets(model, modelResults, df_prediction)
            
        return report.to_json(), 200
    
    def addLinks(self, modelDetails, file_api=pc.URL_CTX_API):
        modelId = str(modelDetails.modelId)
    
        from util.web_utils import append_query
    
        if USE_TEMPORARY_MODEL_PLOTS:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path_scatter = os.path.join(script_dir, "data/plots", f"scatter_plot_{modelId}.png")
            modelDetails.imgSrcPlotScatter = pathlib.Path(file_path_scatter).as_uri()
    
            file_path_histogram = os.path.join(script_dir, "data/plots", f"histogram_{modelId}.png")
            modelDetails.imgSrcPlotHistogram = pathlib.Path(file_path_histogram).as_uri()
    
        elif file_api:
            base = str(file_api)
    
            # ctx search endpoint (must be .../search/ before '?')
            if "chemical/property/model/file/search" in base:
                force_slash = True
                modelDetails.imgSrcPlotScatter = append_query(base, {"typeId": 3, "modelId": modelId}, force_trailing_slash=force_slash)
                modelDetails.imgSrcPlotHistogram = append_query(base, {"typeId": 4, "modelId": modelId}, force_trailing_slash=force_slash)
                modelDetails.urlQMRF = append_query(base, {"typeId": 1, "modelId": modelId}, force_trailing_slash=force_slash)
                modelDetails.urlExcelSummary = append_query(base, {"typeId": 2, "modelId": modelId}, force_trailing_slash=force_slash)
    
            # predictor models endpoint
            elif "api/predictor_models/model/file" in base:
                # This endpoint typically doesn’t require the trailing slash, but you can enforce if needed
                modelDetails.imgSrcPlotScatter = append_query(base, {"type_id": 3, "model_id": modelId})
                modelDetails.imgSrcPlotHistogram = append_query(base, {"type_id": 4, "model_id": modelId})
                modelDetails.urlQMRF = append_query(base, {"type_id": 1, "model_id": modelId})
                modelDetails.urlExcelSummary = append_query(base, {"type_id": 2, "model_id": modelId})
    
            else:
                logging.error(f"Invalid file_api: {file_api}")
    
        else:
            # TODO: generate plots inline and embed as data URLs if no API is available
            pass

    @timer
    def standardizeStructure(self, serverAPIs, smiles, model: Model):
        useFullStandardize = False
        chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                           workflow=model.qsarReadyRuleSet)
        logging.debug(chemicals)
        
        if code == 400:
            return chemicals, code
                
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
        
    def standardizeStructure2(self, serverAPIs, smiles, qsarReadyRuleSet, omitSalts):
        useFullStandardize = False
        chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                           workflow=qsarReadyRuleSet)
        logging.debug(chemicals)

        if code == 500:
            return smiles + ": could not generate QSAR Ready SMILES", code 
                
        if len(chemicals) == 0:
            # logging.debug('Standardization failed')
            return f"{smiles} failed standardization" if smiles else 'No Structure', 400

        if len(chemicals) > 1 and omitSalts:
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

                ad_results = self.determineApplicabilityDomain(model, model.applicabilityDomainName, df_prediction)
                pred_AD = ad_results["AD"]

                line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\t" + str(pred_AD) + "\n"
                print(line)
                file.write(line)
                file.flush()

        return "OK", 200

    @timer
    def determineApplicabilityDomain(self, model: Model, applicabilityDomainName, df_prediction):
        """
        Calculate the applicability domain using the model's training set and the AD measure assigned to the model in the DB
        TODO make sure this works when a model doesnt have a set embedding object
        :param model:
        :param df_prediction:
        :return:
        """

        output, ad_cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
            train_df=model.df_training,
            test_df=df_prediction,
            # test_df=model.df_prediction,  #for testing running batch type ad calc
            remove_log_p=model.remove_log_p_descriptors,
            embedding=model.embedding,
            # applicability_domain=model.applicabilityDomainName,
            applicability_domain=applicabilityDomainName,
            filterColumnsInBothSets=True)

        AD = output['AD'].tolist()[0]
        
        if 'ids' in output.columns and "distances" in output.columns:
            analogsAD = output['ids'].tolist()[0]  # only use first one for singleton prediction
            distances = list(output["distances"][0])
            results = {"AD":AD, "analogs": analogsAD, "distances": distances, "AD_Cutoff": ad_cutoff}
        else:
            results = {"AD":AD}
                
        # print(json.dumps(analogsAD,indent=4))
        # dictsAnalogs = [ast.literal_eval(item) for item in analogsAD]
        
        results["adMethod"] = {}
        results["adMethod"]["name"] = applicabilityDomainName
        # results["adMethod"]["description"] = model.applicabilityDomainDescription

        if applicabilityDomainName == pc.Applicability_Domain_TEST_Embedding_Euclidean\
         or applicabilityDomainName == pc.Applicability_Domain_TEST_All_Descriptors_Euclidean:
            
            results["value"] = sum(distances) / len(distances)
            
            if AD == True: 
                results["conclusion"] = "Inside"
                results["reasoning"] = f"Avg. distance ({results['value']:.2f}) < {ad_cutoff:.2f}" 
            else: 
                results["conclusion"] = "Outside"
                results["reasoning"] = f"Avg. distance ({results['value']:.2f}) > {ad_cutoff:.2f}"
                
            results["analogs"] = self.setExpPredValuesForADAnalogs(model, results["analogs"])
            results["adMethod"]["description"] = 'Whether or not the average Euclidean distance of the three closest training set neighbors exceeds a cutoff defined so that 95% of the training set is within the AD'

            self.addDistances(results["analogs"], results["distances"])        
            del results['distances']
                
        elif applicabilityDomainName == pc.Applicability_Domain_TEST_Fragment_Counts:

            results["adMethod"]["description"] = 'Whether the TEST fragments are within the training set range'
            results["fragment_table"] = output["fragment_table"].tolist()[0]

            if AD == True:
                results["conclusion"] = "Inside"
                results["reasoning"] = "Fragments in test chemical are within the training set ranges"
            else:
                results["conclusion"] = "Outside"
                results["reasoning"] = "Fragments in test chemical are NOT within the training set ranges"

            if 'gcm' in model.modelMethod:
                haveMissingFragmentInModel = False
                for fragment in results["fragment_table"]:
                    if fragment["fragment"] not in model.embedding:
                        haveMissingFragmentInModel = True
                        fragment["fragment"] += "**"                                    
                if haveMissingFragmentInModel:
                    results["conclusion"] = "Outside"
                    results["reasoning"] = "Have fragment in test chemical that is NOT included in the model"

            for fragment in results["fragment_table"]:
                if fragment['test_value'] < fragment['training_min'] or fragment['test_value'] > fragment['training_max']:
                    if "*" not in fragment["fragment"]:
                        fragment["fragment"] += "*"            

        else:
            print("handle " + applicabilityDomainName + " in determineApplicabilityDomain()")
            
        return results  # gives an array instead of each object on separate line
            

def main():
    
    from dotenv import load_dotenv
    load_dotenv('personal.env')
    ######################################################################################################
    excel_file_path = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\hibernate_qsar_model_building\data\reports\prediction reports upload\WebTEST2.1\HLC v1 modeling_RND_REPRESENTATIVE.xlsx"
    mp = ModelPredictor()
    mp.predictSetFromDB_SmilesFromExcel(1065, excel_file_path, 'Test set')
    ######################################################################################################


if __name__ == '__main__':
    main()
