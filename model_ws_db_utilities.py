import concurrent.futures
import functools
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

from db.mongo_cache import (
    cache_predictions,
    get_cached_predictions,
)

from util import predict_constants as pc

from model_ws_utilities import models
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
from util.indigo_utils import IndigoUtils
from util.prediction_cache_key_utils import (
    build_prediction_cache_key,
    ensure_chemical_inchi_key,
    normalize_inchi_key,
)

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
_MISSING_NEIGHBOR_DSS_TOX_CACHE = {}
_MISSING_NEIGHBOR_DSS_TOX_LOCK = threading.Lock()
_INDIGO_UTILS_LOCAL = threading.local()

"""
Not completed:
TODO: make a batch mode
TODO: Add experimental tab with raw data
TODO: Add ability to export report as excel
"""

lock = threading.Lock()
_engine_lock = threading.Lock()
_engine = None
_session_factory = None


@functools.lru_cache(maxsize=4096)
def _render_smiles_to_base64_cached(smiles_string, width=400, height=400):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)

    try:
        mol = indigo.loadMolecule(smiles_string)
        indigo.setOption("render-output-format", "png")
        indigo.setOption("render-image-width", width)
        indigo.setOption("render-image-height", height)
        img_bytes = renderer.renderToBuffer(mol)
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception:
        return None


def _get_indigo_utils():
    indigo_utils = getattr(_INDIGO_UTILS_LOCAL, "indigo_utils", None)
    if indigo_utils is None:
        indigo_utils = IndigoUtils()
        _INDIGO_UTILS_LOCAL.indigo_utils = indigo_utils
    return indigo_utils


@functools.lru_cache(maxsize=50000)
def _inchi_key_from_smiles_cached(smiles_string):
    if smiles_string is None:
        return None

    smiles_text = str(smiles_string).strip()
    if not smiles_text:
        return None

    try:
        return normalize_inchi_key(_get_indigo_utils().inchi_key_from_smiles(smiles_text))
    except Exception:
        logging.exception("Failed to generate InChIKey for SMILES=%s", smiles_text)
        return None


def getEngine():
    global _engine, _session_factory

    if _engine is None:
        with _engine_lock:
            if _engine is None:
                connect_url = URL.create(
                    drivername='postgresql+psycopg2',
                    username=os.getenv('DEV_QSAR_USER'),
                    password=os.getenv('DEV_QSAR_PASS'),
                    host=os.getenv('DEV_QSAR_HOST', 'localhost'),
                    port=os.getenv('DEV_QSAR_PORT', 5432),
                    database=os.getenv('DEV_QSAR_DATABASE')
                )

                _engine = create_engine(connect_url, echo=False, pool_pre_ping=True)
                _session_factory = sessionmaker(bind=_engine)

    return _engine


def getSession():
    global _session_factory

    if _session_factory is None:
        getEngine()

    return _session_factory()


class ModelInitializer:
    
    def init_model(self, model_id):

        with lock:
            if model_id in models:
                logging.debug('have model already initialized')
                model = models[model_id]
                if model is not None:
                    return model

                # Previous initialization failed and stored None; allow fresh retry.
                logging.warning(f"Cached model for {model_id} is None, reloading")
                models.pop(model_id, None)
            else:
                model = None

            model = self.initModel(model_id)
            if model is not None:
                models[model_id] = model
            else:
                logging.error(f"Model initialization failed for model_id={model_id}; not caching")

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
            chunk_count = 0

            # Fetch and write byte data to the output stream
            for record in result:
                if record.bytes is None:
                    logging.error(f"Found NULL model bytes chunk for model_id={model_id}")
                    return None
                output_stream.write(record.bytes)  # Assuming the column name is 'bytes'
                chunk_count += 1

            if chunk_count == 0:
                logging.error(f"No model bytes found in DB for model_id={model_id}")
                return None

            # Return the combined byte array
            model_bytes = output_stream.getvalue()
            logging.debug(f"Loaded model bytes for model_id={model_id}: chunks={chunk_count}, total_bytes={len(model_bytes)}")
            return model_bytes

        except Exception:
            logging.exception(f"Failed to read model bytes for model_id={model_id}")
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
        try:
            model_bytes = self.get_model_bytes(model_id, session)

            if not model_bytes:
                logging.error(f"Couldnt load model_id={model_id} from model bytes")
                return None

            import pickle
            try:
                model = pickle.loads(model_bytes)
            except Exception:
                logging.exception(f"Failed to deserialize model bytes for model_id={model_id}")
                return None

            if not model:
                logging.error(f"Deserialized model is empty for model_id={model_id}")
                return None

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
            logging.info(f"Successfully initialized model_id={model_id}")
            return model

        except Exception:
            logging.exception(f"Unexpected failure while initializing model_id={model_id}")
            return None
        finally:
            try:
                session.close()
            except Exception:
                logging.exception(f"Failed to close DB session after initModel(model_id={model_id})")
    
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

def _sanitize_api_chemical_identifiers(chemical):
    if not isinstance(chemical, dict):
        return chemical
    return {
        key: (None if value == "N/A" else value)
        for key, value in chemical.items()
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
            "chemicalIdentifiers": _sanitize_api_chemical_identifiers(self.chemicalIdentifiers),
            "modelDetails": self.modelDetails.__dict__ if self.modelDetails else None,
            "modelResults": self.modelResults.__dict__ if self.modelResults else None,
            "neighborResultsTraining": self.neighborResultsTraining,
            "neighborResultsPrediction": self.neighborResultsPrediction,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


class NeighborGetter:

    def get_neighbors(self, col_name_id, test_indices, n_neighbors, df_set, return_all=False):
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

        if return_all:
            return ids_combined_transposed

        neighbors = ids_combined_transposed[0]
        return neighbors

    def find_neighbors_in_set(self, model, df_set, df_test_chemicals, precomputed=None, return_all=False):

        if precomputed is not None:
            test_set = df_test_chemicals[model.embedding]
            test_x = precomputed["scaler"].transform(test_set)
            test_distances, test_indices = precomputed["nbrs"].kneighbors(test_x)

            ids = precomputed["ids"]
            if return_all:
                neighbors = [[ids[idx] for idx in row_indices] for row_indices in test_indices]
                distances = [list(distance_row) for distance_row in test_distances]
            else:
                neighbors = [ids[idx] for idx in test_indices[0]]
                distances = list(test_distances[0])
            return neighbors, distances

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
        neighbors = self.get_neighbors(
            col_name_id,
            test_indices,
            n_neighbors,
            df_set,
            return_all=return_all,
        )

        # print(len(test_distances))
        # print(len(neighbors))

        if return_all:
            distances = [list(distance_row) for distance_row in test_distances]
        else:
            distances = list(test_distances[0])

        return neighbors, distances

# cache = {}


class ModelPredictor:

    def __init__(self):
        self._descriptor_api = DescriptorsAPI()
        self._units_converter = UnitsConverter()

    @staticmethod
    def _get_inchi_key_from_smiles(smiles):
        return _inchi_key_from_smiles_cached(smiles)

    @staticmethod
    def _normalize_inchi_key(inchi_key):
        return normalize_inchi_key(inchi_key)

    @classmethod
    def _ensure_chemical_inchi_key(cls, chemical, fallback_smiles=None):
        return ensure_chemical_inchi_key(
            chemical,
            cls._get_inchi_key_from_smiles,
            fallback_smiles=fallback_smiles,
        )

    @classmethod
    def _build_prediction_cache_key(cls, model_id, smiles=None, chemical=None):
        return build_prediction_cache_key(
            model_id,
            cls._get_inchi_key_from_smiles,
            smiles=smiles,
            chemical=chemical,
        )

    @staticmethod
    def _to_lookup(df, key_col, value_cols=None):
        if df is None or df.empty or key_col not in df.columns:
            return {}

        lookup_df = df.drop_duplicates(subset=[key_col], keep='first')

        if value_cols is None:
            return lookup_df.set_index(key_col).to_dict(orient='index')

        existing_value_cols = [col for col in value_cols if col in df.columns]
        if not existing_value_cols:
            return {}

        return lookup_df.set_index(key_col)[existing_value_cols].to_dict(orient='index')

    @staticmethod
    def _prediction_to_obj(prediction):
        def _sanitize_prediction_payload(value):
            if isinstance(value, list):
                return [_sanitize_prediction_payload(item) for item in value]

            if isinstance(value, dict):
                sanitized = {}
                for key, item in value.items():
                    if key == "chemicalIdentifiers":
                        sanitized[key] = _sanitize_api_chemical_identifiers(
                            ModelPredictor._ensure_chemical_inchi_key(item)
                        )
                    else:
                        sanitized[key] = _sanitize_prediction_payload(item)
                return sanitized

            return value

        if isinstance(prediction, (dict, list)):
            return _sanitize_prediction_payload(prediction)

        if isinstance(prediction, (bytes, bytearray)):
            prediction = prediction.decode("utf-8")

        if isinstance(prediction, str):
            try:
                return _sanitize_prediction_payload(json.loads(prediction))
            except json.JSONDecodeError:
                return {"error": prediction}

        return {"error": str(prediction)}

    @staticmethod
    def _prediction_to_json_str(prediction):
        if isinstance(prediction, (dict, list)):
            return json.dumps(prediction)

        if isinstance(prediction, (bytes, bytearray)):
            return prediction.decode("utf-8")

        if isinstance(prediction, str):
            return prediction

        return json.dumps({"error": str(prediction)})

    @staticmethod
    def _preview_log_value(value, max_len: int = 500) -> str:
        if isinstance(value, pd.DataFrame):
            return f"DataFrame rows={len(value.index)} cols={list(value.columns[:10])}"
        if isinstance(value, dict):
            return f"dict keys={list(value.keys())[:10]}"
        if isinstance(value, list):
            return f"list len={len(value)}"

        text = str(value).replace("\n", " ").strip()
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

    @staticmethod
    def _strip_model_details_from_prediction(prediction):
        if isinstance(prediction, list):
            return [ModelPredictor._strip_model_details_from_prediction(item) for item in prediction]

        if isinstance(prediction, dict):
            stripped = dict(prediction)
            stripped.pop("modelDetails", None)
            return stripped

        return prediction

    @staticmethod
    def _attach_model_details_to_prediction(prediction, model_details_dict):
        if not model_details_dict or not isinstance(prediction, dict):
            return prediction

        prediction_with_details = dict(prediction)
        prediction_with_details["modelDetails"] = model_details_dict
        return prediction_with_details

    @staticmethod
    def _prediction_from_cached_value(prediction):
        return ModelPredictor._strip_model_details_from_prediction(
            ModelPredictor._prediction_to_obj(prediction)
        )

    @staticmethod
    def _standardization_match_key(value):
        if value is None:
            return None

        value_text = str(value).strip()
        if not value_text:
            return None

        return value_text

    def _get_standardization_response_match_keys(self, chemical):
        if not isinstance(chemical, dict):
            return []

        match_keys = []

        for field_name, value in (
            ("id", chemical.get("id")),
            ("recordId", chemical.get("recordId")),
        ):
            match_key = self._standardization_match_key(value)
            if match_key:
                match_keys.append((field_name, match_key))

        deduplicated_keys = []
        seen_keys = set()
        for field_name, match_key in match_keys:
            if match_key in seen_keys:
                continue
            seen_keys.add(match_key)
            deduplicated_keys.append((field_name, match_key))

        return deduplicated_keys

    def _build_standardization_expected_id_map(self, smiles_list):
        expected_id_map = {}

        for index, _ in enumerate(smiles_list):
            expected_id = str(index)
            expected_id_map.setdefault(expected_id, []).append(index)

        return expected_id_map

    def _build_standardization_results_by_index(self, grouped_by_id, expected_ids, smiles_list, model):
        results_by_index = {}
        mixture_count = 0
        missing_canonical_count = 0

        for index, smiles in enumerate(smiles_list):
            grouped_chemicals = grouped_by_id.get(expected_ids[index], [])
            if not grouped_chemicals:
                continue

            canonical_chemicals = [
                chemical for chemical in grouped_chemicals
                if isinstance(chemical, dict) and chemical.get("canonicalSmiles")
            ]

            if not canonical_chemicals:
                missing_canonical_count += 1
                results_by_index[index] = (f"{smiles} failed standardization", 400)
                continue

            if model.omitSalts and len(canonical_chemicals) > 1:
                mixture_count += 1
                results_by_index[index] = (f"{smiles}: model can't run mixtures", 400)
                continue

            results_by_index[index] = (canonical_chemicals[0], 200)

        return results_by_index, mixture_count, missing_canonical_count

    def _normalize_standardization_batch_results(self, chemicals, smiles_list, model):
        if not isinstance(chemicals, list):
            diagnostic = f"type={type(chemicals).__name__} preview={self._preview_log_value(chemicals)}"
            return None, diagnostic, {}, list(range(len(smiles_list)))

        if (
            len(chemicals) == len(smiles_list)
            and all(isinstance(chemical, dict) and "canonicalSmiles" in chemical for chemical in chemicals)
            and not (model.omitSalts and any("." in chemical.get("canonicalSmiles", "") for chemical in chemicals))
        ):
            return [(chemical, 200) for chemical in chemicals], None, None, []

        expected_id_map = self._build_standardization_expected_id_map(smiles_list)
        expected_ids = [str(index) for index, _ in enumerate(smiles_list)]
        grouped_by_id = {}
        unidentified_count = 0
        matched_by = {}
        response_key_presence = {}

        for chemical in chemicals:
            response_match_keys = self._get_standardization_response_match_keys(chemical)

            for field_name, _ in response_match_keys:
                response_key_presence[field_name] = response_key_presence.get(field_name, 0) + 1

            matched_id = None
            matched_key_name = None
            for field_name, response_key in response_match_keys:
                if response_key not in expected_id_map:
                    continue
                matched_id = response_key
                matched_key_name = field_name
                break

            if matched_id is None:
                unidentified_count += 1
                continue
            grouped_by_id.setdefault(matched_id, []).append(chemical)
            if matched_key_name:
                matched_by[matched_key_name] = matched_by.get(matched_key_name, 0) + 1

        missing_ids = [
            expected_id
            for expected_id in expected_ids
            if expected_id not in grouped_by_id
        ]
        missing_indices = [
            index
            for index, expected_id in enumerate(expected_ids)
            if expected_id not in grouped_by_id
        ]
        results_by_index, mixture_count, missing_canonical_count = self._build_standardization_results_by_index(
            grouped_by_id,
            expected_ids,
            smiles_list,
            model,
        )

        if grouped_by_id and not missing_ids:
            normalized_results = [
                results_by_index.get(index, (f"{smiles} failed standardization", 400))
                for index, smiles in enumerate(smiles_list)
            ]

            diagnostic_parts = [
                f"expanded_response_grouped result_len={len(chemicals)} expected_len={len(smiles_list)}",
                f"grouped_ids={len(grouped_by_id)}",
            ]

            if unidentified_count:
                diagnostic_parts.append(f"unidentified_items={unidentified_count}")
            if matched_by:
                diagnostic_parts.append(f"matched_by={matched_by}")
            if response_key_presence:
                diagnostic_parts.append(f"response_key_presence={response_key_presence}")
            if mixture_count:
                diagnostic_parts.append(f"mixture_groups={mixture_count}")
            if missing_canonical_count:
                diagnostic_parts.append(f"missing_canonical_groups={missing_canonical_count}")

            return normalized_results, " ".join(diagnostic_parts), None, []

        standardization_reason_parts = [f"result_len={len(chemicals)} expected_len={len(smiles_list)}"]
        if missing_canonical_count:
            standardization_reason_parts.append(
                f"missing_canonicalSmiles={missing_canonical_count}"
            )

        if grouped_by_id:
            standardization_reason_parts.append(f"grouped_ids={len(grouped_by_id)}")
        if missing_ids:
            standardization_reason_parts.append(f"missing_ids={missing_ids[:10]}")
        if unidentified_count:
            standardization_reason_parts.append(f"unidentified_items={unidentified_count}")
        if matched_by:
            standardization_reason_parts.append(f"matched_by={matched_by}")
        if response_key_presence:
            standardization_reason_parts.append(f"response_key_presence={response_key_presence}")

        if model.omitSalts:
            mixture_count = sum(
                1 for chemical in chemicals
                if isinstance(chemical, dict) and "." in chemical.get("canonicalSmiles", "")
            )
            if mixture_count:
                standardization_reason_parts.append(
                    f"mixtures_with_omitSalts={mixture_count}"
                )

        standardization_reason_parts.append(
            f"preview={self._preview_log_value(chemicals)}"
        )
        return None, " ".join(standardization_reason_parts), results_by_index, missing_indices

    def _standardize_smiles_individually(self, serverAPIs, smiles_list, model):
        results = []

        for smiles in smiles_list:
            try:
                results.append(self.standardizeStructure(serverAPIs, smiles, model))
            except Exception as exc:
                logging.exception(
                    "Single-smiles standardization failed; workflow=%s smiles=%s",
                    model.qsarReadyRuleSet,
                    smiles,
                )
                results.append((f"{smiles}: standardization request failed: {exc}", 500))

        return results

    def _retry_missing_standardization_subset(self, serverAPIs, smiles_list, model, split_depth=0):
        if not smiles_list:
            return []

        try:
            chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(
                server_host=serverAPIs,
                smiles=smiles_list,
                full=False,
                workflow=model.qsarReadyRuleSet,
            )
        except Exception as exc:
            logging.warning(
                "Retry batch standardization for missing items raised %s; falling back to single requests; "
                "workflow=%s batch_size=%s split_depth=%s error=%s",
                type(exc).__name__,
                model.qsarReadyRuleSet,
                len(smiles_list),
                split_depth,
                self._preview_log_value(exc),
            )
            return self._standardize_smiles_individually(serverAPIs, smiles_list, model)

        if code != 200:
            logging.warning(
                "Retry batch standardization for missing items failed; falling back to single requests; "
                "code=%s workflow=%s batch_size=%s split_depth=%s response=%s",
                code,
                model.qsarReadyRuleSet,
                len(smiles_list),
                split_depth,
                self._preview_log_value(chemicals),
            )
            return self._standardize_smiles_individually(serverAPIs, smiles_list, model)

        normalized_results, normalization_diagnostic, partial_results_by_index, missing_indices = (
            self._normalize_standardization_batch_results(
                chemicals,
                smiles_list,
                model,
            )
        )

        if normalized_results is not None:
            if normalization_diagnostic:
                logging.info(
                    "Standardization retry batch response normalized; workflow=%s batch_size=%s detail=%s",
                    model.qsarReadyRuleSet,
                    len(smiles_list),
                    normalization_diagnostic,
                )
            return normalized_results

        merged_results = [None] * len(smiles_list)
        partial_results_by_index = partial_results_by_index or {}
        for index, result in partial_results_by_index.items():
            merged_results[index] = result

        if missing_indices:
            logging.warning(
                "Retry batch standardization still missing items; falling back to single requests; "
                "workflow=%s batch_size=%s missing_count=%s split_depth=%s reason=%s",
                model.qsarReadyRuleSet,
                len(smiles_list),
                len(missing_indices),
                split_depth,
                normalization_diagnostic,
            )
            single_results = self._standardize_smiles_individually(
                serverAPIs,
                [smiles_list[index] for index in missing_indices],
                model,
            )
            for index, result in zip(missing_indices, single_results):
                merged_results[index] = result

        return [
            result if result is not None else (f"{smiles_list[index]} failed standardization", 400)
            for index, result in enumerate(merged_results)
        ]

    def _build_neighbor_cache(self, model, df_set):
        if df_set is None or df_set.empty or not model.embedding:
            return None

        if not set(model.embedding).issubset(df_set.columns):
            return None

        n_neighbors = min(10, len(df_set))
        if n_neighbors < 1:
            return None

        train_set = df_set[model.embedding]
        scaler = StandardScaler().fit(train_set)
        train_x = scaler.transform(train_set)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='euclidean')
        nbrs.fit(train_x)

        return {
            "scaler": scaler,
            "nbrs": nbrs,
            "ids": df_set.iloc[:, 0].tolist(),
        }

    def _ensure_model_runtime_cache(self, model: Model):
        if getattr(model, "_runtime_cache_ready", False):
            return

        with lock:
            if getattr(model, "_runtime_cache_ready", False):
                return

            model._training_property_by_id = {}
            model._prediction_property_by_id = {}

            if model.df_training is not None and not model.df_training.empty:
                if {'ID', 'Property'}.issubset(model.df_training.columns):
                    model._training_property_by_id = model.df_training.set_index('ID')['Property'].to_dict()

            if model.df_prediction is not None and not model.df_prediction.empty:
                if {'ID', 'Property'}.issubset(model.df_prediction.columns):
                    model._prediction_property_by_id = model.df_prediction.set_index('ID')['Property'].to_dict()

            model._dsstox_by_smiles = self._to_lookup(model.df_dsstoxRecords, 'canonicalSmiles')
            model._preds_test_by_id = self._to_lookup(model.df_preds_test, 'id', ['exp', 'pred'])
            model._preds_training_cv_by_id = self._to_lookup(model.df_preds_training_cv, 'id', ['exp', 'pred'])

            model._neighbors_prediction_cache = self._build_neighbor_cache(model, model.df_prediction)
            model._neighbors_training_cache = self._build_neighbor_cache(model, model.df_training)
            model._ad_runtime_cache = {}

            model._runtime_cache_ready = True

    @staticmethod
    def _iter_applicability_domains(model: Model):
        return [name for name in (model.applicabilityDomainName or "").split(" and ") if name]

    def _get_test_embedding_ad_runtime(self, model: Model):
        self._ensure_model_runtime_cache(model)

        runtime_cache = getattr(model, "_ad_runtime_cache", None)
        if runtime_cache is None:
            runtime_cache = {}
            model._ad_runtime_cache = runtime_cache

        cache_key = pc.Applicability_Domain_TEST_Embedding_Euclidean
        if cache_key in runtime_cache:
            return runtime_cache[cache_key]

        precomputed = getattr(model, "_neighbors_training_cache", None)
        if precomputed is None:
            return None

        train_set = model.df_training[model.embedding]
        if train_set is None or train_set.empty:
            return None

        cutoff = None
        if len(train_set.index) >= 4:
            train_x = precomputed["scaler"].transform(train_set)
            train_distances, _ = precomputed["nbrs"].kneighbors(train_x, n_neighbors=4)
            train_mean_distances = list(np.mean(train_distances[:, 1:], axis=1))
            cutoff = float(adu.adm.helpers.find_split_value(train_mean_distances, 0.95))

        runtime = {
            "scaler": precomputed["scaler"],
            "nbrs": precomputed["nbrs"],
            "ids": precomputed["ids"],
            "cutoff": cutoff,
            "neighbor_count": min(3, len(precomputed["ids"])),
        }
        runtime_cache[cache_key] = runtime
        return runtime

    def _resolve_model_context(self, model_id, serverAPIs, fileAPI):
        mi = ModelInitializer()
        model = mi.init_model(model_id)

        if model is None or hasattr(model, 'modelId') is False:
            logging.error("Returning Invalid model_id for model_id=%s: model failed to initialize", model_id)
            return None, None, f"Invalid model_id: {model_id}"

        self._ensure_model_runtime_cache(model)

        if serverAPIs == "https://hcd.rtpnc.epa.gov/" and model.qsarReadyRuleSet == 'qsar-ready_04242025_0':
            model.qsarReadyRuleSet = 'qsar-ready_04242025'

        model_details_dict = self._get_model_details_dict(model, fileAPI)
        return model, model_details_dict, None

    def get_model_details_dict_for_model_id(self, model_id, serverAPIs=None, fileAPI=None):
        serverAPIs = serverAPIs or os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
        fileAPI = fileAPI or os.getenv("FILE_API_SERVER", pc.URL_CTX_API)

        _, model_details_dict, model_error = self._resolve_model_context(model_id, serverAPIs, fileAPI)
        return model_details_dict, model_error

    def _get_model_details_dict(self, model, fileAPI):
        cache = getattr(model, "_model_details_cache", None)
        if isinstance(cache, dict) and cache.get("file_api") == fileAPI:
            return cache["payload"]

        modelDetails = ModelDetails(model)

        if 'reg_' in model.modelMethod or 'las_' in model.modelMethod or 'gcm_' in model.modelMethod:
            if not hasattr(model, "_regression_coefficients"):
                y = model.df_training[model.df_training.columns[1]]
                X = model.df_training[model.embedding]
                model._regression_coefficients = json.loads(model.getOriginalRegressionCoefficients2(X, y))
            modelDetails.modelCoefficients = model._regression_coefficients

        self.addLinks(modelDetails, fileAPI)
        self.addPerformance(modelDetails)

        payload = dict(modelDetails.__dict__)
        model._model_details_cache = {"file_api": fileAPI, "payload": payload}
        return payload

    @staticmethod
    def _build_report_dict(
        chemical,
        model_details_dict,
        model_results,
        neighbor_results_training=None,
        neighbor_results_prediction=None,
    ):
        return {
            "chemicalIdentifiers": _sanitize_api_chemical_identifiers(chemical),
            "modelDetails": model_details_dict,
            "modelResults": model_results.to_dict() if hasattr(model_results, "to_dict") else model_results,
            "neighborResultsTraining": neighbor_results_training,
            "neighborResultsPrediction": neighbor_results_prediction,
        }

    def _build_minimal_chemical(self, smiles):
        chemical = {
            "chemId": smiles,
            "smiles": smiles,
        }

        img_base64 = self.smiles_to_base64(smiles)
        chemical["imageSrc"] = f'data:image/png;base64,{img_base64}' if img_base64 else "N/A"
        return self._ensure_chemical_inchi_key(chemical, fallback_smiles=smiles)

    def _prepare_chemical_for_report(self, chemical):
        chemical = dict(chemical)

        if "smiles" in chemical and "cid" not in chemical:
            img_base64 = self.smiles_to_base64(chemical["smiles"])
            chemical["imageSrc"] = f'data:image/png;base64,{img_base64}' if img_base64 else "N/A"
        else:
            chemical["imageSrc"] = imgURLCid + chemical["cid"]

        if "sid" not in chemical:
            chemical["sid"] = "N/A"
            chemical["cid"] = "N/A"
            chemical["name"] = "N/A"

        return self._ensure_chemical_inchi_key(chemical)

    def _build_prediction_error_report(self, chemical, model_details_dict, error):
        if isinstance(chemical, dict):
            prepared_chemical = self._prepare_chemical_for_report(chemical)
        else:
            prepared_chemical = self._build_minimal_chemical(chemical)
        model_results = ModelResults()
        model_results.predictionError = error
        return self._build_report_dict(prepared_chemical, model_details_dict, model_results)

    def _fast_test_embedding_euclidean_ad_batch(self, model, applicability_domain_name, df_prediction):
        runtime = self._get_test_embedding_ad_runtime(model)
        if runtime is None or runtime["neighbor_count"] < 1 or runtime["cutoff"] is None:
            return None

        test_set = df_prediction[model.embedding]
        test_x = runtime["scaler"].transform(test_set)
        test_distances, test_indices = runtime["nbrs"].kneighbors(
            test_x,
            n_neighbors=runtime["neighbor_count"],
        )

        results = []
        ids = runtime["ids"]
        ad_cutoff = runtime["cutoff"]

        for distance_row, index_row in zip(test_distances, test_indices):
            distances = list(distance_row)
            analog_ids = [ids[idx] for idx in index_row]
            analogs = self.setExpPredValuesForADAnalogs(model, analog_ids)
            self.addDistances(analogs, distances)

            value = float(np.mean(distance_row)) if len(distance_row) else None
            is_inside = bool(value is not None and value <= ad_cutoff)
            comparison = "<" if is_inside else ">"

            results.append(
                {
                    "AD": is_inside,
                    "analogs": analogs,
                    "AD_Cutoff": ad_cutoff,
                    "adMethod": {
                        "name": applicability_domain_name,
                        "description": "Whether or not the average Euclidean distance of the three closest training set neighbors exceeds a cutoff defined so that 95% of the training set is within the AD",
                    },
                    "value": value,
                    "conclusion": "Inside" if is_inside else "Outside",
                    "reasoning": f"Avg. distance ({value:.2f}) {comparison} {ad_cutoff:.2f}",
                }
            )

        return results

    def _build_applicability_domain_result_from_row(self, model, applicability_domain_name, row, ad_cutoff):
        ad_value = row.get("AD")
        if isinstance(ad_value, np.generic):
            ad_value = ad_value.item()

        results = {
            "AD": ad_value,
            "adMethod": {"name": applicability_domain_name},
        }

        if 'ids' in row.index and "distances" in row.index:
            analogs_ad = list(row["ids"])
            distances = list(row["distances"])
            results.update({"analogs": analogs_ad, "distances": distances, "AD_Cutoff": ad_cutoff})

        if applicability_domain_name == pc.Applicability_Domain_TEST_Embedding_Euclidean \
         or applicability_domain_name == pc.Applicability_Domain_TEST_All_Descriptors_Euclidean:
            distances = results.get("distances", [])
            results["adMethod"]["description"] = 'Whether or not the average Euclidean distance of the three closest training set neighbors exceeds a cutoff defined so that 95% of the training set is within the AD'

            if distances and ad_cutoff is not None:
                results["value"] = sum(distances) / len(distances)
                if ad_value is True:
                    results["conclusion"] = "Inside"
                    results["reasoning"] = f"Avg. distance ({results['value']:.2f}) < {ad_cutoff:.2f}"
                else:
                    results["conclusion"] = "Outside"
                    results["reasoning"] = f"Avg. distance ({results['value']:.2f}) > {ad_cutoff:.2f}"
            else:
                results["value"] = None
                results["conclusion"] = "Unknown"
                results["reasoning"] = "No neighbor distances available"

            if "analogs" in results:
                results["analogs"] = self.setExpPredValuesForADAnalogs(model, results["analogs"])
                self.addDistances(results["analogs"], results.get("distances", []))
                del results["distances"]

        elif applicability_domain_name == pc.Applicability_Domain_TEST_Fragment_Counts:
            results["adMethod"]["description"] = 'Whether the TEST fragments are within the training set range'
            results["fragment_table"] = row.get("fragment_table")

            if ad_value is True:
                results["conclusion"] = "Inside"
                results["reasoning"] = "Fragments in test chemical are within the training set ranges"
            else:
                results["conclusion"] = "Outside"
                results["reasoning"] = "Fragments in test chemical are NOT within the training set ranges"

            if 'gcm' in model.modelMethod:
                have_missing_fragment_in_model = False
                for fragment in results["fragment_table"]:
                    if fragment["fragment"] not in model.embedding:
                        have_missing_fragment_in_model = True
                        fragment["fragment"] += "**"
                if have_missing_fragment_in_model:
                    results["conclusion"] = "Outside"
                    results["reasoning"] = "Have fragment in test chemical that is NOT included in the model"

            for fragment in results["fragment_table"]:
                if fragment['test_value'] < fragment['training_min'] or fragment['test_value'] > fragment['training_max']:
                    if "*" not in fragment["fragment"]:
                        fragment["fragment"] += "*"

        else:
            print("handle " + applicability_domain_name + " in determineApplicabilityDomain()")

        return results

    def _determine_applicability_domain_batch(self, model, applicability_domain_name, df_prediction):
        if df_prediction is None or df_prediction.empty:
            return []

        if applicability_domain_name == pc.Applicability_Domain_TEST_Embedding_Euclidean:
            fast_results = self._fast_test_embedding_euclidean_ad_batch(model, applicability_domain_name, df_prediction)
            if fast_results is not None:
                return fast_results

        output, ad_cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
            train_df=model.df_training,
            test_df=df_prediction,
            remove_log_p=model.remove_log_p_descriptors,
            embedding=model.embedding,
            applicability_domain=applicability_domain_name,
            filterColumnsInBothSets=True,
        )

        return [
            self._build_applicability_domain_result_from_row(
                model,
                applicability_domain_name,
                output.iloc[row_pos],
                ad_cutoff,
            )
            for row_pos in range(len(output.index))
        ]

    def _determine_applicability_domains_batch(self, model, df_prediction):
        row_count = 0 if df_prediction is None else len(df_prediction.index)
        results_by_row = [[] for _ in range(row_count)]

        for applicability_domain in self._iter_applicability_domains(model):
            ad_results = self._determine_applicability_domain_batch(model, applicability_domain, df_prediction)
            for row_pos, row_result in enumerate(ad_results):
                results_by_row[row_pos].append(row_result)

        return results_by_row

    def _finalize_prediction_report(
        self,
        model,
        model_details_dict,
        chemical,
        df_prediction,
        pred_value,
        generate_report=True,
        ad_results=None,
        neighbor_results_training=None,
        neighbor_results_prediction=None,
        prepared_chemical=None,
    ):
        chemical = prepared_chemical if prepared_chemical is not None else self._prepare_chemical_for_report(chemical)
        model_results = ModelResults()

        if ad_results is not None:
            model_results.adEstimates.extend(ad_results)
        elif model.applicabilityDomainName:
            for applicability_domain in self._iter_applicability_domains(model):
                ad_result = self.determineApplicabilityDomain(model, applicability_domain, df_prediction)
                model_results.adEstimates.append(ad_result)
        else:
            print('AD method for model was not set:', model.modelId)

        if isinstance(pred_value, np.generic):
            pred_value = pred_value.item()

        model_results.predictionValueUnitsModel = pred_value
        model_results.unitsModel = model.unitsModel

        self.setExpValue(chemical, model, model_results)

        uc = self._units_converter
        average_mass = chemical.get("averageMass")

        if model_results.experimentalValueUnitsModel:
            model_results.experimentalValueUnitsDisplay = uc.convert_units(
                model.propertyName,
                model_results.experimentalValueUnitsModel,
                model.unitsModel,
                model.unitsDisplay,
                chemical["sid"],
                average_mass,
            )

        model_results.predictionValueUnitsDisplay = uc.convert_units(
            model.propertyName,
            pred_value,
            model.unitsModel,
            model.unitsDisplay,
            chemical["sid"],
            average_mass,
        )
        model_results.unitsDisplay = model.unitsDisplay

        if generate_report and (neighbor_results_training is None or neighbor_results_prediction is None):
            neighbor_results_training, neighbor_results_prediction = self.addNeighborsFromSets(model, model_results, df_prediction)

        return self._build_report_dict(
            chemical,
            model_details_dict,
            model_results,
            neighbor_results_training=neighbor_results_training,
            neighbor_results_prediction=neighbor_results_prediction,
        )

    def _standardize_smiles_batch_subset(self, serverAPIs, smiles_list, model, split_depth=0):
        if len(smiles_list) == 1:
            return self._standardize_smiles_individually(serverAPIs, smiles_list, model)

        try:
            chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(
                server_host=serverAPIs,
                smiles=smiles_list,
                full=False,
                workflow=model.qsarReadyRuleSet,
            )
            if code == 200:
                normalized_results, normalization_diagnostic, partial_results_by_index, missing_indices = (
                    self._normalize_standardization_batch_results(
                        chemicals,
                        smiles_list,
                        model,
                    )
                )
                if normalized_results is not None:
                    if normalization_diagnostic:
                        logging.info(
                            "Standardization batch response normalized; workflow=%s batch_size=%s detail=%s",
                            model.qsarReadyRuleSet,
                            len(smiles_list),
                            normalization_diagnostic,
                        )
                    return normalized_results

                merged_results = [None] * len(smiles_list)
                partial_results_by_index = partial_results_by_index or {}
                for index, result in partial_results_by_index.items():
                    merged_results[index] = result

                if missing_indices:
                    missing_smiles = [smiles_list[index] for index in missing_indices]
                    logging.warning(
                        "Standardization batch returned partial matches, retrying only missing items; "
                        "workflow=%s batch_size=%s missing_count=%s split_depth=%s reason=%s",
                        model.qsarReadyRuleSet,
                        len(smiles_list),
                        len(missing_indices),
                        split_depth,
                        normalization_diagnostic,
                    )
                    retried_results = self._retry_missing_standardization_subset(
                        serverAPIs,
                        missing_smiles,
                        model,
                        split_depth + 1,
                    )
                    for index, result in zip(missing_indices, retried_results):
                        merged_results[index] = result

                    return [
                        result if result is not None else (f"{smiles_list[index]} failed standardization", 400)
                        for index, result in enumerate(merged_results)
                    ]

                standardization_reason = normalization_diagnostic
            else:
                standardization_reason = (
                    f"type={type(chemicals).__name__} "
                    f"preview={self._preview_log_value(chemicals)}"
                )

            logging.warning(
                "Standardization batch call failed or returned unexpected shape, falling back to single requests; "
                "code=%s workflow=%s batch_size=%s split_depth=%s reason=%s",
                code,
                model.qsarReadyRuleSet,
                len(smiles_list),
                split_depth,
                standardization_reason,
            )
            return self._standardize_smiles_individually(serverAPIs, smiles_list, model)
        except Exception as exc:
            logging.warning(
                "Batch standardization request raised %s, falling back to single requests; "
                "workflow=%s batch_size=%s split_depth=%s error=%s",
                type(exc).__name__,
                model.qsarReadyRuleSet,
                len(smiles_list),
                split_depth,
                self._preview_log_value(exc),
            )
            return self._standardize_smiles_individually(serverAPIs, smiles_list, model)

    def _standardize_smiles_batch(self, serverAPIs, smiles_list, model):
        return self._standardize_smiles_batch_subset(serverAPIs, smiles_list, model)

    def _calculate_descriptors_batch_subset(self, serverAPIs, qsar_smiles_subset, descriptor_service, split_depth=0):
        if not qsar_smiles_subset:
            return []

        if len(qsar_smiles_subset) == 1:
            return [self._descriptor_api.calculate_descriptors(serverAPIs, qsar_smiles_subset[0], descriptor_service)]

        df_batch, code = self._descriptor_api.calculate_descriptors_batch(serverAPIs, qsar_smiles_subset, descriptor_service)
        if code == 200 and isinstance(df_batch, pd.DataFrame) and len(df_batch.index) == len(qsar_smiles_subset):
            return [(df_batch.iloc[[row_pos]].copy(), 200) for row_pos in range(len(qsar_smiles_subset))]

        if code == 400 and len(qsar_smiles_subset) > 1:
            mid = len(qsar_smiles_subset) // 2
            left_subset = qsar_smiles_subset[:mid]
            right_subset = qsar_smiles_subset[mid:]
            logging.warning(
                "Descriptor batch returned HTTP 400; splitting batch and retrying; "
                "descriptor_service=%s batch_size=%s split_depth=%s left_size=%s right_size=%s",
                descriptor_service,
                len(qsar_smiles_subset),
                split_depth,
                len(left_subset),
                len(right_subset),
            )
            return (
                self._calculate_descriptors_batch_subset(serverAPIs, left_subset, descriptor_service, split_depth + 1)
                + self._calculate_descriptors_batch_subset(serverAPIs, right_subset, descriptor_service, split_depth + 1)
            )

        if isinstance(df_batch, pd.DataFrame):
            batch_reason = (
                f"rows={len(df_batch.index)} expected_rows={len(qsar_smiles_subset)} "
                f"columns={list(df_batch.columns[:10])}"
            )
        else:
            batch_reason = (
                f"type={type(df_batch).__name__} "
                f"preview={self._preview_log_value(df_batch)}"
            )

        logging.warning(
            "Descriptor batch call failed or returned unexpected shape for subset, "
            "falling back to per-smiles calls; code=%s descriptor_service=%s batch_size=%s split_depth=%s reason=%s",
            code,
            descriptor_service,
            len(qsar_smiles_subset),
            split_depth,
            batch_reason,
        )

        max_workers = int(os.getenv("PREDICT_DESCRIPTOR_FALLBACK_WORKERS", min(32, (os.cpu_count() or 1) * 5)))
        max_workers = max(1, min(max_workers, len(qsar_smiles_subset)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(
                lambda smiles: self._descriptor_api.calculate_descriptors(serverAPIs, smiles, descriptor_service),
                qsar_smiles_subset,
            ))

    def _calculate_descriptors_batch(self, serverAPIs, qsar_smiles_list, descriptor_service):
        results = [None] * len(qsar_smiles_list)
        valid_indices = []
        valid_smiles = []

        if "test" in descriptor_service.lower():
            for idx, qsar_smiles in enumerate(qsar_smiles_list):
                check_results, code = self._descriptor_api.check_structure(qsar_smiles)
                if code != 200:
                    results[idx] = (check_results, code)
                else:
                    valid_indices.append(idx)
                    valid_smiles.append(qsar_smiles)
        else:
            valid_indices = list(range(len(qsar_smiles_list)))
            valid_smiles = list(qsar_smiles_list)

        if valid_smiles:
            descriptor_results = self._calculate_descriptors_batch_subset(
                serverAPIs,
                valid_smiles,
                descriptor_service,
            )
            for row_pos, idx in enumerate(valid_indices):
                results[idx] = descriptor_results[row_pos]

        return results

    def _predict_model_smiles_batch_from_standardized_results(
        self,
        model,
        serverAPIs,
        smiles_list,
        standardized_results,
        report_model_details=None,
        generate_report=True,
    ):
        results = [None] * len(smiles_list)
        standardized_indices = []
        standardized_smiles = []
        standardized_chemicals = []

        for idx, (chemical, code) in enumerate(standardized_results):
            if code != 200:
                error_chemical = self._build_minimal_chemical(smiles_list[idx])
                results[idx] = self._build_prediction_error_report(error_chemical, report_model_details, chemical)
                continue

            standardized_indices.append(idx)
            standardized_smiles.append(chemical["canonicalSmiles"])
            standardized_chemicals.append(dict(chemical))

        descriptor_results = self._calculate_descriptors_batch(serverAPIs, standardized_smiles, model.descriptorService)

        prediction_frames = []
        prediction_indices = []
        prediction_chemicals = []

        for offset, descriptor_result in enumerate(descriptor_results):
            original_idx = standardized_indices[offset]
            chemical = standardized_chemicals[offset]

            if descriptor_result is None:
                results[original_idx] = self._build_prediction_error_report(
                    self._prepare_chemical_for_report(chemical),
                    report_model_details,
                    "Descriptor calculation did not return a result",
                )
                continue

            df_prediction, code = descriptor_result
            if code != 200:
                results[original_idx] = self._build_prediction_error_report(
                    self._prepare_chemical_for_report(chemical),
                    report_model_details,
                    df_prediction,
                )
                continue

            prediction_frames.append(df_prediction)
            prediction_indices.append(original_idx)
            prediction_chemicals.append(chemical)

        if prediction_frames:
            batch_prediction_df = pd.concat(prediction_frames, ignore_index=True)
            predictions = model.do_predictions(batch_prediction_df)

            if predictions is None:
                for idx, chemical in zip(prediction_indices, prediction_chemicals):
                    results[idx] = self._build_prediction_error_report(
                        chemical,
                        report_model_details,
                        "Model could not generate predictions",
                    )
            else:
                pred_array = np.asarray(predictions).reshape(-1)
                if pred_array.size != len(prediction_indices):
                    for idx, chemical in zip(prediction_indices, prediction_chemicals):
                        results[idx] = self._build_prediction_error_report(
                            chemical,
                            report_model_details,
                            "Model returned an unexpected number of predictions",
                        )
                else:
                    ad_results_by_row = self._determine_applicability_domains_batch(model, batch_prediction_df)
                    neighbor_results_training = None
                    neighbor_results_prediction = None
                    if generate_report:
                        neighbor_results_training, neighbor_results_prediction = self.addNeighborsFromSets_batch(
                            model,
                            batch_prediction_df,
                            model.unitsModel,
                        )
                    prepared_chemicals = [
                        self._prepare_chemical_for_report(chemical)
                        for chemical in prediction_chemicals
                    ]

                    for row_pos, (idx, chemical) in enumerate(zip(prediction_indices, prediction_chemicals)):
                        results[idx] = self._finalize_prediction_report(
                            model,
                            report_model_details,
                            chemical,
                            None,
                            pred_array[row_pos],
                            generate_report=generate_report,
                            ad_results=ad_results_by_row[row_pos],
                            neighbor_results_training=None if neighbor_results_training is None else neighbor_results_training[row_pos],
                            neighbor_results_prediction=None if neighbor_results_prediction is None else neighbor_results_prediction[row_pos],
                            prepared_chemical=prepared_chemicals[row_pos],
                        )

        for idx, prediction in enumerate(results):
            if prediction is None:
                results[idx] = {"smiles": smiles_list[idx], "error": "Prediction failed unexpectedly"}

        return results

    @timer
    def predict_model_smiles_batch(self, model_id, smiles_list, generate_report=True, include_model_details=True):
        serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
        fileAPI = os.getenv("FILE_API_SERVER", pc.URL_CTX_API)

        model, model_details_dict, model_error = self._resolve_model_context(model_id, serverAPIs, fileAPI)
        if model is None:
            return [{"smiles": smiles, "error": model_error} for smiles in smiles_list]

        report_model_details = model_details_dict if include_model_details else None
        standardized_results = self._standardize_smiles_batch(serverAPIs, smiles_list, model)
        return self._predict_model_smiles_batch_from_standardized_results(
            model,
            serverAPIs,
            smiles_list,
            standardized_results,
            report_model_details=report_model_details,
            generate_report=generate_report,
        )

    def _predict_from_db_batch(self, model_id, smiles_list, include_model_details=True):
        if not smiles_list:
            return []

        cache_keys = [self._build_prediction_cache_key(model_id, smiles=smiles) for smiles in smiles_list]
        cached_predictions = get_cached_predictions([key for key in cache_keys if key])

        results = [None] * len(smiles_list)
        missing_indices = []
        missing_smiles = []
        missing_cache_keys = []

        for idx, cache_key in enumerate(cache_keys):
            prediction = cached_predictions.get(cache_key) if cache_key is not None else None
            if prediction is not None:
                results[idx] = self._prediction_from_cached_value(prediction)
                continue

            missing_indices.append(idx)
            missing_smiles.append(smiles_list[idx])
            missing_cache_keys.append(cache_key)

        model_details_dict = None
        if missing_indices or include_model_details:
            serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
            fileAPI = os.getenv("FILE_API_SERVER", pc.URL_CTX_API)

            model, model_details_dict, model_error = self._resolve_model_context(model_id, serverAPIs, fileAPI)
            if model is None:
                for idx, smiles in zip(missing_indices, missing_smiles):
                    results[idx] = self._build_prediction_error_report(smiles, None, model_error)

                if include_model_details and model_details_dict:
                    results = [
                        self._attach_model_details_to_prediction(prediction, model_details_dict)
                        if prediction is not None else prediction
                        for prediction in results
                    ]
                return results

        if missing_indices:
            standardized_results = self._standardize_smiles_batch(serverAPIs, missing_smiles, model)
            generated_results = self._predict_model_smiles_batch_from_standardized_results(
                model,
                serverAPIs,
                missing_smiles,
                standardized_results,
                report_model_details=None,
                generate_report=True,
            )
            cache_entries = []

            for idx, cache_key, prediction in zip(missing_indices, missing_cache_keys, generated_results):
                prediction_obj = self._prediction_from_cached_value(prediction)
                results[idx] = prediction_obj
                if cache_key is not None:
                    cache_entries.append((cache_key, prediction_obj))

            cache_predictions(cache_entries)

        if include_model_details and model_details_dict:
            results = [
                self._attach_model_details_to_prediction(prediction, model_details_dict)
                if prediction is not None else prediction
                for prediction in results
            ]

        return results

    @timer
    def predictFromDB(self, model_id, smiles, include_model_details=True):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        if isinstance(smiles, str):
            predictions = self._predict_from_db_batch(
                model_id,
                [smiles],
                include_model_details=include_model_details,
            )
            return predictions[0] if predictions else {"smiles": smiles, "error": "Prediction failed unexpectedly"}

        smiles_list = list(smiles)
        return self._predict_from_db_batch(
            model_id,
            smiles_list,
            include_model_details=include_model_details,
        )

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
        return _render_smiles_to_base64_cached(smiles_string, width=width, height=height)
    
    def setExpValue(self, chemical, model, modelResults:ModelResults):

        qsarSmiles = chemical["canonicalSmiles"]

        training_lookup = getattr(model, "_training_property_by_id", None)
        if training_lookup and qsarSmiles in training_lookup:
            modelResults.experimentalValueUnitsModel = training_lookup[qsarSmiles]
            modelResults.experimentalValueSet = "Training"
        else:
            matching_row_training = model.df_training[model.df_training['ID'] == qsarSmiles]
            if not matching_row_training.empty:
                modelResults.experimentalValueUnitsModel = matching_row_training['Property'].values[0]
                modelResults.experimentalValueSet = "Training"

        prediction_lookup = getattr(model, "_prediction_property_by_id", None)
        if prediction_lookup and qsarSmiles in prediction_lookup:
            modelResults.experimentalValueUnitsModel = prediction_lookup[qsarSmiles]
            modelResults.experimentalValueSet = "Test"
        else:
            matching_row_test = model.df_prediction[model.df_prediction['ID'] == qsarSmiles]
            if not matching_row_test.empty:
                modelResults.experimentalValueUnitsModel = matching_row_test['Property'].values[0]
                modelResults.experimentalValueSet = "Test"

    def setExpPredValuesForADAnalogs(self, model, analogs):
        self._ensure_model_runtime_cache(model)
        return self.setExpPredValuesForNeighbors(
            model,
            getattr(model, "_preds_training_cv_by_id", None),
            analogs,
            getattr(model, "_dsstox_by_smiles", None),
        )

    def addDistances(self, analogs, distances):
        # print(len(analogs), len(distances))
        for index, analog in enumerate(analogs):  # these are just the qsarSmiles
            analog["distance"] = distances[index] 

    def fixMissingNeighborDsstoxRecord(self, datasetName, qsarSmiles):
        cache_key = (datasetName, qsarSmiles)

        with _MISSING_NEIGHBOR_DSS_TOX_LOCK:
            cached_record = _MISSING_NEIGHBOR_DSS_TOX_CACHE.get(cache_key)
        if cached_record is not None:
            return dict(cached_record)

        row_as_dict = {"canonicalSmiles": qsarSmiles}
        mi = ModelInitializer()

        try:
            session = getSession()

            dtxcid = mi.getQsarDtxcid(qsarSmiles, datasetName, session)

            if dtxcid:
                row_as_dict["cid"] = dtxcid

                if dtxcid in dict_missing_dsstox_records:
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
        except Exception as e:
            print(e)

        with _MISSING_NEIGHBOR_DSS_TOX_LOCK:
            _MISSING_NEIGHBOR_DSS_TOX_CACHE[cache_key] = dict(row_as_dict)

        return dict(row_as_dict)

    def setExpPredValuesForNeighbors(self, model:Model, pred_lookup, neighbors, dsstox_lookup):

        neighbors2 = []

        for neighbor in neighbors:  # these are just the qsarSmiles

            dsstox_row = dsstox_lookup.get(neighbor) if dsstox_lookup else None
            if dsstox_row is not None:
                row_as_dict = dict(dsstox_row)
            else:
                logging.debug("Finding missing dsstox info for " + neighbor)  # only happens for 8 dtxcids
                row_as_dict = self.fixMissingNeighborDsstoxRecord(model.datasetName, neighbor)
                logging.debug(row_as_dict)

            prediction_row = pred_lookup.get(neighbor) if pred_lookup else None
            if prediction_row is not None:
                row_as_dict['exp'] = prediction_row.get('exp')
                row_as_dict['pred'] = prediction_row.get('pred')

            neighbors2.append(row_as_dict)

        return neighbors2

    def _build_neighbor_report(self, model: Model, neighbors, distances, pred_lookup, dsstox_lookup, units_model, set_name, title):
        enriched_neighbors = self.setExpPredValuesForNeighbors(model, pred_lookup, neighbors, dsstox_lookup)
        self.addDistances(enriched_neighbors, distances)

        mae_value = None
        try:
            df_neighbors = pd.DataFrame(enriched_neighbors)
            if set_name == "Training":
                stats_result = stats.calculate_continuous_statistics(df_neighbors, 0, pc.TAG_TRAINING)
                mae_value = stats_result.get(pc.MAE + pc.TAG_TRAINING)
            else:
                stats_result = stats.calculate_continuous_statistics(df_neighbors, 0, pc.TAG_TEST)
                mae_value = stats_result.get(pc.MAE + pc.TAG_TEST)
        except Exception:
            mae_value = None

        return {
            "set": set_name,
            "neighbors": enriched_neighbors,
            "MAE": mae_value,
            "unitNeighbor": units_model,
            "title": title,
        }

    def addNeighborsFromSets_batch(self, model, df_test_chemicals, units_model):
        self._ensure_model_runtime_cache(model)

        ng = NeighborGetter()
        neighbors_test_all, distances_test_all = ng.find_neighbors_in_set(
            model=model,
            df_set=model.df_prediction,
            df_test_chemicals=df_test_chemicals,
            precomputed=getattr(model, "_neighbors_prediction_cache", None),
            return_all=True,
        )
        neighbors_training_all, distances_training_all = ng.find_neighbors_in_set(
            model=model,
            df_set=model.df_training,
            df_test_chemicals=df_test_chemicals,
            precomputed=getattr(model, "_neighbors_training_cache", None),
            return_all=True,
        )

        dsstox_lookup = getattr(model, "_dsstox_by_smiles", None)
        pred_training_lookup = getattr(model, "_preds_training_cv_by_id", None)
        pred_test_lookup = getattr(model, "_preds_test_by_id", None)

        neighbor_results_training = []
        neighbor_results_prediction = []

        for row_pos in range(len(df_test_chemicals.index)):
            neighbor_results_training.append(
                self._build_neighbor_report(
                    model,
                    neighbors_training_all[row_pos],
                    distances_training_all[row_pos],
                    pred_training_lookup,
                    dsstox_lookup,
                    units_model,
                    "Training",
                    "Nearest Neighbors from Training Set (Cross Validation Predictions)",
                )
            )
            neighbor_results_prediction.append(
                self._build_neighbor_report(
                    model,
                    neighbors_test_all[row_pos],
                    distances_test_all[row_pos],
                    pred_test_lookup,
                    dsstox_lookup,
                    units_model,
                    "Test",
                    "Nearest Neighbors from Test Set (External Predictions)",
                )
            )

        return neighbor_results_training, neighbor_results_prediction

    @timer
    def addNeighborsFromSets(self, model:Model, modelResults: ModelResults, df_test_chemicals):
        neighbor_results_training, neighbor_results_prediction = self.addNeighborsFromSets_batch(
            model,
            df_test_chemicals,
            modelResults.unitsModel,
        )
        return neighbor_results_training[0], neighbor_results_prediction[0]
        
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
    def predict_model_smiles(self, model_id, smiles, generate_report=True, include_model_details=True):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """
        batch_result = self.predict_model_smiles_batch(
            model_id,
            [smiles],
            generate_report=generate_report,
            include_model_details=include_model_details,
        )
        if not batch_result:
            return f"Prediction failed for {smiles}", 500

        prediction = batch_result[0]
        if isinstance(prediction, dict) and "error" in prediction and prediction.get("modelResults") is None:
            error = prediction.get("error")
            if isinstance(error, str):
                return error, 400
            return str(error), 400

        return prediction, 200
    
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
        try:
            chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                               workflow=model.qsarReadyRuleSet)
        except Exception as exc:
            logging.exception("Standardization request failed for %s", smiles)
            return f"{smiles}: standardization request failed: {exc}", 500
        logging.debug(chemicals)
        
        if code >= 400:
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
        try:
            chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                               workflow=qsarReadyRuleSet)
        except Exception as exc:
            logging.exception("Standardization request failed for %s", smiles)
            return f"{smiles}: standardization request failed: {exc}", 500
        logging.debug(chemicals)

        if code >= 500:
            return smiles + ": could not generate QSAR Ready SMILES", code 
        if code == 400:
            return chemicals, code
                
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
        batch_results = self._determine_applicability_domain_batch(model, applicabilityDomainName, df_prediction)
        if batch_results:
            return batch_results[0]
        return {"AD": False, "adMethod": {"name": applicabilityDomainName}}
            

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
