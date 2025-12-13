import json
import os
from io import BytesIO

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from API_Utilities import QsarSmilesAPI, DescriptorsAPI
from model_ws_utilities import call_do_predictions_from_df, models
from models import df_utilities as dfu
from models.ModelBuilder import Model
from utils import timer

debug = False

import logging

logging.getLogger('sqlalchemy').setLevel(logging.ERROR)


@timer
def standardizeStructure(serverAPIs, smiles, model: Model):
    useFullStandardize = False
    qsAPI = QsarSmilesAPI()
    chemicals = qsAPI.call_qsar_ready_standardize_post(server_host=serverAPIs, smiles=smiles, full=useFullStandardize,
                                                       workflow=model.qsarReadyRuleSet)
    if "error" in chemicals:
        return chemicals, 400

    if len(chemicals) == 0:
        # logging.debug('Standardization failed')
        return "smiles=" + smiles + " failed standardization", 400

    logging.debug(chemicals)

    if len(chemicals) > 1 and model.omitSalts:
        # print('qsar smiles indicates mixture')
        return "model can't run mixtures", 400

    qsarSmiles = chemicals[0]["canonicalSmiles"]

    logging.debug(f"qsarSmiles: {qsarSmiles}")

    return qsarSmiles, 200


def init_model(model_id):
    if model_id in models:
        logging.debug('have model already initialized')
        model = models[model_id]
    else:
        model = initModel(model_id)
        models[model_id] = model

    return model


@timer
def predictFromDB(model_id, smiles):
    """
    Runs whole workflow: standardize, descriptors, prediction, applicability domain
    :param model_id:
    :param smiles:
    :param mwu:
    :return:
    """

    if isinstance(smiles, str):
        return predict_model_smiles(model_id, smiles)
    else:
        return [predict_model_smiles(model_id, smi) for smi in smiles]


@timer
def predict_model_smiles(model_id, smiles):
    # serverAPIs = "https://hcd.rtpnc.epa.gov" # TODO this should come from environment variable
    serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")

    # initialize model bytes and all details from db:
    model = init_model(model_id)

    # Standardize smiles:
    qsarSmiles, code = standardizeStructure(serverAPIs, smiles, model)
    if code != 200:
        return "error: " + qsarSmiles, code

    # Descriptor calcs:
    descriptorAPI = DescriptorsAPI()
    df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, model.descriptorService)
    if code != 200:
        return df_prediction, 400

    # Run model prediction:
    # df_prediction = model.model_details.predictionSet #all chemicals in the model's prediction set, for testing
    # print("for qsarSmiles="+qsarSmiles+", descriptors="+json.dumps(descriptorsResults,indent=4))
    pred_results = json.loads(call_do_predictions_from_df(df_prediction, model))
    pred_value = pred_results[0]['pred']

    # applicability domain calcs:
    ad_results = None
    if model.applicabilityDomainName:
        str_ad_results = determineApplicabilityDomain(model, df_prediction)
        # str_ad_results = determineApplicabilityDomain(model, model.df_prediction) #testing AD method using multiple chemicals in df

        ad_results = json.loads(str_ad_results)[0]  # TODO check len first?
        # print(ad_results)
    else:
        logging.debug('AD method for model was not set:', model_id)

    # store everything in results:
    model_results = ModelResults(model, ad_results)
    model_results.smiles = smiles
    model_results.qsarSmiles = qsarSmiles
    model_results.predictionValue = pred_value
    model_results.predictionUnits = model.unitsName  # duplicated so displayed near prediction value
    model_results.adResults = ad_results

    return model_results.to_dict()


def predictSetFromDB(model_id, excel_file_path):
    """
    Runs whole workflow: standardize, descriptors, prediction, applicability domain
    :param model_id:
    :param smiles:
    :param mwu:
    :return:
    """

    import model_ws_utilities as mwu
    descriptorAPI = DescriptorsAPI()

    # serverAPIs = "https://hcd.rtpnc.epa.gov" # TODO this should come from environment variable
    serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")

    # initialize model bytes and all details from db:
    model = init_model(model_id, mwu)

    import pandas as pd
    df = pd.read_excel(excel_file_path, sheet_name='Test set')
    smiles_list = df['Smiles'].tolist()  # Extract the 'Smiles' column into a list

    directory = os.path.dirname(excel_file_path)

    # Create a text file path in the same directory
    text_file_path = os.path.join(directory, "output.txt")
    logging.debug(text_file_path)

    with open(text_file_path, 'w') as file:
        file.write("smiles\tqsarSmiles\tpred_value\tpred_AD\n")

        # for smiles, predOld in zip(smiles_list, pred_list):
        for smiles in smiles_list:
            qsarSmiles, code = standardizeStructure(serverAPIs, smiles, model)
            if code != 200:
                logging.warn(smiles, qsarSmiles)
                file.write(smiles + "\terror smiles")
                continue

            df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, model.descriptorService)
            if code != 200:
                logging.warn(smiles, 'error descriptors')
                file.write(smiles + "\terror descriptors")

                continue

            pred_results = json.loads(mwu.call_do_predictions_from_df(df_prediction, model))
            pred_value = pred_results[0]['pred']

            str_ad_results = determineApplicabilityDomain(model, df_prediction)
            ad_results = json.loads(str_ad_results)
            pred_AD = ad_results[0]["AD"]

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


@timer
def determineApplicabilityDomain(model: Model, test_tsv):
    """
    Calculate the applicability domain using the model's training set and the AD measure assigned to the model in the DB
    TODO make sure this works when a model doesnt have a set embedding object
    :param model:
    :param test_tsv:
    :return:
    """
    json_model_description = model.get_model_description()
    model_description = json.loads(json_model_description)
    remove_log_p = model_description["remove_log_p_descriptors"]  # just set to False instead?
    # print("remove_log_p", remove_log_p)

    from applicability_domain import applicability_domain_utilities as adu
    # model.applicabilityDomainName = adu.strOPERA_local_index  # for testing diff number of neighbors

    output = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
        train_df=model.df_training,
        test_df=test_tsv,
        remove_log_p=remove_log_p,
        embedding=model.embedding,
        applicability_domain=model.applicabilityDomainName,
        filterColumnsInBothSets=True)

    # return output.to_json(orient='records', lines=True) # gives each object on separate line
    return output.to_json(orient='records', lines=False)  # gives an array instead of each object on separate line


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


@timer
def initModel(model_id):
    session = getSession()
    model_bytes = get_model_bytes(model_id, session)

    import pickle
    model = pickle.loads(model_bytes)

    if debug:
        print('model_description from pickled model:', model.get_model_description())

    if not hasattr(model, "is_binary"):
        print('model.is_binary is none, setting to false')
        model.is_binary = False

    # Stores model under provided number

    get_model_details(model_id, model, session)

    logging.debug(model.get_model_description_pretty())

    # this wont be necessary if the training/test sets are in the pickled model:
    get_training_prediction_instances(session, model)

    # TODO: for the training/prediction instances, could also query the descriptor api but it would take longer and
    #  sometimes the descriptors will come out different due to the fact that the descriptors will be pulled from the
    #  cache by inchi key (TEST descriptors come out differently sometimes for two different structures with the same inchi key but different smiles

    return model


def get_model_bytes(model_id, session):
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

    # finally:
    # Close the session - closed later
    # print("done")


class ModelDetails:
    def __init__(self, model: Model):
        self.modelId = model.modelId
        self.modelName = model.modelName
        self.propertyName = model.propertyName

        self.regressor_name = model.regressor_name
        # self.version = '0.0.1'
        self.is_binary = model.is_binary
        # self.description = model.description #TODO in database
        # self.description_url = model.description_url #TODO in database
        self.datasetName = model.datasetName
        self.unitsName = model.unitsName
        self.descriptorService = model.descriptorService
        self.applicabilityDomainName = model.applicabilityDomainName
        self.qsarReadyRuleSet = model.qsarReadyRuleSet
        self.embedding = model.embedding


class ModelResults:
    def __init__(self, model: Model, adResults):
        self.smiles = None
        self.qsarSmiles = None
        self.predictionValue = None
        self.predictionUnits = None
        self.adResults = adResults
        self.modelDetails = ModelDetails(model)

    def to_dict(self):
        # Convert the object to a dictionary, including nested objects
        return {
            "smiles": self.smiles,
            "qsarSmiles": self.qsarSmiles,
            "predictionValue": self.predictionValue,
            "predictionUnits": self.predictionUnits,
            "modelDetails": self.modelDetails.__dict__ if self.modelDetails else None,
            "adResults": self.adResults
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


def get_model_details(modelId, model: Model, session):
    """
    Gets model meta data (except training and test set tsvs).
    TODO Should this info be stored directly in model object and then for new models we won't need to query the db since will be already in the pickled object?
    """
    try:
        # SQL query to retrieve model details
        sql = text(getModelMetaDataQuery() + "\nWHERE m.id = :model_id")

        # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
        # print(sql)

        # Execute the query
        result = session.execute(sql, {'model_id': modelId}).fetchone()

        # Process the result
        if result:
            logging.debug(sql)
            row_to_model_details(model, result)

    except Exception as ex:
        print(f"Exception occurred: {ex}")
    # finally:
    # Close the session - close it later after get training/test sets
    # print('done with details')
    # session.close()

    return None


def getModelMetaDataQuery():
    return """
           SELECT m.id,
                  m.name_ccd,
                  m.details,
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
                  adm.name
           FROM qsar_models.models m
                    LEFT JOIN qsar_datasets.datasets d ON d.name = m.dataset_name
                    LEFT JOIN qsar_datasets.units u ON d.fk_unit_id = u.id
                    LEFT JOIN qsar_datasets.properties p ON d.fk_property_id = p.id
                    LEFT JOIN qsar_descriptors.descriptor_sets ds ON m.descriptor_set_name = ds.name
                    LEFT JOIN qsar_datasets.splittings s ON m.splitting_name = s.name
                    LEFT JOIN qsar_models.ad_methods adm ON m.fk_ad_method = adm.id \
           """


def get_available_models():
    """
    Gets  list of available models with meta data
    """
    try:
        session = getSession()

        # SQL query to retrieve model details
        sql = text(getModelMetaDataQuery() + "\nWHERE m.fk_source_id = 3 and m.is_public=true;")

        # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
        # print(sql)

        # Execute the query
        results = session.execute(sql).fetchall()

        models = []
        # Process the result
        for row in results:
            model = Model()
            row_to_model_details(model, row)
            models.append(json.loads(model.get_model_description()))

        return models


    except Exception as ex:
        print(f"Exception occurred: {ex}")
    finally:
        # Close the session - close it later after get training/test sets
        session.close()

    return None

    def get_model_list(self, session):
        """
        Gets model meta data (except training and test set tsvs).
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
            result = session.execute(sql).fetch()

            models = []

            for row in result:
                model_details = self.row_to_model_details(result)

                models.append(model_details)

            return models

        except Exception as ex:
            print(f"Exception occurred: {ex}")
        finally:
            # Close the session
            print('done with details')
            # session.close()

        return None


def row_to_model_details(model: Model, result):
    model.modelId = result[0]
    model.modelName = result[1]
    details = json.loads(result[2].tobytes().decode('utf-8'))  # it's stored as a file object in database for now
    model.datasetId = result[3]
    model.datasetName = result[4]
    model.unitsName = result[5]
    model.dsstox_mapping_strategy = result[6]
    model.propertyName = result[7]
    model.descriptorSetId = result[8]
    model.descriptorSetName = result[9]
    model.descriptorService = result[10]
    model.headersTsv = result[11]
    model.splittingId = result[12]
    model.splittingName = result[13]
    model.applicabilityDomainName = result[14]
    # model_details.descriptorEmbeddingTsv = result[13]  #dont need already have in pickled model

    # print (type(details))
    # print(details)

    model.is_binary = details['is_binary']
    model.remove_log_p_descriptors = details['remove_log_p_descriptors']
    model.embedding = details['embedding']
    model.description = details['description']
    model.description_url = details['description_url']
    model.qsar_method = details['qsar_method']
    model.hyperparameters = details['hyperparameters']
    model.hyperparameter_grid = details['hyperparameter_grid']
    model.qsar_method = details['qsar_method']
    model.use_pmml = details['use_pmml']
    model.version = details['version']
    model.include_standardization_in_pmml = details['include_standardization_in_pmml']

    # omit for now:
    #  'training_stats'
    #  'training_descriptor_std_devs'
    #  'training_descriptor_means'

    # Parse JSON for dsstox_mapping_strategy
    dsstox_mapping = json.loads(model.dsstox_mapping_strategy)
    if 'omitSalts' in dsstox_mapping:
        model.omitSalts = dsstox_mapping.get('omitSalts', False)
    if 'qsarReadyRuleSet' in dsstox_mapping:
        model.qsarReadyRuleSet = dsstox_mapping.get('qsarReadyRuleSet', "qsar-ready")
    else:
        model.qsarReadyRuleSet = "qsar-ready"


def generate_instance(id, qsar_property_value, descriptors):
    # Implement this function based on your requirements
    return f"{id}\t{qsar_property_value}\t{descriptors}\n"


def get_training_prediction_instances(session, model):
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
            id, qsar_property_value, descriptors, split_num = row
            instance = generate_instance(id, qsar_property_value, descriptors)

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

        if debug:
            print('trainingSet shape', model.df_training.shape)
            print('predictionSet shape', model.df_prediction.shape)

    except SQLAlchemyError as ex:
        print(f"An error occurred: {ex}")
    finally:
        # print('done getting tsvs')
        session.close()


if __name__ == '__main__':
    excel_file_path = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\hibernate_qsar_model_building\data\reports\prediction reports upload\WebTEST2.1\HLC v1 modeling_RND_REPRESENTATIVE.xlsx"
    predictSetFromDB(1065, excel_file_path)
