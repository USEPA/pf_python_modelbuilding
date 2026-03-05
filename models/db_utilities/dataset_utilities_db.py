'''
Created on Jan 22, 2026

@author: TMARTI02
'''

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import logging
from io import StringIO
import pandas as pd
import numpy as np


from utils import print_first_row
from model_ws_db_utilities import getSession
import webbrowser      
import os          





    
def _load_df(tsv_string):
    """Loads data from TSV/CSV into a pandas dataframe"""
    if "\t" in tsv_string:
        separator = '\t'
    else:
        separator = ','
        
    # print('separator*',separator,'*')
        
    df = pd.read_csv(StringIO(tsv_string), sep=separator, na_values="null")
    return df


def get_training_cv_instances(session, dataset_name, descriptor_set_name):
    folds = {}
    for i in range(1, 6):
        fold_splitting_name = f'RND_REPRESENTATIVE_CV{i}'
            
        df_training, df_prediction = get_training_prediction_instances(
            session, dataset_name, descriptor_set_name, fold_splitting_name
        )
        folds[i] = {"train": df_training, "pred": df_prediction}
    return folds

def getSqlMappedDataPoints():
        """
        Useful for creating a lookup between canon_qsar_smiles and the dsstox info
        """
        
        return text("""
        WITH filtered_dp AS (
          SELECT
            dp.canon_qsar_smiles,
            dp.qsar_exp_prop_property_values_id,
            dp.qsar_dtxcid,
            TRIM(SPLIT_PART(dp.qsar_exp_prop_property_values_id, '|', 1)) AS qsar_exp_prop_property_values_id_first,
            TRIM(SPLIT_PART(dp.qsar_dtxcid, '|', 1)) AS dtxcid
          FROM qsar_datasets.data_points AS dp
          JOIN qsar_datasets.datasets AS d
            ON dp.fk_dataset_id = d.id
          WHERE d.name = :dataset_name
        )
        SELECT
          fdp.canon_qsar_smiles,
          --fdp.qsar_exp_prop_property_values_id,
          fdp.qsar_exp_prop_property_values_id_first,
          --fdp.qsar_dtxcid,
          fdp.dtxcid,
          r.dtxsid,
          r.casrn,
          r.preferred_name,
          r.smiles,
          r.mol_weight
        FROM filtered_dp AS fdp
        LEFT JOIN qsar_models.dsstox_records AS r
          ON r.dtxcid = fdp.dtxcid
         AND r.fk_dsstox_snapshot_id = 4;
        """)
    
def getMappedDatapoints(session, dataset_name):
    logging.info(f"Getting mapped datapoints for {dataset_name}")
    df_pv = pd.read_sql(getSqlMappedDataPoints(), con=session.get_bind(), params={"dataset_name": dataset_name})
    df_pv = df_pv.replace('', np.nan).dropna(axis=1, how='all')  # drop the columns with no data
    logging.info(f"Done")
    return df_pv





    
    
def get_dataset_details(session, dataset_name):
    """
    Gets m meta data (except training and test set tsvs).
    TODO Should this info be stored directly in m object and then for new models we won't need to query the db since will be already in the pickled object?
    """
    try:
        
        query = """
        SELECT 
            d.id,
            d.name as dataset_name,
            d.description as dataset_description, 
            u.abbreviation_ccd AS units_model,
            u2.abbreviation_ccd AS units_display,
            d.dsstox_mapping_strategy,
            p.name_ccd as property_name,
            p.description as property_description
        FROM qsar_datasets.datasets AS d
        LEFT JOIN qsar_datasets.units AS u ON d.fk_unit_id = u.id
        LEFT JOIN qsar_datasets.units AS u2 ON d.fk_unit_id_contributor = u2.id
        LEFT JOIN qsar_datasets.properties AS p ON d.fk_property_id = p.id
        """
                    # SQL query to retrieve m details
        sql = text(query + "\nWHERE d.name = :dataset_name")

        # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for m)
        # print(sql)

        # Execute the query
        row = session.execute(sql, {'dataset_name': dataset_name}).fetchone()
        row_dict = dict(row._mapping) if row is not None else None

        # Process the result
        return row_dict

    except Exception as ex:
        ex.with_traceback()
        print(f"Exception occurred: {ex}")



def get_training_ids(session, datasetName, descriptorSetName, splittingName):


    sql = text("""
        SELECT dp.id as fk_data_point_id, dp.qsar_property_value
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp on dp.fk_dataset_id = d.id
        JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
        JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
        JOIN qsar_descriptors.descriptor_sets ds on ds.id = dv.fk_descriptor_set_id
        join qsar_datasets.splittings s on s.id = dpis.fk_splitting_id
        WHERE d.name = :datasetName
        AND ds.name = :descriptorSetName
        AND s.name = :splittingName
        AND dpis.split_num = 0;
        """)

    try:
        
        res = session.execute(sql, {'datasetName': datasetName, 'descriptorSetName': descriptorSetName,
                                        'splittingName': splittingName})
        
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
        return df
    
    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None     


# @timer    
def get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName):


    sql = text("""
        select headers_tsv from qsar_descriptors.descriptor_sets ds
        where ds.name=:descriptorSetName
        """)
    
    try:
        results = session.execute(sql, {'descriptorSetName': descriptorSetName}).fetchone()
        instance_header = "ID\tProperty\t" + results[0] + "\r\n"
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
            instance = _generate_instance(chemical_id, qsar_property_value, descriptors)
            
            # print(len(instance.split("\t")))
                        
            if instance is None:
                logging.debug(f"{id}\tnull instance\tdatasetName={datasetName}\tdescriptorSetName={descriptorSetName}")
                continue
        
            if split_num == 0:
                sb_training.append(instance)
        
            elif split_num == 1:
                sb_prediction.append(instance)
                
        df_training = _load_df(''.join(sb_training))
        df_prediction = _load_df(''.join(sb_prediction))
        
        # print(df_training.shape)
        # first_row_dict = df_training.loc[0].to_dict()
        # print(json.dumps(first_row_dict, indent=4))
                
        logging.debug('trainingSet shape' + str(df_training.shape))
        logging.debug('predictionSet shape' + str(df_prediction.shape))
    
        return df_training, df_prediction
    
    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None, None     



from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging

def get_instances_excluding(
    session,
    datasetName,          # the dataset to load
    datasetName2,         # exclude any canon_qsar_smiles present in this dataset
    descriptorSetName
):
    """
    Return a single DataFrame of instances for datasetName/descriptorSetName,
    excluding any canon_qsar_smiles that appear in datasetName2.
    """
    sql_headers = text("""
        SELECT headers_tsv
        FROM qsar_descriptors.descriptor_sets ds
        WHERE ds.name = :descriptorSetName
    """)

    try:
        # Build header
        res_header = session.execute(sql_headers, {'descriptorSetName': descriptorSetName}).fetchone()
        instance_header = "ID\tProperty\t" + res_header[0] + "\r\n"

        # Main query without splittings; exclude rows found in datasetName2
        sql_main = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp
              ON dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv
              ON dv.canon_qsar_smiles = dp.canon_qsar_smiles
            JOIN qsar_descriptors.descriptor_sets ds
              ON ds.id = dv.fk_descriptor_set_id
            WHERE d.name = :datasetName
              AND ds.name = :descriptorSetName
              AND NOT EXISTS (
                    SELECT 1
                    FROM qsar_datasets.datasets d2
                    JOIN qsar_datasets.data_points dp2
                      ON dp2.fk_dataset_id = d2.id
                    WHERE d2.name = :datasetName2
                      AND dp2.canon_qsar_smiles = dp.canon_qsar_smiles
              )
            ORDER BY dp.canon_qsar_smiles
        """)

        sb = [instance_header]

        results = session.execute(sql_main, {
            'datasetName': datasetName,
            'descriptorSetName': descriptorSetName,
            'datasetName2': datasetName2
        })

        for row in results:
            chemical_id, qsar_property_value, descriptors = row
            instance = _generate_instance(chemical_id, qsar_property_value, descriptors)

            if instance is None:
                logging.debug(f"{chemical_id}\tnull instance\tdatasetName={datasetName}\tdescriptorSetName={descriptorSetName}")
                continue

            sb.append(instance)

        df = _load_df(''.join(sb))

        logging.debug('instances shape ' + str(df.shape))
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None
    
    

def _generate_instance(chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\r\n"


def get_descriptor_headers(session, descriptorSetName):
    """
    Fetch and parse descriptor headers for the given descriptor set.

    Returns:
      list[str]: descriptor column names in order.

    Raises:
      ValueError if no headers are found or headers_tsv is NULL/empty.
    """
    headers_sql = text("""
        SELECT headers_tsv
        FROM qsar_descriptors.descriptor_sets ds
        WHERE ds.name = :descriptorSetName
    """)
    row = session.execute(headers_sql, {'descriptorSetName': descriptorSetName}).fetchone()
    if not row or row[0] is None:
        msg = f"No headers found for descriptor set '{descriptorSetName}'"
        logging.error(msg)
        raise ValueError(msg)

    headers_tsv = str(row[0]).strip().rstrip('\r\n')
    # Filter out empty header tokens (in case of trailing tabs)
    descriptor_cols = [h for h in headers_tsv.split('\t') if h != '']
    if not descriptor_cols:
        msg = f"Descriptor headers are empty for descriptor set '{descriptorSetName}'"
        logging.error(msg)
        raise ValueError(msg)

    return descriptor_cols


def get_instances(session, datasetName, descriptorSetName):
    """
    Returns a pandas DataFrame with columns:
      ID, Property, <descriptor_1>, <descriptor_2>, ...

    Behavior:
      - Strict check: if any row's descriptor value count does not match the header count,
        raises ValueError (no padding/truncation).

    Returns:
      pd.DataFrame on success

    On SQLAlchemyError:
      returns (None, None)
    """
    try:
        descriptor_cols = get_descriptor_headers(session, descriptorSetName)

        data_sql = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
            JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
            WHERE d.name = :datasetName
              AND ds.name = :descriptorSetName
            ORDER BY dp.canon_qsar_smiles
        """)

        rows = []
        params = {'datasetName': datasetName, 'descriptorSetName': descriptorSetName}
        for chem_id, prop_val, values_tsv in session.execute(data_sql, params):
            vals = [] if values_tsv is None else str(values_tsv).rstrip('\r\n').split('\t')

            expected = len(descriptor_cols)
            got = len(vals)
            if got != expected:
                msg = (
                    f"Descriptor value count mismatch for ID '{chem_id}': "
                    f"expected {expected}, got {got}."
                )
                logging.error(msg)
                raise ValueError(msg)

            rows.append([chem_id, prop_val] + vals)

        columns = ['ID', 'Property'] + descriptor_cols
        df = pd.DataFrame(rows, columns=columns)

        # Convert numeric columns where possible
        to_convert = ['Property'] + descriptor_cols
        df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')

        logging.debug(f"Number of datapoints: {df.shape}")
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None



def getDatapointsLookup(session, datasetName):
    """
    Returns a pandas DataFrame with columns:
      ID, Property, <descriptor_1>, <descriptor_2>, ...

    Behavior:
      - Strict check: if any row's descriptor value count does not match the header count,
        raises ValueError (no padding/truncation).

    Returns:
      pd.DataFrame on success

    On SQLAlchemyError:
      returns (None, None)
    """
    try:

        data_sql = text("""
            SELECT dp.id, dp.canon_qsar_smiles, dp.qsar_property_value
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            WHERE d.name = :datasetName
            ORDER BY dp.canon_qsar_smiles
        """)

        rows = []
        params = {'datasetName': datasetName}
        
        for fk_data_point_id, qsar_smiles, exp in session.execute(data_sql, params):
            rows.append([fk_data_point_id, qsar_smiles, exp])

        columns = ['fk_data_point_id', 'qsar_smiles',"exp"] 
        df = pd.DataFrame(rows, columns=columns)

        # Convert numeric columns where possible
        # to_convert = ['Property'] + descriptor_cols
        # df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')

        logging.debug(f"Number of datapoints: {df.shape}")
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None












def add_model_prediction_to_df(df, model_id, pred_name, add_squared_column=True):
    """
    #adds model prediction as a descriptor column    
    """
    
    from model_ws_db_utilities import ModelInitializer
    import json
    from model_ws_utilities import call_do_predictions_from_df
    
    mi = ModelInitializer()
    model = mi.init_model(model_id)
    json_test_set_kow = call_do_predictions_from_df(df, model)
    test_set_kow = json.loads(json_test_set_kow)
    
    df_kow = pd.DataFrame(test_set_kow).drop(columns=['exp'], errors='ignore').rename(columns={'id':'ID', 'pred':pred_name})
# Optional: enforce column order
    df_kow = df_kow[['ID', pred_name]]
    
    if add_squared_column:
        df_kow[f'{pred_name}_squared'] = df_kow[pred_name] ** 2
    
# print(test_set_kow[0])
    df = df.merge(df_kow, on='ID', how='left', validate='m:1')
    
    return df


def getLogKowPredictionsForDataset():

    
    dataset_name = "KOC v1 modeling"
    session = getSession()
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    df_training, df_prediction = get_training_prediction_instances(session, dataset_name, descriptor_set_name, splitting_name)

    model_id = str(1069)
    pred_name = 'LOGP_Martin'

    df_prediction = add_model_prediction_to_df(df_prediction, model_id, pred_name)
    df_training = add_model_prediction_to_df(df_training, model_id, pred_name)


if __name__ == '__main__':
    
    # test getting the detailed property data:
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    
    dataset_name = 'KOC v1 modeling'
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    # SplittingGenerator.create_splittings(dataset_name)
    
    # ExpDataGetter.get_mapped_records_by_source(dataset_name)
    
    
    
# *************************************************************************************************
    # from util.database_utilities import getSession
    # session = getSession()
    
    # df_pv = getMappedPropertyValues(session, dataset_name)
    #
    # print('\nMapped property values:')
    # for i in range(1, 3):
    #     print_first_row(df_pv, row=i)
    #
    # print('\nMapped datapoints:')
    # df_dps = getMappedDatapoints(session, dataset_name)
    #
    # for i in range(1, 3):
    #     print_first_row(df_dps, row=i)
        
    # descriptorSetName = 'WebTEST-default'
    # df_dps = getDatapoints(session, dataset_name, descriptorSetName)
    # print_first_row(df_dps)
    # print(df_dps.shape)
    
    # df_lookup = getDatapointsLookup(session, dataset_name)
    # print_first_row(df_lookup)
    # print(df_lookup.shape)

    
    # print_first_row(df_dps, row=1)

