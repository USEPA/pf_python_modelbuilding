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

def _generate_instance(chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\r\n"
    
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



def getSqlPropertyValuesForDataset():
        
        # TODO could get additional mapped identifiers from dsstox by doing a join to qsar_models.dsstox_records

        return text("""
        SELECT
            pv.id AS prop_value_id,
            dp.canon_qsar_smiles,
            sc.source_dtxsid,
            sc.source_casrn,
            sc.source_chemical_name,
            sc.source_smiles,
            --dpc.dtxsid AS mapped_dtxsid,
            --dpc.dtxcid AS mapped_dtxcid,
            --dpc.smiles AS mapped_smiles,
            dr.dtxsid AS mapped_dtxsid, 
            dr.dtxcid AS mapped_dtxcid,
            dr.casrn AS mapped_smiles,
            dr.preferred_name AS mapped_chemical_name, -- get from dsstox records instead of from dpc so that can get the mapped name
            dr.mol_weight as mapped_mol_weight,
            p.name_ccd AS prop_name,
            'experimental' AS prop_type,
            pc.name AS prop_category,
            d."name" AS dataset,
            dpc.property_value AS prop_value,--use the dpc value rather than the pv value fields because it's in the right units
            u.abbreviation_ccd AS prop_unit,
            dp.qsar_property_value,
            u2.abbreviation_ccd AS qsar_property_unit,
            pv.value_original AS prop_value_original,
            pv.value_text AS prop_value_text,
            pvT.value_point_estimate AS exp_details_temperature_c,
            pvP.value_point_estimate AS exp_details_pressure_mmHg,
            pvpH.value_point_estimate AS exp_details_pH, -- note: will convert to lower case
            pvRS.value_text AS exp_details_response_site, -- for BCF, fish tox
            pvSL.value_text AS exp_details_species_latin, -- for BCF, fish tox
            pvSC.value_text AS exp_details_species_common,
            pvSS.value_text AS exp_details_species_supercategory,
            --CASE WHEN ps.name IS NOT NULL THEN ps.name ELSE ls.name END AS source_name,
            --CASE WHEN ps.name IS NOT NULL THEN ps.description ELSE ls.citation END AS source_description,
            --CASE WHEN ps.name IS NOT NULL THEN ps.url ELSE ls.doi END AS source_url,
            ps."name" AS public_source_name,
            ps.description AS public_source_description,
            ps.url AS public_source_url,
            pv.page_url AS direct_url,
            ls."name" AS literature_source_name,
            ls.citation AS literature_source_citation,
            ls.doi AS literature_source_doi,
            pv.document_name AS brief_citation, -- From OPERA2.9 usually
            ps2."name" AS public_source_original_name, -- For sources like toxval, pubchem, sander
            ps2.description AS public_source_original_description,
            ps2.url AS public_source_original_url
        FROM qsar_datasets.data_points AS dp
        JOIN qsar_datasets.data_point_contributors AS dpc
            ON dpc.fk_data_point_id = dp.id
        JOIN exp_prop.property_values AS pv
            ON dpc.exp_prop_property_values_id = pv.id
        LEFT JOIN exp_prop.literature_sources AS ls
            ON pv.fk_literature_source_id = ls.id
        LEFT JOIN exp_prop.public_sources AS ps
            ON pv.fk_public_source_id = ps.id
        LEFT JOIN exp_prop.public_sources AS ps2
            ON pv.fk_public_source_original_id = ps2.id
        LEFT JOIN exp_prop.parameter_values AS pvT
            ON pvT.fk_property_value_id = pv.id AND pvT.fk_parameter_id = 2
        LEFT JOIN exp_prop.parameter_values AS pvP
            ON pvP.fk_property_value_id = pv.id AND pvP.fk_parameter_id = 1
        LEFT JOIN exp_prop.parameter_values AS pvpH
            ON pvpH.fk_property_value_id = pv.id AND pvpH.fk_parameter_id = 3
        LEFT JOIN exp_prop.parameter_values AS pvRS
            ON pvRS.fk_property_value_id = pv.id AND pvRS.fk_parameter_id = 22
        LEFT JOIN exp_prop.parameter_values AS pvSS
            ON pvSS.fk_property_value_id = pv.id AND pvSS.fk_parameter_id = 38
        LEFT JOIN exp_prop.parameter_values AS pvSL
            ON pvSL.fk_property_value_id = pv.id AND pvSL.fk_parameter_id = 21
        LEFT JOIN exp_prop.parameter_values AS pvSC
            ON pvSC.fk_property_value_id = pv.id AND pvSC.fk_parameter_id = 11
        JOIN exp_prop.source_chemicals AS sc
            ON sc.id = pv.fk_source_chemical_id
        JOIN qsar_datasets.datasets AS d
            ON dp.fk_dataset_id = d.id
        JOIN qsar_datasets.properties AS p
            ON d.fk_property_id = p.id
        JOIN qsar_datasets.units AS u
            ON u.id = d.fk_unit_id_contributor
        JOIN qsar_datasets.units AS u2
            ON u2.id = d.fk_unit_id
        LEFT JOIN qsar_datasets.properties_in_categories AS pic
            ON p.id = pic.fk_property_id
        LEFT JOIN qsar_datasets.property_categories AS pc
            ON pic.fk_property_category_id = pc.id
        LEFT JOIN qsar_models.dsstox_records AS dr
          ON dr.dtxcid = dpc.dtxcid
        WHERE d.name = :dataset_name AND dr.fk_dsstox_snapshot_id = 4
          AND keep = TRUE
        """)



def getMappedPropertyValues(session, dataset_name):
    """
    Gets detailed property value records from the database
    """
    
    logging.info(f"Getting mapped property values for {dataset_name}")
    df_pv = pd.read_sql(getSqlPropertyValuesForDataset(), con=session.get_bind(), params={"dataset_name": dataset_name})
    df_pv = df_pv.replace('', np.nan).dropna(axis=1, how='all')  # drop the columns with no data
    df_pv = df_pv.sort_values(["canon_qsar_smiles", "qsar_property_value"]) #sql query could handle the sorting, but easier to handle here
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
            d.name,
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


if __name__ == '__main__':
    
    from utils import print_first_row

    # test getting the detailed property data:
    from dotenv import load_dotenv
    load_dotenv()
    
    from util.database_utilities import getSession
    
    session = getSession()
    dataset_name = 'KOC v1 modeling'
    
    df_pv = getMappedPropertyValues(session, dataset_name)
    
    print('\nMapped property values:')
    for i in range(1, 3):
        print_first_row(df_pv, row=i)

    print('\nMapped datapoints:')
    df_dps = getMappedDatapoints(session, dataset_name)
    
    for i in range(1, 3):
        print_first_row(df_dps, row=i)

    
    # print_first_row(df_dps, row=1)

