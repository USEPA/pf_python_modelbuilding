'''
Created on Jan 22, 2026

@author: TMARTI02
'''

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text, bindparam

import logging
from io import StringIO
import pandas as pd

from utils import timer

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
    
