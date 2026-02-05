'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from models.case_studies.run_model_building_db import run_dataset, Results, ParametersGeneticAlgorithm
from predict_constants import PredictConstants as pc
import logging
import json
from xgboost.dask import da
from sklearn.datasets import descr

# logging.basicConfig(level=logging.ERROR, force=True) #turn off info prints 


def run_example():
    
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False)  # OK


def run_Koc():

    dataset_name = "KOC v1 modeling"

    # ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=False, ad_measure_final=ad_measure_final)
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, add_LOGP_Martin=True)  # Martin LOGP will show up in final descriptors, but error isnt lower!
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, add_LOGP_Martin=True)  # OK

    r = Results()
    r.summarize_model_stats(dataset_name)
    

def run_Koc_knn_ga():
    
    descriptor_set_name = "WebTEST-default"
    dataset_name = "KOC v1 modeling"

    grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']}  # matches AD in terms of using 3
    params = ParametersGeneticAlgorithm(qsar_method='knn', hyperparameter_grid=grid,
                                        descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                        run_rfe=True)
    params.num_optimizers = 100
    params.num_generations = 100
    
    # max_features_array = [3, 5, 10, 15, 20]
    
    max_features_array = [20]
    
    stats_dict = {}
    
    for max_features in max_features_array:
        params.max_features = max_features
        results_dict = run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, params=params)
        MAE_Test = results_dict['test_stats']['MAE_Test']
        MAE_Training_CV = results_dict['cv_stats']['MAE_Test']
        
        logging.info(f"max_features: {max_features}, MAE_Test: {MAE_Test:.2f}, MAE_Training_CV: {MAE_Training_CV:.2f}")
    
        stats = {"max_features": max_features, "MAE_Test":MAE_Test, "MAE_Training_CV":MAE_Training_CV}
        stats_dict[max_features] = stats
    
    print(json.dumps(stats_dict, indent=4))


def run_fish_tox():
    
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3 modeling'
    
    ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    # ad_measure_final = [pc.Applicability_Domain_TEST_All_Descriptors_Euclidean]
        
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=False, ad_measure_final=ad_measure_final)
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="xgb_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

    embedding = ["Mp", "nO", "nS", "ATS1m", "GATS1m", "XLOGP", "Ui"]  # Omit ALOGP
    run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True,
                ad_measure_final=ad_measure_final, embedding=embedding, fs_previous_embedding=False)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_final=ad_measure_final)  # OK    
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

    r = Results()
    r.summarize_model_stats(dataset_name)


def test_create_model():

    from predict_constants import PredictConstants
    pc=PredictConstants    
    from model_ws_db_utilities import getSession    
    session = getSession()
    
    
    from models.case_studies.run_model_building_db import ModelLoader
    ml = ModelLoader()

    from util.database_utilities import DatabaseLoader 

    dl=DatabaseLoader()
        
    user = 'tmarti02'
    qsar_method = 'knn'
    embedding=["col1","col2"]
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = pc.DESCRIPTOR_SET_WEBTEST
    splitting_name = pc.SPLITTING_RND_REPRESENTATIVE
    
    import time    
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
        
    fk_descriptor_embedding_id = ml.create_descriptor_embedding(session, descriptor_embedding)
    
    fk_descriptor_embedding_id = 276

    from models.ModelBuilder import Model
    
    model = Model()
    model.hyperparameter_grid = {"n_estimators": 500, "max_depth": 20}
    model.hyperparameters= {"n_estimators": 500, "max_depth": 20}

    model_name = user+"_"+str(epoch_ms)

    fk_method_id = ml.get_method_id(session, qsar_method) #use the generic method_id that doesnt have a version so can set the hyperparameter_grid
    

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
    fk_model_id = ml.create_model(session, model_row)

    

if __name__ == '__main__':
    # run_example()
    # run_Koc_knn_ga()
    # run_Koc()
    # run_fish_tox()
    test_create_model()
