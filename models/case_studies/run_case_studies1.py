'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from models.case_studies.run_model_building_db import run_dataset, Results, ParametersGeneticAlgorithm
from predict_constants import PredictConstants as pc
import logging
import json

# logging.basicConfig(level=logging.ERROR, force=True) #turn off info prints 

def run_example():
    
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=False) #OK
    r = Results()
    r.summarize_model_stats(dataset_name)

def run_Koc():

    dataset_name = "KOC v1 modeling"

    # ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=False, ad_measure_final=ad_measure_final)
    run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='reg', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='reg', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

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
    
    # ad_measure_final = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_final = [pc.Applicability_Domain_TEST_All_Descriptors_Euclidean]
        
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=False, ad_measure_final=ad_measure_final)
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=False, ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, ad_measure_final=ad_measure_final)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_final=ad_measure_final)  # OK    
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_final=ad_measure_final)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', feature_selection=True, ad_measure_final=ad_measure_final)  # OK

    r = Results()
    r.summarize_model_stats(dataset_name)


def main():
    run_Koc()

if __name__ == '__main__':
    main()
