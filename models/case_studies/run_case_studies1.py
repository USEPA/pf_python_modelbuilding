'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from models.case_studies.run_model_building_db import run_dataset, Results, ParametersGeneticAlgorithm


def run_example():
    
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=False) #OK

def run_Koc():

    dataset_name = "KOC v1 modeling"
    
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=True) #OK
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=False) #OK
    
    run_dataset(dataset_name=dataset_name,qsar_method='xgb',feature_selection=True) #OK
    run_dataset(dataset_name=dataset_name,qsar_method='xgb',feature_selection=False)
    
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=False) #OK
    run_dataset(dataset_name=dataset_name,qsar_method='knn',folder_embedding="rf_WebTEST-default_fs=True") #OK
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=True) #OK

    run_dataset(dataset_name=dataset_name,qsar_method='gcm',feature_selection=False) #OK
    
    run_dataset(dataset_name=dataset_name,qsar_method='reg',folder_embedding="rf_WebTEST-default_fs=True") #OK

    r = Results()
    r.summarize_model_stats(dataset_name)
    

def run_Koc_knn_ga():
    
    descriptor_set_name = "WebTEST-default"
    dataset_name = "KOC v1 modeling"
    

    grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']} # matches AD in terms of using 3
    params = ParametersGeneticAlgorithm(qsar_method='knn', hyperparameter_grid=grid,
                                        descriptor_set_name=descriptor_set_name, dataset_name=dataset_name, 
                                        run_rfe=False)
    params.num_optimizers=1
    params.num_generations=1
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=True, params=params)
    
    params.num_optimizers=1
    params.num_generations=10
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=True, params=params)
    
    params.num_optimizers=10
    params.num_generations=10
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=True, params=params)




def run_fish_tox():
    
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3 modeling'
    
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=False) #OK
    run_dataset(dataset_name=dataset_name,qsar_method='rf',feature_selection=True) #OK

    run_dataset(dataset_name=dataset_name,qsar_method='xgb',feature_selection=False)
    run_dataset(dataset_name=dataset_name,qsar_method='xgb',feature_selection=True) #OK
    
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=False) #OK
    run_dataset(dataset_name=dataset_name,qsar_method='knn',folder_embedding="rf_WebTEST-default_fs=True") #OK
    run_dataset(dataset_name=dataset_name,qsar_method='knn',feature_selection=True) #OK

    run_dataset(dataset_name=dataset_name,qsar_method='gcm',feature_selection=False) #OK    
    
    run_dataset(dataset_name=dataset_name,qsar_method='reg',folder_embedding="rf_WebTEST-default_fs=True") #OK
    run_dataset(dataset_name=dataset_name,qsar_method='reg',feature_selection=True) #OK

    r = Results()
    r.summarize_model_stats(dataset_name)



if __name__ == '__main__':
    run_example()