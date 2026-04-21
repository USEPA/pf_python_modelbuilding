'''
Created on Feb 3, 2026

@author: TMARTI02
'''


from dotenv import load_dotenv
load_dotenv('../../personal.env')

from models.case_studies.run_model_building_db import run_dataset, ParametersGeneticAlgorithm, set_hyper_parameters, Results

from util import predict_constants as pc
from model_ws_db_utilities import getEngine, getSession
from models.ModelToExcel import ModelToExcel
import logging
import json

def run_example():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False)  # OK


def run_Koc():
    create_unique_excel = False
    # write_to_db = True
    write_to_db = False
    # write_to_db=True
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    # append_to_models_folder = ""
    # append_to_models_folder = "_v2.0"
    append_to_models_folder = "_v3.0"
    

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True, 
                append_to_models_folder=append_to_models_folder)  # OK


    # for method in ['xgb']:
    # for method in ['rf','xgb']:
    #
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                         splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     # params.descriptor_coefficient = 0.006
    #     if method == 'rf':
    #         params.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2'],
    #                                      'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
    #                                      'estimator__n_estimators': [10, 100, 250, 500]}
    #     elif method=='xgb':
    #         params.hyperparameter_grid = {'estimator__n_estimators': [50, 100], 'estimator__eta': [0.1, 0.2, 0.3],
    #                                 'estimator__gamma': [0, 1, 10], 'estimator__max_depth': [3, 6, 9, 12],
    #                                 'estimator__min_child_weight': [1, 3, 5], 'estimator__subsample': [0.5, 1]}
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #         create_detailed_excel=True, create_unique_excel=create_unique_excel, 
    #         append_to_models_folder=append_to_models_folder)  
        
        # params.feature_selection = False
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #     create_detailed_excel=True, create_unique_excel=create_unique_excel, 
        #     append_to_models_folder=append_to_models_folder)  
        

        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
        #     ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, 
        #     create_detailed_excel=True,append_to_models_folder=append_to_models_folder)  


    for method in ['reg','knn']:
    # # for method in ['las']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        params.max_features = 12
        params.descriptor_coefficient = 0.006
        run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
            params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
            create_detailed_excel=True, create_unique_excel=create_unique_excel, 
            append_to_models_folder=append_to_models_folder)  
        
        
        
        # for i in range(1, 8):
        #     params.alpha = 0.2+(i-1)*0.1
        #     params.descriptor_coefficient = None
        #
        #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #         create_detailed_excel=False, create_unique_excel=create_unique_excel, 
        #         append_to_models_folder=append_to_models_folder)  # OK

    
        # for i in range(1, 11):
        #     params.descriptor_coefficient = round(0.002 * i, 3)          
        #     # params.n_features_to_select = 10
        #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #                 params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #                 create_detailed_excel=False, create_unique_excel=create_unique_excel, 
        #                 append_to_models_folder=append_to_models_folder)  # OK


    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')
    

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
    
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    write_to_db = False #TODO need to rerun with write = true
    
    create_unique_excel = False
    
    append_to_models_folder = "_bob"
    

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True, 
                append_to_models_folder=append_to_models_folder)  # OK

    # for method in ['rf']:
    # for method in ['xgb']:
    # for method in ['rf','xgb']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                         splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     # params.descriptor_coefficient = 0.006
    #     if method == 'rf':
    #         params.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2'],
    #                                      'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
    #                                      'estimator__n_estimators': [10, 100, 250, 500]}
    #     elif method=='xgb':
    #         params.hyperparameter_grid = {'estimator__n_estimators': [50, 100], 'estimator__eta': [0.1, 0.2, 0.3],
    #                                 'estimator__gamma': [0, 1, 10], 'estimator__max_depth': [3, 6, 9, 12],
    #                                 'estimator__min_child_weight': [1, 3, 5], 'estimator__subsample': [0.5, 1]}
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #         create_detailed_excel=True, create_unique_excel=create_unique_excel, 
    #         append_to_models_folder=append_to_models_folder)  
    
        # params.feature_selection = False
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #     create_detailed_excel=True, create_unique_excel=create_unique_excel, 
        #     append_to_models_folder=append_to_models_folder)  
        

        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
        #     ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, 
        #     create_detailed_excel=True,append_to_models_folder=append_to_models_folder)  


    # for method in ['reg','knn']:
    # # # for method in ['las']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.max_features = 20
    #     params.descriptor_coefficient = 0.006
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #         create_detailed_excel=True, create_unique_excel=create_unique_excel, 
    #         append_to_models_folder=append_to_models_folder)  
    
    
    Results.summarize_model_stats(dataset_name,append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    
    
    
def run_biodeg_rifm():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # automapped one
    write_to_db = False
    create_unique_excel = False
    create_detailed_excel = True
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]


    append_to_models_folder=""
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=create_detailed_excel,
                append_to_models_folder=append_to_models_folder)  # OK

    # Models to upload:
    # for method in ['rf','xgb']:
    # for method in ['reg','knn']:
        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
                    # ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=create_detailed_excel)  # OK


    # for method in ['rf','xgb', 'knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
                
        
    # for method in ['rf','xgb','knn','reg']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     params.run_rfe = False
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel,create_detailed_excel=False)  # OK
    
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def run_pchem():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'HLC v1 modeling' # automapped one
    write_to_db = False
    create_unique_excel = False
    create_detailed_excel = True
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    append_to_models_folder = "_bob"

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True)  # OK

    # Models to upload:
    # for method in ['rf','xgb']:
    # for method in ['reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    # for method in ['rf','xgb', 'knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
                
        
    # for method in ['rf','xgb','knn','reg']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     params.run_rfe = False
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel,create_detailed_excel=False)  # OK
    
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)

def run_biodeg_nite():
    
    dataset_name = 'exp_prop_RBIODEG_NITE_OPPT v1.0'
    append_to_models_folder = ""

    write_to_db = False
    create_unique_excel = False
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]


    # for method in ['rf','xgb','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, ad_measure_model=ad_measure_model,
    #                 write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    #     params = set_hyper_parameters(qsar_method=method, feature_selection=False, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_detailed_excel=False, 
    #                 create_unique_excel=create_unique_excel)  # OK


    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    # # for method in ['rf']:                
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
        
        
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def test_model_summary():
    engine = getEngine()
    session = getSession()
    model_id = 1746
    excel_path = "summary.xlsx"
    test = ModelToExcel(engine, session, model_id, excel_path)
    test.create_excel()


def test_model_summary_local():
    dataset_name = "KOC v1 modeling"
    create_unique_excel = False
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, create_detailed_excel=True, create_unique_excel=create_unique_excel)  # OK


def test_load_model_with_external_set():
    create_unique_excel = False
    write_to_db = False
    # write_to_db = True
    dataset_name = "KOC v1 modeling"
    user = "murdock.weston"
    append_to_models_folder = ""

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True, user=user)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    # embedding = ["ALOGP2","nBnz","MATS6v","ATS1p","nDB","Lop","MATS1p"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)

    r = Results()
    r.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


if __name__ == '__main__':
    
    # run_example()
    # run_Koc_knn_ga()
        
    run_Koc()
    # run_fish_tox()
    # run_biodeg_rifm()
    # run_pchem()
    
    # These 4 should be able to run for the gcm model
    # run_Koc()  # OK
    # run_fish_tox()  # Takes too long to run on my machine? (E.g. started a run at 1:55, errored out at 4:53 because the SQL connection closed automatically)
    run_biodeg_rifm()  # OK
    # run_pchem()  # OK

    # run_biodeg_nite()
    # test_create_model()
    # test_model_summary()

    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()
    # test_model_summary_local()
    # test_load_model_with_external_set()

