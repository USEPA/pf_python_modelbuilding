'''
Created on Feb 3, 2026

@author: TMARTI02
'''


from dotenv import load_dotenv
load_dotenv('../../personal.env')

from models.case_studies.run_model_building_db import run_dataset, ParametersGeneticAlgorithm, set_hyper_parameters, Results

from util import predict_constants as pc
from model_ws_db_utilities import getEngine, getSession
from ModelToExcel import ModelToExcel
import logging
import json

def run_example():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False)  # OK


def run_Koc():
    create_unique_excel = False
    write_to_db = False
    # write_to_db=True
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    # for method in ['rf']:
    # for method in ['knn']:        
        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
        #             ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
    
    # embedding = ["ALOGP2","nBnz","MATS6v","ATS1p","nDB","Lop","MATS1p"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)


    # append_to_models_folder=""
    # append_to_models_folder="_alpha=0.7"
    append_to_models_folder= "_test_cv"
    
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False, append_to_models_folder=append_to_models_folder)  # OK
    
    # for method in ['rf','xgb']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                           splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     params.descriptor_coefficient = 0.006
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #         create_detailed_excel=False, create_unique_excel=create_unique_excel, 
    #         append_to_models_folder=append_to_models_folder)  
        
    # for method in ['reg','knn']:
    # # for method in ['reg']:
    #
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.max_features = 12
    #     params.descriptor_coefficient = 0.006
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #         create_detailed_excel=False, create_unique_excel=create_unique_excel, 
    #         append_to_models_folder=append_to_models_folder)  
        
        
        
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

    write_to_db = False
    create_unique_excel = False
    
    append_to_models_folder ="2"

    # for method in ['rf','xgb', 'reg','knn']:

    for method in ['rf','xgb', 'knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=False, descriptor_set_name="WebTEST-default", 
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        # params.n_features_to_select = 10
        run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=False, params = params, 
                    ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
                    create_detailed_excel=False, create_unique_excel=create_unique_excel, 
                    append_to_models_folder=append_to_models_folder)  # OK

    # # embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'xv1', 'Mp', 'AMW'] 
    # # embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'Mp', 'AMW']
    # embedding = ['ALOGP', 'ALOGP2', 'MW', 'Mp', 'AMW']
    #
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)
    
    # Models to upload:
    
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False,append_to_models_folder=append_to_models_folder)  # OK
    #
    # for method in ['rf','xgb', 'reg','knn']:
    # # for method in ['knn']:
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, 
    #                 create_detailed_excel=False, append_to_models_folder=append_to_models_folder)  # OK
    
    
    Results.summarize_model_stats(dataset_name,append_to_models_folder=append_to_models_folder)
    
    
    
def run_biodeg_rifm():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # automapped one
    write_to_db = False
    create_unique_excel = False
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    # Models to upload:
    for method in ['rf','xgb']:
    # for method in ['reg','knn']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
                    ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

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
    
    
    Results.summarize_model_stats(dataset_name)


def run_pchem():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'HLC v1 modeling' # automapped one
    write_to_db = False
    create_unique_excel = False
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    # Models to upload:
    # for method in ['rf','xgb']:
    # for method in ['reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK

    for method in ['rf','xgb', 'knn']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
                    ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
                
        
    # for method in ['rf','xgb','knn','reg']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     params.run_rfe = False
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel,create_detailed_excel=False)  # OK
    
    
    Results.summarize_model_stats(dataset_name)

def run_biodeg_nite():
    
    dataset_name = 'exp_prop_RBIODEG_NITE_OPPT v1.0'    

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
        
        
    Results.summarize_model_stats(dataset_name)


def test_model_summary():
    engine = getEngine()
    session = getSession()
    model_id = 1065
    excel_path = "summary.xlsx"
    test = ModelToExcel(engine, session, model_id, excel_path)
    test.create_excel()


def test_model_summary_local():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, create_detailed_excel=True)  # OK

# split_num
# fk_data_point_id


if __name__ == '__main__':
    
    # run_example()
    # run_Koc_knn_ga()
        
    run_Koc()
    # run_fish_tox()
    # run_biodeg_rifm()
    
    # run_pchem()
    # run_biodeg_nite()
    # test_create_model()
    # test_model_summary()

    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()
    # test_model_summary_local()

