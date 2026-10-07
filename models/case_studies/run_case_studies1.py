'''
Created on Feb 3, 2026

@author: TMARTI02
'''


import json
import pandas as pd
import os
from pathlib import Path
from io import StringIO

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

from dotenv import load_dotenv
load_dotenv("../../personal.env")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


from models.case_studies.case_study_utilities import (
    _fetch_set_qsar_smiles,
    _get_smiles_for_training_set,
    _calculate_stats_for_binary_test_set,
    _get_qsar_smiles_from_fragrance_spreadsheet,
    _calculate_stats_for_subset,
    _summarize_fragrance_results_as_excel,
    _get_model_ids,
    _getStatsFromDatasets,
    _summarize_fragrance_results,
    _calculate_fragrance_stats,
    _find_model_folder,
    _full_test_mte,
    _predictSetFromDB_SmilesFromExcel,
    _run_test_set,
)


from models.case_studies.run_model_building_db import (
    run_dataset,
    run_dataset_from_dfs,
    ParametersGeneticAlgorithm,
    ParametersImportance,
    ParametersGeneric,
    ParametersGroupContribution,
    Results,
    set_hyper_parameters,
)


from util import predict_constants as pc
from model_ws_db_utilities import getSession, ModelInitializer
from model_ws_utilities import call_do_predictions_from_df
import models.db_utilities.dataset_utilities_db as du  
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics



def run_example():
    write_to_db = False

    dataset_name = "KOC v1 modeling"
    # descriptor_set_name = "WebTEST-default"
    # splitting_name = "RND_REPRESENTATIVE"
    
    append_to_models_folder = "_bob"
    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK


def run_Koc():
    # unique_identifier = 'time'
    unique_identifier = None

    # write_to_db = True
    write_to_db = False
    dataset_name = "KOC v1 modeling"
    # dataset_name = "KOC v2 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  

    # append_to_models_folder = ""
    # append_to_models_folder = "_"+str(descriptor_coefficient)
    # append_to_models_folder = "_ad_test"
    # append_to_models_folder = "_v2.0"
    # append_to_models_folder = "_KOC_v2 external"
    # append_to_models_folder = "_v3.0"
    append_to_models_folder = "_bob"
    descriptor_coefficient = 0.006

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, unique_identifier=unique_identifier,
    #             append_to_models_folder=append_to_models_folder)  # OK
    #
    # for method in ['rf', 'xgb']:
    for method in ['xgb']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
            ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)  
    # #
    # #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
    # #         ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
    # #         append_to_models_folder=append_to_models_folder)  
    #
    #     if method == 'xgb':
    #         continue
    #
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = descriptor_coefficient
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
    #         unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    #
    #
    #
    # for method in ['reg', 'knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     # params.max_features = 12
    #     # params.max_features = 25
    #     params.max_features = 40
    #     params.descriptor_coefficient = descriptor_coefficient
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
    #         unique_identifier=unique_identifier,
    #         append_to_models_folder=append_to_models_folder)  



    # for method in ['rf', 'xgb']:
        # params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
        #                     splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        #
        # if method == 'rf':
        #     params.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2'],
        #                                  'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                                  'estimator__n_estimators': [10, 100, 250, 500]}
        # elif method=='xgb':
        #     params.hyperparameter_grid = {'estimator__n_estimators': [50, 100], 'estimator__eta': [0.1, 0.2, 0.3],
        #                             'estimator__gamma': [0, 1, 10], 'estimator__max_depth': [3, 6, 9, 12],
        #                             'estimator__min_child_weight': [1, 3, 5], 'estimator__subsample': [0.5, 1]}
        #
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #    unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  
        
        # params.feature_selection = False
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #    unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  

    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')
    

def run_BCF():
    unique_identifier = None
    # write_to_db = True
    write_to_db = False
    
    dataset_name = "exp_prop_BCF_v1_modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    # append_to_models_folder = ""
    
    descriptor_coefficient = 0.006
    
    append_to_models_folder = "_" + str(descriptor_coefficient)
    # append_to_models_folder = "_v2.0"
    # append_to_models_folder = "_KOC_v2 external"

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, unique_identifier=unique_identifier,
    #             append_to_models_folder=append_to_models_folder)  # OK

    # for method in ['rf', 'xgb']:
    # for method in ['reg']:
        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
        #     ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
        #     append_to_models_folder=append_to_models_folder)  
    
        # params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
        #                               splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        # params.descriptor_coefficient = descriptor_coefficient
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
        #     unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
        
    # #
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
    #         ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
    #         append_to_models_folder=append_to_models_folder)  
    #
    #
    # for method in ['reg', 'knn']:
    for method in ['knn']: 
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        # params.max_features = 12
        params.max_features = 25
        params.descriptor_coefficient = 0.006
        run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
            params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
            unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)  

    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')


def run_Koc_knn_ga():
    
    descriptor_set_name = "WebTEST-default"
    dataset_name = "KOC v1 modeling"

    grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']}  # matches AD in terms of using 3
    params = ParametersGeneticAlgorithm(qsar_method='knn', hyperparameter_grid=grid,
                                        descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                        run_rfe=True)  # type: ignore
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

# def query_Bcf_rf_models():
#     logging.info("Running query_Bcf_rf_models()")
#     model_ids = []
#     for model_id in model_ids:
#         try:
#             folder_path = os.path.join(PROJECT_ROOT, "data", "models_rf_logp", "exp_prop_BCF_v1_modeling", f"{model_id}") # type: ignore
#             file_path = os.path.join(folder_path, "detailed_summary.xlsx")
#             model_path = os.path.join(folder_path, "model.pkl")
#             results_path = os.path.join(folder_path, "results.json")
#             mdo = ModelDataObjects(model_id=model_id)

#             model = mdo.model
#             with open(model_path, "wb") as f:
#                 f.write(pickle.dumps(model))

#             with open(results_path, 'w') as f:
#                 json.dump(mdo.results_dict, f, indent=4)

#             mte = ModelToExcel(mdo, file_path)
#             mte.create_excel()
#         except Exception as e:
#             logging.error(f"Error occurred while processing model_id {model_id}: {e}")


def run_fish_tox():
    
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling'
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    write_to_db = True  # TODO need to rerun with write = true
    
    unique_identifier = None
    append_to_models_folder = ""
    # append_to_models_folder = ""

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                add_LOGP_Martin=True, logp_columns=['LOGP_Martin'], write_to_db=write_to_db,
                unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK

    # for method in ['rf']:
    # for method in ['xgb']:
    # for method in ['knn']:
    # for method in ['rf','xgb']:
    # for method in ['rf', 'xgb', 'reg','knn']:
        
        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
        #     ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
        
        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
        #     ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  

        # params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
        #                     splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        # params.descriptor_coefficient = 0.003
        
    #     if method == 'rf':
    #         params.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2'],
    #                                      'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
    #                                      'estimator__n_estimators': [10, 100, 250, 500]}
    #     elif method=='xgb':
    #         params.hyperparameter_grid = {'estimator__n_estimators': [50, 100], 'estimator__eta': [0.1, 0.2, 0.3],
    #                                 'estimator__gamma': [0, 1, 10], 'estimator__max_depth': [3, 6, 9, 12],
    #                                 'estimator__min_child_weight': [1, 3, 5], 'estimator__subsample': [0.5, 1]}
    #
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #    unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  
    
        # params.feature_selection = False
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #    unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  

    # for method in ['reg','knn']:
    # # # for method in ['las']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.max_features = 20
    #     params.descriptor_coefficient = 0.006
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
    #        unique_identifier=unique_identifier, 
    #         append_to_models_folder=append_to_models_folder)  
    
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')

    # Results.summarize_model_stats(dataset_name, append_to_models_folder="_0.001", continuous_stat_name='RMSE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder="_0.003", continuous_stat_name='RMSE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder="_0.006", continuous_stat_name='RMSE')


def run_fish_tox_2():
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"

    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    
    unique_identifier = None
    write_to_db = False
    append_to_models_folder = "_bob"

    for method in ["gcm", 'rf', 'xgb']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
            ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)  
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
            ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)
        
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name="RMSE")




    
def run_biodeg_rifm():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # old one from january 26
    dataset_name = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'  # RBIODEG no 10 day window
    # dataset_name = 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG' # RBIODEG with 10 day window
    write_to_db = True
    # write_to_db = False
    
    unique_identifier = None
    # descriptor_set_name = "WebTEST-default"
    descriptor_set_name = "Mordred-default"
    # descriptor_set_name = "PaDEL-default"
    # descriptor_set_name = "RDKit-default"
    
    splitting_name = "RND_REPRESENTATIVE"  

    if descriptor_set_name == 'WebTEST-default':        
        ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    else:
        ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean]
    
    # append_to_models_folder = ""
    append_to_models_folder = "_v3.0"
    # append_to_models_folder = "_0.001"


    descriptor_service = descriptor_set_name.replace("-default", "").lower()
    if dataset_name == 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG': 
        tsv_file_name = f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_{descriptor_service}.tsv"            
    elif dataset_name == 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG':
        tsv_file_name = f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day_{descriptor_service}.tsv"
    else:
        print(f"handle external set name for dataset={dataset_name}")
        return
    tsv_file_path = Path(PROJECT_ROOT) / "data" / "datasets" / "RBIODEG external" / tsv_file_name         


    df_external = pd.read_csv(tsv_file_path, delimiter='\t')
    df_external = df_external.drop_duplicates(subset=["ID"], keep="first").copy()

    session = getSession()
            
    # dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    # df_smiles_subset = _fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    # print(df_smiles_subset)
    

    # model=run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, 
    #             ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    # folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
    # test_stats = run_test_set(df_external,model, folder)    
    # _summarize_fragrance_results_as_excel(session, folder, dataset_name)

    # print('gcm',test_stats)
    
    # # for method in ['rf', 'xgb']:        
    # for method in ['rf']:
    #     model=run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, descriptor_set_name=descriptor_set_name,
    #                 ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    #
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
    #     test_stats = _run_test_set(df_external, model, folder)    
    #     _summarize_fragrance_results_as_excel(session, folder, dataset_name)
        
    # #
    # # # for method in ['reg','knn']:
    for method in ['rf', 'xgb', 'reg', 'knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
                                       splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        params.descriptor_coefficient = 0.001
    
        params.remove_fragment_descriptors = True
        params.remove_acnt_descriptors = True
        params.run_rfe = False
    
        model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
             params=params, descriptor_set_name=descriptor_set_name, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
             unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    
        folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
        test_stats = _run_test_set(df_external, model, folder)    
        _summarize_fragrance_results_as_excel(session, folder, dataset_name)

    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    _summarize_fragrance_results(dataset_name, append_to_models_folder)
    
    # following is deprecated since run_test_set makes needed results files:
    # _calculate_fragrance_stats(dataset_name, append_to_models_folder, 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG')
    # TODO determine how RIFM only models work for test set of ECHA+RIFM set

    
def run_RIFM_model_on_ECHA_test_set():
    
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG'
    
    # dataset_name_ECHA_RIFM = 'exp_prop_RBIODEG_301F v1 modeling'
    dataset_name_ECHA_RIFM = 'exp_prop_RBIODEG_301F v2 modeling'
    
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    _, df_prediction = du.get_training_prediction_instances(getSession(), dataset_name_ECHA_RIFM, descriptor_set_name, splitting_name)

    session = getSession()
    
    model_ids = _get_model_ids(session, dataset_name=dataset_name)
    
    mi = ModelInitializer()
    
    for model_id in model_ids:
        model = mi.initModel(model_id)
        
        # print(model.qsar_method, len(model.embedding))
        if model is not None:
            if len(model.embedding) > 100:
                use_fs = False
            else:
                use_fs = True
        
            run = model.qsar_method + "_WebTEST-default_fs=" + str(use_fs)
            
            json_predictions = call_do_predictions_from_df(df_prediction, model)
            df_predictions_test = pd.read_json(StringIO(json_predictions), orient="records")
            
            # print(df_predictions_test)
            
            test_stats = calculate_binary_statistics(df_predictions_test, 0.5, "_Test")
            print(f"{run}\t{test_stats['BA_Test']:.3f}")

    


def run_continuous_model_on_test_set():
    
    # dataset_name= 'exp_prop_PERCENT_BIODEGRADATION_RIFM_CHEMREG'
    # dataset_name_test = 'exp_prop_RBIODEG_301F v1 modeling'

    dataset_name = 'exp_prop_PERCENT_BIODEGRADATION_301F v1 modeling'
    dataset_name_test = 'exp_prop_RBIODEG_RIFM_CHEMREG'
    
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    _, df_prediction = du.get_training_prediction_instances(getSession(), dataset_name_test, descriptor_set_name, splitting_name)

    session = getSession()
    
    model_ids = _get_model_ids(session, dataset_name=dataset_name)
    
    mi = ModelInitializer()
    
    for model_id in model_ids:
        model = mi.initModel(model_id)
        
        # print(model.qsar_method, len(model.embedding))
        if model is not None:
            if len(model.embedding) > 100:
                use_fs = False
            else:
                use_fs = True
        
            run = model.qsar_method + "_WebTEST-default_fs=" + str(use_fs)
            
            json_predictions = call_do_predictions_from_df(df_prediction, model)
            df_predictions_test = pd.read_json(StringIO(json_predictions), orient="records")
            
            df_predictions_test["pred"] = (df_predictions_test["pred"] >= 60).astype(int)

            # print(df_predictions_test)
            test_stats = calculate_binary_statistics(df_predictions_test, 0.5, "_Test")
            print(f"{run}\t{test_stats['BA_Test']:.3f}")
    


def run_percentage_biodegradation():

    # dataset_name = 'exp_prop_PERCENT_BIODEGRADATION_RIFM_CHEMREG'
    # dataset_name_binary = "exp_prop_RBIODEG_RIFM_CHEMREG"
    
    dataset_name = "exp_prop_PERCENT_BIODEGRADATION_301F v1 modeling"
    dataset_name_binary = "exp_prop_RBIODEG_301F v1 modeling"
    
    write_to_db = True
    # write_to_db = False
    
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    # get the prediction set for the corresponding binary dataset:  
    _, df_prediction_binary = du.get_training_prediction_instances(getSession(), dataset_name_binary, descriptor_set_name, splitting_name)
    
    # append_to_models_folder = ""
    append_to_models_folder = "_0.001"
    
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    log_path = os.path.join(PROJECT_ROOT, "data", "models" + append_to_models_folder, dataset_name, "binary_test_stats.log")  # type: ignore

    model = run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    _calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
    
    for method in ['rf', 'xgb']: 
        model = run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
                    ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
        _calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
                
    for method in ['rf', 'xgb', 'reg', 'knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        if params is not None:
            if not isinstance(params, ParametersGroupContribution) and not isinstance(params, ParametersGeneric):
                params.descriptor_coefficient = 0.001  # type: ignore
            model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
                params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
            _calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)





def run_other_test_set(model, dataset_name_other):

    # description = model.get_model_description_dict()
    # print(json.dumps(description,indent=4))
    
    session=getSession()
    from models.db_utilities.dataset_utilities_db import get_training_prediction_instances2
    _, df_prediction = get_training_prediction_instances2(session, dataset_name_other, model.descriptorService, model.splittingName)
    # print(df_prediction.shape)
    
    json_predictions = call_do_predictions_from_df(df_prediction, model)
    df_preds = pd.read_json(StringIO(json_predictions), orient="records")
    
    # TODO add check for is_binary and generate continuous or binary stats

    stats = calculate_binary_statistics(df_preds, 0.5, "_Test")
    print(stats)







def run_biodeg_301F():
    
    dataset_name = 'exp_prop_RBIODEG_301F v2 modeling'  # automapped one

    write_to_db = True
    # write_to_db = False
   # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean]

    # descriptor_set_name = "WebTEST-default"
    # descriptor_set_name = "Mordred-default"
    # descriptor_set_name = "PaDEL-default"
    descriptor_set_name = "RDKit-default"

    splitting_name = "RND_REPRESENTATIVE"
    unique_identifier = None 


    if descriptor_set_name == 'WebTEST-default':        
        ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    else:
        ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean]

    
    # append_to_models_folder = ""
    # append_to_models_folder = "_0.001"
    append_to_models_folder = "_v3.0"
    # append_to_models_folder = "_v3.0_0.006"
    # append_to_models_folder="_bob"


    descriptor_service = descriptor_set_name.replace("-default", "").lower()
    # tsv_file_path = Path(r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI.tsv")
    tsv_file_name = f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_{descriptor_service}.tsv"            
    tsv_file_path = Path(PROJECT_ROOT) / "data" / "datasets" / "RBIODEG external" / tsv_file_name         
        
    df_external = pd.read_csv(tsv_file_path, delimiter='\t')
    df_external = df_external.drop_duplicates(subset=["ID"], keep="first").copy()
    

    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session = getSession()
    df_smiles_subset = _fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    
    
     # if descriptor_set_name == 'WebTEST-default':    
        # model=run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, 
        #             ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
        # folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
        # test_stats = run_test_set(df_external,model, folder,df_smiles_subset)    
        # _summarize_fragrance_results_as_excel(session, folder, dataset_name)
    
    #
    # # for method in ['rf', 'xgb']:        
    # for method in ['rf']:
    # # for method in ['xgb']:
    #     model=run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, descriptor_set_name=descriptor_set_name,
    #                 ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    #
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     _summarize_fragrance_results_as_excel(session, folder, dataset_name)
        
    #
    # # for method in ['reg','knn']:
    # for method in ['rf', 'xgb', 'reg', 'knn']:
    #
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
    #                                    splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = 0.001
    #
    #     params.remove_fragment_descriptors = True
    #     params.remove_acnt_descriptors = True
    #
    #     model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #          params=params, descriptor_set_name=descriptor_set_name, 
    #          ad_measure_model=ad_measure_model, write_to_db=write_to_db,
    #          unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    #
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     _summarize_fragrance_results_as_excel(session, folder, dataset_name)

    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    _summarize_fragrance_results(dataset_name, append_to_models_folder)
    
    # _calculate_fragrance_stats(dataset_name, append_to_models_folder, 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG')



def run_pchem():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'HLC v1 modeling'  # automapped one
    write_to_db = False
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    append_to_models_folder = "_bob"

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK

    # Models to upload:
    # for method in ['rf','xgb']:
    # for method in ['reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK

    # for method in ['rf','xgb', 'knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK
        
    # for method in ['rf','xgb','knn','reg']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     params.run_rfe = False
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def run_biodeg_nite():
    
    dataset_name = 'exp_prop_RBIODEG_NITE_OPPT v1.0'
    append_to_models_folder = ""

    write_to_db = False
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # for method in ['rf','xgb','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, ad_measure_model=ad_measure_model,
    #                 write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK

    #     params = set_hyper_parameters(qsar_method=method, feature_selection=False, descriptor_set_name="WebTEST-default", 
    #                                 splitting_name="RND_REPRESENTATIVE", dataset_name=dataset_name, ad_measure=ad_measure_model)
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db,
    #                 unique_identifier=unique_identifier)  # OK

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    # # for method in ['rf']:                
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK
        
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def test_model_summary_local():
    dataset_name = "KOC v1 modeling"
    unique_identifier = None
    append_to_models_folder = "_bob"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK


def test_load_model_with_external_set():
    unique_identifier = None
    # write_to_db = False
    write_to_db = True
    dataset_name = "KOC v1 modeling"
    user = "murdock.weston"
    append_to_models_folder = "_bob"

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, unique_identifier=unique_identifier, user=user, append_to_models_folder=append_to_models_folder)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier)  # OK
    
    # embedding = ["ALOGP2","nBnz","MATS6v","ATS1p","nDB","Lop","MATS1p"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, unique_identifier=unique_identifier)

    r = Results()
    r.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def run_rifm_rf_models():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG'  # automapped one
    write_to_db = False
    unique_identifier = "test_stat"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    qsar_method = "rf"
    feature_selection = True
    grid = {}
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_rf_testing"

    for i in range(0, 10):
        for descriptor_coefficient in [0.001, 0.006, 0.01, None]:
            # Jump to correct point in run
            if i < 7:
                continue
            if i == 7 and descriptor_coefficient is not None:
                continue
            
            params = ParametersImportance(qsar_method=qsar_method, feature_selection=feature_selection, hyperparameter_grid=grid,
                                            descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                            splitting_name=splitting_name, ad_measure=ad_measure_model)
            
            params.hyperparameter_grid = {
                'estimator__max_features': ['sqrt', 'log2'],
                'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
                'estimator__n_estimators': [10, 100, 250, 500]
                }
            
            params.min_descriptor_count = i * 10
            params.max_descriptor_count = (i + 1) * 10
            params.descriptor_coefficient = descriptor_coefficient  # type: ignore

            logging.info(f"Running iteration {i}:\n\tmin_descriptor_count: {params.min_descriptor_count},\n\tmax_descriptor_count: {params.max_descriptor_count},\n\tdescriptor_coefficient: {params.descriptor_coefficient}")

            run_dataset(dataset_name=dataset_name, qsar_method=qsar_method, feature_selection=feature_selection, ad_measure_model=ad_measure_model,
                        write_to_db=write_to_db, unique_identifier=unique_identifier,
                        append_to_models_folder=append_to_models_folder,
                        params=params)  # OK
    
    r = Results()
    r.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)




    

def run_test_datasets():
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    descriptor_coefficient = 0.002
    
    append_to_models_folder = f"_{descriptor_coefficient}"
    # append_to_models_folder=""
    
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # endpoint_abbrevs = ["BP", "Density", "FP", "MP", "ST", "VP", "WS", "BCF", "LC50DM", "LC50","LD50"]
    endpoint_abbrevs = ["LD50"]
    
    # run = "rf_WebTEST-default_fs=False"
    run = "rf_WebTEST-default_fs=True"
    # stat="RMSE"
    # stat_dict="test_stats"    
    stat = "MAE_Test_inside_AD"
    stat_dict = "test_stats_AD"
    _getStatsFromDatasets(endpoint_abbrevs, append_to_models_folder, run, stat, stat_dict)
    return
    
    for endpoint_abbrev in endpoint_abbrevs:
    
        dataset_name = 'TEST_' + endpoint_abbrev
    
        training_path = (Path(PROJECT_ROOT) / "data" / "datasets_TEST_export" / endpoint_abbrev / f"{endpoint_abbrev}_training_set-2d.csv")
        df_training = pd.read_csv(training_path, delimiter=',')
        df_training.rename(columns={"CAS": "ID", "Tox": "Property"}, inplace=True)
        # print(df_training.head(5))
        # print(df_training.shape)
        
        prediction_path = (Path(PROJECT_ROOT) / "data" / "datasets_TEST_export" / endpoint_abbrev / f"{endpoint_abbrev}_prediction_set-2d.csv")
        df_prediction = pd.read_csv(prediction_path, delimiter=',')
        df_prediction.rename(columns={"CAS": "ID", "Tox": "Property"}, inplace=True)
        # print(df_prediction.head(5))
        # print(df_prediction.shape)
        
        # run_dataset_from_dfs(property_name=endpoint_abbrev, dataset_name=dataset_name, df_training=df_training, df_prediction=df_prediction, dataset_name_ext=None,
        #                      df_external=None, qsar_method='gcm',  cross_validate=False, feature_selection=False, ad_measure_model=ad_measure_model,
        #             unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)
        
        for method in ['rf']:
        # for method in ['rf', 'xgb']:
        # for method in ['reg']:
            
            # run_dataset_from_dfs(property_name=endpoint_abbrev, dataset_name=dataset_name, df_training=df_training, df_prediction=df_prediction, dataset_name_ext=None,
            #                  df_external=None, qsar_method=method,  cross_validate=False, feature_selection=False, ad_measure_model=ad_measure_model,
            #         unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)        
            
            params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
                                          splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
            
            params.descriptor_coefficient = descriptor_coefficient
            run_dataset_from_dfs(property_name=endpoint_abbrev, dataset_name=dataset_name, df_training=df_training, df_prediction=df_prediction, dataset_name_ext=None,
                            df_external=None, qsar_method=params.qsar_method, cross_validate=False, feature_selection=params.feature_selection,
                params=params, ad_measure_model=ad_measure_model, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    
        Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')








def main():
    
    # pass
    
    # find_model_folder()
    
    # run_test_datasets()
    # run_example()
    # run_Koc_knn_ga()
        
    # run_Koc()
    # run_fish_tox()
    
    # run_biodeg_nite()
    
    run_biodeg_rifm()
    # run_biodeg_301F()

    
    # mi=ModelInitializer()
    # model=mi.initModel(1978)
    # run_other_test_set(model, "exp_prop_RBIODEG_301F v2 modeling")


    # _calc_stats_training_cv_fragrances()

    # run_percentage_biodegradation()
    # run_continuous_model_on_test_set()
    # run_RIFM_model_on_ECHA_test_set()
    
    # # _lookAtModelCoefficients(1847)
    # _lookAtModelCoefficients(1878)
    # testCoefficientFromScratch()
            
    # run_pchem()
    
    # These 4 should be able to run for the gcm model
    # run_fish_tox()  # Takes too long to run on my machine? (E.g. started a run at 1:55, errored out at 4:53 because the SQL connection closed automatically)
    # run_fish_tox_2()  # OK
    
    # test_model_summary_local()
    # test_load_model_with_external_set()
    # test_load_model_with_external_set()
    # run_rifm_rf_models()

    # _full_test_mte()


if __name__ == '__main__':
    main()
