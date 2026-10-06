'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from dotenv import load_dotenv
from models.runGA import descriptor_coefficient
load_dotenv('../../personal.env')

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
from models.ModelToExcel import ModelDataObjects, ModelToExcel
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

import json
import pandas as pd
from sqlalchemy import text
import os
from pathlib import Path
from sqlalchemy.exc import SQLAlchemyError

from openpyxl import load_workbook

import models.db_utilities.dataset_utilities_db as du  
from model_ws_utilities import call_do_predictions_from_df
from io import StringIO
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics

load_dotenv("../../personal.env")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")


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
    write_to_db = True
    # write_to_db = False
    dataset_name = "KOC v1 modeling"
    # dataset_name = "KOC v2 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  

    # append_to_models_folder = ""
    # append_to_models_folder = "_"+str(descriptor_coefficient)
    # append_to_models_folder = "_ad_test"
    # append_to_models_folder = "_v2.0"
    # append_to_models_folder = "_KOC_v2 external"
    append_to_models_folder = "_v3.0"
    descriptor_coefficient = 0.006

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, unique_identifier=unique_identifier,
    #             append_to_models_folder=append_to_models_folder)  # OK
    #
    # for method in ['rf', 'xgb']:
    #
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
    #         ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
    #         append_to_models_folder=append_to_models_folder)  
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
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')
    

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


def predictSetFromDB_SmilesFromExcel(model, smilesCol, excel_file_path, sheetName):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain using smiles in an excel file
        Stores results in tsv file in same folder as excel file
        Runs one at a time since standardizer and descriptors are slow if not cached in mongo (qsar predictions are fast- could aggregate dataframe to run at the end though)
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        from API_Utilities import QsarSmilesAPI, DescriptorsAPI
        from model_ws_db_utilities import ModelInitializer, ModelPredictor
        from model_service_common.config import get_env as _get_env
        STDIZER_API = _get_env('stdizer.url', 'STDIZER_API', default='http://stdizer-api:8200/api/stdizer')
        DESCRIPTORS_API = _get_env('descriptors.url', 'DESCRIPTORS_API', default='http://descriptors-api:8804/api/descriptors')
    
        print(DESCRIPTORS_API)
        descriptorAPI = DescriptorsAPI()

        mp = ModelPredictor()

        # initialize model bytes and all details from db:

        df = pd.read_excel(excel_file_path, sheet_name=sheetName)
        smiles_list = df[smilesCol].tolist()  # Extract the 'Smiles' column into a list

        directory = os.path.dirname(excel_file_path)

        # Create a text file path in the same directory
        text_file_path = os.path.join(directory, "output.txt")
        print(text_file_path)

        with open(text_file_path, 'w') as file:
            file.write("smiles\tqsarSmiles\tpred_value\tpred_AD\n")

            # for smiles, predOld in zip(smiles_list, pred_list):
            for smiles in smiles_list:
                chemical, code = mp.standardizeStructure(STDIZER_API, smiles, model)

                qsarSmiles = chemical["canonicalSmiles"]

                if code != 200:
                    print(smiles, qsarSmiles)
                    file.write(smiles + "\terror smiles")
                    continue

                if model.descriptorSetName == 'WebTEST-default':
                    descriptorSet = 'webtest'
                else:
                    print('couldnt assign descriptorSet for descriptor API')
                    return

                df_prediction, code = descriptorAPI.calculate_descriptors(DESCRIPTORS_API, qsarSmiles, descriptorSet)
                if code != 200:
                    print(smiles, 'error descriptors')
                    file.write(smiles + "\terror descriptors\n")

                    continue

                import model_ws_utilities as mwu

                pred_results = json.loads(mwu.call_do_predictions_from_df(df_prediction, model))
                pred_value = pred_results[0]['pred']

                line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\n"

                # ad_results = mp.determineApplicabilityDomain(model, model.applicabilityDomainName, df_prediction)
                # pred_AD = ad_results["AD"]

                # line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\t" + str(pred_AD) + "\n"
                print(line)
                file.write(line)
                file.flush()

        return "OK", 200

    
def run_biodeg_rifm():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # old one from january 26
    dataset_name = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'  # RBIODEG no 10 day window
    # dataset_name = 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG' # RBIODEG with 10 day window
    # write_to_db = True
    write_to_db = False
    
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    # append_to_models_folder = ""
    append_to_models_folder = "_v3.0"
    # append_to_models_folder = "_0.001"


    if dataset_name == 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG': 
        tsv_file_path = Path(r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI.tsv")
    elif dataset_name == 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG':
        tsv_file_path = Path(r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day.tsv")
    else:
        print(f"handle external set name for dataset={dataset_name}")
        return


    df_external = pd.read_csv(tsv_file_path, delimiter='\t')
    df_external = df_external.drop_duplicates(subset=["ID"], keep="first").copy()
            
    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session = getSession()
    df_smiles_subset = fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    # print(df_smiles_subset)
    
    
    # print('gcm',test_stats)
    
    # # for method in ['rf', 'xgb']:        
    # for method in ['rf']:
    #     model=run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    #
    #     subfolder=f'{method}_WebTEST-default_fs=False'    
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     summarize_fragrance_results_as_excel(session, folder, dataset_name)
    # #
    # # # for method in ['reg','knn']:
    # for method in ['rf', 'xgb', 'reg', 'knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
    #                                    splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = 0.001
    #
    #     params.remove_fragment_descriptors = True
    #     params.remove_acnt_descriptors = True
    #     params.run_rfe = False
    #
    #     model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #          params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
    #          unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    #
    #     subfolder = f'{method}_WebTEST-default_fs=True'    
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     summarize_fragrance_results_as_excel(session, folder, dataset_name)

    
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    
    # dataset_name_subset='exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    # folder = Path(os.getenv("PROJECT_ROOT")) / "data" / ("models"+ append_to_models_folder) / dataset_name
    # session = getSession()
    # df_smiles_subset = fetch_test_set_qsar_smiles(session, dataset_name_subset)
    # print('\n\nSubset stats')
    # for entry in folder.iterdir():
    #     if entry.is_dir():
    #         calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, entry.name)
    
    # summarize_fragrance_results(dataset_name, append_to_models_folder)
    
    
    # TODO determine how RIFM only models work for test set of ECHA+RIFM set
def summarize_fragrance_results(dataset_name, append_to_models_folder):
    folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name

    print("model\tNtest\tBA_Test\tNexternal\tBA_External\tN_both\tBA_both")

    for item in folder.iterdir():
        if item.is_file():
            continue
        
        test_stats = None
        external_stats = None
        test_and_external_stats = None

        for json_file in item.rglob("*.json"):
            if json_file.name == "RIFM test set statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    test_stats = json.load(f)

            elif json_file.name == "external set statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    external_stats = json.load(f)

            # elif json_file.name == "test and external sets statistics2.json":
            elif json_file.name == "test and external sets statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    test_and_external_stats = json.load(f)

        n_test = test_stats["N"] if test_stats else "NA"
        ba_test = f"{test_stats['BA_Test']:.3f}" if test_stats and "BA_Test" in test_stats else "NA"

        n_external = external_stats["N"] if external_stats else "NA"
        ba_external = f"{external_stats['BA_Test']:.3f}" if external_stats and "BA_Test" in external_stats else "NA"

        n_both = test_and_external_stats["N"] if test_and_external_stats else "NA"
        ba_both = f"{test_and_external_stats['BA_Test']:.3f}" if test_and_external_stats and "BA_Test" in test_and_external_stats else "NA"

        print(f"{item.name}\t{n_test}\t{ba_test}\t{n_external}\t{ba_external}\t{n_both}\t{ba_both}")


def get_smiles_for_training_set(session, dataset_name):
    """
    Gets the qsar smiles in the training set for dataset_name
    """
    sql = text("""
        SELECT dp.canon_qsar_smiles
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp
          ON dp.fk_dataset_id = d.id

        join qsar_datasets.data_points_in_splittings dpis on dpis.fk_data_point_id =dp.id
        WHERE d.name = :dataset_name
        and dpis.split_num =0 and dpis.fk_splitting_id =1;
    """)

    result = session.execute(sql, {"dataset_name": dataset_name})
    rows = result.fetchall()

    return pd.DataFrame(rows, columns=["canon_qsar_smiles"])


def recalc_stats(session, dataset_name, subset, append_to_models_folder, dataset_to_exclude):
    
    folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset

    smiles_to_exclude = get_smiles_for_training_set(session, dataset_to_exclude)
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())
    
    file_path_rifm = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset / "RIFM test set predictions.tsv"

    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm = df_test_set_rifm[~df_test_set_rifm["id"].isin(exclude_set)]
        # print('df_test_set_rifm',df_test_set_rifm)

    file_path_external = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset / "external set predictions.tsv"

    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external = df_external[~df_external["id"].isin(exclude_set)]
        # print('df_test_set_external',df_test_set_external)

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    external_stats["N"] = df_external.shape[0]

    filePathOutJson = os.path.join(folder, f"external set statistics exclude {dataset_to_exclude}.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
    
    df_both = pd.concat([df_test_set_rifm, df_external], ignore_index=True).drop_duplicates(subset=["id"])    
    # print(df_both)
    
    both_stats = calculate_binary_statistics(df_both, 0.5, "_Test")
    both_stats["N"] = df_both.shape[0]
    filePathOutJson = os.path.join(folder, f"test and external sets statistics2.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(both_stats, f, indent=4)


def recalc_stats2(session, folder, dataset_to_exclude):
    """
    Recalculates pooled statistics in test set and external set to exclude chemicals in the training set. This method isnt needed if the external set is excluded from the overall set prior to dataset splitting
    """
    smiles_to_exclude = get_smiles_for_training_set(session, dataset_to_exclude)
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())
    
    file_path_rifm = folder / "RIFM test set predictions.tsv"

    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm = df_test_set_rifm[~df_test_set_rifm["id"].isin(exclude_set)]
        # print('df_test_set_rifm',df_test_set_rifm)

    file_path_external = folder / "external set predictions.tsv"

    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external = df_external[~df_external["id"].isin(exclude_set)]
        # print('df_test_set_external',df_test_set_external)

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    external_stats["N"] = df_external.shape[0]

    filePathOutJson = os.path.join(folder, f"external set statistics exclude {dataset_to_exclude}.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
    
    df_both = pd.concat([df_test_set_rifm, df_external], ignore_index=True).drop_duplicates(subset=["id"])    
    # print(df_both)
    
    both_stats = calculate_binary_statistics(df_both, 0.5, "_Test")
    both_stats["N"] = df_both.shape[0]
    filePathOutJson = os.path.join(folder, f"test and external sets statistics2.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(both_stats, f, indent=4)        


def summarize_fragrance_results_as_excel(session, folder, dataset_to_exclude):
    smiles_to_exclude = get_smiles_for_training_set(session, dataset_to_exclude)
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())

    frames = []

    file_path_rifm = folder / "RIFM test set predictions.tsv"
    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm["set"] = "test"
        df_test_set_rifm.loc[df_test_set_rifm["id"].isin(exclude_set), "set"] = "training"
        frames.append(df_test_set_rifm)

    file_path_external = folder / "external set predictions.tsv"
    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external["set"] = "external"
        df_external.loc[df_external["id"].isin(exclude_set), "set"] = "training"
        frames.append(df_external)

    if not frames:
        raise FileNotFoundError("No input prediction files were found.")

    df_all = pd.concat(frames, ignore_index=True)

    # Optional: drop duplicates by id, keeping the first occurrence
    # If the same id appears in both files, the first one in concat order wins.
    df_all = df_all.drop_duplicates(subset=["id"], keep="first")

    # Write a single-tab spreadsheet
    output_path = folder / f"fragrance_results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="results", index=False)

    return df_all

    
    # print(smiles_to_exclude)
    
# RIFM test set predictions.tsv
    
    # print('model\tNtest\tBA_Test\tNexternal\tBA_External\tN_both\tBA_both')    
    #
    #
    #
    # for item in folder.iterdir():
    #     if item.is_file():
    #         continue
    #
    #     # print(item.name)
    #
    #
    #
    #     for json_file in item.rglob("*.json"):
    #
    #         if(json_file.name=='RIFM test set statistics.json'):
    #             with open(json_file, "r", encoding="utf-8") as f:
    #                 test_stats = json.load(f)
    #             # print(test_stats)
    #
    #         if(json_file.name=='external set statistics.json'):
    #             with open(json_file, "r", encoding="utf-8") as f:
    #                 external_stats = json.load(f)
    #             # print(external_stats)
    #
    #
    #         if(json_file.name=='test and external sets statistics.json'):
    #             with open(json_file, "r", encoding="utf-8") as f:
    #                 test_and_external_stats = json.load(f)
    #             # print(test_and_external_stats)
    #
    #     print(f"{item.name}\t{test_stats['N']}\t{test_stats['BA_Test']:.3f}\t{external_stats['N']}\t{external_stats['BA_Test']:.3f}\t{test_and_external_stats['N']}\t{test_and_external_stats['BA_Test']:.3f}")


def run_RIFM_model_on_ECHA_test_set():
    
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG'
    
    # dataset_name_ECHA_RIFM = 'exp_prop_RBIODEG_301F v1 modeling'
    dataset_name_ECHA_RIFM = 'exp_prop_RBIODEG_301F v2 modeling'
    
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    _, df_prediction = du.get_training_prediction_instances(getSession(), dataset_name_ECHA_RIFM, descriptor_set_name, splitting_name)

    session = getSession()
    
    model_ids = get_model_ids(session, dataset_name=dataset_name)
    
    from model_ws_db_utilities import ModelInitializer
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

    
def lookAtModelCoefficients(model_id):
    
    # model_id = 1847 #GCM RBIODEG ECHA+RIFM
    # model_id = 1843   #REG RBIODEG ECHA+RIFM
    # model_id = 1763 #GCM KOC
    
    # model_id = 1877  #GCM RBIODEG RIFM, redone
    
    from model_ws_db_utilities import ModelInitializer
    
    mi = ModelInitializer()
    model = mi.initModel(model_id)

    if model is not None:
        y = model.df_training[model.df_training.columns[1]]
        X = model.df_training[model.embedding]

        modelCoefficients = json.loads(model.getOriginalRegressionCoefficients2(X, y))
        
        print(json.dumps(modelCoefficients, indent=4))


def testCoefficientFromScratch():

    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from models.ModelBuilder import Model
    
    X, y = make_classification(n_samples=500, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # ("clf", LogisticRegression(max_iter=1000))
        # ("clf", LogisticRegression(penalty="l2", solver="liblinear", random_state=0))
        ("clf", LogisticRegression(solver='liblinear', max_iter=1000, dual=False))        
    ]).fit(df, y)
    
    # pipe = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", LogisticRegression(random_state=0))  # doesnt give std_errors
    # ]).fit(df, y)
    
    self_like = type("T", (), {})()
    self_like.get_model = lambda: pipe  # type: ignore
    self_like.embedding = list(df.columns)  # type: ignore
    
    print(Model.getOriginalRegressionCoefficients2(self_like, df, y))  # type: ignore
    

def run_continuous_model_on_test_set():
    
    # dataset_name= 'exp_prop_PERCENT_BIODEGRADATION_RIFM_CHEMREG'
    # dataset_name_test = 'exp_prop_RBIODEG_301F v1 modeling'

    dataset_name = 'exp_prop_PERCENT_BIODEGRADATION_301F v1 modeling'
    dataset_name_test = 'exp_prop_RBIODEG_RIFM_CHEMREG'
    
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    _, df_prediction = du.get_training_prediction_instances(getSession(), dataset_name_test, descriptor_set_name, splitting_name)

    session = getSession()
    
    model_ids = get_model_ids(session, dataset_name=dataset_name)
    
    from model_ws_db_utilities import ModelInitializer
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
    

def get_model_ids(session, dataset_name: str):
    """
    Return a list of model IDs for the given dataset name.
    """
    sql = text("SELECT id FROM qsar_models.models WHERE dataset_name = :dataset_name")
    try:
        result = session.execute(sql, {"dataset_name": dataset_name})
        # Extract the single selected column as a list
        return result.scalars().all()
    except SQLAlchemyError:
        logging.exception("Failed to fetch model ids for dataset_name=%r", dataset_name)
        return []    


def calculate_stats_for_binary_test_set(df_prediction, model, log_path="binary_test_stats.log"):
    json_predictions = call_do_predictions_from_df(df_prediction, model)
    df_predictions_test = pd.read_json(StringIO(json_predictions), orient="records")
    df_predictions_test["pred"] = (df_predictions_test["pred"] >= 60).astype(int)

    test_stats = calculate_binary_statistics(df_predictions_test, 0.5, "_Test")

    # Compose the same line as printed
    line = f"{model.subfolder}\t{test_stats['BA_Test']:.3f}\n"
    print(line.rstrip("\n"))

    # Ensure directory exists (if a path with directories was provided)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Append to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

    return test_stats


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
    calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
    
    for method in ['rf', 'xgb']: 
        model = run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
                    ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
        calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
                
    for method in ['rf', 'xgb', 'reg', 'knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        if params is not None:
            if not isinstance(params, ParametersGroupContribution) and not isinstance(params, ParametersGeneric):
                params.descriptor_coefficient = 0.001  # type: ignore
            model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
                params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
            calculate_stats_for_binary_test_set(df_prediction_binary, model, log_path)
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def fetch_set_qsar_smiles(session, dataset_name, split_num, outer_splitting_name='RND_REPRESENTATIVE', descriptor_set_name='WebTEST-default'):
    
    
    """
    Only includes rows present in the given descriptor set.
    """
    sql = text("""
        SELECT dp.canon_qsar_smiles
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
        JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
        JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
        JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
        JOIN qsar_datasets.splittings s ON s.id = dpis.fk_splitting_id
        WHERE d.name = :datasetName
          AND ds.name = :descriptorSetName
          AND s.name = :outerSplittingName and dpis.split_num=:split_num
        ORDER BY dp.id
    """)
    rows = session.execute(
        sql,
        {
            "datasetName": dataset_name,
            "descriptorSetName": descriptor_set_name,
            "outerSplittingName": outer_splitting_name,
            "split_num": split_num,
        },
    ).fetchall()

    if not rows:
        raise ValueError(
            f"No rows found for dataset='{dataset_name}', descriptorSet='{descriptor_set_name}', "
            f"outerSplitting='{outer_splitting_name}'."
        )

    smiles_list = [r[0] for r in rows]
    return pd.DataFrame({"canon_qsar_smiles": smiles_list})


def calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, run_folder):
    import os
    from StatsCalculator import calculate_binary_statistics
    
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    path_segments = [PROJECT_ROOT, "data", "models" + append_to_models_folder, dataset_name, run_folder, "test set predictions.csv"]
    output_csv_path = os.path.join(*path_segments)
    df_pred = pd.read_csv(output_csv_path)
    
    df_pred_in_rifm_test = df_pred.merge(
        df_smiles_subset[["canon_qsar_smiles"]].drop_duplicates(),
        on="canon_qsar_smiles",
        how="inner")
    test_stats_RIFM = calculate_binary_statistics(df_pred_in_rifm_test, 0.5, "_Test")
    
    df_fragrances_list = getQsarSmilesFromFragranceSpreadsheet()
    df_fragrances_list_only = df_fragrances_list[~df_fragrances_list['canon_qsar_smiles'].isin(df_smiles_subset['canon_qsar_smiles'])]
    
    df_pred_fragrances = df_pred.merge(
        df_fragrances_list_only[["canon_qsar_smiles"]].drop_duplicates(),
        on="canon_qsar_smiles",
        how="inner")
    
    print(df_pred_fragrances)
    frac_exp_1 = (df_pred_fragrances["exp"] == 1).mean()
    print(f"Fraction of rows where exp = 1: {frac_exp_1:.3f}")
    
    test_stats_other_fragrances = calculate_binary_statistics(df_pred_fragrances, 0.5, "_Test")
    # print('Fragrances test chemical stats', json.dumps(test_stats,indent=4))
    # print(df_pred_fragrances.shape[0])
    # print('')

    # print(json.dumps(test_stats_RIFM,indent=4))    
    print(f"{run_folder}\tRIFM_BA_TEST={test_stats_RIFM['BA_Test']:.3f}\tn={df_pred_in_rifm_test.shape[0]}\tOther_Fragrances_BA_TEST={test_stats_other_fragrances['BA_Test']:.3f}\tn={df_pred_fragrances.shape[0]}")


def run_other_test_set(model, descriptor_set_name, splitting_name, dataset_name_other):

    session=getSession()
    from models.db_utilities.dataset_utilities_db import get_training_prediction_instances
    _, df_prediction = get_training_prediction_instances(session, dataset_name_other, descriptor_set_name, splitting_name)
    # print(df_prediction.shape)
    
    json_predictions = call_do_predictions_from_df(df_prediction, model)
    df_preds = pd.read_json(StringIO(json_predictions), orient="records")
    
    # TODO add check for is_binary and generate continuous or binary stats

    stats = calculate_binary_statistics(df_preds, 0.5, "_Test")
    print(stats)


def run_other_test_set2(model_id, dataset_name_other):

    session=getSession()
    mi=ModelInitializer()
    model=mi.init_model(model_id)
    
    descriptor_set_name = "WebTEST-default"#TODO should be in model_details
    splitting_name = "RND_REPRESENTATIVE"#TODO should be in model details
    
    from models.db_utilities.dataset_utilities_db import get_training_prediction_instances
    _, df_prediction = get_training_prediction_instances(session, dataset_name_other, descriptor_set_name, splitting_name)
    # print(df_prediction.shape)
    
    json_predictions = call_do_predictions_from_df(df_prediction, model)
    df_preds = pd.read_json(StringIO(json_predictions), orient="records")
    
    # TODO add check for is_binary and generate continuous or binary stats

    stats = calculate_binary_statistics(df_preds, 0.5, "_Test")
    print(stats)

def run_test_set(df_external, model, folder_path, df_smiles_subset=None):
    '''
    :param df_external: dataframe for external set
    :param model: the model used to run external set
    :param folder_path: model file folder
    :param df_smiles_subset: smiles that appear in RIFM test set (fragrances)
    '''
    
    json_predictions = call_do_predictions_from_df(df_external, model)
    df_external = pd.read_json(StringIO(json_predictions), orient="records")
    
    # TODO add check for is_binary and generate continuous or binary stats

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    filePathOutTsv = os.path.join(folder_path, "external set predictions.tsv")    
    df_external.to_csv(filePathOutTsv, index=False, sep='\t')
    
    filePathOutJson = os.path.join(folder_path, "external set statistics.json")    
    external_stats["N"] = df_external.shape[0]
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
        
    filePathTestSet = os.path.join(folder_path, "test set predictions.csv")
    df_test = pd.read_csv(filePathTestSet)
    df_test = df_test[["canon_qsar_smiles", "exp", "pred"]].copy()
    df_test = df_test.rename(columns={"canon_qsar_smiles": "id"})
    
    if df_smiles_subset is not None:
        # print(df_smiles_subset)
        # print(df_test)
        df_test = df_test.merge(
            df_smiles_subset[["canon_qsar_smiles"]].drop_duplicates().rename(
                columns={"canon_qsar_smiles": "id"}
            ),
            on="id",
            how="inner"
        )
        
    filePathOutTsv = os.path.join(folder_path, "RIFM test set predictions.tsv")
    df_test.to_csv(filePathOutTsv, sep='\t', index=False)

    test_stats = calculate_binary_statistics(df_test, 0.5, "_Test")
    test_stats["N"] = df_test.shape[0]

    filePathOutJson = os.path.join(folder_path, "RIFM test set statistics.json")
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=4)

    df_all = pd.concat([df_test, df_external], ignore_index=True)
    # print(df_all)
    
    test_stats_all = calculate_binary_statistics(df_all, 0.5, "_Test")
    
    test_stats_all["N"] = df_all.shape[0]

    filePathOutJson = os.path.join(folder_path, "test and external sets statistics.json")    
    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(test_stats_all, f, indent=4)
    
    return external_stats


def run_biodeg_301F():
    
    dataset_name = 'exp_prop_RBIODEG_301F v2 modeling'  # automapped one

    write_to_db = True
    # write_to_db = False
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
   # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean]

    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    unique_identifier = None 

    
    # append_to_models_folder = ""
    # append_to_models_folder = "_0.001"
    append_to_models_folder = "_v3.0"
    # append_to_models_folder = "_v3.0_0.006"

    tsv_file_path = Path(r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI.tsv")
    df_external = pd.read_csv(tsv_file_path, delimiter='\t')
    df_external = df_external.drop_duplicates(subset=["ID"], keep="first").copy()
    

    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session = getSession()
    df_smiles_subset = fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    
    # model=run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, 
    #             ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    # folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
    # test_stats = run_test_set(df_external,model, folder,df_smiles_subset)    
    # summarize_fragrance_results_as_excel(session, folder, dataset_name)
    #
    # # print('gcm',test_stats)
    #
    # # for method in ['rf', 'xgb']:        
    # for method in ['rf']:
    #     model=run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    #
    #     subfolder=f'{method}_WebTEST-default_fs=False'    
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     summarize_fragrance_results_as_excel(session, folder, dataset_name)
    #
    # # for method in ['reg','knn']:
    # for method in ['rf', 'xgb', 'reg', 'knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
    #                                    splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = 0.001
    #
    #     params.remove_fragment_descriptors = True
    #     params.remove_acnt_descriptors = True
    #
    #     model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #          params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
    #          unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    #
    #     subfolder = f'{method}_WebTEST-default_fs=True'    
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     summarize_fragrance_results_as_excel(session, folder, dataset_name)

    
    # Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    
    # dataset_name_subset='exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    # folder = Path(os.getenv("PROJECT_ROOT")) / "data" / ("models"+ append_to_models_folder) / dataset_name
    # session = getSession()
    # df_smiles_subset = fetch_test_set_qsar_smiles(session, dataset_name_subset)
    # print('\n\nSubset stats')
    # for entry in folder.iterdir():
    #     if entry.is_dir():
    #         calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, entry.name)
    
    # summarize_fragrance_results(dataset_name, append_to_models_folder)


def run_biodeg_301F_other_descriptors():
    
    dataset_name = 'exp_prop_RBIODEG_301F v2 modeling'  # automapped one

    # write_to_db = True
    write_to_db = False
    descriptor_set_name = "PaDEL-default"
    # descriptor_set_name = "Mordred-default"
    # descriptor_set_name = "RDKit-default"
    descriptor_service = descriptor_set_name.replace("-default", "").lower()
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean] #dont have WebTEST fragments
    splitting_name = "RND_REPRESENTATIVE"
    unique_identifier = None 

    
    # append_to_models_folder = ""
    # append_to_models_folder = "_0.001"
    append_to_models_folder = "_v3.0"
    # append_to_models_folder = "_v3.0_0.006"

    folder = Path(r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files")
    tsv_file_path = folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_{descriptor_service}.tsv"    
    
    df_external = pd.read_csv(tsv_file_path, delimiter='\t')
    df_external = df_external.drop_duplicates(subset=["ID"], keep="first").copy()

    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session = getSession()
    df_smiles_subset = fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    
    # # for method in ['rf', 'xgb']:        
    for method in ['rf']:
        
        params = set_hyper_parameters(qsar_method=method, feature_selection=False, descriptor_set_name=descriptor_set_name,
                                       splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)

        
        model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
             params=params, descriptor_set_name=params.descriptor_set_name,  ad_measure_model=ad_measure_model, write_to_db=write_to_db,
             unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    
        folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / model.subfolder
        test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
        summarize_fragrance_results_as_excel(session, folder, dataset_name)
    
    # # for method in ['reg','knn']:
    # for method in ['rf', 'xgb', 'reg', 'knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name,
    #                                    splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = 0.001
    #
    #     params.remove_fragment_descriptors = True
    #     params.remove_acnt_descriptors = True
    #
    #     model = run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #          params=params, ad_measure_model=ad_measure_model, write_to_db=write_to_db,
    #          unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 
    #
    #     subfolder = f'{method}_WebTEST-default_fs=True'    
    #     folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subfolder
    #     test_stats = run_test_set(df_external, model, folder, df_smiles_subset)    
    #     summarize_fragrance_results_as_excel(session, folder, dataset_name)

    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    
    # dataset_name_subset='exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    # folder = Path(os.getenv("PROJECT_ROOT")) / "data" / ("models"+ append_to_models_folder) / dataset_name
    # session = getSession()
    # df_smiles_subset = fetch_test_set_qsar_smiles(session, dataset_name_subset)
    # print('\n\nSubset stats')
    # for entry in folder.iterdir():
    #     if entry.is_dir():
    #         calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, entry.name)
    
    summarize_fragrance_results(dataset_name, append_to_models_folder)
        
    
def getQsarSmilesFromFragranceSpreadsheet():
    excel_path = Path(os.getenv("PROJECT_ROOT")) / "data" / "models" / "exp_prop_RBIODEG_301F v1 modeling" / "DSSTox_FRAGRANCEBB_20260413_qsar_ready.xlsx"  # type: ignore
    df = pd.read_excel(excel_path)
    df = df.rename(columns={'Structure_qsar_ready': 'canon_qsar_smiles'})
    unique_df = df[['canon_qsar_smiles']].dropna().drop_duplicates().reset_index(drop=True)
    # print(unique_df)
    return unique_df


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


def full_test_mte():
    # Need to test local and database models
    # Need to test binary and continuous models
    # Need to test models with and without an external set

    write_to_db = False
    append_to_models_folder = "_mte_testing"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # LOCAL/BINARY/NO EXTERNAL
    dataset_name = "exp_prop_RBIODEG_301F v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/BINARY/EXTERNAL
    dataset_name = "exp_prop_RBIODEG_RIFM_CHEMREG"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/CONTINUOUS/NO EXTERNAL
    dataset_name = "HLC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/CONTINUOUS/EXTERNAL
    dataset_name = "KOC v1 modeling"    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK
    
    # DATABASE/BINARY/NO EXTERNAL
    # Model id no longer exists in database
    # model_id = 1567
    # file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")
    # mdo = ModelDataObjects(model_id=model_id)
    # mte = ModelToExcel(mdo, file_path)
    # mte.create_excel()

    # DATABASE/BINARY/EXTERNAL
    model_id = 1831
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"binary_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/NO EXTERNAL
    model_id = 1065
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/EXTERNAL (external stats not saved in database)
    model_id = 1753
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()


def getStatsFromDatasets(endpoint_abbrevs, append_to_models_folder, run, stat, stat_dict):
    for endpoint_abbrev in endpoint_abbrevs:
        results_path = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / f"TEST_{endpoint_abbrev}" / run / "results.json"
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
                model_statistics = results.get("model_statistics", {})
                test_val = model_statistics.get(stat_dict).get(stat)
                test_coverage = model_statistics.get(stat_dict).get("Coverage_Test")
                print(f"{endpoint_abbrev}\t{test_val:.3f}\t{test_coverage:.3f}")
        else:
            print(endpoint_abbrev, "missing")
    

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
    getStatsFromDatasets(endpoint_abbrevs, append_to_models_folder, run, stat, stat_dict)
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


def find_model_folder():
    
    # dataset_name = 'KOC v1 modeling'
    dataset_name = 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG'
        
    data_folder = Path(PROJECT_ROOT) / "data"
    
    for folder in data_folder.iterdir():
        
        if folder.is_dir() and "models" in folder.name:
            
            koc_folder = folder / dataset_name
            
            if koc_folder.is_dir():
                
                for folder2 in koc_folder.iterdir():
                    
                    if folder2.is_dir():
                        
                        # results_file = folder2 / "results.json"
                        #
                        # if results_file.is_file():
                        #     with results_file.open("r", encoding="utf-8") as f:
                        #         results = json.load(f)
                        #     # print(results)
                        #     params = results["params"]
                        #     model_details = results["model_details"]
                        #     print(folder.name, params['qsar_method'], model_details["modelId"])
                        # else:
                        #     print("results.json not found")
                            
                        results_file = folder2 / "detailed_summary.xlsx"
                        
                        # print(results_file)
                                                
                        if results_file.is_file():
                            wb = load_workbook(results_file, data_only=True)
                            ws = wb["Summary"]
                            print(folder.name, folder2.name, ws["B1"].value)
                            
                        else:
                            # print("detailed_summary.xlsx not found")
                            pass
                        
                print('\n')


def calc_stats_training_cv_fragrances():
    
    model_ids = [1964, 1965, 1966, 1967, 1968, 1976]
    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session=getSession()
    df_smiles_training_fragrances = fetch_set_qsar_smiles(session, dataset_name_subset, 0)
    
    for model_id in model_ids:
        mi=ModelInitializer()
        model=mi.init_model(model_id)
        # Filter model training CV predictions to only those in the subset smiles list
        subset_smiles_training_fragrances = set(df_smiles_training_fragrances["canon_qsar_smiles"].dropna().astype(str))
        
        df_preds_training_cv_fragrances = model.df_preds_training_cv[
            model.df_preds_training_cv["id"].isin(subset_smiles_training_fragrances)
        ].copy()

        stats_training_cv_fragrances = calculate_binary_statistics(df_preds_training_cv_fragrances, 0.5, "_Training_CV")
        print(model_id, stats_training_cv_fragrances["BA_Training_CV"],df_preds_training_cv_fragrances.shape[0])



def main():
    
    # pass
    
    # find_model_folder()
    
    # run_test_datasets()
    # run_example()
    # run_Koc_knn_ga()
        
    # run_Koc()
    
    # run_BCF()
    
    # run_biodeg_nite()
    
    # run_biodeg_rifm()
    # run_biodeg_301F()
    run_biodeg_301F_other_descriptors()
    # calc_stats_training_cv_fragrances()

    # run_percentage_biodegradation()
    # run_continuous_model_on_test_set()
    # run_RIFM_model_on_ECHA_test_set()
    
    # lookAtModelCoefficients(1847)
    # lookAtModelCoefficients(1878)
    # testCoefficientFromScratch()
            
    # run_pchem()
    
    # These 4 should be able to run for the gcm model
    # run_fish_tox()  # Takes too long to run on my machine? (E.g. started a run at 1:55, errored out at 4:53 because the SQL connection closed automatically)
    # run_fish_tox_2()  # OK
    
    # test_model_summary_local()
    # test_load_model_with_external_set()
    # run_rifm_rf_models()

    # full_test_mte()


if __name__ == '__main__':
    main()
