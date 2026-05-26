'''
Created on Feb 3, 2026

@author: TMARTI02
'''


from dotenv import load_dotenv
load_dotenv('../../personal.env')

from models.case_studies.run_model_building_db import run_dataset, ParametersGeneticAlgorithm, set_hyper_parameters, Results, ParametersImportance

from util import predict_constants as pc
from model_ws_db_utilities import getEngine, getSession
from models.ModelToExcel import ModelDataObjects, ModelToExcel
import logging
import json
import pandas as pd
from sqlalchemy import text
import os
from pathlib import Path

load_dotenv("../../personal.env")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

def run_example():
    write_to_db = False

    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    append_to_models_folder = "_bob"
    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK


def run_Koc():
    unique_identifier = None
    # write_to_db = True
    write_to_db = False
    # write_to_db=True
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    append_to_models_folder = ""
    # append_to_models_folder = "_v2.0"
    # append_to_models_folder = "_KOC_v2 external"
    

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, unique_identifier=unique_identifier,
                append_to_models_folder=append_to_models_folder)  # OK

    for method in ['rf', 'xgb']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
            ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)  
    #
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
            ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
            append_to_models_folder=append_to_models_folder)  
    
    
    for method in ['reg', 'knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        # params.max_features = 12
        params.max_features = 25
        params.descriptor_coefficient = 0.006
        run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
            params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
            unique_identifier=unique_identifier, 
            append_to_models_folder=append_to_models_folder)  


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
    
    unique_identifier = None
    
    append_to_models_folder = "_bob"
    

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, unique_identifier=unique_identifier,
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
    #        unique_identifier=unique_identifier, 
    #         append_to_models_folder=append_to_models_folder)  
    
        # params.feature_selection = False
        # run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
        #     params = params, ad_measure_model=ad_measure_model,write_to_db=write_to_db, 
        #    unique_identifier=unique_identifier, 
        #     append_to_models_folder=append_to_models_folder)  
        

        # run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
        #     ad_measure_model=ad_measure_model,write_to_db=write_to_db, unique_identifier=unique_identifier, 
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
    
    
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')


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
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # automapped one
    
    write_to_db = True
    unique_identifier = None
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"  
    
    append_to_models_folder = ""
    # append_to_models_folder = "_0.006"

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    #
    # for method in ['rf', 'xgb']:        
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
    #                 ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    

    # for method in ['rf']:
    # for method in ['rf', 'xgb', 'reg','knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.descriptor_coefficient = 0.006
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
    #         unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 

        
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)


def fetch_test_set_qsar_smiles(session, dataset_name, outer_splitting_name ='RND_REPRESENTATIVE', descriptor_set_name='WebTEST-default'):
    
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
          AND s.name = :outerSplittingName and dpis.split_num=1
        ORDER BY dp.id
    """)
    rows = session.execute(
        sql,
        {
            "datasetName": dataset_name,
            "descriptorSetName": descriptor_set_name,
            "outerSplittingName": outer_splitting_name,
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
    test_stats_other_fragrances = calculate_binary_statistics(df_pred_fragrances, 0.5, "_Test")
    # print('Fragrances test chemical stats', json.dumps(test_stats,indent=4))
    # print(df_pred_fragrances.shape[0])
    # print('')

    # print(json.dumps(test_stats_RIFM,indent=4))    
    print(f"{run_folder}\tRIFM_BA_TEST={test_stats_RIFM['BA_Test']:.3f}\tOther_Fragrances_BA_TEST={test_stats_other_fragrances['BA_Test']:.3f}")
    


def run_biodeg_301F():
    
    dataset_name = 'exp_prop_RBIODEG_301F v1 modeling' # automapped one
    write_to_db = True
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    unique_identifier=None 

    # append_to_models_folder = ""
    append_to_models_folder = "_desc_coef_0.001"
    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, 
                ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    
    for method in ['rf', 'xgb']:        
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False, 
                    ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder)  # OK
    
    for method in ['rf', 'xgb', 'reg','knn']:
        params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
                                      splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
        params.descriptor_coefficient = 0.001
        run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
            params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
            unique_identifier=unique_identifier, append_to_models_folder=append_to_models_folder) 

        
    Results.summarize_model_stats(dataset_name, append_to_models_folder=append_to_models_folder)
    
    dataset_name_subset='exp_prop_RBIODEG_RIFM_CHEMREG'
    folder = Path(os.getenv("PROJECT_ROOT")) / "data" / "models" / dataset_name
    
    session = getSession()
    df_smiles_subset = fetch_test_set_qsar_smiles(session, dataset_name_subset)
    
    print('\n\nSubset stats')
    for entry in folder.iterdir():
        if entry.is_dir():
            calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, entry.name)
    
    
def getQsarSmilesFromFragranceSpreadsheet():
    excel_path = Path(os.getenv("PROJECT_ROOT")) / "data" / "models" / "exp_prop_RBIODEG_301F v1 modeling" / "DSSTox_FRAGRANCEBB_20260413_qsar_ready.xlsx"
    df=pd.read_excel(excel_path)
    df = df.rename(columns={'Structure_qsar_ready': 'canon_qsar_smiles'})
    unique_df = df[['canon_qsar_smiles']].dropna().drop_duplicates().reset_index(drop=True)
    # print(unique_df)
    return unique_df

def run_pchem():
    
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID' 
    dataset_name = 'HLC v1 modeling' # automapped one
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
    write_to_db = False
    # write_to_db = True
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
    dataset_name = 'exp_prop_RBIODEG_RIFM_CHEMREG' # automapped one
    write_to_db = False
    unique_identifier = "test_stat"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    qsar_method = "rf"
    feature_selection = True
    grid = {}
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder="_rf_testing"

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
            
            params.min_descriptor_count = i*10
            params.max_descriptor_count = (i+1)*10
            params.descriptor_coefficient = descriptor_coefficient

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
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"binary_external.xlsx")
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/NO EXTERNAL
    model_id = 1065
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/EXTERNAL (external stats not saved in database)
    model_id = 1753
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()


def main():
    
    # run_example()
    # run_Koc_knn_ga()
        
    # run_Koc()
    # run_fish_tox()
    
    # run_biodeg_nite()
    
    # run_biodeg_rifm()
    # run_biodeg_301F()
     
    
    # run_pchem()
    
    # These 4 should be able to run for the gcm model
    # run_Koc()  # OK
    # run_fish_tox()  # Takes too long to run on my machine? (E.g. started a run at 1:55, errored out at 4:53 because the SQL connection closed automatically)
    # run_fish_tox_2()  # OK
    
    # test_create_model()
    # test_model_summary()

    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()
    # test_model_summary_local()
    # test_load_model_with_external_set()
    # run_rifm_rf_models()

    full_test_mte()


if __name__ == "__main__":
    main()
