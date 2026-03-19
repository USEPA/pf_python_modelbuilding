'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from dotenv import load_dotenv
load_dotenv('../../personal.env')

from models.case_studies.run_model_building_db import run_dataset, Results, ParametersGeneticAlgorithm, \
    ParametersImportance

from util import predict_constants as pc
from model_ws_db_utilities import getEngine, getSession
from models.ModelToExcel import ModelToExcel
import logging
import json
from typing import List, Dict, Any


def format_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3g}"  # compact numeric formatting
    return str(v)


def compute_columns(items: List[Dict[str, Any]],
                    preferred: List[str]=("exp", "Episuite", "gcm", "rf")) -> List[str]:
    all_keys: List[str] = []
    for it in items:
        vals = it.get("values") or {}
        all_keys.extend(list(vals.keys()))
    seen = set()
    uniq = [k for k in all_keys if not (k in seen or seen.add(k))]
    ordered = [c for c in preferred if c in uniq] + [c for c in uniq if c not in preferred]
    return list(ordered) or list(preferred)


def run_example():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False)  # OK


def run_Koc():
    create_unique_excel = False
    write_to_db = False
    # write_to_db=True
    dataset_name = "KOC v1 modeling"

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=False)  # OK
    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
                write_to_db=write_to_db, create_unique_excel=create_unique_excel, create_detailed_excel=True)  # OK

    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    # embedding = ["ALOGP2","nBnz","MATS6v","ATS1p","nDB","Lop","MATS1p"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)

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
    
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    write_to_db = True
    create_unique_excel = False

    # for method in ['rf','xgb', 'reg','knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name="WebTEST-default", 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     params.n_features_to_select = 6
    #     # params.n_features_to_select = 10
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=True, params = params, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK

    # # embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'xv1', 'Mp', 'AMW'] 
    # # embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'Mp', 'AMW']
    # embedding = ['ALOGP', 'ALOGP2', 'MW', 'Mp', 'AMW']
    #
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
    #                            embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)
    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model, write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK

    for method in ['rf', 'xgb', 'reg', 'knn']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
                    ad_measure_model=ad_measure_model, write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    Results.summarize_model_stats(dataset_name)


def test_model_summary():
    engine = getEngine()
    session = getSession()
    model_id = 1737
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
    # test_create_model()
    # test_model_summary()

    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()
    # test_model_summary_local()

