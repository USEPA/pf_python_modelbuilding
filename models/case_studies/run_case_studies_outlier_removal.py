'''
Created on Sep 29, 2026

@author: TMARTI02
'''

from models.case_studies.run_model_building_db import (
    run_dataset,  Results, set_hyper_parameters,
)


import os
from util import predict_constants as pc

from dotenv import load_dotenv
load_dotenv("../../personal.env")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def run_Bcf_gcm_outlier_testing():
    write_to_db = False
    user = "murdock.weston"
    dataset_name = "exp_prop_BCF_v1_modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_gcm_outlier"

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    outlier_filtering_methods = [
        None,
        ["iqr"],
        ["hampel"],
        ["robust_z"],
        ["esd"],
        ["iqr", "hampel"],
        ["iqr", "robust_z"],
        ["iqr", "esd"],
        ["hampel", "robust_z"],
        ["hampel", "esd"],
        ["robust_z", "esd"],
        ["iqr", "hampel", "robust_z"],
        ["iqr", "hampel", "esd"],
        ["iqr", "robust_z", "esd"],
        ["hampel", "robust_z", "esd"],
        ["iqr", "hampel", "robust_z", "esd"]
    ]

    for filter_set in outlier_filtering_methods:
        if filter_set is None:
            print("Running BCF with GCM method and no outlier filtering")
            run_dataset(
                dataset_name=dataset_name,
                qsar_method="gcm",
                feature_selection=False,
                ad_measure_model=ad_measure_model,
                add_LOGP_Martin=True,
                write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder
            )
        else:
            print(f"Running BCF with GCM method and outlier_filter_methods={filter_set}")
            run_dataset(
                dataset_name=dataset_name,
                qsar_method="gcm",
                feature_selection=False,
                ad_measure_model=ad_measure_model,
                add_LOGP_Martin=True,
                write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder,
                outlier_filter_methods=filter_set
            )

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')

def run_Koc_gcm_outlier_testing():
    write_to_db = False
    user = "murdock.weston"
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_gcm_outlier"

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    outlier_filtering_methods = [
        None,
        ["iqr"],
        ["hampel"],
        ["robust_z"],
        ["esd"],
        ["iqr", "hampel"],
        ["iqr", "robust_z"],
        ["iqr", "esd"],
        ["hampel", "robust_z"],
        ["hamepl", "esd"],
        ["robust_z", "esd"],
        ["iqr", "hampel", "robust_z"],
        ["iqr", "hampel", "esd"],
        ["iqr", "robust_z", "esd"],
        ["hampel", "robust_z", "esd"],
        ["iqr", "hampel", "robust_z", "esd"]
    ]

    for filter_set in outlier_filtering_methods:
        if filter_set is None:
            print("Running KOC with GCM method and no outlier filtering")
            run_dataset(
                dataset_name=dataset_name,
                qsar_method="gcm",
                feature_selection=False,
                ad_measure_model=ad_measure_model,
                add_LOGP_Martin=True,
                write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder
            )
        else:
            print(f"Running KOC with GCM method and outlier_filter_methods={filter_set}")
            run_dataset(
                dataset_name=dataset_name,
                qsar_method="gcm",
                feature_selection=False,
                ad_measure_model=ad_measure_model,
                add_LOGP_Martin=True,
                write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder,
                outlier_filter_methods=filter_set
            )

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')



if __name__ == '__main__':
    run_Koc_gcm_outlier_testing()
    run_Bcf_gcm_outlier_testing()
