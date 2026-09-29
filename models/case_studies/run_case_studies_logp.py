'''
Created on Sep 28, 2026

@author: TMARTI02
'''
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import pickle

from sklearn.metrics import r2_score


from models.db_utilities.dataset_utilities_db import getLogKowPredictionsForDataset, getBcfPredictionsForDataset
from util import predict_constants as pc
from models.ModelToExcel import ModelDataObjects, ModelToExcel
from models.case_studies.run_model_building_db import (
    run_dataset,  Results, set_hyper_parameters,
)


import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv("../../personal.env")
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)


def compare_all_logp_descriptors_to_koc(save_plots=False):
    logp_descriptors = ["ALOGP", "ALOGP2", "XLOGP", "XLOGP2", "LOGP_Martin", "LOGP_Martin2"]
    for descriptor in logp_descriptors:
        compare_logp_descriptors_to_koc(descriptor, save_plot=save_plots)


def compare_logp_descriptors_to_bcf(logp_descriptor, save_plot=True):
    df_training, df_prediction = getBcfPredictionsForDataset()
    df = pd.concat([df_training, df_prediction], ignore_index=True)

    x = df[logp_descriptor]
    y = df.Property

    r2 = r2_score(x, y)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, color="blue", label=f"BCF vs. {logp_descriptor}")

    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    # plt.plot([min_val, max_val], [min_val, max_val], color="red", label="y = x")
    plt.xlabel(logp_descriptor)
    plt.ylabel("BCF")
    plt.title(f"R² = {r2:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        folder_path = os.path.join(PROJECT_ROOT, "data", "logp_vs_bcf")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(folder_path, f"{logp_descriptor}_vs_bcf.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def run_Koc_gcm_alogp():
    write_to_db = False
    user = "murdock.weston"
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_gcm_logp"
    logp_columns = ["ALOGP", "ALOGP2"]

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    run_dataset(
        dataset_name=dataset_name,
        qsar_method="gcm",
        feature_selection=False,
        ad_measure_model=ad_measure_model,
        add_LOGP_Martin=True,
        logp_columns=logp_columns,
        write_to_db=write_to_db,
        append_to_models_folder=append_to_models_folder
    )

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')


def run_Koc_gcm_xlogp():
    write_to_db = False
    user = "murdock.weston"
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_gcm_logp"
    logp_columns = ["XLOGP", "XLOGP2"]

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    run_dataset(
        dataset_name=dataset_name,
        qsar_method="gcm",
        feature_selection=False,
        ad_measure_model=ad_measure_model,
        add_LOGP_Martin=True,
        logp_columns=logp_columns,
        write_to_db=write_to_db,
        append_to_models_folder=append_to_models_folder
    )

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE')
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ')


def run_Koc_logp_custom(logp_columns: list[str], qsar_method: str="gcm", write_to_db: bool=False, params: dict=None, hyperparameters: dict=None, subfolder: str=None):
    user = "murdock.weston"
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    append_to_models_folder = "_gcm_logp"

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    if isinstance(logp_columns, str):
        logp_columns = [logp_columns]

    if params is None:
        params = set_hyper_parameters(
            qsar_method=qsar_method,
            feature_selection=True,
            descriptor_set_name=descriptor_set_name,
            splitting_name=splitting_name,
            dataset_name=dataset_name,
            ad_measure=ad_measure_model
        )

    if hyperparameters is not None:
            for key, value in hyperparameters.items():
                params.hyperparameter_grid[key] = value

    run_dataset(
        dataset_name=params.dataset_name,
        qsar_method=params.qsar_method,
        feature_selection=params.feature_selection,
        ad_measure_model=ad_measure_model,
        add_LOGP_Martin=True,
        logp_columns=logp_columns,
        write_to_db=write_to_db,
        append_to_models_folder=append_to_models_folder,
        params=params,
        user=user,
        subfolder=subfolder
    )

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ', sort_by_stat="External", sort_ascending=False)


def run_Koc_gcm_logp_singles():
    logp_columns = ["ALOGP", "ALOGP2", "XLOGP", "XLOGP2", "LOGP_Martin", "LOGP_Martin2"]
    for combo in logp_columns:
        print(f"Running KOC GCM with logP variables: {combo}")
        run_Koc_logp_custom([combo])


def run_Koc_gcm_logp_pairs():
    logp_columns = [["ALOGP", "ALOGP2"], ["XLOGP", "XLOGP2"], ["LOGP_Martin", "LOGP_Martin2"]]
    for combo in logp_columns:
        print(f"Running KOC GCM with logP variables: {', '.join(combo)}")
        run_Koc_logp_custom(list(combo))


def run_Koc_gcm_logp_all():
    logp_columns = ["ALOGP", "ALOGP2", "XLOGP", "XLOGP2", "LOGP_Martin", "LOGP_Martin2"]
    logp_combos1 = list(combinations(logp_columns, 1))
    logp_combos2 = list(combinations(logp_columns, 2))
    for combo in logp_combos1 + logp_combos2:
        cols = list(combo)
        print(f"Running KOC GCM with logP variables: {', '.join(cols)}")
        run_Koc_logp_custom(cols)


def report_Koc_gcm_logp():
    dataset_name = "KOC v1 modeling"
    append_to_models_folder = "_gcm_logp"

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ', sort_by_stat="External", sort_ascending=False)


def run_Koc_rf_logp():
    qsar_method = "rf"
    write_to_db = False
    logp_columns = [
        "LOGP_Martin",
        "LOGP_Martin2",
        ["LOGP_Martin", "LOGP_Martin2"]
    ]
    
    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    params = set_hyper_parameters(
        qsar_method=qsar_method,
        feature_selection=True,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        dataset_name=dataset_name,
        ad_measure=ad_measure_model
    )

    hyperparameters = {
        'estimator__max_features': ['sqrt', 'log2'],
        'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        'estimator__n_estimators': [10, 100, 250, 500]
    }

    for item in logp_columns:
        if item is None:
            print(f"Running KOC RF with no logP variables")
            run_Koc_logp_custom(logp_columns=[], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)
        else:
            print(f"Running KOC RF with logP variable: {item}")
            run_Koc_logp_custom(logp_columns=item, qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)


def run_Koc_huber_logp():
    qsar_method = "huber"
    write_to_db = False
    logp_columns = [
        # None,
        # "ALOGP",
        # "ALOGP2",
        # "XLOGP",
        # "XLOGP2",
        # "LOGP_Martin",
        # "LOGP_Martin2",
        # ["ALOGP", "ALOGP2"],
        ["XLOGP", "XLOGP2"],
        ["LOGP_Martin", "LOGP_Martin2"]
    ]

    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    params = set_hyper_parameters(
        qsar_method=qsar_method,
        feature_selection=True,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        dataset_name=dataset_name,
        ad_measure=ad_measure_model
    )

    hyperparameters = {
        "estimator__epsilon": [1.35],
        "estimator__alpha": [1e-3],
        "estimator__fit_intercept": [True],
        "estimator__max_iter": [10000],
        "estimator__tol": [1e-3]
    }

    max_descriptors = [5, 10, 20]

    for item in logp_columns:
        for max_features in max_descriptors:
            params.max_features = max_features
            if item is None:
                print(f"Running KOC Huber with no logP variables and max_features = {max_features}")
                run_Koc_logp_custom(logp_columns=[], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)
            else:
                print(f"Running KOC Huber with logP variable: {item} and max_features = {max_features}")
                run_Koc_logp_custom(logp_columns=item, qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)


def run_Koc_ransac_logp():
    qsar_method = "ransac"
    write_to_db = False
    logp_columns = [
        None,
        "ALOGP",
        "ALOGP2",
        "XLOGP",
        "XLOGP2",
        "LOGP_Martin",
        "LOGP_Martin2",
        ["ALOGP", "ALOGP2"],
        ["XLOGP", "XLOGP2"],
        ["LOGP_Martin", "LOGP_Martin2"]
    ]

    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]

    params = set_hyper_parameters(
        qsar_method=qsar_method,
        feature_selection=True,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        dataset_name=dataset_name,
        ad_measure=ad_measure_model
    )
    params.num_generations = 100
    params.num_optimizers = 100
    
    # hyperparameters = {
    #     "": []
    # }
    hyperparameters = None

    for item in logp_columns:
        if item is None:
            print(f"Running KOC RANSAC with no logP variables")
            run_Koc_logp_custom(logp_columns=[], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)
        else:
            print(f"Running KOC RANSAC with logP variable: {item}")
            run_Koc_logp_custom(logp_columns=[item], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)


def run_Koc_theil_sen_logp():
    qsar_method = "theil_sen"
    write_to_db = False
    logp_columns = [
        None,
        "ALOGP",
        "ALOGP2",
        "XLOGP",
        "XLOGP2",
        "LOGP_Martin",
        "LOGP_Martin2",
        ["ALOGP", "ALOGP2"],
        ["XLOGP", "XLOGP2"],
        ["LOGP_Martin", "LOGP_Martin2"]
    ]

    dataset_name = "KOC v1 modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"

    ad_measure_model = [
        pc.Applicability_Domain_TEST_Embedding_Euclidean,
        pc.Applicability_Domain_TEST_Fragment_Counts,
    ]
    
    params = set_hyper_parameters(
        qsar_method=qsar_method,
        feature_selection=True,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        dataset_name=dataset_name,
        ad_measure=ad_measure_model
    )
    params.num_generations = 25
    params.num_optimizers = 25

    # hyperparameters = {
    #     "": []
    # }
    hyperparameters = None

    for item in logp_columns:
        if item is None:
            print(f"Running KOC Theil-Sen with no logP variables")
            run_Koc_logp_custom(logp_columns=[], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)
        else:
            print(f"Running KOC Theil-Sen with logP variable: {item}")
            run_Koc_logp_custom(logp_columns=[item], qsar_method=qsar_method, params=params, hyperparameters=hyperparameters, write_to_db=write_to_db)


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



def compare_all_logp_descriptors_to_bcf(save_plots=False):
    logp_descriptors = ["ALOGP", "ALOGP2", "XLOGP", "XLOGP2", "LOGP_Martin", "LOGP_Martin2"]
    for descriptor in logp_descriptors:
        compare_logp_descriptors_to_bcf(descriptor, save_plot=save_plots)
        
        
def compare_logp_descriptors_to_koc(logp_descriptor, save_plot=True):
    df_training, df_prediction = getLogKowPredictionsForDataset()
    df = pd.concat([df_training, df_prediction], ignore_index=True)

    x = df[logp_descriptor]
    y = df.Property

    r2 = r2_score(x, y)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, color="blue", label=f"KOC vs. {logp_descriptor}")

    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", label="y = x")
    plt.xlabel(logp_descriptor)
    plt.ylabel("log KOC")
    plt.title(f"R² = {r2:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_plot:
        folder_path = os.path.join(PROJECT_ROOT, "data", "logp_vs_koc")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(folder_path, f"{logp_descriptor}_vs_koc.png")
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def query_Koc_gcm_models():
    logging.info("Running query_Koc_gcm_models()")
    # model_ids = [1763, 1754, 1757]
    model_ids = [1757]
    for model_id in model_ids:
        try:
            folder_path = os.path.join(PROJECT_ROOT, "data", "models_gcm_logp", "KOC v1 modeling", f"{model_id}") # type: ignore
            Path(folder_path).mkdir(parents=True, exist_ok=True)

            file_path = os.path.join(folder_path, "detailed_summary.xlsx")
            model_path = os.path.join(folder_path, "model.pkl")
            results_path = os.path.join(folder_path, "results.json")
            mdo = ModelDataObjects(model_id=model_id)

            model = mdo.model
            with open(model_path, "wb") as f:
                f.write(pickle.dumps(model))

            with open(results_path, 'w') as f:
                json.dump(mdo.results_dict, f, indent=4)

            mte = ModelToExcel(mdo, file_path)
            mte.create_excel()
        except Exception as e:
            logging.error(f"Error occurred while processing model_id {model_id}: {e}")
            


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



def run_Bcf():
    unique_identifier = None
    user = "murdock.weston"
    write_to_db = False
    dataset_name = "exp_prop_BCF_v1_modeling"
    descriptor_set_name = "WebTEST-default"
    splitting_name = "RND_REPRESENTATIVE"
    
    append_to_models_folder = "_BCF"
    subfolder = None

    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    qsar_method = "rf"
    feature_selection = True

    params = set_hyper_parameters(
        qsar_method=qsar_method,
        feature_selection=feature_selection,
        descriptor_set_name=descriptor_set_name,
        splitting_name=splitting_name,
        dataset_name=dataset_name,
        ad_measure=ad_measure_model
    )
    params.scale_features = True
    params.hyperparameter_grid = {
        'estimator__max_features': ['sqrt', 'log2'],
        'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        'estimator__n_estimators': [10, 100, 250, 500]
    }

    logp_columns = ["LOGP_Martin", "LOGP_Martin2"]

    run_dataset(
        dataset_name=params.dataset_name,
        qsar_method=params.qsar_method,
        feature_selection=params.feature_selection,
        ad_measure_model=ad_measure_model,
        add_LOGP_Martin=True,
        logp_columns=logp_columns,
        write_to_db=write_to_db,
        append_to_models_folder=append_to_models_folder,
        params=params,
        user=user,
        subfolder=subfolder
    )


    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,
    #             write_to_db=write_to_db, unique_identifier=unique_identifier,
    #             append_to_models_folder=append_to_models_folder)  # OK

    # for method in ['rf', 'xgb']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=False,
    #         ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
    #         append_to_models_folder=append_to_models_folder)  
    # #
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True,
    #         ad_measure_model=ad_measure_model, write_to_db=write_to_db, unique_identifier=unique_identifier,
    #         append_to_models_folder=append_to_models_folder)  
    #
    #
    # for method in ['reg', 'knn']:
    #     params = set_hyper_parameters(qsar_method=method, feature_selection=True, descriptor_set_name=descriptor_set_name, 
    #                                   splitting_name=splitting_name, dataset_name=dataset_name, ad_measure=ad_measure_model)
    #     # params.max_features = 12
    #     params.max_features = 25
    #     params.descriptor_coefficient = 0.006
    #     run_dataset(dataset_name=dataset_name, qsar_method=params.qsar_method, feature_selection=params.feature_selection,
    #         params = params, ad_measure_model=ad_measure_model, write_to_db=write_to_db, 
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

    Results.summarize_model_stats(dataset_name, excel_name="model_stats_rmse.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='RMSE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_mae.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='MAE', sort_by_stat="External", sort_ascending=True)
    Results.summarize_model_stats(dataset_name, excel_name="model_stats_r2.xlsx", append_to_models_folder=append_to_models_folder, continuous_stat_name='PearsonRSQ', sort_by_stat="External", sort_ascending=False)
     

def main():
    query_Koc_gcm_models()
    run_Koc_gcm_alogp()
    run_Koc_gcm_xlogp()
    run_Koc_gcm_logp_singles()
    run_Koc_gcm_logp_pairs()
    run_Koc_gcm_logp_all()
    compare_all_logp_descriptors_to_koc(save_plots=True)
    compare_all_logp_descriptors_to_bcf(save_plots=True)
    run_Koc_rf_logp()
    run_Koc_huber_logp()
    run_Koc_ransac_logp()
    run_Koc_theil_sen_logp()
    report_Koc_gcm_logp()
    run_Koc_gcm_outlier_testing()
    run_Bcf_gcm_outlier_testing()
    run_Bcf()
    

if __name__ == '__main__':
    main()
    
