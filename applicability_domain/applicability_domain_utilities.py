"""
Created on 1/9/23
@author: Todd Martin
"""
from models import df_utilities as dfu
from applicability_domain import ApplicabilityDomain as adm
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics

# strTESTApplicabilityDomainEmbeddingCosine = "TEST Cosine Similarity Embedding Descriptors"
# strTESTApplicabilityDomainEmbeddingEuclidean = "TEST Euclidean Distance Embedding Descriptors"
# strTESTApplicabilityDomainAlLDescriptorsCosine = "TEST Cosine Similarity All Descriptors"
# strTESTApplicabilityDomainAllDescriptorsEuclideanDistance = "TEST Euclidean Distance All Descriptors"
# strOPERA_global_index = "OPERA Global Index"
# strOPERA_local_index = "OPERA Local Index"
# strKernelDensity = "Kernel Density"
# strTESTFragmentCounts = "TEST Fragment Counts"

import requests
import numpy as np
import pandas as pd

debug = False

from util import predict_constants as pc

def generate_applicability_domain_with_preselected_descriptors_from_dfs(train_df, test_df, remove_log_p,
                                                               embedding, applicability_domain,filterColumnsInBothSets=True,
                                                               returnTrainingAD=False):

    if filterColumnsInBothSets:
        train_df = dfu.filter_columns_in_both_sets(train_df, test_df)

    # Need to run get the training column names for alldescriptors AD:
    removeCorr = False  # remove correlated descriptors for all descriptors AD, it's faster without it but doesnt make much difference
    train_ids, train_labels, train_features, train_column_names, is_binary = \
        dfu.prepare_instances(df=train_df, which_set="training", remove_logp= remove_log_p, remove_corr=removeCorr)

    # print(applicability_domain, train_df.shape, train_features.shape, remove_log_p, removeCorr)


    if applicability_domain == pc.Applicability_Domain_TEST_Embedding_Cosine:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine'})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == pc.Applicability_Domain_TEST_Embedding_Euclidean:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'euclidean'})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == pc.Applicability_Domain_TEST_All_Descriptors_Cosine:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine',
                           'train_column_names': train_column_names})
        output = ad.evaluate(embedding=train_column_names)

    elif applicability_domain == pc.Applicability_Domain_TEST_All_Descriptors_Euclidean:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'euclidean',
                           'train_column_names': train_column_names})
        output = ad.evaluate(embedding=train_column_names)

    elif applicability_domain == pc.Applicability_Domain_TEST_Fragment_Counts:
        ad=adm.TESTFragmentCounts(train_df, test_df, is_binary)
        output = ad.evaluate(embedding=embedding)

    elif applicability_domain == pc.Applicability_Domain_OPERA_local_index:
        ad = adm.OPERALocalApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 5, 'exceptionalLocal': 0.6, 'similarity': 'euclidean',
                           'onlyLocal': 0.01, 'fractionTrainingSetInsideAD': 0.95})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == pc.Applicability_Domain_OPERA_global_index:
        ad = adm.OPERAGlobalApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'fractionTrainingSetInsideAD': 0.95, 'train_column_names': train_column_names})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == pc.Applicability_Domain_Kernel_Density:
        ad = adm.KernelDensityApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'fractionTrainingSetInsideAD': 0.95, 'train_column_names': train_column_names})
        output = ad.evaluate(embedding=embedding)
    else:
        raise(f"invalid applicability domain:{applicability_domain}")
        
    df_results_inside = output.loc[output['AD'] == True]
    # print('inside shape=', df_results_inside.shape)
    total_rows = output.shape[0]
    coverage = (df_results_inside.shape[0] / total_rows) if total_rows > 0 else 0.0

    # count_inside_AD = output['AD'].value_counts()[True]
    # countTest = output.shape[0]
    # coverage = count_inside_AD / countTest

    if debug:
        print('\nAD', applicability_domain)
        print('Fraction of test set insideID', coverage)

    if returnTrainingAD:
        col_name_id = ad.TrainSet.columns[0]
        AD_TR = ad.TrainSet[ad.AD_Label]
        idTR = ad.TrainSet[col_name_id]
        output = pd.DataFrame(np.column_stack([idTR, AD_TR]),columns=['idTrain', 'AD'])
        return output, ad.splitting_value
    else:
        # print(output)
        return output, ad.splitting_value



def generate_applicability_domain_with_preselected_descriptors(training_tsv, test_tsv, remove_log_p,
                                                               embedding, applicability_domain,filterColumnsInBothSets=True,
                                                               returnTrainingAD=False):
    train_df = dfu.load_df(training_tsv)
    test_df = dfu.load_df(test_tsv)
    return generate_applicability_domain_with_preselected_descriptors_from_dfs(train_df, test_df, remove_log_p,
                                                               embedding, applicability_domain,filterColumnsInBothSets=True,
                                                               returnTrainingAD=False)

def generate_applicability_domain_with_preselected_descriptors_api_call(training_tsv, test_tsv, remove_log_p,
                                                                        embedding_tsv, applicability_domain, urlHost):
    """
    Test the API call in python
    """
    data = {'training_tsv': training_tsv,
            'test_tsv': test_tsv,
            'remove_log_p': remove_log_p,
            'embedding_tsv': embedding_tsv,
            'applicability_domain': applicability_domain}

    # print(data)

    url = urlHost + 'models/prediction_applicability_domain'
    # print(url)
    # sending post request and saving response as response object
    r = requests.post(url=url, data=data, timeout=999999)
    # print(r.text)
    return r.text


def generate_consensus_ad(df_predictions, stats_dict, ad_measure_final, is_binary=False, is_external=False):
    # Build list of AD columns
    colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]

    # Inside/outside consensus AD masks
    mask_all_true = df_predictions[colsAD].eq(True).all(axis=1)
    mask_outside = ~mask_all_true

    # Coverage of consensus AD
    total_rows = len(df_predictions)
    coverage = (mask_all_true.sum() / total_rows) if total_rows > 0 else float('nan')

    # Prepare subsets for consistency
    df_inside = df_predictions.loc[mask_all_true, ['exp', 'pred']].copy()
    df_outside = df_predictions.loc[mask_outside, ['exp', 'pred']].copy()
    valid_inside = df_inside.dropna(subset=['exp', 'pred'])
    valid_outside = df_outside.dropna(subset=['exp', 'pred'])

    # Check if task is binary (all non-null exp ∈ {0,1})
    # exp_nonnull = df_predictions['exp'].dropna()
    # is_binary = exp_nonnull.isin([0, 1]).all()

    def safe_div(n, d):
        return n / d if d else float('nan')

    ad_measure = " and ".join(ad_measure_final)

    if is_external:
        tag = pc.TAG_EXTERNAL
        stat_insert = "External"
    else:
        tag = pc.TAG_TEST
        stat_insert = "Test"

    if is_binary:
        # Balanced Accuracy path
        cutoff = 0.5  # change if you have a project-wide threshold

        stats_inside = calculate_binary_statistics(valid_inside, cutoff=cutoff, tag=tag)
        stats_outside = calculate_binary_statistics(valid_outside, cutoff=cutoff, tag=tag)

        ba_inside = stats_inside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
        ba_outside = stats_outside.get(pc.BALANCED_ACCURACY + tag, float('nan'))
        ba_ratio = safe_div(ba_inside, ba_outside) 

        # print('for consensus ad:')
        # print('rows outside', df_outside.shape[0])
        # print('stats_outside', stats_outside)

        stats = {
            "ad_measure": ad_measure,
            f"BA_{stat_insert}_inside_AD": ba_inside,
            f"BA_{stat_insert}_outside_AD": ba_outside,
            f"ba_ratio": ba_ratio,
            f"Coverage_{stat_insert}": coverage
        }
    else:
        # Continuous (MAE) path via calculate_continuous_statistics
        # mean_exp_training not needed for MAE; pass NaN and only read MAE from the result
        try:
            stats_inside = calculate_continuous_statistics(df_inside, mean_exp_training=float('nan'), tag=tag)
            mae_inside = stats_inside.get(pc.MAE + tag, float('nan'))
        except Exception:
            mae_inside = float('nan')

        try:
            stats_outside = calculate_continuous_statistics(df_outside, mean_exp_training=float('nan'), tag=tag)
            mae_outside = stats_outside.get(pc.MAE + tag, float('nan'))
                        
        except Exception:
            mae_outside = float('nan')

        mae_ratio = safe_div(mae_outside, mae_inside)

        stats = {
            "ad_measure": ad_measure,
            f"MAE_{stat_insert}_inside_AD": mae_inside,
            f"MAE_{stat_insert}_outside_AD": mae_outside,
            f"mae_ratio": mae_ratio,
            f"Coverage_{stat_insert}": coverage
        }

    stats_dict[f"{ad_measure}{' External' if is_external else ''}"] = stats