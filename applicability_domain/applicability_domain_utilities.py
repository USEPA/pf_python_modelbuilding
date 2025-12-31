"""
Created on 1/9/23
@author: Todd Martin
"""
from models import df_utilities as dfu
from applicability_domain import ApplicabilityDomain as adm

strTESTApplicabilityDomainEmbeddingCosine = "TEST Cosine Similarity Embedding Descriptors"
strTESTApplicabilityDomainEmbeddingEuclidean = "TEST Euclidean Distance Embedding Descriptors"
strTESTApplicabilityDomainAlLDescriptorsCosine = "TEST Cosine Similarity All Descriptors"
strTESTApplicabilityDomainAllDescriptorsEuclideanDistance = "TEST Euclidean Distance All Descriptors"
strOPERA_global_index = "OPERA Global Index"
strOPERA_local_index = "OPERA Local Index"
strKernelDensity = "Kernel Density"

import requests
import numpy as np
import pandas as pd

debug = False

def generate_applicability_domain_with_preselected_descriptors_from_dfs(train_df, test_df, remove_log_p,
                                                               embedding, applicability_domain,filterColumnsInBothSets=True,
                                                               returnTrainingAD=False):

    if filterColumnsInBothSets:
        train_df = dfu.filter_columns_in_both_sets(train_df, test_df)

    # Need to run get the training column names for alldescriptors AD:
    removeCorr = False  # remove correlated descriptors for all descriptors AD, it's faster without it but doesnt make much difference
    train_ids, train_labels, train_features, train_column_names, is_binary = \
        dfu.prepare_instances(df=train_df, which_set="training", remove_logp= remove_log_p, remove_corr=removeCorr)


    if applicability_domain == strTESTApplicabilityDomainEmbeddingCosine:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine'})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == strTESTApplicabilityDomainEmbeddingEuclidean:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'euclidean'})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == strTESTApplicabilityDomainAlLDescriptorsCosine or embedding is None:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine',
                           'train_column_names': train_column_names})
        output = ad.evaluate(embedding=train_column_names)

    elif applicability_domain == strTESTApplicabilityDomainAllDescriptorsEuclideanDistance or embedding is None:
        ad = adm.TESTApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'euclidean',
                           'train_column_names': train_column_names})
        output = ad.evaluate(embedding=train_column_names)

    elif applicability_domain == strOPERA_local_index:
        ad = adm.OPERALocalApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'k': 5, 'exceptionalLocal': 0.6, 'similarity': 'euclidean',
                           'onlyLocal': 0.01, 'fractionTrainingSetInsideAD': 0.95})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == strOPERA_global_index:
        ad = adm.OPERAGlobalApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'fractionTrainingSetInsideAD': 0.95, 'train_column_names': train_column_names})
        output = ad.evaluate(embedding=embedding)
    elif applicability_domain == strKernelDensity:
        ad = adm.KernelDensityApplicabilityDomain(train_df, test_df, is_binary)
        ad.set_parameters({'fractionTrainingSetInsideAD': 0.95, 'train_column_names': train_column_names})
        output = ad.evaluate(embedding=embedding)

    
    


    df_results_inside = output.loc[output['AD'] == True]
    # print('inside shape=', df_results_inside.shape)
    coverage = df_results_inside.shape[0] / output.shape[0]

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
        return output, ad.splitSimilarity
    else:
        # print(output)
        return output, ad.splitSimilarity



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
