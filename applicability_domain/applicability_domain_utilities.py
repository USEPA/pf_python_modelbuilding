"""
Created on 1/9/23
@author: Todd Martin
"""
from models import df_utilities as dfu
from applicability_domain import ApplicabilityDomain as adm

strTESTApplicabilityDomainEmbeddingCosine = "TEST Cosine Similarity Embedding Descriptors"
strTESTApplicabilityDomainEmbeddingEuclidean = "TEST Euclidean Distance Embedding Descriptors"
strTESTApplicabilityDomainAlLDescriptors = "TEST Cosine Similarity All Descriptors"
strOPERA_local_index = "OPERA Local Index"

import requests


def generate_applicability_domain_with_preselected_descriptors(training_tsv, test_tsv, remove_log_p,
                                                               embedding, applicability_domain):
    trainData = dfu.load_df(training_tsv)
    testData = dfu.load_df(test_tsv)

    # Need to run get the training column names for alldescriptors AD:
    removeCorr = False  # remove correlated descriptors for all descriptors AD, it's faster without it but doesnt make much difference
    train_ids, train_labels, train_features, train_column_names, is_binary = \
        dfu.prepare_instances(trainData, "training", remove_log_p, removeCorr)

    if applicability_domain == strTESTApplicabilityDomainEmbeddingCosine:
        ad = adm.TESTApplicabilityDomain(trainData, testData, is_binary)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        output = ad.evaluate2(embedding=embedding)
    elif applicability_domain == strTESTApplicabilityDomainEmbeddingEuclidean:
        ad = adm.TESTApplicabilityDomain(trainData, testData, is_binary)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'euclidean'})
        output = ad.evaluate2(embedding=embedding)
    elif applicability_domain == strTESTApplicabilityDomainAlLDescriptors or embedding is None:
        ad = adm.TESTApplicabilityDomain(trainData, testData, is_binary)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine',
                           'train_column_names': train_column_names})
        output = ad.evaluate2(embedding=train_column_names)
    elif applicability_domain == strOPERA_local_index:
        ad = adm.OPERALocalApplicabilityDomainRevised(trainData, testData, is_binary)
        ad.set_parameters({'k': 5, 'exceptionalLocal': 0.6, 'similarity': 'euclidean',
                           'onlyLocal': 0.01, 'exclusionFraction': 0.05})
        output = ad.evaluate2(embedding=embedding)

    count_inside_AD = output['AD'].value_counts()[True]
    countTest = output.shape[0]
    coverage = count_inside_AD / countTest

    print('\n***', applicability_domain, coverage)
    # print(output)
    return output


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
