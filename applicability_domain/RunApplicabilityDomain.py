"""
Created on 1/9/23
@author: Todd Martin
"""

import applicability_domain_utilities as adu
from models import df_utilities as DFU
import numpy as np
import pandas as pd

class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        self.embedding_df = pd.read_csv(embedding_filepath, delimiter=",")

    def get_embedding(self, endpoint):
        return eval(str(self.embedding_df.loc[self.embedding_df['Property'] == endpoint]['embedding'].iloc[0]))

def generateADs():
    ei = EmbeddingImporter("opera_knn_ga_embeddings.csv")
    # %%
    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "LogOH", "Henry's law constant", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
    #                   "Melting point", "LogOH", "Henry's law constant"]

    # endpointsOPERA = ["LogBCF"]
    endpointsOPERA = ["Water solubility", "LogKmHL"]
    endpointsOPERA = ["Water solubility"]
    # endpointsOPERA = ["Octanol water partition coefficient"]
    endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']

    stats = {}
    scores = {}
    targets = endpointsOPERA
    # targets = endpointsTEST
    # targets = endpointsOPERA + endpointsTEST

    endpoint_data = {}
    np.random.seed(100)
    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    for endpoint in targets:

        endpoint_data[endpoint] = {}

        remove_log_p = False
        if endpoint == "Octanol water partition coefficient":
            remove_log_p = True

        if endpoint in ['LLNA', 'Mutagenicity']:
            is_categorical = True
        else:
            is_categorical = False

        print("Endpoint " + str(endpoint))
        if endpoint in endpointsOPERA:
            PROPERTY = 'Property'
            DELIMITER = '\t'
            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + endpoint + ' OPERA/'
            trainPath = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
            testPath = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

        elif endpoint in endpointsTEST:
            PROPERTY = 'Tox'
            DELIMITER = ','
            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/" + endpoint + "/"
            trainPath = endpoint + "_training_set-2d.csv"
            testPath = endpoint + "_prediction_set-2d.csv"

        trainPath = directory + trainPath
        testPath = directory + testPath
        # %%
        embedding = ei.get_embedding(endpoint)

        ###################################################################################################
        training_tsv = DFU.read_file_to_string(trainPath)
        test_tsv = DFU.read_file_to_string(testPath)
        ###################################################################################################
        # %% TEST AD
        output = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                                test_tsv=test_tsv,
                                                                                remove_log_p=remove_log_p,
                                                                                embedding=embedding,
                                                                                applicability_domain=adu.strTESTApplicabilityDomainEmbeddingCosine)

        output.to_csv('results/AD_' + endpoint + '_' + adu.strTESTApplicabilityDomainEmbeddingCosine + '.csv', index=False)
        output.to_json('results/AD_' + endpoint + '_' + adu.strTESTApplicabilityDomainEmbeddingCosine + '.json',
                       orient='records', lines=True)

        ###################################################################################################
        # # %% TEST All Descriptors
        output = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                                test_tsv=test_tsv,
                                                                                remove_log_p=remove_log_p,
                                                                                embedding=embedding,
                                                                                applicability_domain=adu.strTESTApplicabilityDomainAlLDescriptors)

        output.to_csv('results/AD_' + endpoint + '_' + adu.strTESTApplicabilityDomainAlLDescriptors + '.csv',
                      index=False)

        # ###################################################################################################
        # # %% Local Index

        output = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                                test_tsv=test_tsv,
                                                                                remove_log_p=remove_log_p,
                                                                                embedding=embedding,
                                                                                applicability_domain=adu.strOPERA_local_index)

        output.to_csv('results/AD_' + endpoint + '_' + adu.strOPERA_local_index + '.csv', index=False)


def generateADs_via_API_call():
    """
    Test the API calls in python
    """

    urlHost = 'http://localhost:5004/'

    ei = EmbeddingImporter("opera_knn_ga_embeddings.csv")
    # %%
    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "LogOH", "Henry's law constant", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
    #                   "Melting point", "LogOH", "Henry's law constant"]

    # endpointsOPERA = ["LogBCF"]
    endpointsOPERA = ["Water solubility", "LogKmHL"]
    endpointsOPERA = ["Water solubility"]
    # endpointsOPERA = ["Octanol water partition coefficient"]
    endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']

    stats = {}
    scores = {}
    targets = endpointsOPERA
    # targets = endpointsTEST
    # targets = endpointsOPERA + endpointsTEST

    endpoint_data = {}
    np.random.seed(100)
    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    for endpoint in targets:

        endpoint_data[endpoint] = {}

        remove_log_p = False
        if endpoint == "Octanol water partition coefficient":
            remove_log_p = True

        if endpoint in ['LLNA', 'Mutagenicity']:
            is_categorical = True
        else:
            is_categorical = False

        print("Endpoint " + str(endpoint))
        if endpoint in endpointsOPERA:
            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + endpoint + ' OPERA/'
            trainPath = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
            testPath = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

        elif endpoint in endpointsTEST:
            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/" + endpoint + "/"
            trainPath = endpoint + "_training_set-2d.csv"
            testPath = endpoint + "_prediction_set-2d.csv"

        trainPath = directory + trainPath
        testPath = directory + testPath
        # %%
        embedding = ei.get_embedding(endpoint)
        embedding = "\t".join(embedding)

        # print(type((embedding)),embedding)

        ###################################################################################################
        training_tsv = DFU.read_file_to_string(trainPath)
        test_tsv = DFU.read_file_to_string(testPath)
        ###################################################################################################
        # %% TEST AD
        output = adu.generate_applicability_domain_with_preselected_descriptors_api_call(training_tsv=training_tsv,
                                                                                         test_tsv=test_tsv,
                                                                                         remove_log_p=remove_log_p,
                                                                                         embedding_tsv=embedding,
                                                                                         applicability_domain=adu.strTESTApplicabilityDomainEmbeddingCosine,
                                                                                         urlHost=urlHost)


        # print(output)
        text_file = open('results/AD_' + endpoint + '_' + adu.strTESTApplicabilityDomainEmbeddingCosine + '_API.json', "w")
        text_file.write(output)
        text_file.close()

        ###################################################################################################
        # # %% TEST All Descriptors
        output = adu.generate_applicability_domain_with_preselected_descriptors_api_call(training_tsv=training_tsv,
                                                                                         test_tsv=test_tsv,
                                                                                         remove_log_p=remove_log_p,
                                                                                         embedding_tsv=embedding,
                                                                                         applicability_domain=adu.strTESTApplicabilityDomainAlLDescriptors,
                                                                                         urlHost=urlHost)

        # print(output)

        # ###################################################################################################
        # # %% Local Index

        output = adu.generate_applicability_domain_with_preselected_descriptors_api_call(training_tsv=training_tsv,
                                                                                         test_tsv=test_tsv,
                                                                                         remove_log_p=remove_log_p,
                                                                                         embedding_tsv=embedding,
                                                                                         applicability_domain=adu.strOPERA_local_index,
                                                                                         urlHost=urlHost)

        # print(output)


if __name__ == "__main__":
    generateADs()
    # generateADs_via_API_call()
