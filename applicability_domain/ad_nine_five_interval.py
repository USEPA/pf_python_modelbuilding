# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:14:15 2022

@author: NCHAREST
"""

# -*- coding: utf-8 -*-

"""
Created on Thu May  5 12:09:19 2022

@author: NCHAREST
"""

# from applicability_domain import DataSetManager as dsm
from applicability_domain import ApplicabilityDomain as adm
# from applicability_domain import modelSpace as ms

import numpy as np
# from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'  #TMM suppresses warnings :)
from models import df_utilities as DFU
from models.old.models_old import knn_model as knn


# %%
class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        self.embedding_df = pd.read_csv(embedding_filepath, delimiter=",")

    def get_embedding(self, endpoint):
        return eval(str(self.embedding_df.loc[self.embedding_df['Property'] == endpoint]['embedding'].iloc[0]))



def caseStudies():

    testMetric = 'cosine'
    # testMetric = 'euclidean'

    # descriptor_software = 'WebTEST-default'
    # descriptor_software = 'PaDEL-default'
    descriptor_software = 'ToxPrints-default'

    ei = EmbeddingImporter("../embeddings/"+descriptor_software+"_ga_embeddings.csv")

    # %%
    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "LogOH", "Henry's law constant", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
    #                   "Melting point", "LogOH", "Henry's law constant"]
    # endpointsOPERA = ["Boiling point"]
    # endpointsOPERA = ["LogBCF"]
    # endpointsOPERA = ["LogKoa"]
    # endpointsOPERA=["Water solubility", "LogKmHL"]
    # endpointsOPERA = ["Octanol water partition coefficient"]

    endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']
    # endpointsTEST = ['LLNA']



    stats = {}
    scores = {}
    # targets = endpointsOPERA
    # targets = endpointsTEST
    targets = endpointsOPERA + endpointsTEST

    endpoint_data = {}
    np.random.seed(100)

    for endpoint in targets:

        endpoint_data[endpoint] = {}

        remove_log_p = False


        # if endpoint == "Octanol water partition coefficient":
        #     remove_log_p = True

        if endpoint in ['LLNA', 'Mutagenicity']:
            is_categorical = True
        else:
            is_categorical = False

        print("Endpoint " + str(endpoint))

        if endpoint in endpointsOPERA:
            PROPERTY = 'Property'
            DELIMITER = '\t'
            directory = "../datasets_benchmark/" + endpoint + ' OPERA/'
            trainPath = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
            testPath = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

        elif endpoint in endpointsTEST:
            PROPERTY = 'Property'
            DELIMITER = ','

            # directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/" + endpoint + "/"
            # trainPath = endpoint + "_training_set-2d.csv"
            # testPath = endpoint + "_prediction_set-2d.csv"

            directory = "../datasets_benchmark_TEST/" + endpoint + " TEST/"
            trainPath = endpoint + ' TEST ' + descriptor_software + ' training.tsv'
            testPath = endpoint + ' TEST ' + descriptor_software + ' prediction.tsv'


        trainPath = directory + trainPath
        testPath = directory + testPath
        # %%
        embedding = ei.get_embedding(endpoint)

        print(embedding)

        ###################################################################################################
        trainData = DFU.load_df_from_file(trainPath, sep=DELIMITER)
        testData = DFU.load_df_from_file(testPath, sep=DELIMITER)
        trainData = DFU.filter_columns_in_both_sets(trainData, testData, remove_log_p)


        # testData = testData.loc[[1,2,3,4,5]]  # keep only 3 row to test speed

        # Need to run get the training column names for alldescriptors AD:
        removeCorr = False  # remove correlated descriptors for all descriptors AD
        train_ids, train_labels, train_features, train_column_names, is_binary = \
            DFU.prepare_instances(df=trainData, which_set="training", remove_logp=remove_log_p, remove_corr=removeCorr)

        ###################################################################################################
        # Build the model

        model = knn.Model(trainData, remove_log_p, 1)
        model.build_model_with_preselected_descriptors(embedding)

        modelAll = knn.Model(trainData, remove_log_p, 1)
        modelAll.build_model()

        # model.do_predictions(testData)
        ###################################################################################################
        # %% TEST AD
        ad = adm.TESTApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': testMetric})
        ad.setResponse(PROPERTY)
        ad.setModel(model)
        ad.predict(embedding)
        output = ad.evaluate(embedding)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['TEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                    'coverage_test': train_test_ratios(ad)[1]}

        endpoint_data[endpoint]['TEST'] = ad.getStats()

        ###################################################################################################
        # %% TEST All Descriptors

        useEmbeddingModel = True

        ad = adm.TESTApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': testMetric})
        ad.setResponse(PROPERTY)

        if useEmbeddingModel:
            ad.setModel(model)
            ad.predict(embedding)
        else:
            ad.setModel(modelAll)
            ad.predict(train_column_names)

        # print (trainData)
        # trainData.to_excel("results/trainData2.xlsx", sheet_name='sheet1')
        # print(ds_manager.feature_names)

        output = ad.evaluate(train_column_names)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['AllTEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                       'coverage_test': train_test_ratios(ad)[1]}
        endpoint_data[endpoint]['AllTEST_embedding_model'] = ad.getStats()
        ###################################################################################################
        # %% TEST All Descriptors

        useEmbeddingModel = False

        ad = adm.TESTApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': testMetric})
        ad.setResponse(PROPERTY)

        if useEmbeddingModel:
            ad.setModel(model)
            ad.predict(embedding)
        else:
            ad.setModel(modelAll)
            ad.predict(train_column_names)

        # print (trainData)
        # trainData.to_excel("results/trainData2.xlsx", sheet_name='sheet1')
        # print(ds_manager.feature_names)

        output = ad.evaluate(train_column_names)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['AllTEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                       'coverage_test': train_test_ratios(ad)[1]}
        endpoint_data[endpoint]['AllTEST_all_descriptors_model'] = ad.getStats()

        ###################################################################################################
        # %% Local Index

        print('*********************************************************************************************')
        print('start local index')

        ad = adm.OPERALocalApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 5, 'exceptionalLocal': 0.6, 'similarity': 'euclidean',
                           'onlyLocal': 0.01, 'exclusionFraction': 0.05})

        # print(ad.parameters)
        ad.setResponse(PROPERTY)
        ad.setModel(model)
        ad.predict(embedding)

        ad.evaluate(embedding)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['OPERALocal'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                          'coverage_test': train_test_ratios(ad)[1]}

        try:
            endpoint_data[endpoint]['OPERALocal'] = ad.getStats()

            print ('OPERA results',endpoint_data[endpoint]['OPERALocal'])

        except Exception as e:
            print(e)

        ###################################################################################################
        # # %% Leverage
        # ad = adm.LeverageApplicabilityDomain(trainData, testData, is_categorical)
        # ad.setEmbedding(embedding)  # needs to be in same order as in the built model!
        # ad.setResponse(PROPERTY)
        # ad.setModel(model.getModel2())
        # ad.predict()
        # ad.evaluate()
        # ad.createSets(ad.AD_Label)
        # ad.scoreMetrics()
        # endpoint_data[endpoint]['Leverage'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                        'coverage_test': train_test_ratios(ad)[1]}
        ###################################################################################################
        # # %% Kernel
        # ad = adm.KernelDensityApplicabilityDomain(trainData, testData, is_categorical)
        # ad.setEmbedding(embedding)#needs to be in same order as in the built model!
        # ad.setResponse(PROPERTY)
        # ad.setModel(model.getModel2())
        # ad.predict()
        # ad.evaluate()
        # ad.createSets(ad.AD_Label)
        # ad.scoreMetrics()
        # endpoint_data[endpoint]['KernelDensity'] = {'product': ad.cp_product,
        #                                             'coverage_train': train_test_ratios(ad)[0],
        #                                             'coverage_test': train_test_ratios(ad)[1]}
    # %%

    results = {}


    # keys = ['r2_test', 'coverage_test', 'product']

    # print(endpoint_data)

    # for key in ['TEST', 'AllTEST', 'OPERALocal', 'Leverage', 'KernelDensity']:
    # for key in ['TEST', 'AllTEST_embedding_model', 'AllTEST_all_descriptors_model']:
    for key in ['TEST', 'AllTEST_embedding_model', 'AllTEST_all_descriptors_model', 'OPERALocal']:
        results[key] = {}
        for endpoint in targets:
            try:
                results[key][endpoint] = endpoint_data[endpoint][key]
            except Exception as e:
                print(e)
    # %%
    for key in results.keys():
        pd.DataFrame(results[key]).transpose().to_csv("results/ad_table_" + key + ".csv")




if __name__ == "__main__":
    # caseStudyNate()
    caseStudies()
