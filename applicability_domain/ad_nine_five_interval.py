# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 18:14:15 2022

@author: NCHAREST
"""

# -*- coding: utf-8 -*-
import time

"""
Created on Thu May  5 12:09:19 2022

@author: NCHAREST
"""

import applicability_domain_utilities as adu
import DataSetManager as dsm
import ApplicabilityDomain as adm
import modelSpace as ms
import numpy as np
# from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'  #TMM suppresses warnings :)
from models import df_utilities as DFU
from models import knn_model as knn


# %%
class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        self.embedding_df = pd.read_csv(embedding_filepath, delimiter=",")

    def get_embedding(self, endpoint):
        return eval(str(self.embedding_df.loc[self.embedding_df['Property'] == endpoint]['embedding'].iloc[0]))


def train_test_ratios(ad):
    test_ratio = ad.TestInner.shape[0] / (ad.TestInner.shape[0] + ad.TestOuter.shape[0])
    train_ratio = ad.TrainInner.shape[0] / (ad.TrainInner.shape[0] + ad.TrainOuter.shape[0])
    return train_ratio, test_ratio


def caseStudyNate():
    ei = EmbeddingImporter("opera_knn_ga_embeddings.csv")
    # %%
    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "LogOH", "Henry's law constant"]
    # endpointsOPERA = ["LogBCF"]
    endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']
    stats = {}
    scores = {}
    # targets = endpointsOPERA + endpointsTEST
    targets = endpointsOPERA
    endpoint_data = {}
    np.random.seed(100)
    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    for endpoint in targets:
        print("Endpoint " + str(endpoint))
        if endpoint in endpointsOPERA:
            IDENTIFIER = 'ID'
            PROPERTY = 'Property'
            DELIMITER = '\t'
            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + endpoint + ' OPERA/'
            trainPath = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
            testPath = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

        elif endpoint in endpointsTEST:
            IDENTIFIER = 'CAS'
            PROPERTY = 'Tox'
            DELIMITER = ','
            directory = r"C:\\Users\\NCharest\\OneDrive - Environmental Protection Agency (EPA)\\Profile\\Documents\\OPERA_TEST_DataSetsBenchmark\\DataSetsBenchmarkTEST_Toxicity\\" + endpoint + r"\\" + endpoint
            trainPath = "_training_set-2d.csv"
            testPath = "_prediction_set-2d.csv"

        trainPath = directory + trainPath
        testPath = directory + testPath
        # %%
        embedding = ei.get_embedding(endpoint)

        ds_manager = dsm.DataSetManager()
        ds_manager.importSplitDataset(trainPath, testPath, PROPERTY, identifier=IDENTIFIER, delimiter=DELIMITER)
        ds_manager.applyEmbedding(embedding)
        ds_manager.createStratifiedSplit(test_size=0.3, random_state=1991, scaling_type=None)
        X_train, X_test, Y_train, Y_test = ds_manager.returnActiveData()
        # %%
        data_sets = {'Train': (X_train, Y_train), 'Test': (X_test, Y_test)}

        mCreator = ms.ModelMaker()
        mCreator.init_xgb(scale_method="standardize")
        mCreator.set_hyperparameters({'estimator__random_state': 20, 'estimator__eta': 0.1})
        mCreator.train_model(X_train, Y_train)

        trainData, testData = ds_manager.stratDataframeTrain, ds_manager.stratDataframeTest

        # print(trainData)

        # trainData.to_excel("results/trainData1.xlsx",sheet_name='sheet1')

        endpoint_data[endpoint] = {}

        # %%

        # %% TEST AD
        ad = adm.TESTApplicabilityDomain(trainData, testData)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        ad.setEmbedding(embedding)
        ad.setResponse(PROPERTY)
        ad.setModel(mCreator.model)
        ad.predict()
        output = ad.evaluate()
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        endpoint_data[endpoint]['TEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
                                           'coverage_test': train_test_ratios(ad)[1]}

        # %% TEST All Descriptors
        ad = adm.AllTESTApplicabilityDomain(trainData, testData)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        ad.setEmbedding(embedding)
        ad.setResponse(PROPERTY)
        ad.setModel(mCreator.model)
        ad.predict()

        # print (trainData)
        # trainData.to_excel("results/trainData2.xlsx", sheet_name='sheet1')
        # print(ds_manager.feature_names)

        output = ad.evaluate(ds_manager.feature_names)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        endpoint_data[endpoint]['AllTEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
                                              'coverage_test': train_test_ratios(ad)[1]}
        # %% Local Index
        ad = adm.OPERALocalApplicabilityDomain(trainData, testData)
        ad.setEmbedding(embedding)
        ad.setResponse(PROPERTY)
        ad.setModel(mCreator.model)
        ad.calculate_local(0.05)
        ad.set_parameters({'weakLocal': ad.splitting_value})
        ad.predict()
        ad.evaluate()
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        endpoint_data[endpoint]['OPERALocal'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
                                                 'coverage_test': train_test_ratios(ad)[1]}
        # %% Leverage
        ad = adm.LeverageApplicabilityDomain(trainData, testData)
        ad.setEmbedding(embedding)
        ad.setResponse(PROPERTY)
        ad.setModel(mCreator.model)
        ad.predict()
        ad.evaluate()
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        endpoint_data[endpoint]['Leverage'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
                                               'coverage_test': train_test_ratios(ad)[1]}
        # %% Kernel
        ad = adm.KernelDensityApplicabilityDomain(trainData, testData)
        ad.setEmbedding(embedding)
        ad.setResponse(PROPERTY)
        ad.setModel(mCreator.model)
        ad.predict()
        ad.evaluate()
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        endpoint_data[endpoint]['KernelDensity'] = {'product': ad.cp_product,
                                                    'coverage_train': train_test_ratios(ad)[0],
                                                    'coverage_test': train_test_ratios(ad)[1]}
    # %%
    results = {}
    for key in ['TEST', 'AllTEST', 'OPERALocal', 'Leverage', 'KernelDensity']:
        results[key] = {}
        for endpoint in targets:
            results[key][endpoint] = endpoint_data[endpoint][key]
    # %%
    for key in results.keys():
        pd.DataFrame(results[key]).transpose().to_csv("results/ad_table_" + key + ".csv")


def caseStudyTodd():

    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    descriptor_software = 'WebTEST-default'

    ei = EmbeddingImporter("../embeddings/"+descriptor_software+"_ga_embeddings.csv")

    # %%
    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "LogOH", "Henry's law constant", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Water solubility", "LogKmHL", "LogKoa", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
    #                   "Melting point", "LogOH", "Henry's law constant"]

    # endpointsOPERA = ["LogBCF"]
    # endpointsOPERA=["Water solubility", "LogKmHL"]
    # endpointsOPERA = ["Octanol water partition coefficient"]
    endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']
    # endpointsTEST = ['LC50DM']
    # endpointsTEST = ['LC50']

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
            PROPERTY = 'Property'
            DELIMITER = ','

            # directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/" + endpoint + "/"
            # trainPath = endpoint + "_training_set-2d.csv"
            # testPath = endpoint + "_prediction_set-2d.csv"

            directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark_TEST/" + endpoint + " TEST/"
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
        trainData = DFU.filter_columns_in_both_sets(trainData, testData)

        # testData = testData.loc[[1,2,3,4,5]]  # keep only 3 row to test speed

        # Need to run get the training column names for alldescriptors AD:
        removeCorr = False  # remove correlated descriptors for all descriptors AD
        train_ids, train_labels, train_features, train_column_names, is_binary = \
            DFU.prepare_instances(trainData, "training", remove_log_p, removeCorr)

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
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        ad.setResponse(PROPERTY)
        ad.setModel(model.getModel2())
        ad.predict(embedding)
        output = ad.evaluate2(embedding)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['TEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                    'coverage_test': train_test_ratios(ad)[1]}

        endpoint_data[endpoint]['TEST'] = {'r2_test': ad.scoreTestInner, 'coverage_test': train_test_ratios(ad)[1],
                                           'product': ad.cp_product}

        ###################################################################################################
        # %% TEST All Descriptors

        useEmbeddingModel = True

        ad = adm.AllTESTApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        ad.setResponse(PROPERTY)

        if useEmbeddingModel:
            ad.setModel(model.getModel2())
            ad.predict(embedding)
        else:
            ad.setModel(modelAll.getModel2())
            ad.predict(train_column_names)

        # print (trainData)
        # trainData.to_excel("results/trainData2.xlsx", sheet_name='sheet1')
        # print(ds_manager.feature_names)

        output = ad.evaluate(train_column_names)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['AllTEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                       'coverage_test': train_test_ratios(ad)[1]}
        endpoint_data[endpoint]['AllTEST_embedding_model'] = {'r2_test': ad.scoreTestInner, 'coverage_test': train_test_ratios(ad)[1],
                                           'product': ad.cp_product}
        ###################################################################################################
        # %% TEST All Descriptors

        useEmbeddingModel = False

        ad = adm.AllTESTApplicabilityDomain(trainData, testData, is_categorical)
        ad.set_parameters({'k': 3, 'exclusionFraction': 0.05, 'similarity': 'cosine'})
        ad.setResponse(PROPERTY)

        if useEmbeddingModel:
            ad.setModel(model.getModel2())
            ad.predict(embedding)
        else:
            ad.setModel(modelAll.getModel2())
            ad.predict(train_column_names)

        # print (trainData)
        # trainData.to_excel("results/trainData2.xlsx", sheet_name='sheet1')
        # print(ds_manager.feature_names)

        output = ad.evaluate(train_column_names)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['AllTEST'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                       'coverage_test': train_test_ratios(ad)[1]}
        endpoint_data[endpoint]['AllTEST_all_descriptors_model'] = {'r2_test': ad.scoreTestInner,
                                                              'coverage_test': train_test_ratios(ad)[1],
                                                              'product': ad.cp_product}

        ###################################################################################################
        # %% Local Index
        ad = adm.OPERALocalApplicabilityDomainRevised(trainData, testData, is_categorical)
        ad.set_parameters({'k': 5, 'exceptionalLocal': 0.6, 'similarity': 'euclidean',
                           'onlyLocal': 0.01, 'exclusionFraction': 0.05})

        # print(ad.parameters)
        ad.setResponse(PROPERTY)
        ad.setModel(model.getModel2())
        ad.predict(embedding)
        ad.evaluate2(embedding)
        ad.createSets(ad.AD_Label)
        ad.scoreMetrics()
        # endpoint_data[endpoint]['OPERALocal'] = {'product': ad.cp_product, 'coverage_train': train_test_ratios(ad)[0],
        #                                          'coverage_test': train_test_ratios(ad)[1]}

        endpoint_data[endpoint]['OPERALocal'] = {'r2_test': ad.scoreTestInner, 'coverage_test': train_test_ratios(ad)[1],
                                           'product': ad.cp_product}


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
    for key in ['TEST', 'AllTEST_embedding_model', 'AllTEST_all_descriptors_model', 'OPERALocal']:
        results[key] = {}
        for endpoint in targets:
            results[key][endpoint] = endpoint_data[endpoint][key]
    # %%
    for key in results.keys():
        pd.DataFrame(results[key]).transpose().to_csv("results/ad_table_" + key + ".csv")


if __name__ == "__main__":
    # caseStudyNate()
    caseStudyTodd()
