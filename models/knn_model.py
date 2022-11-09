# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:51:07 2022

@author: CRAMSLAN
"""

import time

# import GeneticOptimizer
# import models.GeneticOptimizer as go
from models import df_utilities as DFU
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import model_ws_utilities as mwu
from os.path import exists
import json

__author__ = "Christian Ramsland"


# class to run random forest using files and be able to use multiple processors. When running using webservice, it
# doesnt seem to let you use multiple threads...
# if n_jobs>1, get UserWarning: Loky-backed parallel loops cannot be nested below threads, setting n_jobs=1
# This class will aid in building models for large data sets like logP which requires one to remove logP based
# descriptors

class Model:
    """Trains and makes predictions with a k nearest neighbors model"""

    def __init__(self, df_training, remove_log_p_descriptors):
        """Initializes the RF model with optimal parameters and provided data in pandas dataframe"""
        self.knn = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.is_binary = None  # Set automatically when training data is loaded
        self.df_training = df_training
        self.version = '1.0'
        self.n_neighbors = 5
        self.qsar_method = 'knn'
        self.description = 'sklearn implementation of k-nearest neighbors ' \
                           'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'
        #todo should url not be a part of description in all methods?

        self.url = 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'

            # self.metric = 'cosine'
        self.metric = 'minkowski'  # default in knn
        self.weights = 'distance'
        # self.weights = 'uniform'

    def getModel(self):

        # print ("here:",self.is_binary)

        if self.is_binary:
            # self.knn = KNeighborsClassifier(self.n_neighbors, weights='distance')
            self.knn = Pipeline([('standardizer', StandardScaler()),
                                 ('estimator', KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights,
                                                                    metric=self.metric))])

        else:
            # self.knn = KNeighborsRegressor(self.n_neighbors, weights='distance')
            self.knn = Pipeline([('standardizer', StandardScaler()),
                                 ('estimator', KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights,
                                                                   metric=self.metric))])
        return self.knn

    def build_model(self):
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        self.getModel()

        # self.knn = KNeighborsRegressor(self.n_neighbors)

        # Train the model on training data
        self.knn.fit(train_features, train_labels)

        print('Score for Training data = ', self.knn.score(train_features, train_labels))

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        return self

    def build_model_with_preselected_descriptors(self, descriptor_names):
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", descriptor_names)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        self.getModel()

        # Train the model on training data
        self.knn.fit(train_features, train_labels)

        print('Score for Training data = ', self.knn.score(train_features, train_labels))

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        return self

    def do_predictions(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')
        if (self.is_binary == True):
            predictions = self.knn.predict_proba(pred_features)[:, 1]
        else:
            predictions = self.knn.predict(pred_features)

        print('Score for Test data = ', self.knn.score(pred_features, pred_labels), '\n')

        # Return predictions
        return predictions

    def do_predictions_score(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')
        if (self.is_binary == True):
            predictions = self.knn.predict_proba(pred_features)[:, 1]
        else:
            predictions = self.knn.predict(pred_features)

        print('Score for Test data = ', self.knn.score(pred_features, pred_labels), '\n')

        # Return predictions
        return self.knn.score(pred_features, pred_labels)


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""

        # kNN specific params:
        self.metric = model.metric
        self.n_neighbors = model.n_neighbors
        self.weights = model.weights
        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors

        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description
        self.url = model.url


    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)


def main():
    """
    Code to run from text files rather than webservice
    :return:
    """
    # endpoint = 'Octanol water partition coefficient'
    # endpoint = 'Water solubility'
    # endpoint = 'Melting point'
    endpoint = 'LogBCF'

    # descriptor_software = 'T.E.S.T. 5.1'
    descriptor_software = 'Padelpy webservice single'

    folder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/'
    folder += 'QSAR_Model_Building/data/datasets_benchmark/' + endpoint + ' OPERA/'
    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    # Parameters needed to build model:
    n_threads = 30
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # df_training = pd.read_csv(training_tsv_path, sep='\t')
    # df_prediction = pd.read_csv(prediction_tsv_path, sep='\t')

    model = Model(df_training, remove_log_p_descriptors, n_threads)
    model.build_model()

    print(ModelDescription(model).to_json())
    model.do_predictions(df_prediction)


def CaseStudyKNN():
    ENDPOINT = "Henry's law constant"

    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKOA", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "Henry's law constant"]
    endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

    if ENDPOINT in endpointsOPERA:
        IDENTIFIER = 'ID'
        PROPERTY = 'Property'
        DELIMITER = '\t'
        directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmark\\" + ENDPOINT + " OPERA" + r"\\"
        trainPath = "training.tsv"
        testPath = "prediction.tsv"
    elif ENDPOINT in endpointsTEST:
        IDENTIFIER = 'CAS'
        PROPERTY = 'Tox'
        DELIMITER = ','
        directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmarkTEST_Toxicity" + ENDPOINT + r"\\" + ENDPOINT
        trainPath = "_training_set-2d.csv"
        testPath = "_prediction_set-2d.csv"

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
    folder = directory

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    print(training_tsv_path)

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    train_ids, train_labels, train_features, train_column_names, is_binary = \
        DFU.prepare_instances(df_training, "training", False, False)

    knn = KNeighborsRegressor(5, weights='distance')

    knn.fit(train_features, train_labels)
    knn.predict(df_prediction)
    pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)
    knn.score(pred_features, pred_labels)


def caseStudyGA():
    # ENDPOINT = "LogKmHL"
    # ENDPOINT = "Henry's law constant"
    # ENDPOINT = "Vapor pressure"
    ENDPOINT = "Water solubility"
    # ENDPOINT = "Boiling point"

    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKOA", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point", "Henry's law constant"]
    endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

    if ENDPOINT in endpointsOPERA:
        IDENTIFIER = 'ID'
        PROPERTY = 'Property'
        # directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmark\\" + ENDPOINT + " OPERA" + r"\\"
        directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + ENDPOINT + ' OPERA/'

    elif ENDPOINT in endpointsTEST:
        IDENTIFIER = 'CAS'
        PROPERTY = 'Tox'
        directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmarkTEST_Toxicity" + ENDPOINT + r"\\" + ENDPOINT

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
    folder = directory

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name
    print(training_tsv_path)

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # train_ids, train_labels, train_features, train_column_names, is_binary = \
    #     DFU.prepare_instances(df_training, "training", False, False)
    # print(train_column_names)

    df_training = df_training.loc[:, (df_training != 0).any(axis=0)]

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    ga_model = Model(df_training, False)
    ga_model.is_binary = False  # todo determine based on df_training
    # features = go.runGA(df_training, IDENTIFIER, PROPERTY, ga_model.getModel())

    features = ['-OH [aromatic attach]', 'Hmax', 'SsOH', 'MDEN11', 'x0', 'Qv', 'nN', 'SaaNH', '-C#N [aliphatic attach]',
                'SdO', '-CHO [aliphatic attach]', 'MDEN33', 'xch6']

    #
    print(features)
    #
    #
    embed_model = Model(df_training, False)
    embed_model.build_model_with_preselected_descriptors(features)
    embed_model_predictions = embed_model.do_predictions(df_prediction)

    full_model = Model(df_training, False)
    full_model.build_model()
    full_model_predictions = full_model.do_predictions(df_prediction)


def caseStudyGA2():

    endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
                      "Vapor pressure", "Water solubility", "Boiling point",
                      "Melting point", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Melting point", "Octanol water partition coefficient"]


    # endpointsOPERA = ["Octanol water partition coefficient"]
    # endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    # Set GA hyperparameters:
    num_generations = 1
    num_optimizers = 10
    qsar_method = 'knn'
    n_threads = 16 # threads to use with kNN- TODO
    num_jobs = 16 # jobs to use for GA search for embedding

    urlHost = 'http://localhost:5004/'

    filename = '../data/opera kNN results NUM_GENERATIONS=' + str(num_generations) + \
               ' NUM_OPTIMIZERS=' + str(num_optimizers) + '.txt'
    f = open(filename, "w")

    dummy_model = Model(None, False)
    model_desc = ModelDescription(dummy_model)
    f.write(model_desc.to_json()+'\n\n')

    f.write('ENDPOINT\tscore\tscore_embed\tlen(features)\tfeatures\tTime(min)\n')
    f.flush()

    for ENDPOINT in endpointsOPERA:
        if ENDPOINT == 'Octanol water partition coefficient':
            remove_log_p = True
        else:
            remove_log_p =False

        # directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmark\\" + ENDPOINT + " OPERA" + r"\\"
        directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + ENDPOINT + ' OPERA/'

        training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
        folder = directory

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        # print(training_tsv_path)

        df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
        df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
        df_training = df_training.loc[:, (df_training != 0).any(axis=0)]

        with open(training_tsv_path, 'r') as file:
            training_tsv = file.read().rstrip()

        # print(training_tsv)

        # **************************************************************************************
        # t1 = time.time()
        # ga_model = Model(df_training, False)
        # ga_model.is_binary = DFU.isBinary(df_training)
        # features = go.runGA(df_training, ga_model.getModel())
        # print(features)
        # print(len(features))
        # t2 = time.time()
        # Run from method
        # json_embedding, timeGA = mwu.call_build_embedding_ga(qsar_method=qsar_method, training_tsv=df_training,
        #                             remove_log_p=remove_log_p, n_threads=n_threads,
        #                             num_generations=num_generations, num_optimizers=num_optimizers)

        #Run from API call:
        str_result = mwu.api_call_build_embedding_ga(qsar_method=qsar_method, training_tsv=training_tsv,
                                    remove_log_p=remove_log_p, n_threads=n_threads,
                                    num_generations=num_generations, num_optimizers=num_optimizers, num_jobs=num_jobs, urlHost=urlHost)

        dict_result = json.loads(str_result)
        features = json.loads(dict_result['embedding'])
        timeGA = dict_result['timeMin']

        print('embedding = ', features)
        print('Time to run ga  = ', timeGA, 'mins')

        # **************************************************************************************
        # Build model based on embedded descriptors: TODO use api to run this code
        embed_model = mwu.call_build_model_with_preselected_descriptors(qsar_method, training_tsv,remove_log_p, features)
        score_embed = embed_model.do_predictions_score(df_prediction)

        # **************************************************************************************
        # Build model based on all descriptors (except correlated and constant ones):
        full_model = mwu.call_build_model(qsar_method, training_tsv,remove_log_p)
        score = full_model.do_predictions_score(df_prediction)

        print(ENDPOINT + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(features)) + '\t' + str(
            features) + '\t' + str(timeGA) + '\n')

        f.write(ENDPOINT + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(features)) + '\t' + str(
            features) + '\t' + str(timeGA) + '\n')
        # f.write(ENDPOINT + '\t' + str(score) + '\n')
        f.flush()

    f.close()


def caseStudyGA3():
    '''
    Runs just all descriptors case
    :return:
    '''
    endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
                      "Vapor pressure", "Water solubility", "Boiling point",
                      "Melting point", "Octanol water partition coefficient"]

    # endpointsOPERA = ["Octanol water partition coefficient"]
    # endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'


    # f = open("todd.txt", "w")
    # f = open("opera results 100 generations.txt", "w")

    filename = 'opera results uniform weighting all descriptors.txt'

    f = open(filename, "w")

    f.write('ENDPOINT\tscore\n')
    f.flush()

    for ENDPOINT in endpointsOPERA:
        IDENTIFIER = 'ID'
        PROPERTY = 'Property'
        # directory = r"C:\Users\CRAMSLAN\OneDrive - Environmental Protection Agency (EPA)\VDI_Repo\python\pf_python_modelbuilding\datasets\DataSetsBenchmark\\" + ENDPOINT + " OPERA" + r"\\"
        directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/" + ENDPOINT + ' OPERA/'

        training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
        folder = directory

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        # print(training_tsv_path)

        df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
        df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
        df_training = df_training.loc[:, (df_training != 0).any(axis=0)]

        # **************************************************************************************
        # Build model based on embedded descriptors:
        # **************************************************************************************
        # Build model based on all descriptors (except correlated and constant ones):
        full_model = Model(df_training, False)
        full_model.build_model()
        print(ENDPOINT)
        score = full_model.do_predictions_score(df_prediction)

        f.write(ENDPOINT + '\t' + str(score) + '\n')
        f.flush()

    f.close()

if __name__ == "__main__":
    # caseStudyGA()
    caseStudyGA2()
    # caseStudyGA3()
