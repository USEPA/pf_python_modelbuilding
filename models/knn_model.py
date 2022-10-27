# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:51:07 2022

@author: CRAMSLAN
"""

import time
from models import df_utilities as DFU
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import GeneticOptimizer as go
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
        self.version = '1.3'
        self.n_neighbors = 5
        self.qsar_method = 'knn'
        self.description = 'sklearn implementation of k-nearest neighbors ' \
                           'https://scikit-learn.org/stable/modules/generated/' \
                           'sklearn.neighbors.KNeighborsClassifier.html' 
                           

    def build_model(self):
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        if self.is_binary:
            self.knn = KNeighborsRegressor(self.n_neighbors, weights='distance')
        else:
            self.knn = KNeighborsClassifier(self.n_neighbors, weights='distance')
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

        if self.is_binary:
            self.knn = KNeighborsRegressor(self.n_neighbors, weights='distance')
        else:
            self.knn = KNeighborsClassifier(self.n_neighbors, weights='distance')
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
            predictions = self.knn.predict_proba(pred_features)[:,1]
        else:
            predictions = self.knn.predict(pred_features)

        print('Score for Test data = ', self.rfr.score(pred_features, pred_labels))

        # Return predictions
        return predictions


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""
        self.is_binary = model.is_binary
        self.n_estimators = model.n_estimators
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description

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

def caseStudyGA():

    # ENDPOINT = "LogKmHL"
    ENDPOINT = "Henry's law constant"

    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKOA", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point","Henry's law constant"]
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
            
        
    print(train_column_names)

    # df_training = df_training.loc[:, (df_training != 0).any(axis=0)]

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    model = Pipeline([('standardizer', StandardScaler()), ('estimator', KNeighborsRegressor())])


    features = go.runGA(df_training, IDENTIFIER, PROPERTY, model)

    print(features)
    
    
    embed_model = Model(df_training, False)
    embed_model.build_model_with_preselected_descriptors(features)
    embed_model_predictions = embed_model.do_predictions(df_prediction)
    
    full_model = Model(df_training, False)
    full_model.build_model()
    full_model_predictions = full_model.do_predictions(df_prediction)



if __name__ == "__main__":
    caseStudyGA()
