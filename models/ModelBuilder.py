# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:14:14 2023

@author: NCHAREST
This is a refactored version of the python model building repo's model object, intended to better utilize OOP and more elegant design patterning.'
"""

import time

from xgboost import XGBRegressor, XGBClassifier

from models import df_utilities as DFU
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import numpy as np

from os.path import exists
import json

__author__ = "Nathaniel Charest, Todd Martin (get it working with webservice)"


# %%
def model_registry(regressor_name, is_categorical):
    try:
        registry = {
            'knn':
                {True: Pipeline([('standardizer', StandardScaler()), ('estimator', KNeighborsClassifier())]),
                 False: Pipeline([('standardizer', StandardScaler()), ('estimator', KNeighborsRegressor())])
                 },
            'rf':
                {True: Pipeline([('standardizer', StandardScaler()), ('estimator', RandomForestClassifier())]),
                 False: Pipeline([('standardizer', StandardScaler()), ('estimator', RandomForestRegressor())])
                 },
            'svm':
                {True: Pipeline([('standardizer', StandardScaler()), ('estimator', SVC())]),
                 False: Pipeline([('standardizer', StandardScaler()), ('estimator', SVR())])
                 },
            'xgb':
                {True: Pipeline([('standardizer', StandardScaler()), ('estimator', XGBClassifier())]),
                 False: Pipeline([('standardizer', StandardScaler()), ('estimator', XGBRegressor())])
                 },
        }

        return registry[regressor_name][is_categorical]
    except:
        raise KeyError(
            r"Instantiating a model from the model_registry has failed. Did you remember to update the registry with your new type of fitting algorithm?")


def scoring_strategy_defaults(is_categorical):
    if is_categorical == True:
        return 'matthews_corrcoef'
    elif is_categorical == False:
        return 'r2'
    else:
        raise ValueError(r"is_categorical has been set to a non-Boolean")


class Model:
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=4):

        self.df_training = df_training
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.n_jobs = n_jobs

        self.regressor_name = ''
        self.version = '0.0.1'
        self.model_obj = None
        self.embedding = None
        self.is_categorical = None
        self.training_stats = {}
        self.seed = 11171992

        self.description = None
        self.description_url = None
        self.hyperparameters = None
        self.params = None  # best params from the hyperparameter gridsearch

    def get_model(self):
        return self.model_obj

    def instantiate_model(self):
        self.model_obj = model_registry(self.regressor_name, self.is_categorical)
        return self.model_obj

    def getModelDescription(self):
        return ModelDescription(self).to_json()

    def build_model(self, descriptor_names=None):
        t1 = time.time()
        self.embedding = descriptor_names

        # Call prepare_instances without removing correlated descriptors
        if self.embedding == None:
            train_ids, train_labels, train_features, train_column_names, self.is_categorical = \
                DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names
        else:
            train_ids, train_labels, train_features, train_column_names, self.is_categorical = \
                DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", self.embedding)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names
        # Instantiate model from the registry
        self.model_obj = model_registry(self.regressor_name, self.is_categorical)

        # Tune hyperparameters
        # TMM: optimizer made local so won't get stored in the database
        # self.fix_hyperparameter_grid()

        optimizer = GridSearchCV(self.model_obj, self.hyperparameters, n_jobs=self.n_jobs,
                                     scoring=scoring_strategy_defaults(self.is_categorical))
        optimizer.fit(train_features, train_labels)

        self.model_obj.set_params(**optimizer.best_params_)
        self.params = optimizer.best_params_

        # Train the model on training data
        self.model_obj.fit(train_features, train_labels)
        training_score = self.model_obj.score(train_features, train_labels)
        self.training_stats['training_score'] = training_score

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        training_time = t2 - t1

        print('\n******************************************************************************************')
        print('Regressor', self.regressor_name)
        print('Best model params', self.params)
        print(r'Score for Training data = {score}'.format(score=training_score))
        print(r'Time to train model  = {training_time} seconds'.format(training_time=training_time))
        print('modelDescription', self.getModelDescription())
        print('******************************************************************************************\n')

        self.training_stats['training_time'] = t2 - t1
        return self

    # def fix_hyperparameter_grid(self):
    #
    #     if 'estimator__max_features' in self.hyperparameters:
    #
    #         for max_features in self.hyperparameters['estimator__max_features']:
    #             if max_features == 4 or max_features == 8:
    #                 if max_features > len(self.embedding):
    #                     self.hyperparameters['estimator__max_features'].remove(max_features)
    #
    #     print(self.hyperparameters)


    def do_predictions(self, df_prediction, return_score=False):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.embedding)
        # print ('pred version 1.4')
        if (self.is_categorical == True):
            predictions = self.model_obj.predict_proba(pred_features)[:, 1]
        else:
            predictions = self.model_obj.predict(pred_features)

        if self.is_categorical == True:
            preds = self.model_obj.predict(pred_features)
            score = balanced_accuracy_score(preds, pred_labels)
        elif self.is_categorical == False:
            score = self.model_obj.score(pred_features, pred_labels)
        else:
            pass

        print(r'Balanced Accuracy or COD for Test data = {score}'.format(score=score))
        # Return predictions
        if return_score == False:
            return predictions
        elif return_score == True:
            return score


class KNN(Model):
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'knn'
        self.version = '1.1'
        self.hyperparameters = {'estimator__n_neighbors': list(range(3, 11)),
                                'estimator__weights': ['uniform', 'distance']}
        self.description = 'sklearn implementation of k-nearest neighbors'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'


class XGB(Model):
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'xgb'
        self.version = '1.1'
        self.hyperparameters = {'estimator__booster':['gbtree', 'gblinear','dart']}
        # print(self.hyperparameters)

        self.description = 'python implementation of extreme gradient boosting'
        self.description_url = 'https://xgboost.readthedocs.io/en/latest/get_started.html'


class SVM(Model):
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=20):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = "svm"
        self.version = '1.3'

        self.c_space = list([10 ** x for x in range(-3, 3)])
        self.gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]
        self.gamma_space.append('scale')
        self.gamma_space.append('auto')
        self.hyperparameters = {"estimator__C": self.c_space, "estimator__gamma": self.gamma_space}

        # self.hyperparameters = {"estimator__C": [10 ** n for n in range(-3, 4)],
        #                         "estimator__kernel": ["linear", "poly", "rbf"],
        #                         "estimator__gamma": [10 ** n for n in range(-3, 4)]}


        self.description = 'sklearn implementation of SVM using NuSVR for regression' \
                           ' or SVC for classification'
        self.description_url = 'https://scikit-learn.org/stable/modules/svm.html'


class RF(Model):
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=20):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = "rf"
        self.version = '1.5'

        self.hyperparameters = {"estimator__max_features": ["sqrt", "log2"]}

        # self.hyperparameters = {"estimator__max_features": ["sqrt", "log2"],
        #                         'estimator__n_estimators': [10, 100, 250, 500]}

        # self.hyperparameters = {'estimator__max_features': ['sqrt', 'log2', 4],
        #                         'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                         'estimator__n_estimators': [10, 100, 250, 500]}

        self.qsar_method = 'Random forest'
        self.description = 'sklearn implementation of random forest'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""

        self.is_binary = model.is_categorical
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.regressor_name
        self.description = model.description
        self.description_url = model.description_url

        self.params = model.params # best params from the hyperparameter gridsearch

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)


def runExamples():
    # %% Test Script
    opera_path = r"C:\Users\ncharest\OneDrive - Environmental Protection Agency (EPA)\Profile\Documents\data_sets\OPERA_TEST_DataSetsBenchmark\DataSetsBenchmark\Water solubility OPERA\{filename}"
    training_df = DFU.load_df_from_file(opera_path.format(filename=r"Water solubility OPERA T.E.S.T. 5.1 training.tsv"),
                                        "\t")
    pred_df = DFU.load_df_from_file(opera_path.format(filename=r"Water solubility OPERA T.E.S.T. 5.1 prediction.tsv"),
                                    "\t")
    # Demonstrate KNN usage
    print(r"Executing KNN")
    model = KNN(training_df, False)
    model.build_model()  # Note we now handle using an embedding by passing a descriptor_names list. By default it is a None type -- this will use all descriptors in df
    predictions = model.do_predictions(pred_df)
    test_score = model.do_predictions(pred_df)
    # Demonstrate RF usage
    print(r"Executing RF")
    model = SVM(training_df, False)
    model.build_model()
    predictions = model.do_predictions(pred_df)
    test_score = model.do_predictions(pred_df)
    # Demonstrate SVM usage
    print(r"Executing SVM")
    model = SVM(training_df, False)
    model.build_model()
    predictions = model.do_predictions(pred_df)
    test_score = model.do_predictions(pred_df)


if __name__ == "__main__":
    runExamples()
