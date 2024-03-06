# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:14:14 2023

@author: NCHAREST
This is a refactored version of the python model building repo's model object, intended to better utilize OOP and more elegant design patterning.'
"""
import math
import time

import pypmml
import sklearn2pmml
from scipy import stats

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn2pmml.pipeline import PMMLPipeline as PMMLPipeline

from models import df_utilities as DFU
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor, XGBClassifier

from sklearn_pmml_model.ensemble import PMMLForestClassifier
from sklearn_pmml_model.ensemble import PMMLForestRegressor
from sklearn_pmml_model.ensemble import PMMLGradientBoostingClassifier
from sklearn_pmml_model.ensemble import PMMLGradientBoostingRegressor
from sklearn_pmml_model.svm import PMMLSVC
from sklearn_pmml_model.svm import PMMLSVR
from sklearn_pmml_model.neighbors import PMMLKNeighborsRegressor
from sklearn_pmml_model.neighbors import PMMLKNeighborsClassifier

from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score

import numpy as np

from os.path import exists
import json

__author__ = "Nathaniel Charest, Todd Martin (modified to work with webservice, added XGB)"


# use_standardizer = True

# %%

def model_registry_model_obj(regressor_name, is_categorical):
    '''
    model registry for getting model_obj (not pipeline)
    :param regressor_name:
    :param is_categorical:
    :return: model obj
    '''

    if regressor_name == 'knn':
        if is_categorical:
            return KNeighborsClassifier()
        else:
            return KNeighborsRegressor()
    elif regressor_name == 'rf':
        if is_categorical:
            return RandomForestClassifier()
        else:
            return RandomForestRegressor()
    elif regressor_name == 'svm':
        if is_categorical:
            return SVC(probability=True)
        else:
            return SVR()
    elif regressor_name == 'xgb':
        if is_categorical:
            return XGBClassifier()
        else:
            return XGBRegressor()
    elif regressor_name == 'reg':
        if is_categorical:
            return LogisticRegression()
        else:
            return LinearRegression()

    else:
        raise KeyError(
            r"Instantiating a model from the model_registry has failed. Did you remember to update the registry with your new type of fitting algorithm?")


def scoring_strategy_defaults(is_categorical):
    # to see available metrics:
    # import sklearn
    # print(sorted(sklearn.metrics.SCORERS.keys()))

    if is_categorical == True:
        return 'balanced_accuracy'  # TMM: Changed to BA since "matthews_corrcoef" was not an available metric
    elif is_categorical == False:
        return 'r2'
    else:
        raise ValueError(r"is_categorical has been set to a non-Boolean")


class Model:
    def __init__(self, df_training, remove_log_p_descriptors, n_jobs=4):

        self.training_descriptor_std_devs = None  # standard deviations of training set descriptors
        self.training_descriptor_means = None  # means of training set descriptors
        self.df_training = df_training
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.n_jobs = n_jobs

        self.regressor_name = ''
        self.version = '0.0.1'
        self.model_obj = None
        self.embedding = None
        self.is_binary = None
        self.training_stats = {}
        self.seed = 11171992

        self.description = None
        self.description_url = None
        self.hyperparameter_grid = None
        self.hyperparameters = None  # best params from the hyperparameter gridsearch
        self.use_pmml = None
        self.use_sklearn2pmml = None


    def get_model(self):
        return self.model_obj

    def set_model_obj_pmml_for_prediction(self, pmml_file_path, qsar_method):

        # print(type(self.model_obj))
        # print(pmml_file_path)
        # self.model_obj = Model_pypmml.fromFile(pmml_file_path)
        # if True:
        #     return

        if self.is_binary:
            if qsar_method == 'rf':
                self.model_obj = PMMLForestClassifier(pmml=pmml_file_path)
            elif qsar_method == 'xgb':
                self.model_obj = PMMLGradientBoostingClassifier(pmml=pmml_file_path)
            elif qsar_method == 'svm':
                self.model_obj = PMMLSVC(pmml=pmml_file_path)
            elif qsar_method == 'knn':
                self.model_obj = PMMLKNeighborsClassifier(pmml=pmml_file_path)
        else:
            if qsar_method == 'rf':
                self.model_obj = PMMLForestRegressor(pmml=pmml_file_path)
            elif qsar_method == 'xgb':
                self.model_obj = PMMLGradientBoostingRegressor(pmml=pmml_file_path)
            elif qsar_method == 'svm':
                self.model_obj = PMMLSVR(pmml=pmml_file_path)
            elif qsar_method == 'knn':
                self.model_obj = PMMLKNeighborsRegressor(pmml=pmml_file_path)

    def set_details(self, details):
        print('\nenter set_details')

        for key in details:
            if key == 'model':
                continue
            print(key, type(details[key]), details[key])
            value = details[key]

            # if key =='embedding':
            #     print(len(value))

            setattr(self, key, value)

    def get_model_description(self):
        return ModelDescription(self).to_json()

    def has_hyperparameter_grid(self):
        """
        Whether or not there are multiple hyperparameter grid options per parameter
        :return:
        """
        for key in self.hyperparameter_grid:
            if len(self.hyperparameter_grid[key]) > 1:
                return True
        return False

    def get_single_parameters(self):
        """
        Get hyperparameters from a grid with a single set of options
        :return:
        """
        parameters = {}

        for key in self.hyperparameter_grid:
            parameters[key] = self.hyperparameter_grid[key][0]
        return parameters

    def build_model(self, use_pmml_pipeline, include_standardization_in_pmml, descriptor_names=None):
        print('enter build model')

        t1 = time.time()
        self.embedding = descriptor_names
        self.use_pmml = use_pmml_pipeline
        self.include_standardization_in_pmml = include_standardization_in_pmml


        # Call prepare_instances without removing correlated descriptors
        if self.embedding is None:
            train_ids, train_labels, train_features, train_column_names, self.is_binary = \
                DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, True)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names

            # print(self.embedding)

        else:
            train_ids, train_labels, train_features, train_column_names, self.is_binary = \
                DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", self.embedding)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names

        # print(train_features)
        if use_pmml_pipeline and include_standardization_in_pmml is False:  # need to handle scaling outside of pipeline
            ss = StandardScaler()
            train_features = pd.DataFrame(ss.fit_transform(train_features), columns=train_features.columns)
            self.training_descriptor_means = ss.mean_.tolist()
            self.training_descriptor_std_devs = (ss.var_ ** 0.5).tolist()

        # print(train_features)

        # Instantiate model from the registry
        # self.instantiate_model() # already called in mwsutility instantiatemodel

        # Tune hyperparameters
        # TMM: optimizer made local so won't get stored in the database
        # self.fix_hyperparameter_grid()

        if self.regressor_name == 'svm':
            if len(train_labels) > 20000:
                self.c_space = [100]
                self.gamma_space = ['auto']
                self.hyperparameter_grid = {"estimator__C": self.c_space, "estimator__gamma": self.gamma_space}
                print('using single set of hyperparameters for SVM due to large data set')

        print('hyperparameter_grid', self.hyperparameter_grid)

        if self.has_hyperparameter_grid():
            print('Hyperparameter grid has multiple sets of parameters, running grid search')

            kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)

            optimizer = GridSearchCV(self.model_obj, self.hyperparameter_grid, n_jobs=self.n_jobs,
                                     scoring=scoring_strategy_defaults(self.is_binary), cv=kfold_splitter)
            # optimizer = GridSearchCV(self.model_obj, self.hyperparameter_grid, n_jobs=self.n_jobs,
            #                              scoring=scoring_strategy_defaults(self.is_categorical))

            optimizer.fit(train_features, train_labels)

            self.model_obj.set_params(**optimizer.best_params_)
            self.hyperparameters = optimizer.best_params_

        else:
            print('Hyperparameter grid only has a single set of parameters, skipping grid search')
            self.hyperparameters = self.get_single_parameters()
            self.model_obj.set_params(**self.hyperparameters)

        # self.generate_cv_predictions(kfold_splitter,train_ids, train_features, train_labels)

        # Train the model on training data

        self.model_obj.fit(train_features, train_labels)

        # training_score = self.model_obj.score(train_features, train_labels)
        # self.training_stats['training_score'] = training_score
        # df_results=pd.DataFrame(results.cv_results_)
        # self.training_stats['training_cv_score'] = optimizer.best_score_
        # cv_score = cross_val_score(self.model_obj, train_features, train_labels, cv=kfold_splitter)
        # self.training_stats['cross_val_score'] = list(cv_score)
        # self.training_stats['oob_score'] = self.model_obj.oob_score_  #only for RF
        # print(self.training_stats)

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        training_time = t2 - t1

        print('\n******************************************************************************************')
        print('Regressor', self.regressor_name)
        print('Best model params', self.hyperparameters)
        # print('training_cv_r2',self.training_stats['training_cv_r2'])
        # print('training_cv_q2', self.training_stats['training_cv_q2'])

        # print(r'Score for Training data = {score}'.format(score=training_score))
        print(r'Time to train model  = {training_time} seconds'.format(training_time=training_time))
        print('modelDescription', self.get_model_description())
        print('******************************************************************************************\n')

        self.training_stats['training_time'] = t2 - t1
        return self

    def build_cv_model(self, use_pmml_pipeline, descriptor_names=None, params={}):
        t1 = time.time()

        self.embedding = descriptor_names

        # Call prepare_instances without removing correlated descriptors
        if self.embedding is None:
            train_ids, train_labels, train_features, train_column_names, self.is_binary = \
                DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, True)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names
        else:
            train_ids, train_labels, train_features, train_column_names, self.is_binary = \
                DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", self.embedding)
            # Use columns selected by prepare_instances (in case logp descriptors were removed)
            self.embedding = train_column_names

        if use_pmml_pipeline: # do scaling here since not part of pipeline
            ss = StandardScaler()
            train_features = pd.DataFrame(ss.fit_transform(train_features), columns=train_features.columns)
            self.training_descriptor_means = ss.mean_.tolist()
            self.training_descriptor_std_devs = (ss.var_ ** 0.5).tolist() #TODO should it instead use the overall training set stdev instead of the CV training set stdev? Sometimes the CV will have a zero stddev during CV

        self.model_obj.set_params(**params)

        # Train the model on training data
        self.model_obj.fit(train_features, train_labels)

    # def generate_cv_predictions(self, kfold_splitter, train_ids, train_features, train_labels):
    #     """
    #     @author Todd Martin
    #     Does 5 fold CV of training set using final params and splitting
    #     Reports stats and the experimental and predicted values
    #
    #     This method currently doesnt not generate the same CV split every time. It has been replaced by running build_cv_modell
    #
    #     training_cv_r2  = Squared Pearson coeff = coefficient of determination
    #     training_cv_q2  = Q2_F3 see eqn 2 of Consonni et al, 2019 (https://onlinelibrary.wiley.com/doi/full/10.1002/minf.201800029)
    #
    #     training_cv_q2 = average value of Q2_F3 for the 5 folds
    #
    #     From OPERA paper:
    #     Different ways of calculating Q2 are available in the literature [50, 61, 62]. However, because RMSEP (and RMSE)
    #     depends on the scale reference, Q2 must fulfill the ability of R2 to be independent of the response scale [51].
    #     Also, to be a subjective representative of the true predictivity of a model, Q2 must be invariant for a fixed
    #     RMSEP value, be invariant to the splitting of the external data into subsets (ergodic principle), and correlate
    #     perfectly with RMSEP. This study used the Q2 formula below demonstrated by Todeschini et al. [51] because it is
    #     the only formula that fulfils all these requirements.
    #
    #     :param train_ids: smiles or DTXCIDs for training set
    #     :param kfold_splitter: splitter used to split into 5 folds
    #     :param train_features: X matrix for training set as dataframe
    #     :param train_labels: Experimental values for training set as list
    #     :return:
    #     """
    #     df_results_cv = pd.DataFrame()
    #
    #     q2 = 0  # initialize average q2
    #
    #     for i, (train_index, test_index) in enumerate(kfold_splitter.split(train_features)):
    #         df_exp = pd.DataFrame(train_labels, columns=['exp'])
    #         df_exp_train = df_exp.filter(items=train_index, axis=0)
    #         df_exp_test = df_exp.filter(items=test_index, axis=0)
    #
    #         df_ids = pd.DataFrame(train_ids, columns=['id'])
    #         df_ids_test = df_ids.filter(items=test_index, axis=0)
    #
    #         X_training = train_features.filter(items=train_index, axis=0)
    #         X_prediction = train_features.filter(items=test_index, axis=0)
    #
    #         self.model_obj.fit(X_training, df_exp_train)
    #         pred = self.model_obj.predict(X_prediction)
    #
    #         df_pred = pd.DataFrame(pred, columns=['pred'])
    #         df_exp_test.reset_index(drop=True, inplace=True)  # need to reset index otherwise it wont line up rows correctly
    #         df_pred.reset_index(drop=True, inplace=True)
    #         df_ids_test.reset_index(drop=True, inplace=True)
    #
    #         Ybar_tr = df_exp_train['exp'].sum()/df_exp_train.shape[0]
    #         sumN = ((df_exp_test['exp'] - df_pred['pred']) ** 2).sum() / df_pred.shape[0]
    #         sumD = ((df_exp_train['exp'] - Ybar_tr) ** 2).sum() / df_exp_train.shape[0]
    #
    #         q2i = 1 - sumN / sumD
    #
    #         q2 = q2 + q2i
    #
    #         colId = list(df_ids_test['id'])
    #         colExp = list(df_exp_test['exp'])
    #         colPred = list(df_pred['pred'])
    #         df_results = pd.DataFrame(list(zip(colId, colExp, colPred)), columns=['id', 'exp', 'pred'])
    #         df_results['split'] = i + 1
    #         r = df_results['exp'].corr(df_results['pred'])
    #
    #         r2 = r**2
    #         # print('r2 fold' + str(i), r2)
    #
    #         df_results_cv = df_results_cv.append(df_results)
    #
    #     # print(df_results_cv)
    #
    #     q2 = q2 / 5 # average value over 5 folds
    #     r = df_results_cv['exp'].corr(df_results_cv['pred'])
    #     self.training_stats['training_cv_q2'] = q2
    #     self.training_stats['training_cv_r2'] = r**2
    #     self.training_stats['training_cv_predictions'] = df_results_cv.to_dict('records')

    # print(df_results_cv.to_csv(index=False))

    # def fix_hyperparameter_grid(self):
    #
    #     if 'estimator__max_features' in self.hyperparameter_grid:
    #
    #         for max_features in self.hyperparameter_grid['estimator__max_features']:
    #             if max_features == 4 or max_features == 8:
    #                 if max_features > len(self.embedding):
    #                     self.hyperparameter_grid['estimator__max_features'].remove(max_features)
    #
    #     print(self.hyperparameter_grid)

    def do_predictions(self, df_prediction, return_score=False):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.embedding)

        # print('pred_labels',pred_labels)

        # print('Enter model.do_predictions')
        # print('Model description = ',self.get_model_description())
        # print('Model type = ', type(self.model_obj))

        # print(len(self.training_descriptor_means), len(self.training_descriptor_stds))

        # Perform scaling

        if hasattr(self, 'training_descriptor_means') and self.training_descriptor_means:
            print('normalizing prediction set using means and stdevs')

            # print(self.training_descriptor_means)
            # print(self.training_descriptor_std_devs)

            pred_features = pd.DataFrame(
                (np.array(pred_features) - self.training_descriptor_means) / self.training_descriptor_std_devs,
                columns=self.embedding)

        # print('pred_features after scaling')
        # print(pred_features)
        # print(predictions)
        # print('is_categorical', self.is_binary)

        if self.is_binary:

            if isinstance(self.model_obj, pypmml.Model):
                predictions = self.model_obj.predict(pred_features)
                predictions = np.array(predictions[predictions.columns[1]])  # probability of score=1 (continuous value) # TODO this might not work directly with kNN
                preds = np.rint(predictions)  # convert to integer to allow BA calculation to work
                # print(preds)
                score = balanced_accuracy_score(pred_labels, preds)
            elif isinstance(self.model_obj, Pipeline) or isinstance(self.model_obj, PMMLPipeline) or 'PMML' in type(
                    self.model_obj).__name__:
                predictions = self.model_obj.predict_proba(pred_features)[:,1]  # probability of score=1 (continuous value)

                # preds = self.model_obj.predict(pred_features) # generate integer values to allow BA calculation to work
                preds = np.rint(
                    predictions)  # convert to integer to allow BA calculation to work (faster than running predict)
                # print(preds)
                score = balanced_accuracy_score(pred_labels, preds)
            else:
                print("Cant handle ", type(self.model_obj))

            print(r'Balanced Accuracy for Test data = {score}'.format(score=score))

        elif not self.is_binary:

            predictions = self.model_obj.predict(pred_features)
            # print([predictions])
            # print(type(self.model_obj).__name__)

            if isinstance(self.model_obj, pypmml.Model):
                predictions = np.array(predictions[predictions.columns[0]]) # TODO this might not work directly with kNN
            elif isinstance(self.model_obj, Pipeline) or isinstance(self.model_obj, PMMLPipeline) or 'PMML' in type(
                    self.model_obj).__name__:
                predictions = np.array(predictions)
            else:
                print("Cant handle ", type(self.model_obj))

            # print(predictions)
            # print(pred_labels)

            score = stats.pearsonr(predictions, pred_labels)[0]
            score = score * score

            print(r'R2 for Test data = {score}'.format(score=score))

        else:
            print("is_categorical is null")  # does this happen?
            pass

        # df = pd.DataFrame()
        # df['exp'] = pred_labels
        # df['pred'] = predictions
        # print(df.to_csv(index=False))

        # print(predictions)

        # Return predictions
        if not return_score:
            return predictions
        elif return_score:
            return score

    def do_predictions_RMSE(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.embedding)
        # print ('pred version 1.4')

        predictions = self.model_obj.predict(pred_features)

        RMSE = 0
        for index, pred in enumerate(predictions):
            exp = pred_labels[index]
            error = (exp - pred) * (exp - pred)
            RMSE = RMSE + error
            # print(exp,pred)

        RMSE = math.sqrt(RMSE / len(predictions))
        return RMSE


class KNN(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)

        self.regressor_name = 'knn'
        self.version = '1.2'

        # self.hyperparameter_grid = {'estimator__n_neighbors': [5],
        #                         'estimator__weights': ['uniform', 'distance']}
        self.hyperparameter_grid = {'estimator__n_neighbors': [5], 'estimator__weights': [
            'distance']}  # keep it consistent between endpoints, match OPERA

        self.description = 'sklearn implementation of k-nearest neighbors'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html'


class REG(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'reg'
        self.version = '1.0'
        self.hyperparameter_grid = {}  # keep it consistent between endpoints, match OPERA

        self.description = 'python implementation of regression'
        self.description_url = 'https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model'


class XGB(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'xgb'
        self.version = '1.3'

        # self.hyperparameter_grid = {'estimator__booster':['gbtree', 'gblinear','dart']}  #other two make it run a lot slower
        self.hyperparameter_grid = {'estimator__booster': ['gbtree']}

        self.description = 'python implementation of extreme gradient boosting'
        self.description_url = 'https://xgboost.readthedocs.io/en/latest/get_started.html'


class SVM(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=20):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = "svm"
        self.version = '1.4'

        # Following grid takes way too long:
        # self.c_space = list([10 ** x for x in range(-3, 3)])
        # self.gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]
        # self.gamma_space.append('scale')
        # self.gamma_space.append('auto')

        # self.hyperparameter_grid = {"estimator__C": [10 ** n for n in range(-3, 4)],
        #                         "estimator__kernel": ["linear", "poly", "rbf"],
        #                         "estimator__gamma": [10 ** n for n in range(-3, 4)]}

        self.c_space = [1, 10, 100]
        self.gamma_space = ['scale', 'auto']
        self.hyperparameter_grid = {"estimator__C": self.c_space, "estimator__gamma": self.gamma_space}

        self.description = 'sklearn implementation of SVM using SVR for regression' \
                           ' or SVC for classification'
        self.description_url = 'https://scikit-learn.org/stable/modules/svm.html'


class RF(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=20):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = "rf"

        # following didnt seem to help at all for predicting PFAS properties:
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"],
        #                         'estimator__n_estimators': [10, 100, 200, 400],
        #                         'estimator__min_samples_leaf': [1, 2, 4, 8]}

        # self.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2', 4],
        #                         'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],   append 0!
        #                         'estimator__n_estimators': [10, 100, 250, 500]}

        self.version = '1.6'
        self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"]}

        # self.version = '1.7'
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"], "estimator__max_depth": [1, 5, 10, 50,100,500]}
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"],
        #                             "estimator__n_estimators": [10, 50, 100]}

        self.description = 'sklearn implementation of random forest'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the current method"""

        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.regressor_name
        self.description = model.description
        self.description_url = model.description_url
        self.hyperparameter_grid = model.hyperparameter_grid
        self.hyperparameters = model.hyperparameters  # final hyperparameters
        self.training_stats = model.training_stats
        self.embedding = model.embedding
        self.use_pmml = model.use_pmml

        if hasattr(model, "training_descriptor_std_devs"):
            self.include_standardization_in_pmml = False
            self.training_descriptor_std_devs = model.training_descriptor_std_devs
            self.training_descriptor_means = model.training_descriptor_means
        else:
            self.include_standardization_in_pmml = True

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
