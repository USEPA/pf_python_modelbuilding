# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:14:14 2023

@author: NCHAREST
This is a refactored version of the python model building repo's model object, intended to better utilize OOP and more elegant design patterning.'
"""
import logging
import math
import time

import pypmml

from scipy import stats

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn2pmml.pipeline import PMMLPipeline as PMMLPipeline

import model_ws_utilities
from models import df_utilities as DFU, df_utilities
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
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

importance_type = 'weight' #default, TODO make a passable parameter to app when generating embedding
# importance_type = 'gain'
# importance_type = 'cover'
# importance_type = 'total_gain'
# importance_type ='total_cover'


def model_registry_model_obj(regressor_name, is_categorical):
    '''
    model registry for getting model_obj (not pipeline)
    :param regressor_name:
    :param is_categorical:
    :return: model obj
    TODO move to each model class and remove this method?
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
            # return XGBClassifier()
            # return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            return XGBClassifier(eval_metric='logloss')
            # return XGBClassifier(use_label_encoder=False, eval_metric='auc')
        else:
            return XGBRegressor(importance_type=importance_type)
    elif regressor_name == 'lgb':
        if is_categorical:
            # return XGBClassifier()
            # return XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            return LGBMClassifier(eval_metric='logloss')
            # return XGBClassifier(use_label_encoder=False, eval_metric='auc')
        else:
            return LGBMRegressor(eval_metric='rmse')
    elif regressor_name == 'reg':
        # return LinearRegression()
        if is_categorical:
            # return LogisticRegression(penalty='none')
            return LogisticRegression(max_iter=1000)
        else:
            return LinearRegression()
    elif regressor_name == 'las':
        if is_categorical:
            return Lasso()
        else:
            return Lasso()

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
        # return 'r2'
        return 'neg_root_mean_squared_error'
    else:
        raise ValueError(r"is_categorical has been set to a non-Boolean")


class Model:


    def __init__(self, df_training=None, remove_log_p_descriptors=None, n_jobs=4):

        self.training_descriptor_std_devs = None  # standard deviations of training set descriptors
        self.training_descriptor_means = None  # means of training set descriptors
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.n_jobs = n_jobs

        self.regressor_name = '' #TODO set to None instead?
        self.qsar_method = None
        self.qsar_method_version = None

        self.model_obj = None
        self.embedding = None
        # self.coeff = None
        self.is_binary = None
        self.training_stats = {}
        self.seed = 11171992

        self.description = None
        self.description_url = None
        self.hyperparameter_grid = None
        self.hyperparameters = None  # best params from the hyperparameter gridsearch
        self.use_pmml = None
        self.use_sklearn2pmml = None

        #Extra metadata 2025-12-08
        self.modelId = None
        self.modelName = None
        self.modelSource = None
        self.modelStatistics = None
        
        self.modelMethod = None
        self.modelMethodDescription = None
        self.modelMethodDescriptionURL = None
        
        self.datasetId = None
        self.datasetName = None
        self.unitsModel = None
        self.unitsDisplay = None
        self.dsstoxMappingStrategy = None
        
        self.propertyName = None
        self.propertyDescription = None

        self.descriptorSetId = None
        self.descriptorSetName = None
        self.descriptorService = None
        self.headersTsv = None
        self.splittingId = None
        self.splittingName = None
        self.applicabilityDomainName = None
        self.applicabilityDomainDescription = None
        self.omitSalts = None
        self.qsarReadyRuleSet = None
                

        self.df_dsstoxRecords = None
        self.df_training = df_training
        self.df_prediction = None
        
        self.num_training = None
        self.num_prediction = None
                
        self.df_preds_test = None # external predictions for test set
        self.df_preds_training_cv = None #cross validation predictions for training set
        
        self.detailsFile = None


    def get_model(self):
        return self.model_obj


    def to_dict(self):
        """Convert the object to a dictionary.
        \nCommenting out the ones we dont need to see in ModelResults"""

        return {
            "modelId": self.modelId,
            "modelName": self.modelName,
            # "datasetId": self.datasetId,
            "datasetName": self.datasetName,
            "unitsName": self.unitsName,
            # "dsstox_mapping_strategy": self.dsstox_mapping_strategy,
            "propertyName": self.propertyName,
            # "descriptorSetId": self.descriptorSetId,
            # "descriptorSetName": self.descriptorSetName,
            "descriptorService": self.descriptorService,
            # "headersTsv": self.headersTsv,  #dont need to export via API
            # "splittingId": self.splittingId,
            "splittingName": self.splittingName,
            "applicabilityDomainName": self.applicabilityDomainName,
            "descriptorEmbeddingTsv": self.descriptorEmbeddingTsv,
            "omitSalts": self.omitSalts,
            "qsarReadyRuleSet": self.qsarReadyRuleSet,
            # "trainingSet": self.trainingSet,
            # "predictionSet": self.predictionSet

            #TODO add following
            # self.regressor_name = ''
            # self.version = '0.0.1'
            # self.model_obj = None
            # self.embedding = None
            # self.is_binary = None
            # self.training_stats = {}
            # self.description = None
            # self.description_url = None
            # self.hyperparameter_grid = None
            # self.hyperparameters = None  # best params from the hyperparameter gridsearch
            # self.use_pmml = None
            # self.use_sklearn2pmml = None


        }


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
        logging.debug('\nenter set_details')

        for key in details:
            if key == 'model':
                continue
            print(key, type(details[key]), details[key])
            value = details[key]

            # if key =='embedding':
            #     print(len(value))

            setattr(self, key, value)

    def get_model_description(self):
        modelDescription = ModelDescription(self)
        return json.dumps(modelDescription.__dict__)
    
    def get_model_description_dict(self):
        modelDescription = ModelDescription(self)
        return modelDescription.__dict__


    def get_model_description_pretty(self):
        modelDescription = ModelDescription(self)
        return json.dumps(modelDescription.__dict__, indent=4)


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
        logging.debug('enter build model')

        t1 = time.time()
        self.embedding = descriptor_names
        self.use_pmml = use_pmml_pipeline
        self.include_standardization_in_pmml = include_standardization_in_pmml

        # Call prepare_instances without removing correlated descriptors
        if self.embedding is None:
            train_ids, train_labels, train_features, train_column_names, self.is_binary = \
                DFU.prepare_instances(self.df_training, "training", remove_logp=self.remove_log_p_descriptors, remove_corr=True)
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

        if use_pmml_pipeline:  # do scaling here since not part of pipeline
            ss = StandardScaler()
            train_features = pd.DataFrame(ss.fit_transform(train_features), columns=train_features.columns)
            self.training_descriptor_means = ss.mean_.tolist()
            self.training_descriptor_std_devs = (
                    ss.var_ ** 0.5).tolist()  # TODO should it instead use the overall training set stdev instead of the CV training set stdev? Sometimes the CV will have a zero stddev during CV

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



    def create_plot(self, df_training, df_prediction, model_name, plot_type):

        train_ids, train_labels, train_features = DFU.prepare_prediction_instances(df_training, self.embedding) #only uses model descriptors
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.embedding)

        # train_ids, train_labels, train_features,train_columns,train_is_binary = DFU.prepare_instances(df_training, 'training', remove_logp=False, remove_corr=False, remove_constant=False)
        # pred_ids, pred_labels, pred_features,pred_columns,pred_is_binary = DFU.prepare_instances(df_prediction, 'training', remove_logp=False, remove_corr=False, remove_constant=False)

        print('running the fit...')

        if plot_type == 'PCA':
            from sklearn.decomposition import PCA
            # Initialize PCA
            pca = PCA(n_components=2)
            # Fit PCA on the training data
            X_train = pca.fit_transform(train_features)
            # Transform the test data
            X_test = pca.transform(pred_features)

        elif plot_type == 'UMAP':
            import umap
            umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15,min_dist=0.1)
            # Fit and transform the training data
            X_train = umap_model.fit_transform(train_features)
            # Transform the test data
            X_test = umap_model.transform(pred_features)

        elif plot_type =='UMAP3d':
            # Fit UMAP to the training set
            import umap
            umap_3d = umap.UMAP(n_components=3, random_state=42)

            X_train = umap_3d.fit_transform(train_features)

            # Transform the prediction set using the fitted UMAP
            X_test = umap_3d.transform(pred_features)


        elif plot_type == 't-SNE':
            from sklearn.manifold import TSNE

            perplexity = 10
            # perplexity = 30
            # if df_prediction.shape[0] < perplexity:
            #     perplexity = df_prediction.shape[0] / 2
            # Initialize t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)

            print('running the tsne fit')
            # Fit and transform the data
            X_train = tsne.fit_transform(train_features)

            # Separate Transformations: t-SNE is typically not used to transform new data after fitting because
            # it doesn't maintain an internal model for further transformations. However, for visualization, you can
            # fit-transform separately to compare sets

            X_test = tsne.fit_transform(pred_features)

        print('done')

        # Create a scatter plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))

        fontsize = 16
        # Add legend, labels, and title
        plt.legend(fontsize=fontsize)

        title = plot_type + ' Plot for ' + model_name
        plt.title(title, fontsize=fontsize)


        useLogScale = False

        if useLogScale:
            plt.xscale('log')
            plt.yscale('log')

        if '3d' in plot_type:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot training data
            ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=train_labels, cmap='viridis', alpha=0.7,
                        label='Training Set')
            # Plot test data
            ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=pred_labels, cmap='plasma', alpha=1.0, marker='x',
                        label='Prediction Set')

        else:
            # Plot training data
            plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='viridis', alpha=0.7, label='Training Set')

            # Plot test data
            plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_labels, cmap='plasma', alpha=1.0, marker='x',
                        label='Test Set')

        output_folder = 'datasets_v1_modeling/plots '+plot_type

        if plot_type == 'PCA':
            plt.xlabel('Principal Component 1', fontsize=fontsize)
            plt.ylabel('Principal Component 2', fontsize=fontsize)
        elif plot_type == 'UMAP':
            plt.xlabel('UMAP Component 1', fontsize=fontsize)
            plt.ylabel('UMAP Component 2', fontsize=fontsize)

        elif plot_type == 'UMAP3d':
            ax.set_xlabel('UMAP Component 1', fontsize=fontsize)
            ax.set_ylabel('UMAP Component 2', fontsize=fontsize)
            ax.set_zlabel('UMAP Component 3', fontsize=fontsize)
            ax.set_title(title, fontsize=fontsize)
            ax.legend()

        elif plot_type == 't-SNE':
            plt.xlabel('UMAP Component 1', fontsize=fontsize)
            plt.ylabel('UMAP Component 2', fontsize=fontsize)


        import os
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, title + '.png')
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')

        if 'All But' in model_name:
            plt.show()
        # Show the plot

        return "ok"


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
                predictions = np.array(predictions[predictions.columns[
                    1]])  # probability of score=1 (continuous value) # TODO this might not work directly with kNN
                preds = np.rint(predictions)  # convert to integer to allow BA calculation to work
                # print(preds)
                score = balanced_accuracy_score(pred_labels, preds)
            elif isinstance(self.model_obj, Pipeline) or isinstance(self.model_obj, PMMLPipeline) or 'PMML' in type(
                    self.model_obj).__name__:
                predictions = self.model_obj.predict_proba(pred_features)[:,
                              1]  # probability of score=1 (continuous value)

                # preds = self.model_obj.predict(pred_features) # generate integer values to allow BA calculation to work
                preds = np.rint(
                    predictions)  # convert to integer to allow BA calculation to work (faster than running predict)
                # print(preds)
                score = balanced_accuracy_score(pred_labels, preds)
            else:
                print("Cant handle ", type(self.model_obj))

            print(r'Balanced Accuracy for Test data = {score}'.format(score=score))

        elif not self.is_binary:

            # inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
            # pred_features.to_csv(inputFolder+'predfeatures.csv')

            predictions = self.model_obj.predict(pred_features)
            # print([predictions])
            # print(type(self.model_obj).__name__)

            if isinstance(self.model_obj, pypmml.Model):
                predictions = np.array(
                    predictions[predictions.columns[0]])  # TODO this might not work directly with kNN
            elif isinstance(self.model_obj, Pipeline) or isinstance(self.model_obj, PMMLPipeline) or 'PMML' in type(
                    self.model_obj).__name__:
                predictions = np.array(predictions)
            else:
                print("Cant handle ", type(self.model_obj))

            # print(predictions)
            # print(pred_labels)

            if df_prediction.shape[0] > 1:
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
        self.qsar_method_version = '1.2'

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

    def getOriginalRegressionCoefficients(self):

        # print('enter getOriginalRegressionCoefficients')

        model_obj = self.get_model()
        reg = model_obj.steps[1][1]
        scale = model_obj.steps[0][1]
        
        # Get the scaled coefficients and intercept
        beta_scaled = reg.coef_
        intercept_scaled = reg.intercept_

        # Get the means and standard deviations used by the StandardScaler
        means = scale.mean_
        stds = scale.scale_

        # Transform the coefficients to the unscaled version
        beta_unscaled = beta_scaled / stds

        # Transform the intercept to the unscaled version
        intercept_unscaled = intercept_scaled - np.sum((means * beta_scaled) / stds)

        # Report the unscaled coefficients and intercept
        # print("Intercept (unscaled):", intercept_unscaled)
        # print("Coefficients (unscaled):", beta_unscaled)
        # print(self.embedding)

        from collections import OrderedDict
        coefficients_dict = OrderedDict()

        # Create a dictionary for the coefficients, starting with the intercept
        coefficients_dict['Intercept'] = intercept_unscaled
        
        # Add the coefficients in the order of embedding
        coefficients_dict.update(dict(zip(self.embedding, beta_unscaled)))        
        # print(coefficients_dict)
        # return coefficients_dict
        return json.dumps(coefficients_dict,indent=4)


class LAS(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'las'
        self.qsar_method_version = '1.0'
        # self.hyperparameter_grid = {'estimator__alpha': [np.round(i, 5) for i in np.logspace(-5, 0, num=26)],'estimator__max_iter': [1000000]}
        self.hyperparameter_grid = {'estimator__alpha': [np.round(i, 5) for i in np.logspace(-4, 0, num=20)],
                                    'estimator__max_iter': [1000000]}
        # self.hyperparameter_grid = {'estimator__alpha': [np.round(i, 4) for i in np.linspace(0,1,10000)],'estimator__max_iter': [1000000]}
        # self.hyperparameter_grid = {'estimator__alpha': [np.round(i, 5) for i in np.logspace(-4, 0, num=50)],
        #                            'estimator__max_iter': [1000000], 'estimator__tol': [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]}
        # self.hyperparameter_grid = {}

        print(self.hyperparameter_grid)

        self.description = 'python implementation of lasso'
        self.description_url = 'https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model'


class XGB(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'xgb'

        # self.hyperparameter_grid = {'estimator__booster':['gbtree', 'gblinear','dart']}  #other two make it run a lot slower

        self.version = '1.3'
        self.hyperparameter_grid = {'estimator__booster': ['gbtree']}

        # self.version = '1.4'
        # self.hyperparameter_grid = {'estimator__n_estimators': [50, 100], 'estimator__eta': [0.1, 0.2, 0.3],
        #                             'estimator__gamma': [0, 1, 10], 'estimator__max_depth': [3, 6, 9, 12],
        #                             'estimator__min_child_weight': [1, 3, 5], 'estimator__subsample': [0.5, 1]}

        self.description = 'python implementation of extreme gradient boosting'
        self.description_url = 'https://xgboost.readthedocs.io/en/stable/tutorials/model.html'


class LGB(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=1):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = 'lgb'

        # self.hyperparameter_grid = {'estimator__booster':['gbtree', 'gblinear','dart']}  #other two make it run a lot slower

        self.self.qsar_method_version = '1.0'

        # 'weight': The default for , this represents the number of times a feature is used to split data across all trees.
        # 'gain': The default for the scikit-learn API's attribute, this is the average gain across all splits where the feature is used. Gain is the improvement in accuracy from a feature on its branches.
        # 'cover': Represents the average coverage across all splits where the feature is used. Coverage is the number of samples affected by the split.
        # 'total_gain': The total gain across all splits where the feature is used. It's the sum of gain across all splits in all trees.
        # 'total_cover': The total coverage across all splits where the feature is used. It's the sum of cover across all splits in all trees.

        # print('importance_type',importance_type)
        self.hyperparameter_grid = {}
        self.description = 'python implementation of lgb'
        self.description_url = 'https://lightgbm.readthedocs.io/en/latest/index.html'


class SVM(Model):
    def __init__(self, df_training=None, remove_log_p_descriptors=False, n_jobs=20):
        Model.__init__(self, df_training, remove_log_p_descriptors, n_jobs=n_jobs)
        self.regressor_name = "svm"

        # Following grid takes way too long:
        # self.c_space = list([10 ** x for x in range(-3, 3)])
        # self.gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]
        # self.gamma_space.append('scale')
        # self.gamma_space.append('auto')

        # self.hyperparameter_grid = {"estimator__C": [10 ** n for n in range(-3, 4)],
        #                         "estimator__kernel": ["linear", "poly", "rbf"],
        #                         "estimator__gamma": [10 ** n for n in range(-3, 4)]}

        self.self.qsar_method_version = '1.4'
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

        # # self.version = '1.4'
        # self.hyper_parameter_grid = {'max_features': ['sqrt', 'log2'],
        #                              'min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                              'n_estimators': [10, 100, 250, 500]}

        # following didnt seem to help at all for predicting PFAS properties:
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"],
        #                         'estimator__n_estimators': [10, 100, 200, 400],
        #                         'estimator__min_samples_leaf': [1, 2, 4, 8]}

        # self.hyperparameter_grid = {'estimator__max_features': ['sqrt', 'log2', 4],
        #                         'estimator__min_impurity_decrease': [10 ** x for x in range(-5, 0)],   append 0!
        #                         'estimator__n_estimators': [10, 100, 250, 500]}

        # self.version = '1.6'
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2"]}

        # self.version = '1.7'
        # self.hyperparameter_grid = {"estimator__max_features": ["sqrt", 1],
        #                             "estimator__n_estimators": [50, 100, 150, 300],
        #                             "estimator__max_depth": [50, 100, 200],
        #                             "estimator__min_samples_split": [2, 5, 10],
        #                             "estimator__max_samples": [0.25, 0.50, 1.0]}

        self.version = '1.8'

        min_impurity_decrease = [10 ** x for x in range(-5, 0)]
        min_impurity_decrease.append(0)

        self.hyperparameter_grid = {"estimator__max_features": ["sqrt", "log2", None],
                                    "estimator__n_estimators": [50, 100, 150, 300],
                                    "estimator__min_impurity_decrease": min_impurity_decrease
                                    }

        self.description = 'sklearn implementation of random forest'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'


class ModelDescription:
    def __init__(self, model: Model):
        """Describes parameters of the current method"""

        self.modelId = model.modelId
        self.modelName = model.modelName
        self.modelSource = model.modelSource
        
        
        # self.qsar_method = model.regressor_name

        if model.regressor_name:
            self.qsar_method = model.regressor_name
        elif model.qsar_method:
            self.qsar_method = model.qsar_method

        if hasattr(model, "version"):
            self.qsar_method_version = model.version
        if hasattr(model, "qsar_method_version"):
            self.qsar_method_version = model.qsar_method_version


        if model.modelStatistics:
            self.modelStatistics = model.modelStatistics

        self.description = model.description
        self.description_url = model.description_url
        self.datasetName = model.datasetName
        
        self.embedding = model.embedding
        
        self.unitsModel = model.unitsModel
        self.unitsDisplay = model.unitsDisplay
        
        self.propertyName = model.propertyName
        self.propertyDescription = model.propertyDescription
        
        self.descriptorService = model.descriptorService
        self.splittingName = model.splittingName
        self.applicabilityDomainName = model.applicabilityDomainName
        
        self.omitSalts = model.omitSalts
        self.qsarReadyRuleSet = model.qsarReadyRuleSet
        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.hyperparameter_grid = model.hyperparameter_grid
        self.hyperparameters = model.hyperparameters  # final hyperparameters
        self.training_stats = model.training_stats
        self.use_pmml = model.use_pmml

        if hasattr(model, "training_descriptor_std_devs"):
            self.include_standardization_in_pmml = False
            self.training_descriptor_std_devs = model.training_descriptor_std_devs
            self.training_descriptor_means = model.training_descriptor_means
        else:
            self.include_standardization_in_pmml = True

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__) # TODO make dict to put them in custom order


def runExamples():
    # %% Test Script
    # opera_path = r"C:\Users\ncharest\OneDrive - Environmental Protection Agency (EPA)\Profile\Documents\data_sets\OPERA_TEST_DataSetsBenchmark\DataSetsBenchmark\Water solubility OPERA\{filename}"
    # training_df = DFU.load_df_from_file(opera_path.format(filename=r"Water solubility OPERA T.E.S.T. 5.1 training.tsv"),
    # pred_df = DFU.load_df_from_file(opera_path.format(filename=r"Water solubility OPERA T.E.S.T. 5.1 prediction.tsv"),

    # mainFolder = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/"
    # endpoint ="LC50"
    # training_tsv_path = mainFolder + endpoint + '/' + endpoint + '_training_set-2d.csv'
    # prediction_tsv_path = mainFolder + endpoint + '/' + endpoint + '_prediction_set-2d.csv'

    mainFolder = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 python/modeling services/pf_python_modelbuilding/datasets_exp_prop/"
    training_tsv_path = mainFolder + 'exp_prop_96HR_FHM_LC50_v1 modeling_WebTEST-default_RND_REPRESENTATIVE_training.tsv'
    prediction_tsv_path = mainFolder + 'exp_prop_96HR_FHM_LC50_v1 modeling_WebTEST-default_RND_REPRESENTATIVE_prediction.tsv'

    training_df = DFU.load_df_from_file(training_tsv_path, sep='\t')
    pred_df = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # Demonstrate KNN usage
    print(r"Executing KNN")
    model = model_ws_utilities.instantiateModel(training_df, n_jobs=8, qsar_method='knn', remove_log_p=False,
                                                use_pmml_pipeline=False, include_standardization_in_pmml=False)
    model.build_model(False, False,
                      None)  # Note we now handle using an embedding by passing a descriptor_names list. By default it is a None type -- this will use all descriptors in df
    test_score = model.do_predictions(pred_df, return_score=True)

    # Demonstrate RF usage
    print(r"Executing RF")
    model = model_ws_utilities.instantiateModel(training_df, n_jobs=8, qsar_method='rf', remove_log_p=False,
                                                use_pmml_pipeline=False, include_standardization_in_pmml=False)
    model.build_model(False, False, None)
    test_score = model.do_predictions(pred_df, return_score=True)

    # Demonstrate SVM usage
    print(r"Executing SVM")
    model = model_ws_utilities.instantiateModel(training_df, n_jobs=8, qsar_method='svm', remove_log_p=False,
                                                use_pmml_pipeline=False, include_standardization_in_pmml=False)

    model.build_model(False, False, None)
    test_score = model.do_predictions(pred_df, return_score=True)

    print(r"Executing XGB")
    model = model_ws_utilities.instantiateModel(training_df, n_jobs=8, qsar_method='xgb', remove_log_p=False,
                                                use_pmml_pipeline=False, include_standardization_in_pmml=False)
    model.build_model(False, False, None)
    test_score = model.do_predictions(pred_df, return_score=True)


if __name__ == "__main__":
    runExamples()


