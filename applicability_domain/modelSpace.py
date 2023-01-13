# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 17:20:39 2022

@author: NCHAREST
@versionDate: v5.9.2022
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, cross_validate, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import random as rnd
import copy
#%%

class ModelMaker:
    def __init__(self, is_categorical=False):
        self.model = None
        self.score_types_regressor =  ['pearson', 'cod', 'MAE']
        self.is_categorical = is_categorical
        self.validation_scores_ = {}
    
    def model_validate(self, X_train, Y_train, X_val, Y_val):
        model = self.model
        validation_scores_ = {}
        ### Runs n_fold validation
        for n_fold in [3,5,10]:
            scores = cross_val_score(self.model, X_train, Y_train, cv=n_fold)
            validation_scores_[n_fold] = {'fold_scores':scores}
            validation_scores_[n_fold]['mean'] = np.mean(scores)
            validation_scores_[n_fold]['std'] = np.std(scores)
            validation_scores_[n_fold]['MinMax'] = max(scores) - min(scores)
        ### Runs Y-randomization
        randomYData = list(copy.deepcopy(Y_train))
        rnd.shuffle(randomYData)
        model.fit(X_train, randomYData)
        Y_ran_pred_train = model.predict(X_train)
        Y_ran_pred_val = model.predict(X_val)
        Y_ran_train = pearsonr(Y_train, Y_ran_pred_train)[0]**2
        Y_ran_val = pearsonr(Y_val, Y_ran_pred_val)[0]**2
        validation_scores_['YRanTrain'] = Y_ran_train
        validation_scores_['YRanVal'] = Y_ran_val
        ### Update Dictionary
        self.validation_scores_ = self.validation_scores_ | validation_scores_
        
    def group_kfold_validate(self, X_train, Y_train):
        pass
    
    def init_xgb(self, scale_method=None):
        if self.is_categorical == False:
            if scale_method == None:
                pipe = Pipeline([('estimator', XGBRegressor())])
                self.model = pipe
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', XGBRegressor())])
                self.model = pipe
        elif self.is_categorical == True: 
            if scale_method == None:
                self.model = XGBClassifier()
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', XGBClassifier())])
                self.model = pipe
        else:
            raise ValueError("is_categorical must be set to boolean. Re: modelMaker.")
            
    def init_dt(self, is_categorical=False):
        if self.is_categorical == False:
            self.model = Pipeline([('scaler', StandardScaler())('estimator',DecisionTreeRegressor())])
        elif self.is_categorical == True:
            self.model = Pipeline([('scaler', StandardScaler())('estimator',DecisionTreeClassifier())])
            
    def init_rf(self, scale_method=None):
        if self.is_categorical == False:
            if scale_method == None:
                self.model = Pipeline([('estimator', RandomForestRegressor())])
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', RandomForestRegressor())])
                self.model = pipe
        elif self.is_categorical == True: 
            if scale_method == None:
                self.model = Pipeline([('estimator', RandomForestClassifier())])
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', RandomForestClassifier())])
                self.model = pipe

    def init_knn(self, scale_method=None):
        if self.is_categorical == False:
            if scale_method == None:
                self.model = KNeighborsRegressor()
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', KNeighborsRegressor())])
                self.model = pipe
        elif self.is_categorical == True: 
            if scale_method == None:
                self.model = KNeighborsClassifier()
            elif scale_method == 'standardize':
                pipe = Pipeline([('scaler', StandardScaler()), ('estimator', KNeighborsClassifier())])
                self.model = pipe
            
    def set_hyperparameters(self, hyperparameters):
        self.model.set_params(**hyperparameters)
            
    def train_model(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
        
    def predict(self, X):
        preds = self.model.predict(X)
        return preds
    
    def score_model_regressor(self, data_sets):
        """
        Parameters
        ----------
        data_sets : dictionary
            keys should be strings naming the dataset to be scored
            values should be a tuple of (features, target variable)

        Returns
        -------
        Pandas DataFrame
            Summary of Scores

        """

        scores = {}

        for score_type in self.score_types_regressor:
            for data_set in data_sets.keys():
                preds = self.model.predict(data_sets[data_set][0])

                if score_type == 'pearson':                   
                    score = pearsonr(preds, data_sets[data_set][1])[0]**2
                elif score_type == 'cod':
                    score =  self.model.score(data_sets[data_set][0], data_sets[data_set][1])
                elif score_type == 'MAE':
                    score = mean_absolute_error(preds, data_sets[data_set][1])
                scores[score_type+data_set] = [round(score,2)]
        scores_df = pd.DataFrame(data=scores)
           
        return scores_df

        
    def converge_solution(self, hyperparameter, values, X_train, Y_train, X_test, Y_test, tolerance=0.05):
        """
        Iterates through hyperparameters to find the best converged solution.
        
        Hyperparameter should be the name of the hyperparameter to be scanned
        Values should be a list of acceptable values for the hyperparameter
        
        Tolerance should be a float or None. If None, then all values will be scanned.
        If Tolerance is a float, then the scanning process will stop when a model is found with a train/test score difference within the tolerance.
        """
        diff_scores = {}
        acceptable_parameters = []
        data_sets = {'Train':(X_train, Y_train), 'Test':(X_test, Y_test)}
        for parameter in values:
            self.set_hyperparameters({hyperparameter:parameter})
            self.train_model(X_train, Y_train)
            if self.is_categorical == False:
                dfScores = self.score_model_regressor(data_sets)
            diff_scores[parameter] = {'Train':float(dfScores['pearsonTrain']), 'Test':float(dfScores['pearsonTest'])}
        while len(acceptable_parameters) == 0:
            for param in diff_scores.keys():
                Delta = abs(diff_scores[param]['Train'] - diff_scores[param]['Test'])
                if Delta < tolerance:
                    acceptable_parameters.append((param, Delta, diff_scores[param]['Test']))
                acceptable_parameters.sort(key=lambda x:x[2], reverse=True)
            if len(acceptable_parameters) == 0:
                tolerance += 0.005
        
        
        return acceptable_parameters
        
    
    
            
    
            
    
            
    
        