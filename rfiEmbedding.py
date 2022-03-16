# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:28:25 2022

@author: NCHAREST
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy import spearmanr, hierarchy, squareform
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

class RFICalculator:
    ## Class containing functions to create an embedding for a data set using the importances of a random forest trained on the data
    def __init__(self):
        ## Basic Constructor
        self.df = None        
        pass
        
    def importSplit(self, tsvPath, yLabel, **kwargs):
        ## Import method designed to handle tsvs
        self.dfFull = pd.read_csv(tsvPath, delimiter='\t', **kwargs)
        self.X_full = self.dfFull.drop(['ID', yLabel], axis=1)
        self.Y_full = self.dfFull[yLabel]
        
        self.feature_names = list(self.X_train.columns)
        
    def basicClean(self):
        ## Rudimentary function to clean out null descriptor columns
        self.zeroColumns = self.X_full.columns[(self.X_full==0).all()]
        self.X_full = self.X_full.loc[:, (self.X_full != 0).any(axis=0)]
        self.X_full = self.X_full.drop(self.zeroColumns, axis=1)
        self.feature_names = list(self.X_train.columns)
        
    def wardsMethod(self, threshold, test=False):  
        ## This method implements Ward's hierarchical clustering on the distance matrix derived from Spearman's correlations between descriptors.
        ## Inputs: threshold (float) -- this is the cutoff t-value that determines the size and number of colinearity clusters.
        ########## test (Boolean) -- if True then trains a RF model with default hyperparameters using Ward embedding
        ## Output: sets self.wardsFeatures -- the list of features that have been identified as non-colinear.
        ## Source documentation: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
        
        ## We standardize data
        scaler = preprocessing.StandardScaler().fit(self.X_full)
        ## Compute spearman's r and ensure symmetry of correlation matrix
        corr = spearmanr(scaler.transform(self.X_full)).correlation
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        ## Compute distance matrix and form hierarchical clusters
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))        
        cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
        self.clusters = cluster_ids
        ## Pull out one representative descriptor from each cluster
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)            
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        named_features = [self.feature_names[i] for i in selected_features]
        ## Test Ward embedding using RF model
        if test == True:
            X_train_sel = self.X_full[named_features]
            Y_train_sel = self.Y_full
        
            rfr = RandomForestRegressor(n_estimators=100, random_state=42)
            rfr.fit(X_train_sel, Y_train_sel)
        
        ## Set attribute with features that are not colinear
        self.wardsFeatures = [self.feature_names[i] for i in selected_features]
        
    def hyperparameterOptimizationRFRegressor(self, embedding, nEstimators, featuresPerTree, depthOfTrees):
        ## This is a method for optimizing the hyperparameters of a Random Forest model
        ## Inputs:
        ########## embedding (list of strings) -- the descriptor names for the embedding of the model
        ########## nEstimators (list of integers) -- range of number of trees per forest
        ########## featuresPerTree (list of integers, floats or supported strings) -- see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        ########## depthOfTrees (list of integers) -- maximum depths of trees to be considers
        ## Outputs:
        ########## sets optimizerResults attribute with best performing hyperparameters tested
            
        hyperParameterGrid = {'n_estimators' : nEstimators, 'max_features' : featuresPerTree, 'max_depth':depthOfTrees}
        scaler = preprocessing.StandardScaler().fit(self.X_full[embedding])
        X_train_scaled = scaler.transform(self.X_full[embedding])
        Y_train = self.Y_full
        
        rf = RandomForestRegressor()
        rfOptimizer = RandomizedSearchCV(estimator=rf, param_distributions=hyperParameterGrid, cv=5, random_state = 1117, n_iter = 25)
        rfOptimizer.fit(X_train_scaled, Y_train)
        
        self.optimizerResults = rfOptimizer.best_params_
         
     
    def randomForestImportance(self, embedding, hyperparameters):
        ### Trains a random forest regressor with the argued hyperparameters and returns the 12 most important descriptors based on mean decrease in impurity
        numDesc = 12
        rfr = RandomForestRegressor(n_estimators = hyperparameters['n_estimators'], max_features = hyperparameters['max_features'], max_depth=hyperparameters['max_depth'], random_state = 924)
        scaler = preprocessing.StandardScaler().fit(self.X_train[embedding])
        
        X_train_scaled = scaler.transform(self.X_train[embedding])
        Y_train = self.Y_train
        
        rfr.fit(X_train_scaled, Y_train)
        
        importances = rfr.feature_importances_
        scoredFeatures = []
        
        for i in range(len(embedding)):
            scoredFeatures.append([embedding[i], importances[i]])
        
        scoredFeatures.sort(key=lambda x:x[1], reverse=True)
        
        mdiEmbedding = [i[0] for i in scoredFeatures[0:numDesc]]
        
        return mdiEmbedding, scoredFeatures
      
    def findEmbedding(self, filePath):
        ## Prime Method to execute RFI selection process for a regression problem
        self.importSplit(filePath, ylabel = 'Property')
        self.basicClean()
        self.wardsMethod(0.5)
        self.hyperparameterOptimizationRFRegressor(self.wardsFeatures, [120], [0.2, 0.4, 0.6, 0.8], [2,6,10], searchMethod = 'random')
        mdiEmbedding, mdiScored = self.randomForestImportance(self.wardsFeatures, self.expAD.optimizerResults)
        json = {'embedding' : mdiEmbedding, 'importances' : mdiScored }
        return json
              