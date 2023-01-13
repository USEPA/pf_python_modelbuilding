# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 07:38:04 2022

@author: NCHAREST
@author: GSINCL01
"""
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from sklearn.cluster import AffinityPropagation, FeatureAgglomeration, AgglomerativeClustering
from scipy.stats import iqr
import numpy as np
from sklearn.manifold import TSNE
import numpy as np

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, train_test_split

#%%
""" Gabriel Sinclair Code"""
def discretize(y, bins='auto', strategy='quantile'):
    n_bins = np.histogram_bin_edges(y, bins=bins).size - 1
    return KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy=strategy
        ).fit_transform(np.array(y).reshape(-1, 1))
    
def continuous_stratified_train_test_split(X, y, test_size=None, train_size=None, 
                                           bins='auto', strategy='quantile', random_state=None):
    y_class = discretize(y, bins=bins, strategy=strategy)
    return train_test_split(X, y, test_size=test_size, train_size=train_size, 
                            random_state=random_state, stratify=y_class)

class ContinuousStratifiedKFold():
    def __init__(self, n_splits=5, bins='auto', strategy='quantile', random_state=None):
        self.n_splits = n_splits
        self.bins = bins
        self.strategy = strategy
        self.random_state = random_state
            
    def split(self, X, y, groups=None):
        y_class = discretize(y, bins=self.bins, strategy=self.strategy)
        return StratifiedKFold(
            n_splits=self.n_splits, random_state=self.random_state, shuffle=True
            ).split(X, y_class)

""" Nathaniel Charest Code """
class DataSetManager:
    def __init__(self):
        self.accepted_scaler_types = ['standardize', 'normalize']
        self.active_XTrain, self.active_XTest, self.active_YTrain, self.active_YTest = None, None, None, None
        self.response = None
    
    def returnActiveData(self):
        return self.active_XTrain, self.active_XTest, self.active_YTrain, self.active_YTest
    
    def applyEmbedding(self, embedding):
        self.X_full = self.X_full[embedding]
    
    def importSplitDataset(self, trainPath, testPath, yLabel, identifier = 'ID', delimiter='\t', **kwargs):
        ## Import method designed to handle the OPERA benchmark tsvs
        self.import_dfTrain = pd.read_csv(trainPath, delimiter=delimiter, **kwargs)
        self.import_dfTest = pd.read_csv(testPath, delimiter=delimiter, **kwargs)
        self.import_X_train = self.import_dfTrain.drop([identifier, yLabel], axis=1)
        self.import_Y_train = self.import_dfTrain[yLabel]
        self.import_X_test = self.import_dfTest.drop([identifier, yLabel], axis=1)
        self.import_Y_test = self.import_dfTest[yLabel]
        self.dfFull = pd.concat([self.import_dfTrain, self.import_dfTest])
        self.dfFull['index'] = list(range(self.dfFull.shape[0]))
        self.dfFull = self.dfFull.set_index('index')
        self.X_full = self.dfFull.drop([yLabel, identifier], axis=1)
        self.Y_full = self.dfFull[yLabel]       
        self.feature_names = list(self.import_X_train.columns) 
        self.response = yLabel
        self.identifier = identifier
    
    def set_data(self, dfFull, yLabel, identifier = 'ID'):
        self.dfFull = dfFull
        self.X_full = self.dfFull.drop([yLabel, identifier], axis=1)
        self.Y_full = self.dfFull[yLabel]
        self.response = yLabel
        self.identifier = identifier
        
    def instantiateScaler(self, type_of_scaler):
        if type_of_scaler not in self.accepted_scaler_types:
            raise ValueError('Desired Scaling is not supported')
        else:
            if type_of_scaler == 'standardize':
                return StandardScaler()
            elif type_of_scaler == 'normalize':
                return MinMaxScaler()
            
    def scaleData(self, scaler, trainData, testData):
        scaler.fit(trainData)
        newTrain = scaler.transform(trainData)
        newTest = scaler.transform(testData)
        return newTrain, newTest
    
    def affinity_propagation(self, preference=None, random_state=2020):
        affinity_prop = AffinityPropagation(preference=preference, random_state=2020, max_iter=1000)
        affinity_prop.fit(self.X_full)
        affinity_prop_embedding = affinity_prop.predict(self.X_full)
        self.dfFull['ap_cluster'] = affinity_prop_embedding
        self.num_clusters = max(affinity_prop_embedding)
        
    def agglomeration(self, n_clusters=10):
        agglomeration = AgglomerativeClustering(n_clusters=n_clusters)
        agglomeration_label = agglomeration.fit_predict(self.X_full)
        self.dfFull['ag_cluster'] = agglomeration_label
        self.num_clusters = max(agglomeration_label)+1
    
    def tsne_embedding(self, random_state=2020):
        tsne_manifold = TSNE(random_state=random_state)
        embedding = tsne_manifold.fit_transform(self.X_full)
        self.dfFull['tsne_embedding_1'] = embedding[:,0]
        self.dfFull['tsne_embedding_2'] = embedding[:,1]
    
    def createStratifiedSplit(self, test_size, random_state=1991, bin_calculator = 'fd', scaling_type = False):
                
        def freedman_diaconis_n_bins(data, bin_calculator):
            """
            Computes the number of bins to be given to the discretizer based on freedman-diaconis width
            Parameters
            ----------
            data : numpy array
                Array of response variable to be stratified

            Returns
            -------
            n_bins
                the number of recommended bins

            """
            edges = np.histogram_bin_edges(data, bins=bin_calculator)
            fd_width = edges[0] - edges[1]
            minMax = np.amax(data) - np.amin(data)
            n_bins = int(minMax/fd_width)
            return np.abs(n_bins)
        
        y_data = np.array(self.Y_full)
        n_bins = freedman_diaconis_n_bins(y_data, bin_calculator)
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        transform = discretizer.fit_transform(y_data.reshape(-1,1))
        transform_list = [int(i) for i in transform]
        #%%
        self.n_bins = n_bins
        self.dfFull['discreteResponse'] = transform_list
        
        self.stratXTrain, self.stratXTest, self.stratYTrain, self.stratYTest = train_test_split(self.X_full, self.Y_full, test_size=test_size, random_state=random_state, stratify=self.dfFull['discreteResponse'])
        self.active_XTrain, self.active_XTest, self.active_YTrain, self.active_YTest = self.stratXTrain, self.stratXTest, self.stratYTrain, self.stratYTest
    
        self.stratDataframeTrain = self.dfFull.iloc[list(self.stratXTrain.index)]
        self.stratDataframeTest = self.dfFull.iloc[list(self.stratXTest.index)]
      
    def createRandomSplit(self, test_size, random_state=1991, scaling_type = None):
        self.randomXTrain, self.randomXTest, self.randomYTrain, self.randomYTest = train_test_split(self.X_full, self.Y_full, test_size=test_size, random_state=random_state)
        if scaling_type != None:
            self.scaler = self.instantiateScaler(scaling_type)
            scaledTrain, scaledTest = self.scaleData(self.scaler, self.randomXTrain, self.randomXTest)
            self.active_XTrain, self.active_XTest, self.active_YTrain, self.active_YTest = scaledTrain, scaledTest, self.randomYTrain, self.randomYTest
        else:
            self.active_XTrain, self.active_XTest, self.active_YTrain, self.active_YTest = self.randomXTrain, self.randomXTest, self.randomYTrain, self.randomYTest
        self.randomDataframeTrain = self.dfFull.iloc[list(self.randomXTrain.index)]
        self.randomDataframeTest = self.dfFull.iloc[list(self.randomXTest.index)]