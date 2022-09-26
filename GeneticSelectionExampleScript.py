# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:57:48 2022

@author: NCHAREST
"""

import GeneticOptimizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from collections import defaultdict

NUM_GENERATIONS = 10
NUM_OPTIMIZERS = 10
NUM_PARENTS = 10
MINIMUM_LENGTH = 4
MAXIMUM_LENGTH = 24
MUTATION_PROBABILITY = 0.001
NUMBER_SURVIVORS = 10
ENDPOINT = "LogKmHL"
THRESHOLD = 2
# %% Data Import
"""
This code block just establishes the internal set (X_internal, y_internal) and the external set (X_external, Y_external)

We will only use the internal data for descriptor selection, leaving the external data for our final testing.
"""

endpointsOPERA = ["Water solubility", "LogKmHL", "LogKOA", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                  "Melting point"]
endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

if ENDPOINT in endpointsOPERA:
    IDENTIFIER = 'ID'
    PROPERTY = 'Property'
    DELIMITER = '\t'
    directory = r"C:\\Users\Weeb\\Documents\\QSARmod\\data\\DataSetsBenchmark\\" + ENDPOINT + " OPERA\\" + ENDPOINT + " OPERA T.E.S.T. 5.1 "
    trainPath = "training.tsv"
    testPath = "prediction.tsv"
elif ENDPOINT in endpointsTEST:
    IDENTIFIER = 'CAS'
    PROPERTY = 'Tox'
    DELIMITER = ','
    directory = r"C:\\Users\Weeb\\Documents\\QSARmod\\dataDataSetsBenchmarkTEST_Toxicity\\" + ENDPOINT + r"\\" + ENDPOINT
    trainPath = "_training_set-2d.csv"
    testPath = "_prediction_set-2d.csv"


def wardsMethod(train_tsv, threshold, yLabel):
    ## This method implements Ward's hierarchical clustering on the distance matrix derived from Spearman's correlations between descriptors.
    ## Inputs: threshold (float) -- this is the cutoff t-value that determines the size and number of colinearity clusters.
    ########## test (Boolean) -- if True then trains a RF model with default hyperparameters using Ward embedding
    ## Output: sets self.wardsFeatures -- the list of features that have been identified as non-colinear.
    ## Source documentation: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html

    ## We standardize data
    train_tsv2 = train_tsv.drop(['ID'], axis=1)
    train_tsv = train_tsv2.drop(['Property'], axis=1)

    feature_names = list(train_tsv.columns)

    scaler = preprocessing.StandardScaler().fit(train_tsv)
    ## Compute spearman's r and ensure symmetry of correlation matrix
    corr = spearmanr(scaler.transform(train_tsv)).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    ## Compute distance matrix and form hierarchical clusters
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    clusters = cluster_ids
    ## Pull out one representative descriptor from each cluster
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    named_features = [feature_names[i] for i in selected_features]
    ## Set attribute with features that are not colinear
    wardsFeatures = [feature_names[i] for i in selected_features]

    return wardsFeatures

#%% Parameter Configuration

if __name__ == '__main__':


    trainPath = directory+trainPath
    testPath = directory+testPath
    ### Apply Ward filtering to eliminate colinear data

    df_train = pd.read_csv(trainPath, delimiter='\t')
    df_predict = pd.read_csv(testPath, delimiter='\t')
    y_internal = df_train[PROPERTY]

    df_train = df_train.loc[:, (df_train != 0).any(axis=0)]
    features = wardsMethod(df_train, 0.5, IDENTIFIER)
    descriptor_pool = features
    print(descriptor_pool)
    x_internal = df_train[descriptor_pool]
    model = Pipeline([('standardizer', StandardScaler()), ('estimator', KNeighborsRegressor())])
    fitness_calculator = GeneticOptimizer.FiveFoldFitness(X_train = x_internal, y_train = y_internal, model = model)
    ensemble_selector = GeneticOptimizer.GeneticSelector(descriptor_pool, fitness_calculator)

    ensemble_selector.ensemble_evolution(num_optimizers=NUM_OPTIMIZERS, num_generations=NUM_GENERATIONS,
                                         num_parents=NUM_PARENTS, min_length=MINIMUM_LENGTH,
                                         max_length=MAXIMUM_LENGTH, mutation_probability=MUTATION_PROBABILITY,
                                         num_survivors=NUMBER_SURVIVORS)
   
    high_count_descriptors = ensemble_selector.descriptor_threshold(THRESHOLD)
    print(high_count_descriptors)
    final_selection = GeneticOptimizer.GeneticOptimizer(high_count_descriptors, fitness_calculator)
    final_selection.run_evolution(num_generations=NUM_GENERATIONS, num_parents=NUM_PARENTS, min_length=MINIMUM_LENGTH,
                                  max_length=MAXIMUM_LENGTH, mutation_probability=MUTATION_PROBABILITY,
                                  num_survivors=NUMBER_SURVIVORS)
    final_embedding = final_selection.optimal_sequence
    print(final_embedding)


"""
### Use the homebrewed DataSetManager object to stratify split sets --
ds_manager = DataSetManager()
ds_manager.importSplitDataset(trainPath, testPath, PROPERTY, identifier=IDENTIFIER, delimiter=DELIMITER)
ds_manager.applyEmbedding(exp.wardsFeatures)
ds_manager.createStratifiedSplit(test_size=0.2, random_state=1991, scaling_type=None)
X_internal, X_external, y_internal, y_external = ds_manager.returnActiveData()
#%% Genetic Selection
"""
# This block of code configures and executes the ensemble selection process, generating a new pool of descriptors that emerged from
# multiple rounds of genetic selection on the initial pool
"""
###
###
###
###
#%%


# The final code block takes the descriptors that were counted the threshold number of times from the prior selection and
# runs an optimization to produce the final selected embedding


###

"""
