# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 07:18:59 2022

@author: NCHAREST
"""
#%%
import numpy as np
from sklearn.model_selection import cross_val_score
from models import df_utilities as dfu
from sklearn import preprocessing
from scipy.stats import spearmanr, pearsonr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np
import pandas as pd
from collections import defaultdict
from models import ModelBuilder
from models import df_utilities as DFU


debug = True

NUM_GENERATIONS = 100 # kamel used 100
NUM_OPTIMIZERS = 10 # kamel used 100
NUM_PARENTS = 10
MINIMUM_LENGTH = 4
MAXIMUM_LENGTH = 24
MUTATION_PROBABILITY = 0.001
NUMBER_SURVIVORS = 10
# THRESHOLD = 2  # Nate
THRESHOLD = 1  # Todd: keeps more descriptors from stage 1 of GA optimization

# NUM_JOBS = 16
NUM_JOBS = 4  # calculation time is barely reduced by using more than 4 threads
DESCRIPTOR_COEFFICIENT = 0.002


# def wardsMethod(df, threshold):
#     ## This method implements Ward's hierarchical clustering on the distance matrix derived from Spearman's correlations between descriptors.
#     ## Inputs: threshold (float) -- this is the cutoff t-value that determines the size and number of colinearity clusters.
#     ########## test (Boolean) -- if True then trains a RF model with default hyperparameters using Ward embedding
#     ## Output: sets self.wardsFeatures -- the list of features that have been identified as non-colinear.
#     ## Source documentation: https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html
#
#     ## We standardize data
#
#
#     ids = np.array(df[df.columns[0]])
#     col_name_id = df.columns[0]
#     col_name_property = df.columns[1]
#
#     # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
#     df = df.drop(col_name_property, axis=1)
#
#     # drop ID column:
#     df = df.drop(col_name_id, axis=1)
#
#     # print(df.shape)
#     # #drop constant columns:
#     # df = dfu.do_remove_constant_descriptors(df)
#     # print(df.shape)
#
#
#
#     feature_names = list(df.columns)
#
#     df = df.loc[:, (df != 0).any(axis=0)]
#
#     scaler = preprocessing.StandardScaler().fit(df)
#
#     ## Compute spearman's r and ensure symmetry of correlation matrix
#     corr = spearmanr(scaler.transform(df)).correlation
#     corr = (corr + corr.T) / 2
#     np.fill_diagonal(corr, 1)
#     ## Compute distance matrix and form hierarchical clusters
#     distance_matrix = 1 - np.abs(corr)
#     dist_linkage = hierarchy.ward(squareform(distance_matrix))
#     cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
#     clusters = cluster_ids
#     ## Pull out one representative descriptor from each cluster
#     cluster_id_to_feature_ids = defaultdict(list)
#     for idx, cluster_id in enumerate(cluster_ids):
#         cluster_id_to_feature_ids[cluster_id].append(idx)
#     selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
#     named_features = [feature_names[i] for i in selected_features]
#     ## Set attribute with features that are not colinear
#     wardsFeatures = [feature_names[i] for i in selected_features]
#
#     return wardsFeatures


def runGA(df_training, model, use_wards, remove_log_p_descriptors=False):

    if use_wards:
        # Using ward's method removes too descriptors for PFAS only training sets:
        train_ids, train_labels, train_features, train_column_names, model.is_binary = \
            DFU.prepare_instances_wards(df_training, "training", remove_log_p_descriptors,
                                        0.5)  # uses wards method to remove extra descriptors
    else:
        train_ids, train_labels, train_features, train_column_names, model.is_binary = \
            DFU.prepare_instances(df_training, "training", remove_log_p_descriptors,
                                  True)  # removes descriptors which are correlated by 0.95


    # train_ids, train_labels, train_features, train_column_names, model.is_categorical = \
    #     DFU.prepare_instances_wards(df_train, "training", remove_log_p_descriptors, 0.5) # uses wards method to remove extra descriptors

    # print(type(model))

    print('use_wards = ',use_wards)
    print('after initial feature selection, # features = ',len(train_column_names))
    # print('Number of rows = ', len(train_ids))


    # model.model_obj = ModelBuilder.model_registry_pipeline(model.regressor_name, model.is_binary)  #already done earlier in instantiate_model method

    # print(model.model_obj.steps[1][1])

    if model.regressor_name == 'rf':
        model.hyperparameter_grid = {
            "estimator__max_features": ["sqrt"]}  # just use a single set of hyperparameters to speed up

    model.hyperparameters = model.get_single_parameters()
    model.model_obj.set_params(**model.hyperparameters)

    print (model.hyperparameters)

    # features = wardsMethod(df_train, 0.5)
    # y_internal = df_train.iloc[:,1]
    # print(y_internal)

    y_internal = train_labels
    x_internal = train_features
    descriptor_pool = train_column_names

    fitness_calculator = FiveFoldFitness(X_train=x_internal, y_train=y_internal, model=model.model_obj)

    go = GeneticOptimizer(descriptor_pool, fitness_calculator)

    ensemble_selector = GeneticSelector(descriptor_pool, fitness_calculator)

    print('NUM_GENERATIONS',NUM_GENERATIONS)

    ensemble_selector.ensemble_evolution(num_optimizers=NUM_OPTIMIZERS, num_generations=NUM_GENERATIONS,
                                         num_parents=NUM_PARENTS, min_length=MINIMUM_LENGTH,
                                         max_length=MAXIMUM_LENGTH, mutation_probability=MUTATION_PROBABILITY,
                                         num_survivors=NUMBER_SURVIVORS)


    high_count_descriptors = ensemble_selector.descriptor_threshold(THRESHOLD)
    print(high_count_descriptors)
    go2 = GeneticOptimizer(high_count_descriptors, fitness_calculator)
    go2.run_evolution(num_generations=NUM_GENERATIONS, num_parents=NUM_PARENTS,
                                  min_length=MINIMUM_LENGTH,
                                  max_length=MAXIMUM_LENGTH, mutation_probability=MUTATION_PROBABILITY,
                                  num_survivors=NUMBER_SURVIVORS)
    return go2.optimal_sequence
#%% 
class FitnessFunctions:    
    @staticmethod
    def five_fold_cv(descriptors, X_train, Y_train, model):
        results = cross_val_score(model, X_train[descriptors], Y_train, cv=5, n_jobs=NUM_JOBS)
        score = np.mean(results) - np.std(results) - DESCRIPTOR_COEFFICIENT * len(descriptors)  # Objective function
        return score
        
    @staticmethod
    def rational_split(descriptors, X_train, Y_train, X_val, Y_val, model):
        model.fit(X_train[descriptors], Y_train)
        score = model.score(X_val[descriptors], Y_val)
        return score
#%%
class FitnessCalculator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class FiveFoldFitness(FitnessCalculator):
    def __init__(self, **kwargs):
        FitnessCalculator.__init__(self, **kwargs)
        
    def evaluate(self, descriptors):
        score = FitnessFunctions.five_fold_cv(descriptors, X_train = self.kwargs['X_train'], 
                                             Y_train = self.kwargs['y_train'], model = self.kwargs['model'])
        return score
    
class RationalSplit(FitnessCalculator):
    def __init__(self, **kwargs):
        FitnessCalculator.__init__(self, **kwargs)
        
    def evaluate(self, descriptors):
        score = FitnessFunctions.rational_split(descriptors, X_train = self.kwargs['X_train'], 
                                               Y_train = self.kwargs['y_train'], X_val = self.kwargs['X_val'],  
                                               Y_val = self.kwargs['y_val'], model = self.kwargs['model'])  
        return score     
#%%        
class GeneticOptimizer:
    def __init__(self, descriptor_pool, fitness_calculator):
        self.set_fitness_function(fitness_calculator)
        self.descriptor_pool = descriptor_pool
    
    def set_fitness_function(self, fitness_calculator):
        self.fitness_calculator = fitness_calculator  
    
    def breed(self, A, B):
        ### 
        if len(A) <= len(B):
            embedding_A = B
            embedding_B = A
        else:
            embedding_A = A
            embedding_B = B
        ###
        len_A = len(embedding_A)
        len_B = len(embedding_B)
        ###
        unique_B = [desc for desc in embedding_B if desc not in embedding_A] 
        ###
        lengths = [len_A, len_B, int((len_A + len_B)/2), len_A + len(unique_B)]
        ###
        child_length = np.random.choice(lengths)
        ###
        num_A = int(child_length/2)
        num_B = child_length - num_A
        ###
        if num_B <= len(unique_B):
            A_gamete = list(np.random.choice(embedding_A, num_A, replace=False))
            B_gamete = list(np.random.choice(unique_B, num_B, replace=False))
            
        else:
            A_gamete = list(np.random.choice(embedding_A, num_A, replace=False))
            B_gamete = unique_B
            
        child = A_gamete + B_gamete
        if len(child) <= 3:
            child = np.random.choice(self.descriptor_pool, 4, replace=False) 
        return child
    
    def mutate(self, A, index, probability):
        mutation = A[index]
        while mutation in A:
            mutation = np.random.choice(self.descriptor_pool, replace=False)
        if np.random.random() < probability:
            A[index] = mutation
            
    def initialize_parents(self, num_parents, minimum_length, maximum_length):

        # print('Here1 maximum_length=',maximum_length)
        generation = []

        while len(generation) < num_parents:
            # size = np.random.randint(minimum_length, maximum_length)
            size = np.random.randint(minimum_length, min(len(self.descriptor_pool), maximum_length)) #Fix added by TMM
            # print('Here2 maximum_length=', maximum_length)
            # print('Here size=', size)

            candidate = list(np.random.choice(self.descriptor_pool, size, replace=False))
            if candidate not in generation:
                generation.append(candidate)
        return generation

    
    def apply_fitness(self, generation):
        scored_generation = []
        for i in generation:
            scored_generation.append((i, self.fitness_calculator.evaluate(i)))
        return scored_generation
    
    def breed_generation(self, generation, breed_chance = 0.5):
        children = []
        for A in range(len(generation)):
            chance_of_breed = np.random.random()
            if chance_of_breed >= breed_chance:
                for B in range(len(generation)):
                    if A == B:
                        pass
                    elif (set(generation[B]) == set(generation[A])) == False:
                        child = self.breed(generation[A],generation[B])
                        children.append(child)
                    elif ((set(generation[B]) == set(generation[A])) == True) and A != B:
                        child = np.random.choice(self.descriptor_pool, len(generation[A]), replace=False)
                        children.append(child)
            else:
                children.append(generation[A])
        return children
    
    def fitness_filter(self, scored_generation, num_survivors):
        scored_generation.sort(key=lambda x:x[1], reverse=True)
        self.history['prime_scores'].append(scored_generation[0][1])
        self.history['prime_genes'].append(scored_generation[0][0])
        survivors = [i[0] for i in scored_generation[:num_survivors]]
        if len(survivors) < num_survivors:
            print("Too few survivors | only {num_survivers} re:fitness_filter".format(num_survivers=len(survivors)))
        return survivors

    def run_generation(self, parents, mutation_probability, num_survivors, return_prime=False):

        scored_parents = self.apply_fitness(parents)
    
        survivors = self.fitness_filter(scored_parents, num_survivors)
        children = self.breed_generation(survivors)
        for child in children:
            for index in range(len(child)):
                self.mutate(child, index, mutation_probability)
        if return_prime == True:
            return children, survivors[0]
        if return_prime == False:            
            return children

    def run_evolution(self, num_generations, num_parents, min_length, max_length, mutation_probability, num_survivors):

        self.history = {}
        self.history['prime_scores'] = []
        self.history['prime_genes'] = []
        primes = []
        self.generations = {}


        primordials = self.initialize_parents(num_parents, min_length, max_length)
        children, prime = self.run_generation(primordials, mutation_probability, num_survivors, return_prime=True)

        for i in range(num_generations):
            if debug:
                print('\tgeneration',i+1)
            children, prime = self.run_generation(children, mutation_probability, num_survivors, return_prime = True)
            self.generations[i] = children
            primes.append(prime)



        max_fitness = max(self.history['prime_scores'])
        for i in range(len(self.history['prime_scores'])):
            if self.history['prime_scores'][i] == max_fitness:
                max_index = i
        self.optimal_sequence = self.history['prime_genes'][max_index]
        self.optimal_score = self.history['prime_scores'][max_index]
        
#%%        
class GeneticSelector:
    def __init__(self, descriptor_pool, fitness_calculator):
        self.descriptor_pool = descriptor_pool
        self.fitness_calculator = fitness_calculator
    
    def ensemble_evolution(self, num_optimizers, num_generations, num_parents, min_length, max_length, mutation_probability, num_survivors):
        self.genetic_optimizers = {}
        self.num_optimizers = num_optimizers

        for identification_number in range(num_optimizers):

            if debug:
                print ('optimizer',identification_number+1)

            self.genetic_optimizers[identification_number] = GeneticOptimizer(self.descriptor_pool, self.fitness_calculator)
            self.genetic_optimizers[identification_number].run_evolution(num_generations=num_generations, num_parents=num_parents, min_length=min_length, max_length=max_length, mutation_probability=mutation_probability, num_survivors=num_survivors)
        
        self.prime_sequences = []
        for key in self.genetic_optimizers.keys():
           self.prime_sequences += self.genetic_optimizers[key].optimal_sequence
            
        self.descriptor_counts = {}
        for descriptor in self.prime_sequences:
            if descriptor not in self.descriptor_counts.keys():
                self.descriptor_counts[descriptor] = self.prime_sequences.count(descriptor)
                
    def descriptor_threshold(self, threshold = 'default'):
        if threshold == 'default':
            threshold = int(self.num_optimizers/2)
        high_count_descriptors = []
        for desc in self.descriptor_counts.keys():
            if self.descriptor_counts[desc] >= threshold:
                high_count_descriptors.append(desc)
        return high_count_descriptors
