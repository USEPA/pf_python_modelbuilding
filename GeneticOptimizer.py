# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 07:18:59 2022

@author: NCHAREST
"""
#%%
import numpy as np
from sklearn.model_selection import cross_val_score

#%% 
class FitnessFunctions:    
    @staticmethod
    def five_fold_cv(descriptors, X_train, Y_train, model):
        results = cross_val_score(model, X_train[descriptors], Y_train, cv=5, n_jobs=16)
        score = np.mean(results) - np.std(results) - 0.002*len(descriptors)
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
        generation = []
        while len(generation) < num_parents:
            size = np.random.randint(minimum_length, maximum_length)
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
#%%  