# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:40:56 2022

@author: NCHAREST
"""
import json
import time

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import KernelDensity
from sklearn.metrics import balanced_accuracy_score
import pandas as pd


class helpers:

    @staticmethod
    def find_split_value(list_to_split, split_percentile):
        index = int(len(list_to_split) * split_percentile)
        list_to_split.sort(key=lambda x: x)
        # splitting_value = (list_to_split[index] + list_to_split[index + 1]) / 2.0 # why take an average???
        splitting_value = list_to_split[index]
        return splitting_value


class ApplicabilityDomainStrategy:
    """
    Parent class for executing and evaluating a particular applicability_domain strategy
    """

    def __init__(self, TrainSet, TestSet, is_categorical=False):
        self.TrainSet = TrainSet
        self.TestSet = TestSet
        self.model = None
        self.embedding = None
        self.response = None
        self.parameters = None
        self.is_categorical = is_categorical

    def setEmbedding(self, embedding):
        self.embedding = embedding

    def setResponse(self, response):
        self.response = response

    def train_test_ratios(self):
        """
        Used to calculate coverage
        :return:
        """
        try:
            test_ratio = self.TestInner.shape[0] / (self.TestInner.shape[0] + self.TestOuter.shape[0])
            train_ratio = self.TrainInner.shape[0] / (self.TrainInner.shape[0] + self.TrainOuter.shape[0])
            return train_ratio, test_ratio
        except Exception as e:
            return -9999, -9999

    def getStats(self):
        """
        Results for a calculation run
        :return: Results dictionary
        """
        # results = {'r2_test_inner': self.scoreTestInner, 'coverage_test': self.train_test_ratios()[1],
        #            'product': self.cp_product}

        results = {'r2_test_inner': self.scoreTestInner, 'r2_test_outer': self.scoreTestOuter}

        return results

    def computeSpreadOfResiduals(self):
        stdTrainInner = np.std(self.TrainInner['ResidualSquared'])
        stdTrainOuter = np.std(self.TrainOuter['ResidualSquared'])
        stdTestInner = np.std(self.TestInner['ResidualSquared'])
        stdTestOuter = np.std(self.TestOuter['ResidualSquared'])
        self.spreadRatioTrain = stdTrainInner / stdTrainOuter
        self.spreadRatioTest = stdTestInner / stdTestOuter

    def computeInnerOuterRatio(self):

        try:

            self.scoreTestOuter = 'NA'

            # if len(self.TrainInner[self.response])<2 and len(self.TrainInner['Prediction']):
            #     print('cant calculate inner outer stats')
            #     return

            if not self.is_categorical:

                self.scoreTrainInner = pearsonr(self.TrainInner[self.response], self.TrainInner['Prediction'])[0] ** 2
                self.scoreTrainOuter = pearsonr(self.TrainOuter[self.response], self.TrainOuter['Prediction'])[0] ** 2

                self.scoreTestInner = pearsonr(self.TestInner[self.response], self.TestInner['Prediction'])[0] ** 2
                self.scoreTestOuter = pearsonr(self.TestOuter[self.response], self.TestOuter['Prediction'])[0] ** 2

                self.TrainInnerOuter = self.scoreTrainInner / self.scoreTrainOuter
                self.TestInnerOuter = self.scoreTestInner / self.scoreTestOuter

            elif self.is_categorical:

                print('here its categorical')

                self.scoreTrainInner = balanced_accuracy_score(self.TrainInner[self.response],
                                                               round(self.TrainInner['Prediction']))
                self.scoreTrainOuter = balanced_accuracy_score(self.TrainOuter[self.response],
                                                               round(self.TrainOuter['Prediction']))

                self.scoreTestInner = balanced_accuracy_score(self.TestInner[self.response],
                                                              round(self.TestInner['Prediction']))
                self.scoreTestOuter = balanced_accuracy_score(self.TestOuter[self.response],
                                                              round(self.TestOuter['Prediction']))

                self.TrainInnerOuter = self.scoreTrainInner / self.scoreTrainOuter
                self.TestInnerOuter = self.scoreTestInner / self.scoreTestOuter
            else:
                raise (ValueError("is_categorical is not set to a boolean"))
        except Exception as e:
            print(e)

    def computeCoveragePerformanceProduct(self):

        try:
            coverage = self.TestInner.shape[0] / (self.TestOuter.shape[0] + self.TestInner.shape[0])

            self.coverage = coverage

            print('coverage', coverage)

            # print("Yexp", self.TestInner[self.response])
            # print("Ypred", self.TestInner['Prediction'])

            if self.is_categorical == False:
                self.scoreTestInner = pearsonr(self.TestInner[self.response], self.TestInner['Prediction'])[0] ** 2
            elif self.is_categorical == True:

                Ypred = [round(x) for x in self.TestInner['Prediction']]
                # print('Ypredint',Ypred)
                self.scoreTestInner = balanced_accuracy_score(self.TestInner[self.response], Ypred)

                # self.scoreTestInner = self.calcBA(self.TestInner[self.response], self.TestInner['Prediction'])

            self.cp_product = coverage * self.scoreTestInner

            print('self.scoreTestInner', self.scoreTestInner)
            print('self.cp_product', self.cp_product)

        except Exception as e:
            print('***error=', e)

    def calcBA(self, Yexp, Ypred):
        # print ('enter calcBA')
        cutOff = 0.5

        countPositive = 0
        countNegative = 0
        countTruePositive = 0
        countTrueNegative = 0

        for index, exp in enumerate(Yexp):
            pred = Ypred[index]

            if pred >= cutOff:
                predBinary = 1.0
            else:
                predBinary = 0.0

            print(index, exp, pred, predBinary)

            if exp == 1.0:
                countPositive += 1
                if predBinary == 1:
                    countTruePositive += 1

            elif exp == 0.0:
                countNegative += 1
                if predBinary == 0:
                    countTrueNegative += 1

        sensitivity = countTruePositive / countPositive
        specificity = countTrueNegative / countNegative
        balancedAccuracy = (sensitivity + specificity) / 2.0

        print('BA', balancedAccuracy)
        return balancedAccuracy

    def scoreMetrics(self):
        # self.predict()
        # self.computeSpreadOfResiduals()
        self.computeInnerOuterRatio()
        self.computeCoveragePerformanceProduct()

    def predict(self, model_features):
        print('enter predict in AD class')

        # train_preds = self.model.predict(self.TrainSet[model_features])
        # test_preds = self.model.predict(self.TestSet[model_features])

        train_preds = self.model.do_predictions(self.TrainSet)
        test_preds = self.model.do_predictions(self.TestSet)

        # print(test_preds)

        # print('test_preds',test_preds)

        self.TrainSet['Prediction'] = train_preds
        self.TestSet['Prediction'] = test_preds
        train_residuals = (self.TrainSet['Prediction'] - self.TrainSet[self.response]) ** 2
        test_residuals = (self.TestSet['Prediction'] - self.TestSet[self.response]) ** 2
        self.TrainSet['ResidualSquared'] = train_residuals
        self.TestSet['ResidualSquared'] = test_residuals

        train_residuals = self.TrainSet['Prediction'] - self.TrainSet[self.response]
        test_residuals = self.TrainSet['Prediction'] - self.TestSet[self.response]

    def setModel(self, model):
        self.model = model

    def createSets(self, label):
        self.TrainInner = self.TrainSet.loc[self.TrainSet[label] == True]
        self.TrainOuter = self.TrainSet.loc[self.TrainSet[label] == False]
        self.TestInner = self.TestSet.loc[self.TestSet[label] == True]
        self.TestOuter = self.TestSet.loc[self.TestSet[label] == False]

    def setParameters(self, parameters):
        self.parameters = parameters


# %% TEST AD
class TESTApplicabilityDomain(ApplicabilityDomainStrategy):
    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
        self.parameters = {'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine'}
        self.AD_Label = 'WITHIN_TEST_AD'
        self.is_categorical = is_categorical

    def set_parameters(self, parameters):
        self.parameters = parameters

    def evaluate(self, embedding):
        """
        :return: returns results in dataframe with the cids of the neighbors and the applicability domain result
        """
        # print(embedding)
        t1 = time.time()

        ###
        self.nbrs = NearestNeighbors(n_neighbors=self.parameters['k'] + 1, algorithm='brute',
                                     metric=self.parameters['similarity'])

        print('metric', self.parameters['similarity'])

        # embedding.append('QSAR_READY_SMILES')

        nbrs = self.nbrs
        TrainSet = self.TrainSet[embedding]
        TestSet = self.TestSet[embedding]

        scaler = StandardScaler().fit(TrainSet)
        train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
        ###
        nbrs.fit(train_x)

        train_distances, train_indices = nbrs.kneighbors(train_x)
        test_distances, test_indices = nbrs.kneighbors(test_x)


        train_indices = train_indices[:, 1:]  #this omits the chemical itself from being a neighbor
        train_distances = train_distances[:, 1:]

        test_indices = test_indices[:, :-1]  #this assumes that we dont need the 4th neighbor because we arent checking for exact match like we did with training set
        test_distances = test_distances[:, :-1]  #returns all elements [:] except the last one -1

        # print(TrainSet[test_indices[0][1]][0])
        col_name_id = self.TrainSet.columns[0]

        neighbor1 = test_indices[:, 0]
        neighbor2 = test_indices[:, 1]
        neighbor3 = test_indices[:, 2]

        # print(neighbor1[0],neighbor2[0],neighbor3[0])
        # print(self.TrainSet[col_name_id].loc[219])

        id = self.TestSet[col_name_id]
        id1 = self.TrainSet[col_name_id].loc[neighbor1]
        id2 = self.TrainSet[col_name_id].loc[neighbor2]
        id3 = self.TrainSet[col_name_id].loc[neighbor3]

        train_TESTSimilarity, test_TESTSimilarity = list(np.mean(train_distances, axis=1)), list(
            np.mean(test_distances, axis=1))
        ###
        self.TrainSet['TESTSimilarity'] = train_TESTSimilarity
        self.TestSet['TESTSimilarity'] = test_TESTSimilarity
        ###

        # print(train_TESTSimilarity)

        self.splitSimilarity = helpers.find_split_value(train_TESTSimilarity, self.parameters['fractionTrainingSetInsideAD'])

        print('splitSimilarity', self.splitSimilarity)

        self.TrainSet[self.AD_Label] = True
        self.TestSet[self.AD_Label] = True

        # Sklearn similarities are actually distances (1-SC), so if distance > cutoff it's outside AD:
        self.TrainSet.loc[self.TrainSet['TESTSimilarity'] > self.splitSimilarity, self.AD_Label] = False
        self.TestSet.loc[self.TestSet['TESTSimilarity'] > self.splitSimilarity, self.AD_Label] = False

        # print(self.TestSet[self.AD_Label])  # array of whether or not it's in AD

        AD = self.TestSet[self.AD_Label]
        # print(AD.value_counts()[False])

        # print(AD_TR.value_counts()[False])

        results = pd.DataFrame(np.column_stack([id, id1, id2, id3, AD]),
                               columns=['idTest', 'idNeighbor1', 'idNeighbor2', 'idNeighbor3', 'AD'])
        # print(results)

        t2 = time.time()
        # print((t2-t1),' secs to evaluate')

        return results


# %%
# class AllTESTApplicabilityDomain(ApplicabilityDomainStrategy):
#     def __init__(self, TrainSet, TestSet, is_categorical):
#         ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
#         self.is_categorical = is_categorical
#         self.parameters = {'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine'}
#         self.AD_Label = 'WITHIN_TEST_AD'
#
#     def set_parameters(self, parameters):
#         self.parameters = parameters
#
#     def evaluate(self, alternate_embedding):
#         ###
#         self.nbrs = NearestNeighbors(n_neighbors=self.parameters['k'] + 1, algorithm='brute',
#                                      metric=self.parameters['similarity'])
#         nbrs = self.nbrs
#         TrainSet = self.TrainSet[alternate_embedding]
#         TestSet = self.TestSet[alternate_embedding]
#         ###
#         scaler = StandardScaler().fit(TrainSet)
#         train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
#         ###
#         nbrs.fit(train_x)
#         train_neighbors, test_neighbors = nbrs.kneighbors(train_x), nbrs.kneighbors(test_x)
#         ###
#         train_distances, test_distances = train_neighbors[0][:, 1:], test_neighbors[0][:, :-1]
#         ###
#         train_TESTSimilarity, test_TESTSimilarity = list(np.mean(train_distances, axis=1)), list(
#             np.mean(test_distances, axis=1))
#         ###
#         self.TrainSet['TESTSimilarity'] = train_TESTSimilarity
#         self.TestSet['TESTSimilarity'] = test_TESTSimilarity
#         ###
#         train_TESTSimilarity.sort(key=lambda x: x)
#         splitIndex = int(len(train_TESTSimilarity) * self.parameters['exclusionFraction'])
#         splitSimilarity = train_TESTSimilarity[splitIndex]
#         self.splitSimilarity = splitSimilarity
#         ###
#         self.TrainSet[self.AD_Label] = True
#         self.TrainSet.loc[self.TrainSet['TESTSimilarity'] <= splitSimilarity, self.AD_Label] = False
#
#         self.TestSet[self.AD_Label] = True
#         self.TestSet.loc[self.TestSet['TESTSimilarity'] <= splitSimilarity, self.AD_Label] = False


# %%
class KernelDensityApplicabilityDomain(ApplicabilityDomainStrategy):
    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
        self.parameters = {'k': 3, 'fractionTrainingSetInsideAD': 0.95, 'similarity': 'cosine'}
        self.AD_Label = 'WITHIN_TEST_AD'
        self.is_categorical = is_categorical

    def set_parameters(self, parameters):
        self.parameters = parameters

    def evaluate(self, embedding):
        ###
        TrainSet = self.TrainSet[embedding]
        TestSet = self.TestSet[embedding]

        scaler = StandardScaler().fit(TrainSet)
        train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)

        density_estimator = KernelDensity()

        train_densities = list(np.exp(density_estimator.fit(TrainSet).score_samples(train_x)))
        test_densities = list(np.exp(density_estimator.fit(TrainSet).score_samples(test_x)))

        ###
        self.TrainSet['KESimilarity'] = train_densities
        self.TestSet['KESimilarity'] = test_densities
        ###

        # print(train_densities)

        self.splitting_value = helpers.find_split_value(train_densities, 1-self.parameters['fractionTrainingSetInsideAD'])

        # print (self.splitting_value)
        # print (self.TestSet['KESimilarity'])

        ###
        self.TrainSet[self.AD_Label] = True
        self.TrainSet.loc[self.TrainSet['KESimilarity'] <= self.splitting_value, self.AD_Label] = False

        self.TestSet[self.AD_Label] = True
        self.TestSet.loc[self.TestSet['KESimilarity'] <= self.splitting_value, self.AD_Label] = False

        AD = self.TestSet[self.AD_Label]

        col_name_id = self.TrainSet.columns[0]
        ids = self.TestSet[col_name_id]


        results = pd.DataFrame(np.column_stack([ids, AD]),
                               columns=['idTest', 'AD'])

        # print (results)
        return results




# %% OPERA AD
class OPERAApplicabilityDomain(ApplicabilityDomainStrategy):
    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
        self.parameters = {'k': 5, 'weakLocal': 0.4, 'exceptionalLocal': 0.6,
                           'leverageThreshold': 3 * (TrainSet.shape[1] / TrainSet.shape[0]),
                           'similarity': 'euclidean', 'onlyLocal': 0.01}
        self.AD_Label = 'WITHIN_OPERA_AD'

    @staticmethod
    def MansouriSimilarity(Similarities):
        weights = OPERAApplicabilityDomain.MansouriWeights(Similarities)
        return 1 / (1 + np.dot(weights, Similarities))

    @staticmethod
    def SmoothingFactor(S):
        if S <= 0.1:
            f = 5e-4
        else:
            f = 5e-2
        return f

    @staticmethod
    def MansouriWeights(Similarities):
        denominator = np.sum(OPERAApplicabilityDomain.MansouriReciprocal(Similarities))
        fs = np.array([OPERAApplicabilityDomain.SmoothingFactor(i) for i in Similarities])
        weights = (fs + Similarities) / denominator
        return weights

    @staticmethod
    def MansouriReciprocal(S):
        f = np.array([OPERAApplicabilityDomain.SmoothingFactor(i) for i in S])
        return 1 / (f + S)

    @staticmethod
    def LocalIndexConversion(SimilarityArray):
        return np.array([OPERAApplicabilityDomain.MansouriSimilarity(Similarity) for Similarity in SimilarityArray])

    @staticmethod
    def Leverage(X):
        core = np.linalg.inv(np.matmul(X.transpose(), X))
        hat = np.matmul(X, np.matmul(core, X.transpose()))
        leverages = np.diagonal(hat)
        return leverages

    def set_local_parameters(self, weak, strong):
        self.parameters['weakLocal'] = float(weak)
        self.parameters['exceptionalLocal'] = float(strong)

    def evaluate(self):
        ###
        self.nbrs = NearestNeighbors(n_neighbors=self.parameters['k'] + 1, algorithm='brute',
                                     metric=self.parameters['similarity'])
        nbrs = self.nbrs
        TrainSet = self.TrainSet[self.embedding]
        TestSet = self.TestSet[self.embedding]
        ###
        scaler = StandardScaler().fit(TrainSet)
        train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
        ###
        nbrs.fit(train_x)
        train_neighbors, test_neighbors = nbrs.kneighbors(train_x), nbrs.kneighbors(test_x)
        ###
        train_distances, test_distances = train_neighbors[0][:, 1:], test_neighbors[0][:, :-1]
        train_local_index, test_local_index = OPERAApplicabilityDomain.LocalIndexConversion(
            train_distances), OPERAApplicabilityDomain.LocalIndexConversion(test_distances)
        ###
        train_leverages, test_leverages = OPERAApplicabilityDomain.Leverage(train_x), OPERAApplicabilityDomain.Leverage(
            test_x)
        ###
        self.TrainSet['Leverage'] = train_leverages
        self.TrainSet['OPERALocalIndex'] = train_local_index
        self.TestSet['Leverage'] = test_leverages
        self.TestSet['OPERALocalIndex'] = test_local_index

        ###
        def determineLocalReliability(localIndex):
            if localIndex <= self.parameters['weakLocal']:
                return 'WEAK'
            elif localIndex > self.parameters['weakLocal'] and localIndex <= self.parameters['exceptionalLocal']:
                return 'MODERATE'
            else:
                return 'STRONG'

        def determineLeverageReliability(leverage):
            if leverage >= self.parameters['leverageThreshold']:
                return False
            else:
                return True

        def combinedCriteria(numericLocalIndex, numericLeverage):
            localIndex = determineLocalReliability(numericLocalIndex)
            leverage = determineLeverageReliability(numericLeverage)
            if localIndex == 'WEAK' and leverage == False:
                return False
            elif localIndex == 'MODERATE' and leverage == False:
                return False
            elif localIndex == 'STRONG' and leverage == False:
                return False
            elif localIndex == 'WEAK' and leverage == True:
                return False
            else:
                return True
                ###

        combined_train_assignment = [combinedCriteria(i, j) for i, j in
                                     zip(self.TrainSet['OPERALocalIndex'], self.TrainSet['Leverage'])]
        combined_test_assignment = [combinedCriteria(i, j) for i, j in
                                    zip(self.TestSet['OPERALocalIndex'], self.TestSet['Leverage'])]
        self.TrainSet[self.AD_Label] = combined_train_assignment
        self.TestSet[self.AD_Label] = combined_test_assignment


# %% Leverage
# class LeverageApplicabilityDomain(ApplicabilityDomainStrategy):
#     def __init__(self, TrainSet, TestSet, is_categorical):
#         ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
#         self.parameters = {
#             'leverageThreshold': 3 * (TrainSet.shape[1] / TrainSet.shape[0]),
#             'similarity': 'euclidean'}
#         self.AD_Label = 'WITHIN_OPERA_AD'
#
#     def set_local_parameters(self, parameters):
#         self.parameters = parameters
#
#     def evaluate(self):
#
#         ###
#         TrainSet = self.TrainSet[self.embedding]
#         TestSet = self.TestSet[self.embedding]
#         ###
#         scaler = StandardScaler().fit(TrainSet)
#         train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
#         ###
#         #TODO test_leverages are wrong since invXTX should always be based on training set
#         train_leverages, test_leverages = OPERAApplicabilityDomain.Leverage(train_x), OPERAApplicabilityDomain.Leverage(
#             test_x)
#         ###
#         self.TrainSet['Leverage'] = train_leverages
#         self.TestSet['Leverage'] = test_leverages
#
#         splitting_value = helpers.find_split_value(list(train_leverages), 0.95)
#         self.set_local_parameters({'leverageThreshold': splitting_value, 'similarity': 'euclidean'})
#         self.splitting_value = splitting_value
#
#         def determineLeverageReliability(leverage):
#             if leverage >= self.parameters['leverageThreshold']:
#                 return False
#             else:
#                 return True
#
#         combined_train_assignment = [determineLeverageReliability(j) for j in self.TrainSet['Leverage']]
#         combined_test_assignment = [determineLeverageReliability(j) for j in self.TestSet['Leverage']]
#
#         self.TrainSet[self.AD_Label] = combined_train_assignment
#         self.TestSet[self.AD_Label] = combined_test_assignment


# %% OPERA Local AD
# class OPERALocalApplicabilityDomain(ApplicabilityDomainStrategy):
#     def __init__(self, TrainSet, TestSet, is_categorical):
#         ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
#         self.parameters = {'k': 5, 'weakLocal': 0.4, 'exceptionalLocal': 0.6,
#                            'similarity': 'euclidean', 'onlyLocal': 0.01}
#         self.AD_Label = 'WITHIN_OPERA_AD'
#
#     def set_parameters(self, parameters):
#         self.parameters = parameters
#
#     def calculate_local(self, percentage_exclusion):
#         ###
#         self.nbrs = NearestNeighbors(n_neighbors=self.parameters['k'] + 1, algorithm='brute',
#                                      metric=self.parameters['similarity'])
#         nbrs = self.nbrs
#         TrainSet = self.TrainSet[self.embedding]
#         TestSet = self.TestSet[self.embedding]
#         ###
#         scaler = StandardScaler().fit(TrainSet)
#         train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
#         ###
#         nbrs.fit(train_x)
#         train_neighbors, test_neighbors = nbrs.kneighbors(train_x), nbrs.kneighbors(test_x)
#         ###
#         train_distances, test_distances = train_neighbors[0][:, 1:], test_neighbors[0][:, :-1]
#         train_local_index, test_local_index = OPERAApplicabilityDomain.LocalIndexConversion(
#             train_distances), OPERAApplicabilityDomain.LocalIndexConversion(test_distances)
#         ###
#         self.TrainSet['OPERALocalIndex'] = train_local_index
#         self.TestSet['OPERALocalIndex'] = test_local_index
#         ###
#         index = int(self.TrainSet.shape[0] * percentage_exclusion)
#         train_local_index = list(train_local_index)
#         train_local_index.sort(key=lambda x: x)
#         self.splitting_value = (train_local_index[index] + train_local_index[index + 1]) / 2.0
#
#     def evaluate(self):
#         def determineLocalReliability(localIndex):
#             if localIndex <= self.parameters['weakLocal']:
#                 return False
#             else:
#                 return True
#
#         ###
#         combined_train_assignment = [determineLocalReliability(i) for i in self.TrainSet['OPERALocalIndex']]
#         combined_test_assignment = [determineLocalReliability(i) for i in self.TestSet['OPERALocalIndex']]
#         self.TrainSet[self.AD_Label] = combined_train_assignment
#         self.TestSet[self.AD_Label] = combined_test_assignment


# %% OPERA Local AD
class OPERALocalApplicabilityDomain(ApplicabilityDomainStrategy):
    """
    Revised OPERA local AD to make it consistent with other ADs (TMM)
    """

    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)

        self.AD_Label = 'WITHIN_OPERA_AD'

    def set_parameters(self, parameters):
        self.parameters = parameters


    def evaluate(self, embedding):
        """
        Returns dataframe with neighbors and AD result
        """
        self.nbrs = NearestNeighbors(n_neighbors=self.parameters['k'] + 1, algorithm='brute',
                                     metric=self.parameters['similarity'])
        nbrs = self.nbrs
        TrainSet = self.TrainSet[embedding]
        TestSet = self.TestSet[embedding]
        ###
        scaler = StandardScaler().fit(TrainSet)
        train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)
        ###
        nbrs.fit(train_x)

        train_distances, train_indices = nbrs.kneighbors(train_x)
        test_distances, test_indices = nbrs.kneighbors(test_x)

        train_distances = train_distances[:, 1:]
        test_distances = test_distances[:, :-1]

        train_indices = train_indices[:, 1:]
        test_indices = test_indices[:, :-1]

        # print(TrainSet[test_indices[0][1]][0])
        col_name_id = self.TrainSet.columns[0]

        #TODO there must be a more compact way to to do this- but this works:
        neighbor1 = test_indices[:, 0]
        neighbor2 = test_indices[:, 1]
        neighbor3 = test_indices[:, 2]
        neighbor4 = test_indices[:, 3]
        neighbor5 = test_indices[:, 4]

        id = self.TestSet[col_name_id]
        id1 = self.TrainSet[col_name_id].loc[neighbor1]
        id2 = self.TrainSet[col_name_id].loc[neighbor2]
        id3 = self.TrainSet[col_name_id].loc[neighbor3]
        id4 = self.TrainSet[col_name_id].loc[neighbor4]
        id5 = self.TrainSet[col_name_id].loc[neighbor5]

        train_local_index, test_local_index = OPERAApplicabilityDomain.LocalIndexConversion(
            train_distances), OPERAApplicabilityDomain.LocalIndexConversion(test_distances)
        ###
        self.TrainSet['OPERALocalIndex'] = train_local_index
        self.TestSet['OPERALocalIndex'] = test_local_index
        ###

        #Use 1-fraction because train_local_index is in terms of similarity instead of distance
        self.splitting_value = helpers.find_split_value(list(train_local_index), 1-self.parameters['fractionTrainingSetInsideAD'])

        # print('splitting value=',self.splitting_value)

        ###
        self.TrainSet[self.AD_Label] = True
        self.TrainSet.loc[self.TrainSet['OPERALocalIndex'] <= self.splitting_value, self.AD_Label] = False

        self.TestSet[self.AD_Label] = True
        self.TestSet.loc[self.TestSet['OPERALocalIndex'] <= self.splitting_value, self.AD_Label] = False

        AD = self.TestSet[self.AD_Label]

        results = pd.DataFrame(np.column_stack([id, id1, id2, id3, id4, id5, AD]),
                               columns=['idTest', 'idNeighbor1', 'idNeighbor2',
                                        'idNeighbor3', 'idNeighbor4', 'idNeighbor5', 'AD'])

        # print (results)
        return results

class OPERAGlobalApplicabilityDomain(ApplicabilityDomainStrategy):
    """
    Leverage based AD that OPERA uses for Global AD
    """

    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)

        self.AD_Label = 'WITHIN_OPERA_AD'

    def set_parameters(self, parameters):
        self.parameters = parameters

    def evaluate(self, embedding):
        """
        Returns dataframe with neighbors and AD result
        """
        TrainSet = self.TrainSet[embedding]
        TestSet = self.TestSet[embedding]
        ###
        scaler = StandardScaler().fit(TrainSet)
        train_x, test_x = scaler.transform(TrainSet), scaler.transform(TestSet)

        # print(train_x)
        # https: // en.wikipedia.org / wiki / Leverage_(statistics)
        invXTX = np.linalg.inv(np.matmul(train_x.transpose(), train_x))# invXTX should be based on training set (since model is based on training set)

        leverages_train = np.matmul(train_x, np.matmul(invXTX, train_x.transpose())).diagonal()
        leverages_test = np.matmul(test_x, np.matmul(invXTX, test_x.transpose())).diagonal()

        self.splitting_value = helpers.find_split_value(list(leverages_train), self.parameters['fractionTrainingSetInsideAD'])

        cutoff_OPERA = 3 * train_x.shape[1]/train_x.shape[0] # 3 x p /n used by OPERA and in stats books

        print('cutoff_OPERA=',cutoff_OPERA, 'splittingValue=',self.splitting_value)  #they come out pretty similar

        self.TestSet[self.AD_Label] = True
        self.TestSet.loc[leverages_test > self.splitting_value, self.AD_Label] = False
        AD = self.TestSet[self.AD_Label]

        col_name_id = self.TrainSet.columns[0]
        ids = self.TestSet[col_name_id]

        # print(AD)
        results = pd.DataFrame(np.column_stack([ids, AD]), columns=['idTest', 'AD'])
        # print(hat.shape)
        return results

class OPERAStatic:
    @staticmethod
    def MansouriSimilarity(Similarities):
        weights = OPERAApplicabilityDomain.MansouriWeights(Similarities)
        return 1 / (1 + np.dot(weights, Similarities))

    @staticmethod
    def SmoothingFactor(S):
        if S <= 0.1:
            f = 5e-4
        else:
            f = 5e-2
        return f

    @staticmethod
    def MansouriWeights(Similarities):
        denominator = np.sum(OPERAApplicabilityDomain.MansouriReciprocal(Similarities))
        fs = np.array([OPERAApplicabilityDomain.SmoothingFactor(i) for i in Similarities])
        weights = (fs + Similarities) / denominator
        return weights

    @staticmethod
    def MansouriReciprocal(S):
        f = np.array([OPERAApplicabilityDomain.SmoothingFactor(i) for i in S])
        return 1 / (f + S)

    @staticmethod
    def LocalIndexConversion(SimilarityArray):
        return np.array([OPERAApplicabilityDomain.MansouriSimilarity(Similarity) for Similarity in SimilarityArray])
