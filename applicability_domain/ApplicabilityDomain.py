# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:40:56 2022

@author: NCHAREST
"""
import time

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr
from sklearn.neighbors import KernelDensity
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
# from afxres import AFX_IDS_COMPANY_NAME


debug = False

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
        self.splitting_value = None

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


    def get_results_df(self, AD, col_name_id, id, test_indices, test_distances=None):
        """Get AD dataframe for k neighbors"""

        # Recoding so that can have arbitrary number of neighbors (i.e. store as list), TODO check Java AD code for places this breaks things
        # Extract neighbors
        neighbors = [test_indices[:, i] for i in range(self.parameters['k'])]
        # Retrieve IDs for each neighbor and combine them into a list
        ids_combined = [self.TrainSet[col_name_id].loc[neighbor].tolist() for neighbor in neighbors]

        # Transpose ids_combined to align with test_indices rows
        ids_combined_transposed = np.array(ids_combined).T.tolist()
        # ids_combined_transposed = [list(col) for col in zip(*ids_combined)] # make it json serializable by not using np.array
        
        # print(ids_combined_transposed)
        
        # makes results list instead of df (if want to try to add similarity/distance values later)
        # results = [{'idTest': id[i],'ids': list(map(str, ids_combined_transposed[i])),'AD': AD[i]}
        #     for i in range(len(id))
        # ]
        # Prepare the DataFrame
        
        # distances = np.array(test_distances).T.tolist()
        # print(test_distances)
        # print(ids_combined_transposed)
        # print(distances)
        
        if test_distances is not None:
            results = pd.DataFrame({'idTest': id, 'ids': ids_combined_transposed, 'distances': list(test_distances), 'AD': AD})
            # Ensure 'ids' is stored as a list of strings instead of as comma delimited string
            results['ids'] = results['ids'].apply(lambda x: list(map(str, x)))
        
        else:
            results = pd.DataFrame({'idTest': id, 'ids': ids_combined_transposed, 'AD': AD})
            # Ensure 'ids' is stored as a list of strings instead of as comma delimited string
            results['ids'] = results['ids'].apply(lambda x: list(map(str, x)))
            
        
        return results


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

        # print('metric', self.parameters['similarity'])

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

        # following is awkward but only way it works correctly:
        train_indices = train_indices[:, 1:]  #this omits the chemical itself from being a neighbor
        train_distances = train_distances[:, 1:]

        test_indices = test_indices[:, :-1]  #this assumes that we dont need the 4th neighbor because we arent checking for exact match like we did with training set
        test_distances = test_distances[:, :-1]  #returns all elements [:] except the last one -1

        # print(TrainSet[test_indices[0][1]][0])
        col_name_id = self.TrainSet.columns[0]


        # print(neighbor1[0],neighbor2[0],neighbor3[0])

        id = self.TestSet[col_name_id]


        train_TESTSimilarity, test_TESTSimilarity = list(np.mean(train_distances, axis=1)), list(
            np.mean(test_distances, axis=1))
        ###
        self.TrainSet['TESTSimilarity'] = train_TESTSimilarity
        self.TestSet['TESTSimilarity'] = test_TESTSimilarity
        ###

        self.splitting_value = helpers.find_split_value(train_TESTSimilarity, self.parameters['fractionTrainingSetInsideAD'])

        if debug:
            print('splitting_value', self.splitting_value)
            
        self.TrainSet[self.AD_Label] = True
        self.TestSet[self.AD_Label] = True

        # Sklearn similarities are actually distances (1-SC), so if distance > cutoff it's outside AD:
        self.TrainSet.loc[self.TrainSet['TESTSimilarity'] > self.splitting_value, self.AD_Label] = False
        self.TestSet.loc[self.TestSet['TESTSimilarity'] > self.splitting_value, self.AD_Label] = False

        # print(self.TestSet[self.AD_Label])  # array of whether or not it's in AD

        AD = self.TestSet[self.AD_Label]
        # print(AD.value_counts()[False])

        # print(AD_TR.value_counts()[False])

        # neighbor1 = test_indices[:, 0]
        # neighbor2 = test_indices[:, 1]
        # neighbor3 = test_indices[:, 2]
        # id1 = self.TrainSet[col_name_id].loc[neighbor1]
        # id2 = self.TrainSet[col_name_id].loc[neighbor2]
        # id3 = self.TrainSet[col_name_id].loc[neighbor3]
        #
        # results = pd.DataFrame(np.column_stack([id, id1, id2, id3, AD]),
        #                        columns=['idTest', 'idNeighbor1', 'idNeighbor2', 'idNeighbor3', 'AD'])

        results = self.get_results_df(AD, col_name_id, id, test_indices,test_distances)

        # TODO ideally results would also store the similarity for each neighbor and the cutoff value for inside AD but more complicated...

        # pd.set_option('display.max_columns', None)
        # print(results)

        # print(results)

        if debug:
            print(results)

        # results.to_json('bob.json')

        t2 = time.time()
        # print((t2-t1),' secs to evaluate')

        # adResults = ADResults(AD, self.splitting_value, results)
        # print(json.dumps(adResults.__dict__,indent=4))

        return results


# TODO convert to dataclass:
class ADResults:
    def __init__(self, AD, splitSimilarity,results):
        self.AD = AD
        self.splitting_value = splitSimilarity
        self.results = results


class TESTFragmentCounts(ApplicabilityDomainStrategy):
    
    def __init__(self, TrainSet, TestSet, is_categorical):
        ApplicabilityDomainStrategy.__init__(self, TrainSet, TestSet, is_categorical)
        
    def evaluate(self, embedding):
        # Define the fragment range
        start_column = "As [+5 valence, one double bond]"
        stop_column = "-N=S=O"
    
        # Robustly get the slice between start and stop by position (inclusive),
        # regardless of column order
        cols = self.TestSet.columns
        if start_column not in cols or stop_column not in cols:
            raise KeyError("Start or stop column not found in TestSet.")
        lo = cols.get_loc(start_column)
        hi = cols.get_loc(stop_column)
        lo, hi = sorted([lo, hi])
        frag_cols = cols[lo:hi+1]
    
        # Subset the test set to the fragment range
        df_new = self.TestSet.loc[:, frag_cols]
    
        # Keep only fragments that also exist in the training set
        common_columns = frag_cols.intersection(self.TrainSet.columns)
    
        # Training stats for each fragment
        min_values = (
            self.TrainSet[common_columns]
            .apply(lambda col: col[col > 0].min())
            .fillna(0)
        )
        max_values = self.TrainSet[common_columns].max()
    
        # Count how many training chemicals contain each fragment
        training_count = (self.TrainSet[common_columns] > 0).sum()
    
        # Build a long-format dataframe of test counts for all chemicals/fragments
        results_df = (
            df_new[common_columns]
            .reset_index()  # preserves the original test chemical index
            .rename(columns={'index': 'test_chemical'})  # name the index column
            .melt(id_vars=['test_chemical'], var_name='fragment', value_name='test_value')
        )
    
        # Keep only fragments that are present in the test chemical (count > 0)
        results_df = results_df[results_df['test_value'] > 0]
    
        # Attach training stats per fragment
        train_stats = pd.DataFrame({
            'fragment': common_columns,
            'training_min': min_values.values,
            'training_max': max_values.values,
            'training_count': training_count.values
        })
        results_df = results_df.merge(train_stats, on='fragment', how='left')
    
        # Correctly attach idTest per row by mapping test_chemical -> first column of TestSet
        id_map = self.TestSet.iloc[:, 0]  # Series indexed by test chemical index
        results_df['idTest'] = results_df['test_chemical'].map(id_map)
    
        # Build per_chemical_df:
        # - fragment_table: list of dicts with fragment-level details
        # - AD: True if all test_value values are within [training_min, training_max]
        def _build_row(group: pd.DataFrame) -> dict:
            frag_table = group[['fragment', 'test_value', 'training_min', 'training_max', 'training_count']] \
                .to_dict(orient='records')
            in_bounds = (group['test_value'] >= group['training_min']) & (group['test_value'] <= group['training_max'])
            return {
                'idTest': group['idTest'].iloc[0],
                'fragment_table': frag_table,
                'AD': bool(in_bounds.all())
            }
    
        if results_df.empty:
            per_chemical_df = pd.DataFrame(columns=['idTest', 'fragment_table', 'AD'])
        else:
            rows = [_build_row(g) for _, g in results_df.groupby('idTest', sort=False)]
            per_chemical_df = pd.DataFrame(rows, columns=['idTest', 'fragment_table', 'AD'])
    
        return per_chemical_df
        

# At this point:
# - results_df contains long-form rows for all nonzero fragments per test chemical.
# - per_chemical is a dict keyed by ID, each value includes ID and lists of fragments/stats.
# - per_chemical_df is the dataframe version of per_chemical, with an AD column indicating
#   whether all fragment test_values are within training bounds.

# At this point:
# - results_df contains long-form rows for all nonzero fragments per test chemical.
# - per_chemical is a dict keyed by ID, each value includes ID and lists of fragments/stats.
# - per_chemical_df is the dataframe version of per_chemical; first row printed above.        

        
        # start_column = "As [+5 valence, one double bond]"
        # stop_column = "-N=S=O"
        #
        # # TODO technically it should check if the test chemical has a fragment not in the embedding and not just in the training set range
        #
        #
        # df_new = self.TestSet.loc[:, start_column:stop_column]
        # columns_greater_than_zero = df_new.iloc[0] > 0
        # df_new = df_new.loc[:, columns_greater_than_zero]
        # common_columns = df_new.columns.intersection(self.TrainSet.columns)
        #
        # # # Calculate min and max for each common column in df_training
        # # min_values = df_training[common_columns].apply(lambda col: col[col > 0].min())
        # min_values = (self.TrainSet[common_columns].apply(lambda col: col[col > 0].min()).fillna(0))        
        #
        # # check the count of chemicals containing the fragment in the training set
        #
        #
        # max_values = self.TrainSet[common_columns].max()
        #
        # results = {
        #     "test_chemical": df_new.loc[0, common_columns].to_dict(),
        #     "training_min": min_values.to_dict(),
        #     "training_max": max_values.to_dict()
        # }
        #
        # # modelResults.adResultsFrag = results
        # print(json.dumps(results, indent=4))
        #
        # outside_ad = False
        #
        # for col_name in results["test_chemical"].keys():
        #     test_value = int(results["test_chemical"][col_name])
        #     training_min = int(results["training_min"][col_name])    
        #     training_max = int(results["training_max"][col_name])
        #
        #             # Determine if the row should be highlighted
        #     if test_value < training_min or test_value > training_max:
        #         outside_ad = True
        #
        # adResultsFrag = {}
        # adResultsFrag["adMethod"] = {}
        # adResultsFrag["adMethod"]["name"] = PredictConstants.TEST_FRAGMENTS
        # adResultsFrag["adMethod"]["description"] = "Whether or not the fragment counts are within the range for chemicals in the training set"
        #
        # adResultsFrag["AD"] = not outside_ad
        # adResultsFrag["fragmentTable"] = results     
        #
        # if outside_ad: 
        #     adResultsFrag["reasoning"] = "fragment counts were not within the training set range"
        #     adResultsFrag["conclusion"] = "Outside"
        # else:
        #     adResultsFrag["reasoning"] = "fragment counts were within the training set range"
        #     adResultsFrag["conclusion"] = "Inside"
        #
        # return "TODO"
        #


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
        id = self.TestSet[col_name_id]

        train_local_index, test_local_index = OPERAApplicabilityDomain.LocalIndexConversion(
            train_distances), OPERAApplicabilityDomain.LocalIndexConversion(test_distances)
        ###
        self.TrainSet['OPERALocalIndex'] = train_local_index
        self.TestSet['OPERALocalIndex'] = test_local_index
        ###

        #Use 1-fraction because train_local_index is in terms of similarity instead of distance
        self.splitting_value = helpers.find_split_value(list(train_local_index), 1-self.parameters['fractionTrainingSetInsideAD'])

        # self.splitting_value=0.065

        # print('cutoff opera local index=', self.splitting_value)


        # print('splitting value=',self.splitting_value)

        ###
        self.TrainSet[self.AD_Label] = True
        self.TrainSet.loc[self.TrainSet['OPERALocalIndex'] <= self.splitting_value, self.AD_Label] = False

        self.TestSet[self.AD_Label] = True
        self.TestSet.loc[self.TestSet['OPERALocalIndex'] <= self.splitting_value, self.AD_Label] = False

        AD = self.TestSet[self.AD_Label]

        # neighbor1 = test_indices[:, 0]
        # neighbor2 = test_indices[:, 1]
        # neighbor3 = test_indices[:, 2]
        # neighbor4 = test_indices[:, 3]
        # neighbor5 = test_indices[:, 4]
        #
        # id1 = self.TrainSet[col_name_id].loc[neighbor1]
        # id2 = self.TrainSet[col_name_id].loc[neighbor2]
        # id3 = self.TrainSet[col_name_id].loc[neighbor3]
        # id4 = self.TrainSet[col_name_id].loc[neighbor4]
        # id5 = self.TrainSet[col_name_id].loc[neighbor5]
        #
        # results = pd.DataFrame(np.column_stack([id, id1, id2, id3, id4, id5, AD]),
        #                        columns=['idTest', 'idNeighbor1', 'idNeighbor2',
        #                                 'idNeighbor3', 'idNeighbor4', 'idNeighbor5', 'AD'])

        results = self.get_results_df(AD, col_name_id, id, test_indices)

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
        
        # print(leverages_train)
        # print(leverages_test)

        self.splitting_value = helpers.find_split_value(list(leverages_train), self.parameters['fractionTrainingSetInsideAD'])

        cutoff_OPERA = 3 * train_x.shape[1]/train_x.shape[0] # 3 x p /n used by OPERA and in stats books
        # self.splitting_value=cutoff_OPERA

        # print('cutoff_OPERA=',cutoff_OPERA, 'splittingValue=',self.splitting_value)  #they come out pretty similar

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
