# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:13 2021

@author: CRupakhe
@author: Todd Martin (conversion to Python 3.7)
@author: Gabriel Sinclair (refactoring)

"""
# import pickle
import time
import copy
import threading
import json
import numpy as np
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.svm import NuSVR, SVC
from sklearn.metrics import r2_score, roc_curve, auc
from sklearn.decomposition import PCA
from models import df_utilities as DFU


# Stripped-down class to run the single optimal SVM model for webservice: no consensus, predetermined optimal
# parameters, etc. Use SVM_Full_Python.py for consensus modeling, parameter testing, or other tasks
# requiring full functionality.
class Model:
    def __init__(self, df_training, remove_log_p_descriptors, n_threads):
        # Model description
        self.is_binary = None  # Set automatically when training data is processed
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.remove_corr = True
        self.df_training = df_training
        self.version = '1.1'
        self.n_threads = n_threads
        self.qsar_method = 'SVM'
        self.description = 'sklearn implementation of SVM using NuSVR for regression' \
                           ' or SVC for classification ' \
                           '(https://scikit-learn.org/stable/modules/svm.html),' \
                           ' no applicability domain'

        # SVM model parameters
        # self.corr_limit = 0.95
        self.n_folds = 5
        self.n_feats = 0.95

        # Ranges for cross-validation
        self.c_space = np.logspace(-1, 1, 50)
        self.gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]  # [0.01]
        self.nu_space = [i / 100.0 for i in range(5, 100, 15)]  # nu-svr error params (0,1]
        self.kernel = "rbf"

        # To store transformations of whole data set
        self.scaler_fit = None

        # To store transformations of data set by fold
        self.pca_fit_by_fold = []
        self.covar_x_inv_by_fold = []
        self.training_data_by_fold = []
        self.test_data_by_fold = []

        # To store generated model
        self.model = None
        self.params = None

    def build_model(self):
        t1 = time.time()
        self.prepare_training_data()
        score = self.cross_validate()
        print('Score for Training data = ', score)
        self.df_training = None  # Delete training data to save space in database
        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')
        # pickle.dump(self, open("svm.p", "wb"))

    def prepare_training_data(self):
        training_ids, training_labels, training_features, training_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, True)
        self.descriptor_names = training_column_names
        if self.is_binary:
            self.nu_space = [0]  # Parameter nu is not used in SVC

        training_features = pd.DataFrame(training_features)

        # Removes correlated features
        # corr = training_features.corr().abs()
        # pper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        # self.corr_to_drop = [column for column in upper.columns if any(upper[column] > self.corr_limit)]
        # training_features.drop(self.corr_to_drop, axis=1, inplace=True)

        # Scales descriptors
        scaler = pp.MinMaxScaler()
        training_features = scaler.fit_transform(training_features)
        self.scaler_fit = scaler

        # Performs PCA with automatic selection of n_feats
        folds = [i % self.n_folds for i in range(len(training_labels))]
        np.random.shuffle(folds)

        self.n_feats = max(self.auto_select_n_feats(folds, training_features, training_labels), 40)

        for i in range(self.n_folds):
            training_features_i, training_labels_i, test_features_i, test_labels_i \
                = self.split_data_by_fold(folds, training_features, training_labels, i)
            self.training_data_by_fold.append([training_features_i, training_labels_i])
            self.test_data_by_fold.append([test_features_i, test_labels_i])
            self.do_pca_on_fold(i)

    def do_pca_on_fold(self, fold):
        pca = PCA(n_components=self.n_feats)
        self.training_data_by_fold[fold][0] = pca.fit_transform(self.training_data_by_fold[fold][0])
        self.test_data_by_fold[fold][0] = pca.transform(self.test_data_by_fold[fold][0])
        self.pca_fit_by_fold.append(pca)

        covar_x_inv = np.linalg.inv(self.training_data_by_fold[fold][0].transpose()
                                    .dot(self.training_data_by_fold[fold][0]))
        self.covar_x_inv_by_fold.append(covar_x_inv)

    @staticmethod
    def split_data_by_fold(folds, descriptors, exp_vals, fold):
        in_fold_idx = []
        not_in_fold_idx = []
        for i in range(len(folds)):
            if folds[i] == fold:
                in_fold_idx.append(i)
            else:
                not_in_fold_idx.append(i)
        training_descriptors = [descriptors[i] for i in not_in_fold_idx]
        training_exp_vals = [exp_vals[i] for i in not_in_fold_idx]
        test_descriptors = [descriptors[i] for i in in_fold_idx]
        test_exp_vals = [exp_vals[i] for i in in_fold_idx]

        return training_descriptors, training_exp_vals, test_descriptors, test_exp_vals

    def auto_select_n_feats(self, folds, descriptors, exp_vals):
        scree = []
        for i in range(self.n_folds):
            training_descriptors, training_exp_vals, test_descriptors, test_exp_vals \
                = self.split_data_by_fold(folds, descriptors, exp_vals, i)
            exp_var_rat = PCA().fit(training_descriptors).explained_variance_ratio_
            for j in range(1, len(exp_var_rat)):
                exp_var_rat[j] = exp_var_rat[j - 1] + exp_var_rat[j]
            scree.append(exp_var_rat)
        scree_avg = pd.DataFrame(scree).mean(axis=0)
        scree_avg_where = scree_avg[scree_avg > self.n_feats]
        n_feats = scree_avg_where.idxmin() + 1
        return n_feats

    def cross_validate(self):
        my_threads = []
        active_threads = []
        for c in self.c_space:
            for g in self.gamma_space:
                for n in self.nu_space:
                    thread = CrossValThread(self, c, g, n, self.kernel)
                    my_threads.append(thread)
                    active_threads.append(thread)

                    if len(active_threads) >= self.n_threads:
                        self.kill_threads(active_threads, 1)

                    thread.start()

        # Makes sure main-thread waits until workers are finished
        threads = copy.copy(my_threads)
        self.kill_threads(my_threads, 0)

        all_models = {}
        all_score_avgs = {}
        for th in threads:
            all_models[th.params] = th.models_q
            all_score_avgs[th.params] = np.average(th.score_q, axis=0)

        self.params = self.find_best_params(all_score_avgs)
        self.model = all_models[self.params]

        return all_score_avgs[self.params]

    @staticmethod
    def kill_threads(threads, limit):
        while len(threads) > limit:
            for th in threads:
                # th.is_alive() for 3.9, th.isAlive() for 3.7
                if not th.is_alive():
                    threads.pop(threads.index(th))

    @staticmethod
    def find_best_params(score_avgs):
        sorted_score_avgs = dict(sorted(score_avgs.items(), key=lambda item: item[1], reverse=True))
        return list(sorted_score_avgs)[0]

    def do_predictions(self, df_prediction):
        test_descriptors_by_fold, test_labels = self.prepare_test_data(df_prediction)

        predictions = []
        for i in range(len(test_labels)):
            chemical_predictions = []
            for j in range(self.n_folds):
                chemical_features = test_descriptors_by_fold[j][i]
                single_chemical_prediction = self.model[j].predict([chemical_features])
                chemical_predictions.append(single_chemical_prediction)
            avg_chemical_prediction = np.sum(chemical_predictions) / len(chemical_predictions) * 1.0
            if self.is_binary:
                final_chemical_prediction = 1.0 if avg_chemical_prediction >= 0.5 else 0.0
            else:
                final_chemical_prediction = avg_chemical_prediction
            predictions.append(final_chemical_prediction)

        if self.is_binary:
            fpr, tpr, thr = roc_curve(test_labels, predictions, pos_label=1)
            score = auc(fpr, tpr)
        else:
            score = r2_score(test_labels, predictions)
        print('Score for Test data = ', score)

        return predictions

    def prepare_test_data(self, df_prediction):
        test_ids, test_labels, test_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)

        # Removes correlated descriptors
        # test_features = pd.DataFrame(test_features)
        # test_features.drop(self.corr_to_drop, axis=1, inplace=True)

        # Scales descriptors
        test_features = self.scaler_fit.transform(test_features)

        # Performs PCA
        test_descriptors_by_fold = []
        for i in range(self.n_folds):
            if self.n_feats > 0:
                pca = self.pca_fit_by_fold[i]
                pca_test_descriptors = pca.transform(test_features)
                test_descriptors_by_fold.append(pca_test_descriptors)
            else:
                test_descriptors_by_fold.append(test_features)

        return test_descriptors_by_fold, test_labels


class CrossValThread(threading.Thread):
    def __init__(self, model, c, g, n, kernel):
        threading.Thread.__init__(self)

        self.is_binary = model.is_binary
        self.training_data_by_fold = model.training_data_by_fold
        self.test_data_by_fold = model.test_data_by_fold
        self.c = c
        self.g = g
        self.n = n
        self.kernel = kernel

        if self.is_binary:
            if self.kernel == "linear":
                self.params = str(self.c)
            else:
                self.params = str(self.c) + "_" + str(self.g)
        else:
            if self.kernel == "linear":
                self.params = str(self.c) + "_" + str(self.n)
            else:
                self.params = str(self.c) + "_" + str(self.g) + "_" + str(self.n)

        self.score_q = []
        self.models_q = []

    def run(self):
        for i in range(len(self.training_data_by_fold)):
            score, clf = self.train_and_test_fold(i)
            self.score_q.append(score)
            self.models_q.append(clf)

    def train_and_test_fold(self, fold):
        clf = None
        if self.is_binary:
            if self.kernel == "rbf":
                clf = SVC(C=self.c, gamma=self.g, kernel=self.kernel)
            elif self.kernel == "linear":
                clf = SVC(C=self.c, kernel=self.kernel)
        else:
            if self.kernel == "rbf":
                clf = NuSVR(C=self.c, nu=self.n, gamma=self.g, kernel=self.kernel)
            elif self.kernel == "linear":
                clf = NuSVR(C=self.c, nu=self.n, kernel=self.kernel)

        if clf is None:
            print("***clf=none")

        clf.fit(self.training_data_by_fold[fold][0], self.training_data_by_fold[fold][1])
        test_exp_vals_predicted = clf.predict(self.test_data_by_fold[fold][0])

        if self.is_binary:
            fpr, tpr, thr = roc_curve(self.test_data_by_fold[fold][1], test_exp_vals_predicted, pos_label=1)
            score = auc(fpr, tpr)
        else:
            score = r2_score(self.test_data_by_fold[fold][1], test_exp_vals_predicted)

        if np.isnan(score):
            score = 0.0
        return score, clf


class ModelDescription:
    def __init__(self, model):
        # Model description
        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.remove_corr = model.remove_corr
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description

        # SVM model parameters
        self.n_folds = model.n_folds
        self.n_feats = int(model.n_feats)
        self.kernel = "rbf"
        self.params = model.params

    def to_json(self):
        return json.dumps(self.__dict__)


def main():
    """
    Code to run from text files rather than webservice
    :return:
    """
    # endpoint = 'Octanol water partition coefficient'
    # endpoint = 'Water solubility'
    # endpoint = 'Melting point'
    # endpoint = 'LLNA'
    endpoint = 'LogKmHL'

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'

    folder = 'C:/Users/GSincl01/OneDrive - Environmental Protection Agency (EPA)/Python/pf-python-modelbuilding/data/'
    folder += 'DataSetsBenchmark/' + endpoint + ' OPERA/'
    # folder += 'datasets_benchmark_toxicity/' + endpoint + '/'
    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
    # training_file_name = endpoint + '_training_set-2d.csv'
    # prediction_file_name = endpoint + '_prediction_set-2d.csv'
    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path)
    df_prediction = DFU.load_df_from_file(prediction_tsv_path)

    model = Model(df_training, remove_log_p_descriptors, n_threads)
    model.build_model()

    print(ModelDescription(model).to_json())
    model.do_predictions(df_prediction)


if __name__ == "__main__":
    main()
