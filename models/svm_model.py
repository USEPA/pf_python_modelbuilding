# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:27:00 2022

@author: NCHAREST
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:13 2021

@author: CRupakhe
@author: Todd Martin (conversion to Python 3.7)
@author: Gabriel Sinclair (refactoring)
@author: Nathaniel Charest (refactor; modernizing to sklearn)

"""
# import pickle
import time
import json
import numpy as np
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from models import df_utilities as DFU


# %%

# Stripped-down class to run the single optimal SVM model for webservice: no consensus, predetermined optimal
# parameters, etc. Use SVM_Full_Python.py for consensus modeling, parameter testing, or other tasks
# requiring full functionality.
class Model:
    """Trains and makes predictions with an optimized support vector machines model"""

    def __init__(self, df_training, remove_log_p_descriptors, n_threads, modelid):
        """Initializes the SVM model with optimal parameters and provided data in pandas dataframe"""
        # Model description
        self.is_binary = None  # Set automatically when training data is processed
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.remove_corr = True
        self.df_training = df_training
        self.version = '1.2'
        self.n_threads = n_threads
        self.qsar_method = 'SVM'
        self.description = 'sklearn implementation of SVM using NuSVR for regression' \
                           ' or SVC for classification'

        self.description_url = 'https://scikit-learn.org/stable/modules/svm.html'

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

    def getModel(self):

        if self.is_binary == True:
            self.model = Pipeline([('scaler', StandardScaler()), ('estimator', SVC())])
        else:
            self.model = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])

        return self.model

    def getModel2(self):
        return  self.model



    def build_model(self):
        """Trains the SVM model on provided data"""
        t1 = time.time()
        # Do data prep
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        self.descriptor_names = train_column_names
        # configure model

        self.getModel()

        optimizer = GridSearchCV(self.model, {'estimator__C': list([10 ** x for x in range(-3, 4)]),
                                              'estimator__gamma': list([10 ** x for x in range(-3, 4)])},
                                 n_jobs=self.n_threads)

        optimizer.fit(train_features, train_labels)
        best_params = optimizer.best_params_
        self.model.set_params(**best_params)
        self.model.fit(train_features, train_labels)
        score = self.model.score(train_features, train_labels)
        print('Score for Training data = ', score)
        self.df_training = None  # Delete training data to save space in database
        t2 = time.time()
        self.training_time = t2 - t1
        print('Time to train model  = ', t2 - t1, 'seconds')
        # pickle.dump(self, open("svm.p", "wb"))

    def build_model_with_preselected_descriptors(self, descriptor_names):
        """Trains the RF model on provided data"""
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", descriptor_names)
        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        # configure model
        if self.is_binary == True:
            self.model = Pipeline([('scaler', StandardScaler()), ('estimator', SVC())])
        else:
            self.model = Pipeline([('scaler', StandardScaler()), ('estimator', SVR())])

        optimizer = GridSearchCV(self.model,
                                 {'estimator__C': np.arange(-3, 4, 0.5), 'estimator__gamma': self.gamma_space})
        optimizer.fit(train_features, train_labels)

        self.best_params = optimizer.best_params_
        self.model.set_params(**self.best_params)
        self.model.fit(train_features, train_labels)
        score = self.model.score(train_features, train_labels)
        print('Score for Training data = ', score)
        self.df_training = None  # Delete training data to save space in database
        t2 = time.time()
        self.training_time = t2 - t1
        print('Time to train model  = ', t2 - t1, 'seconds')

    def do_predictions(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')
        if (self.is_binary == True):
            predictions = self.model.predict_proba(pred_features)[:, 1]
        else:
            predictions = self.model.predict(pred_features)

        print('Score for Test data = ', self.model.score(pred_features, pred_labels))

        # Return predictions
        return predictions


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""
        # Model description
        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.remove_corr = model.remove_corr
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description
        self.description_url = model.description_url

        # SVM model parameters
        self.n_folds = model.n_folds
        self.n_feats = int(model.n_feats)
        self.kernel = "rbf"
        # self.params = model.params

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)

