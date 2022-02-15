# from URL https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

import time
from pprint import pprint

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from models import df_utilities as DFU
import json
import numpy as np

__author__ = "Todd Martin"


# class to run random forest using files and be able to use multiple processors. When running using webservice, it
# doesnt seem to let you use multiple threads...
# if n_jobs>1, get UserWarning: Loky-backed parallel loops cannot be nested below threads, setting n_jobs=1
# This class will aid in building models for large data sets like logP which requires one to remove logP based
# descriptors

class Model:
    """Trains and makes predictions with a random forest model"""
    def __init__(self, df_training, remove_log_p_descriptors, n_threads):
        """Initializes the RF model with optimal parameters and provided data in pandas dataframe"""
        self.rfr = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.is_binary = None  # Set automatically when training data is loaded
        self.df_training = df_training
        self.version = '1.1'
        self.n_estimators = 100
        self.min_impurity_decrease = 1e-5
        self.n_threads = n_threads
        self.qsar_method = 'Random forest'
        self.description = 'sklearn implementation of random forest ' \
                           '(https://scikit-learn.org/stable/modules/generated/' \
                           'sklearn.ensemble.RandomForestClassifier.html)'

    def build_model(self):
        """Trains the RF model on provided data"""
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]

        # Number of features to consider at every split
        max_features = ['sqrt']
        # max_features = [0.01]

        # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        # max_depth.append(None)
        # # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        # Create the random grid

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features}


        # random_grid = {'n_estimators': n_estimators,
        #                'max_features': max_features,
        #                'max_depth': max_depth,
        #                'min_samples_split': min_samples_split,
        #                'min_samples_leaf': min_samples_leaf,
        #                'bootstrap': bootstrap}



        pprint(random_grid)


        if self.is_binary:
            self.rfr = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=self.n_threads,
                                              min_impurity_decrease=self.min_impurity_decrease)
            # Train the model on training data
            self.rfr.fit(train_features, train_labels) #TODO implement CV search for classifier


        else:
            # self.rfr = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=self.n_threads,
            #                                  min_impurity_decrease=self.min_impurity_decrease)

            rf = RandomForestRegressor()
            rf.min_impurity_decrease =  self.min_impurity_decrease
            # Random search of parameters, using 3 fold cross validation,
            # search across 100 different combinations, and use all available cores
            rf_random = RandomizedSearchCV(n_jobs=self.n_threads, estimator=rf, param_distributions=random_grid, cv=5, verbose=2,
                                           random_state=42)
            # Fit the random search model
            rf_random.fit(train_features, train_labels)

            self.rfr = rf_random.best_estimator_

            self.n_estimators = rf_random.best_params_['n_estimators']

            print("best features:")
            pprint(rf_random.best_params_)



        print('Score for Training data = ', self.rfr.score(train_features, train_labels))

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        # pickle.dump(rfr, open("rfr.p", "wb"))
        return self

    def do_predictions(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')
        predictions = self.rfr.predict(pred_features)

        print('Score for Test data = ', self.rfr.score(pred_features, pred_labels))

        # Return predictions
        return predictions


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""
        self.is_binary = model.is_binary
        self.n_estimators = model.n_estimators
        self.min_impurity_decrease = model.min_impurity_decrease
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)


def main():
    """
    Code to run from text files rather than webservice
    :return:
    """
    # endpoint = 'Octanol water partition coefficient'
    # endpoint = 'Water solubility'
    # endpoint = 'Melting point'
    endpoint = 'LogBCF'

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'

    folder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/'
    folder += 'QSAR_Model_Building/data/DataSetsBenchmark/' + endpoint + ' OPERA/'
    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    # Parameters needed to build model:
    n_threads = 30
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # df_training = pd.read_csv(training_tsv_path, sep='\t')
    # df_prediction = pd.read_csv(prediction_tsv_path, sep='\t')

    model = Model(df_training, remove_log_p_descriptors, n_threads)
    model.build_model()

    print(ModelDescription(model).to_json())
    model.do_predictions(df_prediction)


if __name__ == "__main__":
    main()
