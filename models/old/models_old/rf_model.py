# from URL https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split, RandomizedSearchCV, cross_validate, \
    GridSearchCV
from models import df_utilities as DFU
# from util import sqlalchemyutilities as SAU
import json
from qsar_models import Statistic, Model, ModelStatistic

__author__ = "Todd Martin"


# class to run random forest using files and be able to use multiple processors. When running using webservice, it
# doesnt seem to let you use multiple threads...
# if n_jobs>1, get UserWarning: Loky-backed parallel loops cannot be nested below threads, setting n_jobs=1
# This class will aid in building models for large data sets like logP which requires one to remove logP based
# descriptors

class Model:
    """Trains and makes predictions with a random forest model"""

    def __init__(self, df_training, remove_log_p_descriptors, n_threads, modelid):
        """Initializes the RF model with optimal parameters and provided data in pandas dataframe"""
        self.rfr = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.is_binary = None  # Set automatically when training data is loaded
        self.df_training = df_training
        self.version = '1.4'

        # For build_model() and for GA embedding calculations:
        self.n_estimators = 200
        self.max_samples = 0.99
        self.min_impurity_decrease = 1e-5
        # self.max_features='auto'

        # For build_model_with_grid_search():
        # self.hyper_parameter_grid = {'max_features': [4, 8, 'sqrt', 'log2', None],
        #                              'min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                              'n_estimators': [10 ** x for x in range(2, 4)], 'max_samples': [0.66, 0.99]}

        # self.hyper_parameter_grid = {'max_features': ['sqrt', 'log2'],
        #                              'min_impurity_decrease': [10 ** x for x in range(-5, 0)],
        #                              'n_estimators': [10 ** x for x in range(1, 4)]}

        self.hyper_parameter_grid = {'max_features': ['sqrt', 'log2'],
                                     'min_impurity_decrease': [10 ** x for x in range(-5, 0)],
                                     'n_estimators': [10, 100, 250, 500]}

        self.hyper_parameter_grid['min_impurity_decrease'].append(0)


        self.modelid = modelid
        self.n_threads = n_threads
        self.qsar_method = 'Random forest'
        self.description = 'sklearn implementation of random forest'
        self.description_url = 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'

    def getModel(self):
        """Used in build_model and GA embedding calculation to construct a default model with no grid search"""

        if self.is_binary:
            self.rfr = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42, n_jobs=self.n_threads,
                                              min_impurity_decrease=self.min_impurity_decrease,
                                              max_samples=self.max_samples)
        else:
            self.rfr = RandomForestRegressor(n_estimators=self.n_estimators, random_state=42, n_jobs=self.n_threads,
                                             min_impurity_decrease=self.min_impurity_decrease,
                                             max_samples=self.max_samples)

        return self.rfr

    def getModel2(self):
        return  self.rfr


    def build_model_no_grid_search(self):
        """Trains the RF model on provided data"""
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        self.getModel()

        # Train the model on training data
        self.rfr.fit(train_features, train_labels)

        print('Score for Training data = ', self.rfr.score(train_features, train_labels))

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        # pickle.dump(rfr, open("rfr.p", "wb"))
        return self

    def build_model_with_preselected_descriptors_no_grid_search(self, descriptor_names):

        t1 = time.time()
        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", descriptor_names)

        self.descriptor_names = train_column_names

        self.getModel()

        # Train the model on training data
        self.rfr.fit(train_features, train_labels)

        print('Score for Training data = ', self.rfr.score(train_features, train_labels))

        # Save space in database:
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        pass



    def build_model_with_preselected_descriptors(self, descriptor_names):

        """Trains the RF model on provided data"""
        t1 = time.time()
        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances_with_preselected_descriptors(self.df_training, "training", descriptor_names)
        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names
        # Instantiate Estimator
        if self.is_binary:
            self.rfr = RandomForestClassifier(random_state=42, oob_score=True)
        else:
            self.rfr = RandomForestRegressor(random_state=42, oob_score=True)
            # Basic hyperparameter optimization

        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_splitter.get_n_splits(train_features, train_labels)

        self.fix_hyperparameter_grid(descriptor_names, train_labels)

        # print(self.hyper_parameter_grid['max_features'])


        optimizer = GridSearchCV(self.rfr, self.hyper_parameter_grid, n_jobs=self.n_threads, return_train_score=True,
                                 cv=kfold_splitter, verbose=3)

        # optimizer = GridSearchCV(self.rfr, self.hyper_parameter_grid, n_jobs=self.n_threads,
        #                          return_train_score=True,verbose=3)

        optimizer.fit(train_features, train_labels)

        # Set hyperparameters
        self.rfr.set_params(**optimizer.best_params_)

        print('best params=',self.rfr.get_params())

        # Train the model on training data
        self.rfr.fit(train_features, y=train_labels)


        print('Score for Training data = ', self.rfr.score(train_features, train_labels))

        # Run final cross validation
        cv_score = cross_val_score(self.rfr, train_features, train_labels, cv=kfold_splitter)

        # This dictionary will contain enough internal validation to write a qmrf:
        self.internal_validation_scores = {'cross_val': cv_score, 'oob_score': self.rfr.oob_score_}

        # Save space in database:
        self.df_training = None
        # upload CV stats to the database
        # starting with cv_scores
        genericStatisticName = 'CVscore_fold'
        modelid = self.modelid
        i = 0

        # for score in cv_score:
        #     i= i + 1
        #     statisticName = genericStatisticName + str(i)
        #     session = SAU.connectSession()
        #     statistic = session.query(Statistic).filter_by(name=statisticName).first()
        #     if statistic is None:
        #         SAU.createStatistic(statisticName, False, 'give a description for this statistic')
        #         statistic = session.query(Statistic).filter_by(name=statisticName).first()
        #     SAU.createModelStatistic(statisticName, score, modelid)

        # and then for the out of bag estimates
        # statisticName = 'OOB_score'
        # SAU.createModelStatistic(statisticName, self.rfr.oob_score_, modelid)

        # Finalize time to train
        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')
        print(r"Out-of-Bag estimates: {oob_score}".format(oob_score=round(self.rfr.oob_score_, 2)))
        print(r"CV Scores: {scores}".format(scores=str(cv_score)))


        return self

    def fix_hyperparameter_grid(self, descriptor_names, train_labels):

        for max_features in self.hyper_parameter_grid['max_features']:
            if max_features == 4 or max_features == 8:
                if max_features > len(descriptor_names):
                    self.hyper_parameter_grid['max_features'].remove(max_features)

        for n_estimators in self.hyper_parameter_grid['n_estimators']:
            if n_estimators / len(train_labels) < 0.01:
                self.hyper_parameter_grid['n_estimators'].remove(n_estimators)

        print(self.hyper_parameter_grid)

    def build_model(self):

        """Trains the RF model on provided data"""
        t1 = time.time()
        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names
        # Instantiate Estimator
        if self.is_binary:
            self.rfr = RandomForestClassifier(random_state=42, oob_score=True)
        else:
            self.rfr = RandomForestRegressor(random_state=42, oob_score=True)
        # Basic hyperparameter optimization
        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        kfold_splitter.get_n_splits(train_features, train_labels)
        optimizer = GridSearchCV(self.rfr, self.hyper_parameter_grid, n_jobs=self.n_threads, return_train_score=True,
                                 cv=kfold_splitter, verbose=3)
        optimizer.fit(train_features, train_labels)


        # Set hyperparameters
        self.rfr.set_params(**optimizer.best_params_)
        # Train the model on training data
        self.rfr.fit(train_features, y=train_labels)
        # Run final cross validation
        cv_score = cross_val_score(self.rfr, train_features, train_labels, cv=kfold_splitter)
        # This dictionary will contain enough internal validation to write a qmrf
        self.internal_validation_scores = {'cross_val': cv_score, 'oob_score': self.rfr.oob_score_}
        # Save space in database:
        self.df_training = None
        # upload CV stats to the database
        # starting with cv_scores
        genericStatisticName = 'CVscore_fold'
        modelid = self.modelid
        i = 0

        print('Score for Training data = ', self.rfr.score(train_features, train_labels))  # TMM

        # for score in cv_score:
        #     i= i + 1
        #     statisticName = genericStatisticName + str(i)
        #     session = SAU.connectSession()
        #     statistic = session.query(Statistic).filter_by(name=statisticName).first()
        #     if statistic is None:
        #         SAU.createStatistic(statisticName, False, 'give a description for this statistic')
        #         statistic = session.query(Statistic).filter_by(name=statisticName).first()
        #     SAU.createModelStatistic(statisticName, score, modelid)

        # and then for the out of bag estimates
        statisticName = 'OOB_score'
        # SAU.createModelStatistic(statisticName, self.rfr.oob_score_, modelid)

        # Finalize time to train
        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')
        print(r"Out-of-Bag estimates: {oob_score}".format(oob_score=round(self.rfr.oob_score_, 2)))
        print(r"CV Scores: {scores}".format(scores=str(cv_score)))
        return self

    def do_predictions(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')
        if (self.is_binary == True):
            predictions = self.rfr.predict_proba(pred_features)[:, 1]
        else:
            predictions = self.rfr.predict(pred_features)

        print('Score for Test data = ', self.rfr.score(pred_features, pred_labels))

        # Return predictions
        return predictions

    def do_predictions2(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, self.descriptor_names)
        # print ('pred version 1.4')

        return self.rfr.score(pred_features, pred_labels)


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""
        self.is_binary = model.is_binary
        self.n_estimators = model.n_estimators
        self.min_impurity_decrease = model.min_impurity_decrease
        # self.params = model.params

        self.modelid = model.modelid
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description
        self.description_url = model.description_url

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)


def main():
    """
    Code to run from text files rather than webservice
    :return:
    """
    # endpoint = 'Octanol water partition coefficient'
    endpoint = 'Water solubility'
    # endpoint = 'Vapor pressure'
    # endpoint = 'Henry\'s law constant'

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'

    folder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 python/pf_python_modelbuilding/datasets/'
    training_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_training.tsv'
    prediction_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_prediction.tsv'
    prediction_file_name2 = 'Data from Standard ' + endpoint + ' from exp_prop external to ' + endpoint + ' OPERA_T.E.S.T. 5.1_full.tsv'

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name
    prediction_tsv_path2 = folder + prediction_file_name2

    # Parameters needed to build model:
    n_threads = 30
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')

    df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera

    model = Model(df_training, remove_log_p_descriptors, n_threads, 1)
    model.build_model()
    # model.build_model_with_grid_search()

    print(ModelDescription(model).to_json())
    # TODO store final hyperparameters in the model description...

    predictions = model.do_predictions(df_prediction)
    predictions = model.do_predictions(df_prediction2)


if __name__ == "__main__":
    main()
