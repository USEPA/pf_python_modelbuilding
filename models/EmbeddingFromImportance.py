import logging
import time
import numpy as np
import pandas


from models import df_utilities as DFU

from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from models import ModelBuilder
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def generateEmbedding(model, df_training, df_prediction, fraction_of_max_importance,min_descriptor_count, max_descriptor_count,
                      num_generations=5, n_threads=4,remove_log_p_descriptors=False,
                      use_permutative=False, use_wards=False):
    '''
    Generates embedding based on importance

    :param model: the model we are using to build the embedding
    :param df_training: the training data frame
    :param df_prediction: the prediction data frame
    :param fraction_of_max_importance: the fraction of the maximum descriptor importance we need to have for a descriptor to be kept from a run
    :param min_descriptor_count: the minimum number of descriptors to retain from a run- fraction_of_max_importance will be slowly reduced until we have this many descriptors retained
    :param num_generations: number of runs where we build a model and check the importances of the descriptors for them to be retained
    :param n_threads: number of threads for building the model
    :param remove_log_p_descriptors: whether or not to keep log_p descriptors when building a model
    :param use_permutative: whether or not to use permutative importance or regular importance
    :return: Note: final descriptors are stored in model.embedding
    '''

    # TODO Make an objective function that keeps adding important descriptors until CV is maximized? rather than just using ones with high enough importance from each run?


    # print(min_descriptor_count,min_descriptor_count_2)

    logging.debug(f"use_wards = {use_wards}")

    if use_wards:
        # Using ward's method removes too descriptors for PFAS only training sets:
        train_ids, train_labels, train_features, train_column_names, model.is_binary = \
                DFU.prepare_instances_wards(df_training, "training", remove_log_p_descriptors, 0.5) # uses wards method to remove extra descriptors
    else:
        train_ids, train_labels, train_features, train_column_names, model.is_binary = \
            DFU.prepare_instances(df_training, "training", remove_log_p_descriptors, True)  # removes descriptors which are correlated by 0.95

    # print('train_labels',train_labels)
    # print(train_features)


    if model.regressor_name == 'rf':
        model.hyperparameter_grid = {
            "estimator__max_features": ["sqrt"]}  # just use a single set of hyperparameters to speed up

    model.hyperparameters = model.get_single_parameters()
    model.model_obj.set_params(**model.hyperparameters)


    model.model_obj.fit(train_features, train_labels)
    model.embedding = train_column_names

    if df_prediction.shape[0] != 0:
        logging.debug('\nprediction set results for non embedded model as benchmark:')
        score = model.do_predictions(df_prediction, return_score=True)  # results for non embedded model as benchmark

    # Loop until number of descriptors stops changing
    while True:

        new_descriptors = []

        for run_num in range(num_generations):

            logging.debug(f"Run number = {run_num + 1}")

            model.model_obj.fit(train_features, train_labels)


            sorted_importances, sorted_names = get_important_descriptors(model, n_threads, train_column_names,
                                                                         train_features, train_labels, use_permutative)

            add_new_descriptors(fraction_of_max_importance, max_descriptor_count, min_descriptor_count,
                                new_descriptors, sorted_importances, sorted_names)

            # print(count,fraction_of_max_importance)
        logging.debug(f"len(new_descriptors)={len(new_descriptors)}")
        logging.debug(f"new_descriptors={new_descriptors}")

        if (len(new_descriptors) == len(train_column_names)):
            model.embedding = train_column_names # *** store final descriptors here ***
            logging.debug("Number of descriptors didnt change, stopping")
            break

        train_ids, train_labels, train_features, = \
            DFU.prepare_prediction_instances(df_training, new_descriptors)
        train_column_names = new_descriptors

    # plt.barh(sorted_names,sorted_importances)
    # plt.xlabel("Random Forest Feature Importance")
    # plt.show()


def get_important_descriptors(model, n_threads, train_column_names, train_features, train_labels, use_permutative):

    if use_permutative:
        # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
        start_time = time.time()

        # Can use test set to make it run faster (as was mentioned in link above)-
        # test_ids, test_labels, test_features, = \
        #     DFU.prepare_prediction_instances(df_prediction, train_column_names)
        #
        # feature_importances = \
        # permutation_importance(estimator=model.model_obj, X=test_features, y=test_labels, n_jobs=n_threads,
        #                        n_repeats=5)['importances_mean']

        # Using training set since runs only slightly slower but probably works better in terms of finding best descriptors
        feature_importances = \
            permutation_importance(estimator=model.model_obj, X=train_features, y=train_labels,
                                   n_jobs=n_threads,
                                   n_repeats=5)['importances_mean']

        # print(feature_importances)
        sorted_idx = np.argsort(feature_importances)[::-1][::]
        # print(sorted_idx)
        sorted_importances = np.array(feature_importances)[sorted_idx]
        sorted_names = np.array(train_column_names)[sorted_idx]
        elapsed_time = time.time() - start_time
        # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    else:
        # Using built in feature importances:
        feature_importances = model.model_obj.steps[1][1].feature_importances_

        sorted_idx = feature_importances.argsort()[::-1][::]
        sorted_importances = np.array(feature_importances)[sorted_idx]
        sorted_names = np.array(train_column_names)[sorted_idx]
        # print('sorted importances',sorted_importances)
        # print(sorted_names)
    return sorted_importances, sorted_names


def add_new_descriptors(fraction_of_max_importance, max_descriptor_count, min_descriptor_count,
                        new_descriptors, sorted_importances, sorted_names):

    max_importance = sorted_importances[0]

    count = 0
    for index, importance in enumerate(sorted_importances):
        if importance > fraction_of_max_importance * max_importance:
            count = count + 1

    logging.debug(f"count exceeding fraction:{count}")

    if count < min_descriptor_count:  # just take the first min_descriptor_count descriptors
        logging.debug(f"count < min, using min={min_descriptor_count}")

        for index, importance in enumerate(sorted_importances):
            descriptor = sorted_names[index]

            if descriptor not in new_descriptors:
                new_descriptors.append(descriptor)
            if index == min_descriptor_count - 1:
                break
    elif count > max_descriptor_count:  # just take the first max_descriptor_count descriptors
        logging.debug(f"count > max, using max={max_descriptor_count}")

        for index, importance in enumerate(sorted_importances):
            descriptor = sorted_names[index]

            if descriptor not in new_descriptors:
                new_descriptors.append(descriptor)
            if index == max_descriptor_count - 1:
                break
    else:
        logging.debug(f"min <= count <= max, using count={count}")

        for index, importance in enumerate(sorted_importances): # take the ones exceeding the fraction of the max importance
            if importance > fraction_of_max_importance * max_importance:
                # if importance > min_importance:
                descriptor = sorted_names[index]
                if descriptor not in new_descriptors:
                    new_descriptors.append(descriptor)


def perform_sequential_feature_selection(model,df_training):
    """
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
    # if tol is not None, then features are selected while the score change does not exceed tol.
    # otherwise, half of the features are selected!
    """
    
    from sklearn.feature_selection import SequentialFeatureSelector
    
    _, y, X, _ = DFU.prepare_instances2(df_training, model.embedding,True)


    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("estimator", model.model_obj.named_steps['estimator'])
    ])

    frac = 0.0001 # fraction of initial score to add a parameter

    if model.is_binary:
        scoring='balanced_accuracy'
        scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
        initial_ba = np.mean(scores) # TODO: is the sign correct?
        logging.debug(f"Initial BA before SFS: {initial_ba}")
        tol = frac * initial_ba #1% improvement in BA         
    else:
        scoring ='neg_mean_squared_error'
            # Calculate initial cross-validated MSE
        scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
        initial_mse = -np.mean(scores) # Convert neg_mse to mse
        logging.debug(f"Initial MSE before SFS: {initial_mse}")
        tol = frac * initial_mse #1% improvement in neg error

    logging.info(f"tol for SFS: {tol}")
        
    #model.model_obj is PMMLPipeline which may not work, convert to Pipeline:

    sfs = SequentialFeatureSelector(
        # model.model_obj.steps[1][1], # this is just the estimator w/o the scalar (needed for knn)
        estimator=pipe,
        n_features_to_select='auto', # or integer, e.g., 3
        tol = tol,
        # direction='backward',         # 'forward' or 'backward'
        direction='forward',         # 'forward' or 'backward'
        scoring=scoring,
        cv=5,
        n_jobs=-1
    )
    
    # 5. Fit SFS
    sfs.fit(X, y)
    
    model.embedding = sfs.get_feature_names_out().tolist()
    
    return model.embedding


def perform_iterative_recursive_feature_elimination(model, df_training, n_threads, n_steps=1):
    start_time = time.time()
    
    embedding_old = model.embedding
    
    while True:  # need to get more aggressive (remove 2 at a time) since first RFE didnt remove enough
        perform_recursive_feature_elimination(model, df_training, n_threads, n_steps)
        
        logging.debug(f"After RFE iteration, {len(model.embedding)}, descriptors: {model.embedding}")
        
        if len(model.embedding) == len(embedding_old):
            break
        embedding_old = model.embedding
        
    stop_time = time.time()
        
    calc_time = stop_time - start_time
    return model.embedding, calc_time
        

def perform_recursive_feature_elimination(model, df_training, n_threads, n_steps=1):
    '''
    Runs CV recursive_feature_elimination
    :param n_steps:
    :param model:
    :param df_training:
    :param n_threads:
    :return: Note: final descriptors are stored in model.embedding
    '''
    # print('Here1, Before RFE, ', len(model.embedding), "descriptors", model.embedding)

    pandas.options.mode.chained_assignment = None  # avoids weird message about using a slice of df- TODO fix so dont need?

    train_ids, train_labels, train_features, train_column_names = \
        DFU.prepare_instances2(df_training, model.embedding,True)


    if model.is_binary:
        scoring = 'balanced_accuracy'
        cv = StratifiedKFold(5)
    else:
        scoring = 'neg_root_mean_squared_error'
        cv = KFold(5)

    # Recursive feature elimination using 5 fold CV:
    # Check if estimator supports feature importance
    # estimator = model.model_obj.steps[1][1]  #this is the estimator with no scalar
    
#    Adding scaling

    #model.model_obj is PMMLPipeline which may not work, convert to Pipeline:
    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("estimator", model.model_obj.named_steps['estimator'])
    ])

    
    estimator_name = model.regressor_name.lower() if hasattr(model, 'regressor_name') else ''
    
    # Define a custom importance getter for models without feature_importances_
    # This will use absolute coefficient values for linear models, or uniform importance for KNN
    def get_feature_importance(est):
        if hasattr(est, 'feature_importances_'):
            return est.feature_importances_
        elif hasattr(est, 'coef_'):
            return np.abs(est.coef_).flatten()
        else:
            # For models without feature importance (KNN), return uniform importance
            # This makes RFECV use cross-validation score to select features
            return np.ones(est.n_features_in_)
    
    rfecv = RFECV(
        # estimator=estimator,
        estimator=pipe,
        step=n_steps,
        cv=cv,
        scoring=scoring,
        n_jobs=n_threads,
        importance_getter=get_feature_importance
    )
    
    rfecv.fit(train_features, train_labels)
    mask = rfecv.get_support(indices=True)

    # print(rfecv.grid_scores_)
    # print(rfecv.score)

    # print(mask)

    features = np.array(train_column_names)
    model.embedding = features[mask].tolist()  # Final descriptor list

