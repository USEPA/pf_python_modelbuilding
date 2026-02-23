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


import logging


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


def perform_iterative_sequential_feature_selection(
    model,
    df_training,
    max_iters=10,
    min_features=2,
    cv=5,
    improvement_threshold=0.01,  # require at least 1% relative improvement to keep going
    patience=1,                  # stop after this many consecutive iterations without enough improvement
    shuffle_cv=False,            # optional: explore different CV splits
    random_state=None,           # used if shuffle_cv=True
    logger=None
):
    """
    Iteratively runs perform_sequential_feature_selection while keeping the
    candidate universe fixed. Logs progress via logging.info, tracks the
    best-scoring embedding, and stops when relative improvement over the best
    is < improvement_threshold for 'patience' iterations.

    The score used is:
      - balanced_accuracy for classification (higher is better)
      - neg_mean_squared_error for regression (higher is better; i.e., lower MSE)

    Returns the best embedding found.
    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # Preserve original candidate universe (None => DFU uses all features)
    original_universe = getattr(model, "embedding", None)

    # Optionally enable shuffled CV folds if your perform_* uses cross_val_score defaults.
    # Note: perform_sequential_feature_selection must honor this if you want variability.
    if shuffle_cv and hasattr(model, "cv_splitter"):
        # If your pipeline can accept a splitter, set it here.
        # Otherwise, consider modifying perform_* to pass a KFold with shuffle=True.
        pass

    best_score = -np.inf
    best_embedding = None
    no_improve_count = 0
    eps = 1e-12

    for it in range(1, max_iters + 1):
        # Reset candidate universe before each SFS run
        model.embedding = (list(original_universe) if original_universe is not None else None)

        logger.info(
            f"Iterative SFS: iter={it} | starting candidate_universe="
            f"{'ALL' if original_universe is None else len(original_universe)}"
        )

        # Run SFS; this logs pre/post scores and sets model._last_sfs_cv_score
        perform_sequential_feature_selection(model, df_training, cv=cv)

        # Read post-selection score (higher is better) and the embedding
        current_score = getattr(model, "_last_sfs_cv_score", None)
        current_embedding = list(model.embedding or [])
        current_count = len(current_embedding)

        logger.info(f"Iterative SFS: iter={it} | selected {current_count} features")
        logger.info(f"Iterative SFS: iter={it} | embedding={current_embedding}")
        logger.info(f"Iterative SFS: iter={it} | post-score={current_score}")

        # Guard against collapsing to too few features
        if current_count < min_features:
            logger.info(
                f"Iterative SFS: iter={it} | selection dropped to {current_count} < min_features={min_features}; stopping."
            )
            # Restore to widest reasonable set (original universe if available)
            model.embedding = (list(original_universe) if original_universe is not None else current_embedding)
            break

        # Update best if improved
        if current_score is not None and current_score > best_score:
            rel_impr = (current_score - best_score) / max(abs(best_score), eps) if best_score != -np.inf else np.inf
            logger.info(f"Iterative SFS: iter={it} | relative improvement over best={rel_impr if np.isfinite(rel_impr) else float('inf'):.4%}")
            best_score = current_score
            best_embedding = current_embedding
            no_improve_count = 0
        else:
            # Not improved vs. best
            rel_impr = 0.0
            logger.info(f"Iterative SFS: iter={it} | relative improvement over best={rel_impr:.4%}")
            no_improve_count += 1

        # Early stop if improvement is below threshold for 'patience' iterations
        # Note: We treat any non-increase as < threshold; if you want strict thresholding,
        #       track the last improvement magnitude and compare to improvement_threshold.
        if no_improve_count >= patience:
            logger.info(
                f"Iterative SFS: iter={it} | no sufficient improvement (< {improvement_threshold:.2%}) "
                f"for {patience} iteration(s); stopping."
            )
            break

    else:
        logger.info("Iterative SFS: reached max_iters without sufficient improvement.")

    # Restore best embedding before returning
    if best_embedding is not None:
        model.embedding = best_embedding
        logger.info(f"Iterative SFS: best embedding selected (n={len(best_embedding)}), best_score={best_score}")

    return model.embedding

def perform_sequential_feature_selection(model, df_training, cv=5):
    """
    Runs SFS on the current candidate feature set in model.embedding, logs
    CV score before and after selection, logs the selected embedding, and stores
    the post-selection score (higher-is-better) in model._last_sfs_cv_score.

    For classification: balanced_accuracy (higher is better).
    For regression: neg_mean_squared_error (higher is better, i.e., lower MSE).
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    logger = logging.getLogger(__name__)

    # Prepare inputs based on current candidate set
    _, y, X, _ = DFU.prepare_instances2(df_training, model.embedding, True)

    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("estimator", model.model_obj.named_steps['estimator'])
    ])

    frac = 0.0001  # fraction of initial score for tol

    # Compute initial CV score
    if model.is_binary:
        scoring = 'balanced_accuracy'
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        initial_mean = float(np.mean(scores))
        initial_std = float(np.std(scores))
        tol = frac * max(initial_mean, 1e-12)
        logger.info(f"SFS (pre): n_features={X.shape[1]}, CV balanced_accuracy={initial_mean:.6f} ± {initial_std:.6f}")
    else:
        scoring = 'neg_mean_squared_error'
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        # Positive for logging; tol uses same scale factor
        initial_mse = float(-np.mean(scores))
        initial_mse_std = float(np.std(-scores))
        tol = frac * max(initial_mse, 1e-12)
        logger.info(f"SFS (pre): n_features={X.shape[1]}, CV MSE={initial_mse:.6f} ± {initial_mse_std:.6f}")

    # logger.info(f"SFS tol: {tol}")

    # Run SFS
    sfs = SequentialFeatureSelector(
        estimator=pipe,
        n_features_to_select='auto',
        tol=tol,
        direction='forward',
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    sfs.fit(X, y)

    # Update embedding with selected features
    model.embedding = sfs.get_feature_names_out().tolist()

    # Evaluate and log score on selected set
    _, y_sel, X_sel, _ = DFU.prepare_instances2(df_training, model.embedding, True)
    sel_scores = cross_val_score(pipe, X_sel, y_sel, cv=cv, scoring=scoring)

    if model.is_binary:
        sel_mean = float(np.mean(sel_scores))
        sel_std = float(np.std(sel_scores))
        logger.info(f"SFS (post): n_features={len(model.embedding)}, CV balanced_accuracy={sel_mean:.6f} ± {sel_std:.6f}")
        model._last_sfs_cv_score = sel_mean  # higher is better
    else:
        sel_neg_mse = float(np.mean(sel_scores))      # negative MSE (higher is better)
        sel_mse = float(-sel_neg_mse)                 # positive MSE for logging
        sel_mse_std = float(np.std(-sel_scores))
        logger.info(f"SFS (post): n_features={len(model.embedding)}, CV MSE={sel_mse:.6f} ± {sel_mse_std:.6f}")
        model._last_sfs_cv_score = sel_neg_mse        # higher is better

    # Log the selected embedding (feature names)
    logger.info(f"SFS (post): embedding={model.embedding}")

    return model.embedding



