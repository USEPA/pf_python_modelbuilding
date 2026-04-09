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
from sklearn.feature_selection import SequentialFeatureSelector

def generateEmbedding(model, df_training, df_prediction, fraction_of_max_importance,
                      min_descriptor_count, max_descriptor_count,
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


def perform_iterative_recursive_feature_elimination(model, df_training, n_threads, n_steps=1,cv=5):
    start_time = time.time()
    
    embedding_old = model.embedding
        
    # TODO use scoring function to penalize using more variables like in SFS?

    while True:  # need to get more aggressive (remove 2 at a time) since first RFE didnt remove enough
        perform_recursive_feature_elimination(model, df_training, n_threads, n_steps, cv)
        
        logging.debug(f"After RFE iteration, {len(model.embedding)}, descriptors: {model.embedding}")
        
        if len(model.embedding) == len(embedding_old):
            break
        embedding_old = model.embedding
        
    stop_time = time.time()
        
    calc_time = stop_time - start_time
    return model.embedding, calc_time


def perform_recursive_feature_elimination(model, df_training, n_threads, n_steps=1, cv=5):
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
        
    # if model.is_binary:
    #     scoring = 'balanced_accuracy'
    #     cv = StratifiedKFold(5)
    # else:
    #     scoring = 'neg_root_mean_squared_error'
    #     cv = KFold(5)

    if model.is_binary:
        scoring = 'balanced_accuracy'
    else:
        scoring = 'neg_root_mean_squared_error'

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



def perform_iterative_sequential_feature_selection_old(
    model,
    df_training,
    cv=5,
    n_min=2,
    n_max=30,
    step=2,
    direction='forward',
    DESCRIPTOR_COEFFICIENT = 0.0025
):
    """
    Iteratively runs Sequential Feature Selection, increasing the number of
    selected features from n_min by 'step' until n_max (if provided) or until
    the total number of features is reached. Automatically breaks the loop if
    k exceeds the available number of features.

    For each k:
      - Runs SFS with k features (if k < n_features)
      - Logs CV RMSE (mean ± std)
    If k reaches n_features, it evaluates the full set (no SFS) as a baseline,
    logs it, then stops.

    Notes:
    - This function is intended for regression (RMSE). Raises if model.is_binary is True.
    - Keeps the initial candidate set fixed throughout the sweep.
    - Updates model.embedding to the best-performing selection and returns it.

    Parameters
    ----------
    model : object
        Must expose:
          - model.embedding: list of candidate feature names
          - model.model_obj: a Pipeline with named steps 'standardizer' and 'estimator'
          - model.is_binary: boolean
    df_training : DataFrame
        Training data (passed to DFU.prepare_instances2).
    cv : int or CV splitter, default=5
        Cross-validation specification.
    n_min : int, default=1
        Starting number of features to select.
    n_max : int or None, default=None
        Optional upper limit on number of features to try. If None, will iterate
        until hitting the total feature count (which triggers a baseline eval).
    step : int, default=1
        Increment for k between iterations. Must be >= 1.
    direction : {'forward', 'backward'}, default='forward'
        SFS direction.

    Returns
    -------
    best_embedding : list[str]
        Feature names at the k that produced the lowest mean CV RMSE.
    """
    logger = logging.getLogger(__name__)

    if getattr(model, "is_binary", False):
        raise ValueError("perform_iterative_sequential_feature_selection is intended for regression (RMSE).")

    # Preserve the original candidate feature set
    candidate_features = list(model.embedding)

    # Prepare data once
    _, y, X, _ = DFU.prepare_instances2(df_training, candidate_features, True)
    n_features_total = X.shape[1]

    # Build the same pipeline as the single-run SFS
    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("estimator", model.model_obj.named_steps['estimator'])
    ])

    # Edge case: not enough features for SFS to run
    if n_features_total <= 1:
        scoring = 'neg_mean_squared_error'
        base_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        rmse_folds = np.sqrt(np.maximum(0.0, -base_scores))
        base_rmse = float(np.mean(rmse_folds))
        model.embedding = candidate_features
        model._last_sfs_cv_score = -(base_rmse ** 2)  # store as negative MSE (higher is better)
        logger.info(
            f"SFS iterative (degenerate): n_features={n_features_total}, "
            f"CV RMSE={base_rmse:.2f}, embedding={candidate_features}"
        )
        return candidate_features

    # Defaults and bounds
    n_min = max(1, int(n_min))
    step = int(step)
    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}.")
    if n_max is not None:
        n_max = int(n_max)
        if n_min > n_max:
            raise ValueError(f"n_min ({n_min}) cannot be greater than n_max ({n_max}).")

    # Use neg_mean_squared_error for selection, report RMSE for interpretability
    scoring = 'neg_mean_squared_error'

    # A small tolerance based on initial MSE scale
    frac = 1e-4
    init_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    init_mse_mean = float(-np.mean(init_scores))
    tol = frac * max(init_mse_mean, 1e-12)

    import math

    logger.info(
        f"SFS iterative (pre): total_features={n_features_total}, "
        f"baseline CV RMSE={math.sqrt(init_mse_mean):.3f}, tol={tol:.3e}"
    )

    best_rmse = float("inf")
    best_score = float("inf") # want to minimize this
    best_embedding = None

    # Iterate k from n_min by step; break when exceeding limits
    k = n_min
    while True:
        # Respect user-provided n_max (if any)
        if n_max is not None and k > n_max:
            break

        # If k reaches or exceeds the total number of features, evaluate baseline and stop
        if k >= n_features_total:
            # Only evaluate baseline if the sweep intends to reach all features
            if n_max is None or n_max >= n_features_total:
                base_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                rmse_folds = np.sqrt(np.maximum(0.0, -base_scores))
                base_rmse = float(np.mean(rmse_folds))
                base_rmse_std = float(np.std(rmse_folds))
                logger.info(
                    f"SFS iterative (baseline all features): n_features={n_features_total}, "
                    f"CV RMSE={base_rmse:.3f} ± {base_rmse_std:.3f}"
                )
                
                base_score = base_rmse + base_rmse_std + DESCRIPTOR_COEFFICIENT * n_features_total
                 
                if base_score < best_score:
                    best_score = base_score
                    best_embedding = candidate_features
            break

        # Run SFS for k < n_features_total
        sfs = SequentialFeatureSelector(
            estimator=pipe,
            n_features_to_select=int(k),
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X, y)

        # Extract selected feature names
        try:
            selected_names = sfs.get_feature_names_out().tolist()
        except AttributeError:
            mask = sfs.get_support()
            if hasattr(X, "columns"):
                selected_names = list(X.columns[mask])
            else:
                selected_names = [f"f{i}" for i, m in enumerate(mask) if m]

        # Evaluate CV RMSE on the selected subset
        if hasattr(X, "loc"):
            X_sel = X[selected_names]
        else:
            X_sel = X[:, sfs.get_support()]

        sel_scores = cross_val_score(pipe, X_sel, y, cv=cv, scoring=scoring)
        rmse_folds = np.sqrt(np.maximum(0.0, -sel_scores))
        rmse_mean = float(np.mean(rmse_folds))
        rmse_std = float(np.std(rmse_folds))
        
        score = rmse_mean + rmse_std + DESCRIPTOR_COEFFICIENT * len(selected_names) #penalize models which have high std dev over the folders and have many variables
        

        # logger.info(
        #     f"SFS iterative: n_features={k}, CV RMSE={rmse_mean:.3f} ± {rmse_std:.3f}, embedding={selected_names}"
        # )
        
        logger.info(
            f"SFS iterative: n_features={k}, score={score:.3f}, CV RMSE={rmse_mean:.3f} ± {rmse_std:.3f}, embedding={selected_names}"
        )

    
        # if rmse_mean < best_rmse:
        #     best_rmse = rmse_mean
        #     best_embedding = selected_names
        #

        if score < best_score:
            best_score = score
            best_embedding = selected_names


        # Next k
        k += step

    # Finalize and return
    if best_embedding is None:
        # Fallback to baseline if no selection was made/evaluated
        base_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        rmse_folds = np.sqrt(np.maximum(0.0, -base_scores))
        best_rmse = float(np.mean(rmse_folds))
        best_embedding = candidate_features

    model.embedding = best_embedding
    # Store as negative MSE (higher is better), consistent with existing convention
    model._last_sfs_cv_score = -(best_rmse ** 2)

    logger.info(
        f"SFS iterative (best): n_features={len(best_embedding)}, "
        f"CV RMSE={best_rmse:.2f}, embedding={best_embedding}"
    )

    return best_embedding


def perform_sequential_feature_selection(model, df_training, cv=5, n_features_to_select='auto'):
    """
    Runs SFS on the current candidate feature set in model.embedding, logs
    CV score before and after selection, logs the selected embedding, and stores
    the post-selection score (higher-is-better) in model._last_sfs_cv_score.

    For classification: balanced_accuracy (higher is better).
    For regression: neg_mean_squared_error (higher is better, i.e., lower MSE).
    """

    logger = logging.getLogger(__name__)
    
    print(f"perform_sequential_feature_selection, n_features_to_select: {n_features_to_select}")

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
        # n_features_to_select=n_features_to_select,
        n_features_to_select=n_features_to_select,
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
        logger.info(f"SFS (post): n_features={len(model.embedding)}, CV balanced_accuracy={sel_mean:.3f} ± {sel_std:.3f}")
        model._last_sfs_cv_score = sel_mean  # higher is better
    else:
        sel_neg_mse = float(np.mean(sel_scores))      # negative MSE (higher is better)
        sel_mse = float(-sel_neg_mse)                 # positive MSE for logging
        sel_mse_std = float(np.std(-sel_scores))
        logger.info(f"SFS (post): n_features={len(model.embedding)}, CV RMSE={sel_mse:.3f} ± {sel_mse_std:.3f}")
        model._last_sfs_cv_score = sel_neg_mse        # higher is better

    # Log the selected embedding (feature names)
    logger.info(f"SFS (post): embedding={model.embedding}")

    return model.embedding

def perform_iterative_sequential_feature_selection(
    model,
    df_training,
    cv=5,
    n_min=2,
    n_max=20,
    step=1,
    direction='forward',
    # descriptor_coefficient=0.0025
    descriptor_coefficient=0.002,
    alpha=0.7 # 0.2 yields descriptor_coefficient around 0.0025, alpha was 0.7 from AI
):
    """
    Iteratively runs Sequential Feature Selection, increasing the number of
    selected features from n_min by 'step' until n_max (if provided) or until
    the total number of features is reached. Automatically breaks the loop if
    k exceeds the available number of features.

    Unified objective across tasks (relative units):
      - error = RMSE (regression) or (1 - BAC) (binary)
      - objective = (error_mean / base_error) + (error_std / base_error) + C * n_features

    If descriptor_coefficient is None, it is auto-selected based on:
      - rel_noise = std(error_folds) / base_error
      - C = max(c_min, alpha * rel_noise * sqrt(log(1+p)/n)), where
        n = #samples, p = #candidate features, alpha=0.7, c_min=1e-3 by default.

    Parameters
    ----------
    model : object
        Must expose:
          - model.embedding: list of candidate feature names
          - model.model_obj: a Pipeline with named steps 'standardizer' and 'estimator'
          - model.is_binary: boolean
    df_training : DataFrame
        Training data.
    cv : int or CV splitter, default=5  TODO assign a PredefinedSplit instead of just number of folds so get consistent results
    n_min, n_max, step, direction : SFS controls
    descriptor_coefficient : float or None
        If None, a value is auto-selected as described above.

    Returns
    -------
    best_embedding : list[str]
        Feature names that minimize the penalized, relative objective.
    """
    import math
        
    logger = logging.getLogger(__name__)
    is_binary = bool(getattr(model, "is_binary", False))

    # Preserve the original candidate feature set
    candidate_features = list(model.embedding)

    # Prepare data once
    _, y, X, _ = DFU.prepare_instances2(df_training, candidate_features, True)
    n_samples = X.shape[0]
    n_features_total = X.shape[1]

    # Build pipeline
    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("estimator", model.model_obj.named_steps['estimator'])
    ])

    # Bounds
    n_min = max(1, int(n_min))
    step = int(step)
    if step < 1:
        raise ValueError(f"'step' must be >= 1, got {step}.")
    if n_max is not None:
        n_max = int(n_max)
        if n_min > n_max:
            raise ValueError(f"n_min ({n_min}) cannot be greater than n_max ({n_max}).")

    # Scoring and baseline
    scoring = "balanced_accuracy" if is_binary else "neg_mean_squared_error"
    frac = 1e-4
    eps = 1e-12

    init_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    if is_binary:
        # init_mean is BAC; base_error is (1 - BAC0)
        init_mean = float(np.mean(init_scores))
        base_error = max(1.0 - init_mean, eps)
        tol = frac * max(init_mean, eps)  # tol in scoring units (BAC)
        logger.info(
            "SFS iterative (pre, binary): total_features=%d, baseline CV BAC=%.3f, "
            "base_error (1-BAC0)=%.6f, tol=%.3e",
            n_features_total, init_mean, base_error, tol
        )
    else:
        # init_mean is MSE; base_error is RMSE0
        init_mean = float(-np.mean(init_scores))  # MSE
        base_error = max(math.sqrt(init_mean), eps)  # RMSE0
        tol = frac * max(init_mean, eps)  # tol in scoring units (MSE)
        logger.info(
            "SFS iterative (pre): total_features=%d, baseline CV RMSE=%.3f, "
            "base_error=%.6f, tol=%.3e",
            n_features_total, base_error, base_error, tol
        )

    # Convert CV scores to per-fold errors in absolute units
    def scores_to_error_folds(scores):
        scores = np.asarray(scores, dtype=float)
        if is_binary:
            return 1.0 - scores                       # (1 - BAC_i)
        else:
            return np.sqrt(np.maximum(0.0, -scores))  # RMSE_i

    # Auto-select descriptor_coefficient if None
    if descriptor_coefficient is None:
        error_folds0 = scores_to_error_folds(init_scores)
        rel_noise = float(np.std(error_folds0) / max(base_error, eps))

        # Heuristic parameters
        c_min = 1e-3
        dim_factor = math.sqrt(max(math.log(1 + max(n_features_total, 1)) / max(n_samples, 1), 1e-6))

        descriptor_coefficient = max(c_min, alpha * rel_noise * dim_factor)
        logger.info(
            "Auto-selected descriptor_coefficient=%.6f (alpha=%.2f, rel_noise=%.6f, dim_factor=%.4f, n=%d, p=%d)",
            descriptor_coefficient, alpha, rel_noise, dim_factor, n_samples, n_features_total
        )
    else:
        logger.info("Using provided descriptor_coefficient=%.6f", float(descriptor_coefficient))

    best_embedding = None
    best_penalized_score = float("inf")  # lower is better
    best_error_mean_abs = None
    best_error_std_abs = None

    # Iterate k from n_min by step; break when exceeding limits
    k = n_min
    while True:
        # Respect user-provided n_max (if any)
        if n_max is not None and k > n_max:
            break

        # If k reaches or exceeds the total number of features, evaluate baseline and stop
        if k >= n_features_total:
            if n_max is None or n_max >= n_features_total:
                base_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                error_folds = scores_to_error_folds(base_scores)
                error_mean = float(np.mean(error_folds))
                error_std = float(np.std(error_folds))
                rel_mean = error_mean / max(base_error, eps)
                rel_std = error_std / max(base_error, eps)
                objective = rel_mean + rel_std + descriptor_coefficient * n_features_total
                
                if is_binary:
                    bac_mean = 1.0 - error_mean
                    logger.info(
                        "SFS iterative (baseline all, binary): n_features=%d, "
                        "objective_rel=%.4f, CV BAC=%.3f ± %.3f, rel_err=%.4f, rel_std=%.4f",
                        n_features_total, objective, bac_mean, error_std, rel_mean, rel_std
                    )
                else:
                    logger.info(
                        "SFS iterative (baseline all): n_features=%d, "
                        "objective_rel=%.4f, CV RMSE=%.3f ± %.3f, rel_mean=%.4f, rel_std=%.4f",
                        n_features_total, objective, error_mean, error_std, rel_mean, rel_std
                    )

                if objective < best_penalized_score:
                    best_penalized_score = objective
                    best_embedding = candidate_features
                    best_error_mean_abs = error_mean
                    best_error_std_abs = error_std
            break

        # Run SFS for k < n_features_total
        sfs = SequentialFeatureSelector(
            estimator=pipe,
            n_features_to_select=int(k),
            tol=tol,
            direction=direction,
            scoring=scoring,
            cv=cv, #need fixed cv or will get different cv each iteration
            n_jobs=-1
        )
        sfs.fit(X, y)

        # Extract selected feature names
        try:
            selected_names = sfs.get_feature_names_out().tolist()
        except AttributeError:
            mask = sfs.get_support()
            if hasattr(X, "columns"):
                selected_names = list(X.columns[mask])
            else:
                selected_names = [f"f{i}" for i, m in enumerate(mask) if m]
        else:
            mask = sfs.get_support()

        # Evaluate CV on the selected subset
        if hasattr(X, "loc"):
            X_sel = X[selected_names]
        else:
            X_sel = X[:, mask]

        cv_scores = cross_val_score(pipe, X_sel, y, cv=cv, scoring=scoring)
        error_folds = scores_to_error_folds(cv_scores)
        error_mean = float(np.mean(error_folds))
        error_std = float(np.std(error_folds))
        rel_mean = error_mean / max(base_error, eps)
        rel_std = error_std / max(base_error, eps)
        objective = rel_mean + rel_std + descriptor_coefficient * len(selected_names)


        # penalty = descriptor_coefficient * len(selected_names)
        # err_term = rel_mean + rel_std
        # logger.info("k=%d: err_term=%.4f, penalty=%.4f (C=%.4f), objective=%.4f", 
        #             k, err_term, penalty, descriptor_coefficient, err_term + penalty)

        if is_binary:
            bac_mean = 1.0 - error_mean
            logger.info(
                "SFS iterative (binary): n_features=%d, objective_rel=%.4f, "
                "CV BA=%.3f ± %.3f, rel_err=%.4f, rel_std=%.4f, embedding=%s",
                k, objective, bac_mean, error_std, rel_mean, rel_std, selected_names
            )
        else:
            logger.info(
                "SFS iterative: n_features=%d, objective_rel=%.4f, "
                "CV RMSE=%.3f ± %.3f, rel_mean=%.4f, rel_std=%.4f, embedding=%s",
                k, objective, error_mean, error_std, rel_mean, rel_std, selected_names
            )

        if objective < best_penalized_score:
            best_penalized_score = objective
            best_embedding = selected_names
            best_error_mean_abs = error_mean
            best_error_std_abs = error_std

        # Next k
        k += step

    from sklearn.base import clone
    
    def _compute_pooled_rmse(pipe, X, y, cv, is_binary):
        # Build/accept a splitter
        if hasattr(cv, "split"):
            splitter = cv
        else:
            splitter = StratifiedKFold(n_splits=int(cv), shuffle=False) if is_binary \
                       else KFold(n_splits=int(cv), shuffle=False)
    
        y_true_all = []
        y_pred_all = []
        for train_idx, test_idx in splitter.split(X, y):
            pipe_fold = clone(pipe)
            # Slice X/y whether DataFrame or ndarray
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
            X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            y_te = y.iloc[test_idx] if hasattr(y, "iloc") else y[test_idx]
    
            pipe_fold.fit(X_tr, y_tr)
            y_hat = pipe_fold.predict(X_te)
    
            y_true_all.append(np.asarray(y_te))
            y_pred_all.append(np.asarray(y_hat))
    
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)
        return float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    
    
    # ---------------- existing code block with additions ----------------
    
    if best_embedding is None:
        base_scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
        error_folds = scores_to_error_folds(base_scores)
        best_embedding = candidate_features
        best_error_mean_abs = float(np.mean(error_folds))   # mean-of-folds RMSE
        best_error_std_abs = float(np.std(error_folds))     # std across folds
    
    # Finalize and compute pooled RMSE over the selected features
    model.embedding = best_embedding
    
    # Build X_best using the selected feature names (works for both DataFrame and ndarray X)
    if hasattr(X, "loc"):
        X_best = X[best_embedding]
    else:
        # Map feature names to column positions based on candidate_features order
        name_to_pos = {name: idx for idx, name in enumerate(candidate_features)}
        cols_idx = [name_to_pos[n] for n in best_embedding]
        X_best = X[:, cols_idx]
    
    if is_binary:
        bac_best = 1.0 - best_error_mean_abs
        model._last_sfs_cv_score = float(bac_best)  # keep existing convention for binary
        logger.info(
            "SFS iterative (best, binary): n_features=%d, CV BAC=%.3f ± %.3f, embedding=%s",
            len(best_embedding), bac_best, best_error_std_abs, best_embedding
        )
    else:
        # Mean-of-folds RMSE you already computed (best_error_mean_abs ± best_error_std_abs)
        # Also compute pooled RMSE by concatenating fold predictions:
        pooled_rmse = _compute_pooled_rmse(pipe, X_best, y, cv, is_binary=False)
        model._last_sfs_cv_score = -(float(best_error_mean_abs) ** 2)  # keep existing convention
        model._last_sfs_cv_rmse_pooled = float(pooled_rmse)            # new: pooled RMSE
    
        logger.info(
            "SFS iterative (best): n_features=%d, CV RMSE (mean-of-folds)=%.3f ± %.3f, pooled_RMSE=%.3f, embedding=%s",
            len(best_embedding), best_error_mean_abs, best_error_std_abs, pooled_rmse, best_embedding
        )
    
    return best_embedding
    