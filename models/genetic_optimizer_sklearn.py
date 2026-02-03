'''
Created on Feb 3, 2026

@author: TMARTI02
'''
import models.df_utilities as DFU
import logging
from sklearn_genetic import GAFeatureSelectionCV as ga
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
# from utils import print_first_row
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

def runExample():
    
    
    # Data
    data = load_diabetes()
    
    # print(type(data.data))
    
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y_series = pd.Series(data.target, name="target")
    
    # Model (KNN + scaling)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", KNeighborsRegressor(n_neighbors=7, weights="distance"))
    ])
    
    # CV and scoring for regression
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "neg_root_mean_squared_error"  # or "neg_mean_squared_error", "neg_mean_absolute_error"
    
    # GA selector
    selector = ga(
        estimator=pipe,
        cv=cv,
        scoring=scoring,
        max_features=min(20, X_df.shape[1]),
        population_size=40,
        crossover_probability=0.5,
        mutation_probability=0.2,
        generations=20,
        verbose=0,
        n_jobs=-1
    )
    
    X = pd.DataFrame(X_df)
    y = pd.Series(y_series, name="target")
    
    print(type(X), type(y))
    
    selector.fit(X, y)
    
    support = selector.support_
    selected_columns = X.columns[support]
    print("Selected features (regression):", list(selected_columns))

def runGA(df_training, model, params):
    """
    See https://sklearn-genetic-opt.readthedocs.io/en/stable/    
    """

    # print("remove_fragment_descriptors",remove_fragment_descriptors)

    if params.use_wards:
        # Using ward's method removes too descriptors for PFAS only training sets:
        _, train_labels, train_features, train_column_names, model.is_binary = \
            DFU.prepare_instances_wards(df_training, "training", params.remove_log_p_descriptors,
                                        0.5)  # uses wards method to remove extra descriptors
    else:
        _, train_labels, train_features, train_column_names, model.is_binary = \
            DFU.prepare_instances(df=df_training, which_set="training", remove_logp=params.remove_log_p_descriptors,
                                  remove_corr=True, remove_constant=True,
                                  remove_fragment_descriptors=params.remove_fragment_descriptors,
                                  remove_acnt_descriptors=params.remove_acnt_descriptors)  # removes descriptors which are correlated by 0.95
    # print(train_features.shape)
    # print(f"model.hyperparameters:{model.hyperparameters}")

    logging.debug(f"use_wards: {params.use_wards}")
    logging.info('after initial feature selection, # features: ' + str(len(train_column_names)))
    logging.info(f"NUM_GENERATIONS:{params.num_generations}")
    logging.info(f"NUM_OPTIMIZERS:{params.num_optimizers}")

    model.hyperparameters = model.get_single_parameters()
    model.model_obj.set_params(**model.hyperparameters)
    
    if model.is_binary:
        scoring = "balanced_accuracy"
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        scoring = "neg_mean_squared_error"
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Convert PMMLPipeline to Pipeline (otherwise it throws a lot of warnings):
    pipe = Pipeline([
        ("scaler", model.model_obj.named_steps['standardizer']),
        ("reg", model.model_obj.named_steps['estimator'])
    ])
    
    # Genetic algorithm-based feature selector
    selector = ga(
        estimator=pipe,
        cv=cv,  # TODO needs to use KFoldStratified if model_is_binary == True
        scoring=scoring,  # or 'neg_root_mean_squared_error', 'neg_mean_absolute_error'
        max_features=20,
        population_size=params.num_optimizers,  # population size
        crossover_probability=params.crossover_probability,
        mutation_probability=params.mutation_probability,
        generations=params.num_generations,
        elitism=params.elitism,
        tournament_size=3,
        verbose=0,
        n_jobs=-1
    )
    
    X = pd.DataFrame(train_features)
    y = pd.Series(train_labels, name="target")

    # print(type(X),type(y))
    # print(y)
    # print_first_row(X)
    logging.info(f"shape of starting training feature before GA optimization: {X.shape}")

    selector.fit(X, y)
    
    # Selected features
    support = selector.support_  # boolean mask of selected features
    selected_columns = train_features.columns[support]
    print("Selected features:", list(selected_columns))
    
    return selected_columns

if __name__ == '__main__':
    runExample()