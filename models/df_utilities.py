import logging

import numpy as np
import pandas as pd
from io import StringIO

from sklearn import preprocessing
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import json

def load_df(tsv_string):
    """Loads data from TSV/CSV into a pandas dataframe"""
    if "\t" in tsv_string:
        separator = '\t'
        # print('separator=tab')
    else:
        separator = ','

    # cant have these in variable names - messes up xgboost fit call:
    # tsv_string = tsv_string.replace("[","").replace("]","").replace("<","").replace(">","")

    df = pd.read_csv(StringIO(tsv_string), sep=separator, na_values="null")

    # print('0 col:')
    # print(df.iloc[:,0])
    #
    # print('1 col:')
    # print(df.iloc[:, 1])
    # print('')


    # df = df.replace('null', np.nan).replace('{}', np.nan)


    # Remove special chars from column names or it causes issues with pmml:
    # df.columns = df.columns.str.replace('[', '')
    # df.columns = df.columns.str.replace(']', '')
    # df.columns = df.columns.str.replace('<', '')
    # df.columns = df.columns.str.replace('>', '') # TODO need to be be careful with old embeddings

    # print(df.shape)
    # Deletes rows with bad values CR 4/20/2022: descriptors with full nulls are more frequent in descriptor packages like Mordred than individual compounds with full nulls.
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    return df


def read_file_to_string(file_path):
    text_file = open(file_path, "r")
    # read whole file to a string
    return text_file.read()

def load_df_from_file(filepath, sep='\t'):
    """Loads data from delimited file into a pandas dataframe
    Automatically reads .csv and .tsv
    Otherwise specify delimiter (e.g. for tab-delimited .txt)"""
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath, delimiter=',')
    elif filepath.endswith(".tsv"):
        df = pd.read_csv(filepath, delimiter='\t')
    else:
        df = pd.read_csv(filepath, delimiter=sep)
    # print(df.shape)

    #############################################################################################################
    # Deletes rows with bad values
    # TMM: this is not what we want to do. We want to delete bad columns not rows.
    # We need to delete bad columns from prediction set as well- see remove_null_columns_in_both_sets() method
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    #############################################################################################################

    # Deletes columns with bad values
    # df = df.dropna(axis=1)
    # df = df.reset_index(drop=True)

    # print(df.shape)
    return df


def is_descriptor_nearly_constant(df, column_name):
    """Checks if descriptor is ~nearly~ constant (for a given threshold) over the provided data"""
    mean = df[column_name].mean()
    stdev = df[column_name].std()
    constant_threshold = 0.000001

    if stdev < mean * constant_threshold:
        return True
    else:
        return False


def remove_log_p_descriptors(df, which_set):
    """Removes descriptor columns that contain logp"""
    drop_column_names = []
    for column in df:
        if 'logp' in column.lower():
            drop_column_names.append(column)
            # print(column)
    # for dropColumnName in dropColumnNames:
    #     df = df.drop(dropColumnName, axis=1)

    df = df.drop(drop_column_names, axis=1)

    logging.debug(f"{which_set}: The shape of our features after removing logp descriptors is: {df.shape}")
    return df


def do_remove_fragment_descriptors(df):
    
    start_column = "As [+5 valence, one double bond]"
    stop_column = "-N=S=O"
    i = df.columns.get_loc(start_column)
    j = df.columns.get_loc(stop_column)
    lo, hi = sorted([i, j])
    to_drop = df.columns[lo:hi+1] 
    return df.drop(columns=to_drop)

def do_remove_acnt_descriptors(df):
    cols_to_drop = [c for c in df.columns if '_acnt' in c.lower()]  # case-insensitive    
    # print(cols_to_drop)
    return df.drop(columns=cols_to_drop)  #    return df.drop(columns=to_drop)

def do_remove_non_double_descriptors(df):
    df_non_num = df.select_dtypes(exclude=[np.number])
    df = df.drop(df_non_num, axis=1)
    return df


# def prepare_prediction_instances(df, train_column_names):
#     """Prepares a pandas df of prediction data using descriptor set from training data"""
#     ids = np.array(df[df.columns[0]])
#     labels = np.array(df[df.columns[1]])
#
#     # Remove ids and exp vals
#     df.drop(df.columns[0], axis=1)
#     df.drop(df.columns[1], axis=1)
#
#     drop_column_names = []
#
#     for column in df:
#         if column not in train_column_names:
#             drop_column_names.append(column)
#
#     # Drop the same columns that were dropped from training data
#     df = df.drop(drop_column_names, axis=1)
#
#     print('The shape of features is:', df.shape)
#
#     features = np.array(df)
#
#     # Return ids, exp vals, and descriptors as separate arrays
#     return ids, labels, features

def prepare_prediction_instances(df, train_column_names):
    """Prepares a pandas df of prediction data using descriptor set from training data
    Uses a one liner to drop all the columns: df = df[train_column_names]
    """
    ids = df[df.columns[0]]
    # labels = np.array(df[df.columns[1]])
    labels = df[df.columns[1]]

    df = df[train_column_names]

    # print(train_column_names)

    # df.to_excel("predset.xlsx")

    # print('The shape of prediction features is:', df.shape)

    # features = np.array(df)
    features = df  # scikit learn converts it to numpy array later anyways

    # print(features.shape)

    # Return ids, exp vals, and descriptors as separate arrays
    return ids, labels, features




def filter_columns_in_both_sets(df_training, df_prediction):

    # Deletes columns with null values:

    df_training.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df_training.to_csv("C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/Profile/Documents/bob.csv")
    df_training = df_training.dropna(axis=1)

    # print(df_training.columns[df_training.isna().any()].tolist())

    # df_training = do_remove_non_double_descriptors(df_training)
    # df_training = df_training[~df_training.isin([np.nan, np.inf, -np.inf]).any(1)]

    # Deletes columns with null values:

    df_prediction.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prediction = df_prediction.dropna(axis=1)
    # print('shape1', df_prediction.shape)
    # df_prediction = df_prediction[~df_prediction.isin([np.nan, np.inf, -np.inf]).any(1)]
    # print('shape2', df_prediction.shape)

    # df_prediction = do_remove_non_double_descriptors(df_prediction)

    # Need to only include the columns in common:
    column_names = df_training.columns.intersection(df_prediction.columns)
    df_training = df_training[column_names]
    return df_training

# def prepare_prediction_instances(df, train_column_names):
#     """Prepares a pandas df of prediction data using descriptor set from training data"""
#     ids = np.array(df[df.columns[0]])
#     labels = np.array(df[df.columns[1]])
#
#     df = df[train_column_names]
#     print('The shape of features is:', df.shape)
#
#     # features = np.array(df)
#     features = df
#
#     # Return ids, exp vals, and descriptors as separate arrays
#     return ids, labels, features


# def prepare_instances(df, which_set, remove_logp=False, remove_corr=True):
#     """Prepares a pandas df of training data by removing logp and correlated descriptors"""
#     df_labels = df[df.columns[1]]
#     if df_labels.isin([0, 1]).all():
#         is_binary = True
#     else:
#         is_binary = False
#
#     # Labels are the values we want to predict
#     # labels = np.array(df_labels)
#     labels = df_labels
#
#     ids = df[df.columns[0]]
#
#     col_name_id = df.columns[0]
#
#     # print('col_name_id',col_name_id)
#
#     col_name_property = df.columns[1]
#
#     # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
#     df = df.drop(col_name_property, axis=1)
#
#     # drop ID column:
#     df = df.drop(col_name_id, axis=1)
#
#     df = do_remove_non_double_descriptors(df)
#
#     # Remove constant descriptors:
#     df = do_remove_constant_descriptors(df)
#
#     if remove_logp:
#         df = remove_log_p_descriptors(df, which_set)
#
#     if remove_corr:
#         do_remove_correlated_descriptors(df, 0.95)
#
#     # print(which_set + ': The shape of features is:', df.shape)
#
#     # Convert to numpy array
#     # features = np.array(df)
#
#     features = df  # scikit learn converts it to numpy array later anyways
#
#     column_names = list(df.columns)
#
#     return ids, labels, features, column_names, is_binary


def prepare_instances(df, which_set, remove_logp=False, remove_corr=True, 
                      remove_constant=True,
                      remove_fragment_descriptors=False,remove_acnt_descriptors=False):
    
    
    # #df = df.copy() # make sure we dont change it
    
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""
    df_labels = df[df.columns[1]]
    if df_labels.isin([0, 1]).all():
        is_binary = True
    else:
        is_binary = False

    # Labels are the values we want to predict
    # labels = np.array(df_labels)
    labels = df_labels

    ids = df[df.columns[0]]

    col_name_id = df.columns[0]

    # print('col_name_id',col_name_id)

    col_name_property = df.columns[1]

    # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
    df = df.drop(col_name_property, axis=1)

    # drop ID column:
    df = df.drop(col_name_id, axis=1)

    if remove_fragment_descriptors:
        df = do_remove_fragment_descriptors(df)

    if remove_acnt_descriptors:
        df = do_remove_acnt_descriptors(df)

    df = do_remove_non_double_descriptors(df)

    # Remove constant descriptors:
    if remove_constant:
        df = do_remove_constant_descriptors(df)

    if remove_logp:
        df = remove_log_p_descriptors(df, which_set)

    if remove_corr:
        do_remove_correlated_descriptors(df, 0.95)
        

    # print(which_set + ': The shape of features is:', df.shape)

    # Convert to numpy array
    # features = np.array(df)

    features = df  # scikit learn converts it to numpy array later anyways

    column_names = list(df.columns)

    return ids, labels, features, column_names, is_binary



def prepare_instances_with_preselected_descriptors_gcm(df, which_set, min_count=3, inv_diag_threshold=1000):
    """Prepares a pandas df of training data by removing low-count and highly collinear descriptors.
       Ensures labels and ids drop rows corresponding to removed feature rows.
    """
    try:
        # Labels and IDs from the first two columns
        df_labels = df[df.columns[1]]
        is_binary = df_labels.isin([0, 1]).all()

        labels = df_labels.copy()
        ids = df[df.columns[0]].copy()


        start = "As [+5 valence, one double bond]"
        stop = "-N=S=O"

        # Keep only fragment columns in the specified range
        features = keep_columns_between(df, start, stop, True)
        
        # print(features)
        #
        # # Keep only fragment columns in the specified range (preselected via embedding)
        # features = df[embedding].copy()

        # Drop low nonzero columns and rows (first pass)
        features, cols_dropped_low, rows_dropped_low = drop_low_nonzero_cols_and_rows(features, min_nonzero=min_count)

        # Align labels and ids to remaining feature rows
        labels = labels.loc[features.index]
        ids = ids.loc[features.index]

        # Iteratively drop columns with large diag(inv(X^T X)) and any rows that had nonzero values in those columns
        all_high_cols = []
        all_high_rows = pd.Index([])

        while True:
            filtered, high_cols, high_rows, diag_df = drop_high_invdiag_cols_and_rows(
                features, threshold=inv_diag_threshold, numeric_only=True
            )

            if not high_cols:
                # No more problematic columns; done
                break

            # print(f"Dropping {len(high_cols)} features with inv_diag > {inv_diag_threshold}: {high_cols}")

            # Accumulate dropped info and update features for the next iteration
            all_high_cols.extend(high_cols)
            all_high_rows = all_high_rows.union(high_rows)
            features = filtered

            # Keep labels and ids aligned with the remaining feature rows after this iteration
            labels = labels.loc[features.index]
            ids = ids.loc[features.index]

        # Final feature list after all drops
        column_names = list(features.columns)

        # Optional: report what was dropped
        # if cols_dropped_low:
        #     print(f"Dropped low-nonzero columns (< {min_count} nonzeros): {cols_dropped_low}")
        # if len(rows_dropped_low) > 0:
        #     print(f"Dropped {len(rows_dropped_low)} rows due to low-nonzero column removal.")

        if all_high_cols:
            print(f"Dropped high-inv_diag columns (> {inv_diag_threshold}): {all_high_cols}")
        
        # if len(all_high_rows) > 0:
        #     print(f"Dropped {len(all_high_rows)} rows due to high-inv_diag column removal.")

        # If you have an ID column in the original df, you can show which IDs were dropped:
        if 'ID' in df.columns and len(all_high_rows) > 0:
            print(f"Dropped {len(all_high_rows)} row IDs (high-inv_diag removal):{df.loc[all_high_rows, 'ID'].tolist()}")

        return ids, labels, features, column_names, is_binary

    except Exception as ex:
        ex.with_traceback()
        return "error finding fragments with min_count"


def drop_high_invdiag_cols_and_rows(df, threshold=1000, numeric_only=True):
    """
    Drop columns where diag(inv(X^T X)) > threshold and remove rows that had nonzero values in those columns,
    similar to drop_low_nonzero_cols_and_rows row handling.

    Returns:
    - result: filtered DataFrame
    - cols_to_drop: list of dropped columns
    - rows_dropped_index: index of dropped rows
    - diag_df: DataFrame with per-feature inverse-diagonal values, sorted descending
    """
    # Choose numeric columns for the X^T X computation
    cols_to_check = df.select_dtypes(include="number").columns if numeric_only else df.columns
    if len(cols_to_check) == 0:
        # Nothing numeric to check
        return df.copy(), [], pd.Index([]), pd.DataFrame(columns=['feature', 'inv_diag'])

    X = df[cols_to_check].to_numpy(dtype=float)

    # Compute inverse (or pseudo-inverse if singular)
    try:
        invXTX = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        invXTX = np.linalg.pinv(X.T @ X)

    diag_vals = np.diag(invXTX)
    diag_df = (
        pd.DataFrame({'feature': cols_to_check, 'inv_diag': diag_vals})
        .sort_values('inv_diag', ascending=False)
        .reset_index(drop=True)
    )

    # Columns to drop based on threshold
    high_mask = diag_vals > threshold
    cols_to_drop = list(cols_to_check[high_mask])

    # Rows to drop: any nonzero value in the columns to drop (ignore NaN)
    if cols_to_drop:
        nz_mask = df[cols_to_drop].ne(0) & df[cols_to_drop].notna()
        rows_to_drop_mask = nz_mask.any(axis=1)
    else:
        rows_to_drop_mask = pd.Series(False, index=df.index)

    # First drop rows, then drop columns (same pattern as drop_low_nonzero_cols_and_rows)
    result = df.loc[~rows_to_drop_mask].drop(columns=cols_to_drop)
    rows_dropped_index = df.index[rows_to_drop_mask]

    return result, cols_to_drop, rows_dropped_index, diag_df


def drop_low_nonzero_cols_and_rows(df, min_nonzero=3, numeric_only=True):
    """
    Remove columns with < min_nonzero nonzero values and also remove rows
    that had nonzero values in those columns.

    Nonzero is defined as value != 0 and not NaN. NaN does not count as nonzero.
    """
    # Choose which columns to evaluate
    cols_to_check = df.select_dtypes(include="number").columns if numeric_only else df.columns

    # Boolean mask of nonzero values (ignoring NaN)
    nonzero_mask = df[cols_to_check].ne(0) & df[cols_to_check].notna()

    # Count nonzero values per column
    nonzero_counts = nonzero_mask.sum(axis=0)

    # Columns to drop: fewer than min_nonzero nonzero entries
    cols_to_drop = nonzero_counts.index[nonzero_counts < min_nonzero].tolist()

    # Rows to drop: any nonzero in those columns to drop
    if cols_to_drop:
        rows_to_drop_mask = nonzero_mask[cols_to_drop].any(axis=1)
    else:
        rows_to_drop_mask = pd.Series(False, index=df.index)

    # First drop rows, then drop columns
    result = df.loc[~rows_to_drop_mask].drop(columns=cols_to_drop)

    # Optionally return info about what was dropped
    return result, cols_to_drop, df.index[rows_to_drop_mask]


def keep_columns_between(df, start, stop, inclusive=True):
    if start not in df.columns or stop not in df.columns:
        raise KeyError("Start or stop column not found in DataFrame.")
    start_i = df.columns.get_loc(start)
    stop_i  = df.columns.get_loc(stop)
    lo, hi = sorted([start_i, stop_i])
    if inclusive:
        return df.iloc[:, lo:hi+1]
    else:
        return df.iloc[:, lo+1:hi]    

def prepare_instances_wards(df, which_set, remove_logp, threshold):
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""
    
    # df=df.copy()
    
    df_labels = df[df.columns[1]]
    if df_labels.isin([0, 1]).all():
        is_binary = True
    else:
        is_binary = False

    # Labels are the values we want to predict
    labels = df_labels
    ids = df[df.columns[0]]

    col_name_id = df.columns[0]
    col_name_property = df.columns[1]

    # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
    df = df.drop(col_name_property, axis=1)

    # drop ID column:
    df = df.drop(col_name_id, axis=1)

    if remove_logp:
        df = remove_log_p_descriptors(df, which_set)

    df = do_remove_non_double_descriptors(df)

    # Remove constant descriptors:
    df = do_remove_constant_descriptors(df)
    # df = df.loc[:, (df != 0).any(axis=0)] # deletes ones with all zeros but removing constant descriptors covers that

    feature_names = list(df.columns)

    scaler = preprocessing.StandardScaler().fit(df)

    ## Compute spearman's r and ensure symmetry of correlation matrix
    corr = spearmanr(scaler.transform(df)).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    ## Compute distance matrix and form hierarchical clusters
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, threshold, criterion="distance")
    clusters = cluster_ids
    ## Pull out one representative descriptor from each cluster
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    # named_features = [feature_names[i] for i in selected_features]

    ## Set attribute with features that are not colinear
    wardsFeatures = [feature_names[i] for i in selected_features]

    # print(list(df.columns))

    features = df[wardsFeatures]

    return ids, labels, features, wardsFeatures, is_binary


def prepare_instances2(df, embedding, remove_corr):
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""

    
    #df = df.copy()
    df_labels = df[df.columns[1]]

    # Labels are the values we want to predict
    labels = df_labels
    ids = df[df.columns[0]]

    # drop ID column:
    df = df[embedding]

    if remove_corr:
        do_remove_correlated_descriptors(df, 0.95)

    features = df  # scikit learn converts it to numpy array later anyways

    column_names = list(df.columns)

    return ids, labels, features, column_names


def isBinary(df):
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""
    df_labels = df[df.columns[1]]

    if df_labels.isin([0, 1]).all():
        return True
    else:
        return False


# def prepare_instances_with_preselected_descriptors(df, which_set, descriptor_names):
#     """Prepares a pandas df of training data by removing logp and correlated descriptors"""
#     df_labels = df[df.columns[1]]
#     if df_labels.isin([0, 1]).all():
#         is_binary = True
#     else:
#         is_binary = False
#
#     # Labels are the values we want to predict
#     labels = np.array(df_labels)
#     ids = np.array(df[df.columns[0]])
#     col_name_id = df.columns[0]
#     col_name_property = df.columns[1]
#
#     # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
#     df = df.drop(col_name_property, axis=1)
#
#     # drop ID column:
#     df = df.drop(col_name_id, axis=1)
#
#     df = do_remove_non_double_descriptors(df)
#
#     drop_column_names = []
#
#     for column in df:
#         if column not in descriptor_names:
#             drop_column_names.append(column)
#
#     # Drop the same columns that were dropped from training data
#     df = df.drop(drop_column_names, axis=1)
#
#     print(which_set + ': The shape of features is:', df.shape)
#
#     # Convert to numpy array
#     features = np.array(df)
#
#     column_names = list(df.columns)
#
#     # print('col names=',column_names)
#     # print('features', features)
#
#     return ids, labels, features, column_names, is_binary


def prepare_instances_with_preselected_descriptors(df, which_set, descriptor_names):
    """Prepares a pandas df of training data only including features in descriptor_names
    """
    
    #df = df.copy()
    
    df_labels = df[df.columns[1]]

    if df_labels.isin([0, 1]).all():
        is_binary = True
    else:
        is_binary = False

    # Labels are the values we want to predict
    labels = df_labels

    ids = df[df.columns[0]]
    
    # Use one liner to drop columns:
    df = df[descriptor_names]

    logging.debug(which_set + ': The shape of features is:', df.shape)

    # Convert to numpy array
    # features = np.array(df)
    features = df  # scikit learn converts it to numpy array later anyways
    # features.to_excel("train_set_embedding.xlsx")

    column_names = list(df.columns)
    # print('col names=',column_names)
    # print('features', features)
    return ids, labels, features, column_names, is_binary


def do_remove_constant_descriptors(df):
    """Removes constant descriptors"""

    drop_column_names = []
    num_rows = len(df.index)

    for column in df:
        if df[column].count() != num_rows:
            # also remove if there are null values (number of values doesnt match # of rows
            drop_column_names.append(column)
            # print(column, df[column].count())

        if df[column].std() == 0:
            # if is_descriptor_constant(df, column):
            drop_column_names.append(column)
    df = df.drop(drop_column_names, axis=1)

    # print('done')
    return df


def do_remove_correlated_descriptors(df, threshold):
    """Removes descriptors correlated above a certain threshold
    Adapted from https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
    """
    # corr = df.corr().abs()
    # upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # corr_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # df.drop(corr_to_drop, axis=1, inplace=True)
    
    # to avoid "A value is trying to be set on a copy of a slice from a DataFrame" error:
    # Ensure df is a full, explicit copy of the data, not a view/slice
    df = df.copy() 
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the columns without using inplace=True
    df = df.drop(corr_to_drop, axis=1) 
