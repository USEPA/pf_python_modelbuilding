import numpy as np
import pandas as pd
from io import StringIO


def load_df(tsv_string):
    """Loads data from TSV/CSV into a pandas dataframe"""
    if "\t" in tsv_string:
        separator = '\t'
    else:
        separator = ','

    df = pd.read_csv(StringIO(tsv_string), sep=separator)

    # Deletes rows with bad values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df


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
    print(df.shape)

    # Deletes rows with bad values
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # Deletes columns with bad values
    # df = df.dropna(axis=1)
    # df = df.reset_index(drop=True)

    print(df.shape)
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

    print(which_set + ': The shape of our features after removing logp descriptors is:', df.shape)
    return df


def prepare_prediction_instances(df, train_column_names):
    """Prepares a pandas df of prediction data using descriptor set from training data"""
    ids = np.array(df[df.columns[0]])
    labels = np.array(df[df.columns[1]])

    # Remove ids and exp vals
    df.drop(df.columns[0], axis=1)
    df.drop(df.columns[1], axis=1)

    drop_column_names = []

    for column in df:
        if column not in train_column_names:
            drop_column_names.append(column)

    # Drop the same columns that were dropped from training data
    df = df.drop(drop_column_names, axis=1)

    print('The shape of features is:', df.shape)

    features = np.array(df)

    # Return ids, exp vals, and descriptors as separate arrays
    return ids, labels, features


def prepare_instances(df, which_set, remove_logp, remove_corr):
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""
    df_labels = df[df.columns[1]]
    if df_labels.isin([0, 1]).all():
        is_binary = True
    else:
        is_binary = False

    # Labels are the values we want to predict
    labels = np.array(df_labels)
    ids = np.array(df[df.columns[0]])
    col_name_id = df.columns[0]
    col_name_property = df.columns[1]

    # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
    df = df.drop(col_name_property, axis=1)

    # drop ID column:
    df = df.drop(col_name_id, axis=1)

    drop_column_names = []

    # Remove constant descriptors:
    df = do_remove_constant_descriptors(df, drop_column_names)

    if remove_logp:
        df = remove_log_p_descriptors(df, which_set)

    if remove_corr:
        do_remove_correlated_descriptors(df, 0.95)

    print(which_set + ': The shape of features is:', df.shape)

    # Convert to numpy array
    features = np.array(df)

    column_names = list(df.columns)

    return ids, labels, features, column_names, is_binary


def prepare_instances_with_preselected_descriptors(df, which_set, descriptor_names):
    """Prepares a pandas df of training data by removing logp and correlated descriptors"""
    df_labels = df[df.columns[1]]
    if df_labels.isin([0, 1]).all():
        is_binary = True
    else:
        is_binary = False

    # Labels are the values we want to predict
    labels = np.array(df_labels)
    ids = np.array(df[df.columns[0]])
    col_name_id = df.columns[0]
    col_name_property = df.columns[1]

    # drop Property column with experimental property we are trying to correlate (# axis 1 refers to the columns):
    df = df.drop(col_name_property, axis=1)

    # drop ID column:
    df = df.drop(col_name_id, axis=1)

    drop_column_names = []

    for column in df:
        if column not in descriptor_names:
            drop_column_names.append(column)

    # Drop the same columns that were dropped from training data
    df = df.drop(drop_column_names, axis=1)

    print(which_set + ': The shape of features is:', df.shape)

    # Convert to numpy array
    features = np.array(df)

    column_names = list(df.columns)

    return ids, labels, features, column_names, is_binary


def do_remove_constant_descriptors(df, drop_column_names):
    """Removes constant descriptors"""
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
    Adapted from https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/"""
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    corr_to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(corr_to_drop, axis=1, inplace=True)
