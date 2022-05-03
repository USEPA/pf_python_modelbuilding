from models import df_utilities as dfu
from models import rf_model as rf
from models import svm_model as svm
# from models import dnn_model as dnn
from models import xgb_model as xgb


import numpy as np
import pandas as pd

from flask import abort

models = {}


def get_model_info(qsar_method):
    """Returns a short, generic description for each QSAR method"""
    qsar_method = qsar_method.lower()

    if qsar_method == 'rf':
        return 'sklearn implementation of random forest ' \
                           '(https://scikit-learn.org/stable/modules/generated/' \
                           'sklearn.ensemble.RandomForestClassifier.html)'
    elif qsar_method == 'svm':
        return 'sklearn implementation of SVM using NuSVR for regression' \
                           ' or SVC for classification ' \
                           '(https://scikit-learn.org/stable/modules/svm.html),' \
                           ' no applicability domain'
    elif qsar_method == 'dnn':
        return 'tensorflow/keras implementation of DNN'
    elif qsar_method == 'xgb':
        return 'python implementation of extreme gradient boosting ' \
                           '(https://xgboost.readthedocs.io/en/latest/get_started.html)'
    else:
        return qsar_method + ' not implemented'


def call_build_model(qsar_method, training_tsv, remove_log_p):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    df_training = dfu.load_df(training_tsv)
    qsar_method = qsar_method.lower()
    
    model = None
    if qsar_method == 'svm':
        model = svm.Model(df_training, remove_log_p, 30)
    elif qsar_method == 'rf':
        model = rf.Model(df_training, remove_log_p, 30)
    elif qsar_method == 'xgb':
        model = xgb.Model(df_training, remove_log_p)
    # elif qsar_method == 'dnn':
    #     model = dnn.Model(df_training, remove_log_p)
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented')

    # Returns trained model
    model.build_model()
    return model


def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, descriptor_names_tsv):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    df_training = dfu.load_df(training_tsv)
    qsar_method = qsar_method.lower()

    model = None
    if qsar_method == 'rf':
        model = rf.Model(df_training, False, 30)
    elif qsar_method == 'xgb':
        model = xgb.Model(df_training, False)
    elif qsar_method == 'svm':
        model = svm.Model(df_training, False, 30)
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented with preselected descriptors')

    # Returns trained model
    model.build_model_with_preselected_descriptors(descriptor_names_tsv)
    return model


def call_do_predictions(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""
    df_prediction = dfu.load_df(prediction_tsv)
    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])

    predictions = model.do_predictions(df_prediction)

    if predictions is None:
        return None

    # Pulls together IDs, exp vals, and predictions into JSON format
    results = pd.DataFrame(np.column_stack([pred_ids, pred_labels, predictions]), columns=['ID', 'exp', 'pred'])
    results_json = results.to_json(orient='records')
    return results_json


def get_model_details(qsar_method, model):
    """Returns detailed description of models, with version and parameter info, for each QSAR method"""
    if qsar_method.lower() == 'rf':
        return rf.ModelDescription(model).to_json()
    elif qsar_method.lower() == 'svm':
        return svm.ModelDescription(model).to_json()
    # elif qsar_method.lower() == 'dnn':
    #     return dnn.ModelDescription(model).to_json()
    elif qsar_method.lower() == 'xgb':
        return xgb.ModelDescription(model).to_json()
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented')
