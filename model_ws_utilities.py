from models import df_utilities as dfu
# from models import dnn_model as dnn
# from models_old import knn_model as knn, svm_model as svm, rf_model as rf, xgb_model as xgb
from models_old import knn_model as knn, rf_model as rf

from models import ModelBuilder as mb

from models import df_utilities as DFU
import time as time
import numpy as np
import pandas as pd
from models import GeneticOptimizer as go
from flask import abort
import requests


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
    elif qsar_method == 'knn':
        return 'sklearn implementation of KNN ' \
               '(https://scikit-learn.org/stable/modules/generated/' \
               'sklearn.neighbors.KNeighborsClassifier.html)'
    else:
        return qsar_method + ' not implemented'


def call_build_model(qsar_method, training_tsv, remove_log_p):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    df_training = dfu.load_df(training_tsv)

    qsar_method = qsar_method.lower()

    n_jobs = 30

    model = None
    if qsar_method == 'svm':
        model = mb.SVM(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'rf':
        model = mb.RF(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'xgb':
        model = mb.XGB(df_training, remove_log_p,n_jobs)
    elif qsar_method == 'knn':
        model = mb.KNN(df_training, remove_log_p,n_jobs)
    # elif qsar_method == 'dnn':
    #     model = dnn.Model(df_training, remove_log_p)
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented')

    # Returns trained model
    model.build_model()
    return model


def call_build_embedding_ga(qsar_method, training_tsv, prediction_tsv, remove_log_p, n_threads,
                            num_generations, num_optimizers, num_jobs, descriptor_coefficient, max_length, threshold,
                            model_id):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    df_training = DFU.load_df(training_tsv)
    df_prediction = DFU.load_df(prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction, remove_log_p)
    print('training shape=', df_training.shape)

    qsar_method = qsar_method.lower()

    ga_model = None

    # if qsar_method == 'rf':
    #     ga_model = rf.Model(df_training=df_training, remove_log_p_descriptors=remove_log_p, n_threads=n_threads,
    #                         modelid=model_id)
    # elif qsar_method == 'knn':
    #     ga_model = knn.Model(df_training=df_training,
    #                          remove_log_p_descriptors=remove_log_p,
    #                          modelid=model_id)  # TODO should we add threads to knn?
    # else:
    #     # 404 NOT FOUND if requested QSAR method has not been implemented
    #     abort(404, qsar_method + ' not implemented')


    if qsar_method == 'knn':
        ga_model = knn.Model(df_training=df_training,
                             remove_log_p_descriptors=remove_log_p,
                             modelid=model_id)  # TODO should we add threads to knn?
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented')



    ga_model.is_binary = DFU.isBinary(df_training)

    go.NUM_GENERATIONS = num_generations
    go.NUM_OPTIMIZERS = num_optimizers
    go.NUM_JOBS = num_jobs
    go.DESCRIPTOR_COEFFICIENT = descriptor_coefficient
    go.MAXIMUM_LENGTH = max_length
    go.THRESHOLD = threshold

    t1 = time.time()
    descriptor_names = go.runGA(df_training, ga_model.getModel())
    t2 = time.time()

    timeMin = (t2 - t1) / 60

    # embedding = json.dumps(descriptor_names)

    # print('embedding='+embedding)

    # Returns embedding
    return descriptor_names, timeMin




def api_call_build_embedding_ga(qsar_method, training_tsv, prediction_tsv, remove_log_p, n_threads, num_generations, num_optimizers,
                                num_jobs, descriptor_coefficient, max_length, threshold, urlHost):
    data = {'qsar_method': qsar_method,
            'training_tsv': training_tsv,
            'prediction_tsv': prediction_tsv,
            'remove_log_p': remove_log_p,
            'n_threads': n_threads,
            'num_generations': num_generations,
            'num_optimizers': num_optimizers,
            'descriptor_coefficient': descriptor_coefficient,
            'threshold': threshold,
            'max_length': max_length,
            'num_jobs': num_jobs}

    # print(data)

    url = urlHost + 'models/' + qsar_method + '/embedding'

    # print(url)
    # sending post request and saving response as response object
    r = requests.post(url=url, data=data, timeout=999999)
    # print(r.text)
    return r.text


# def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, descriptor_names_tsv,
#                                                   model_id):
#     """Loads TSV training data into a pandas DF and calls the appropriate training method"""
#
#     df_training = dfu.load_df(training_tsv)
#
#     qsar_method = qsar_method.lower()
#
#     model = None
#     if qsar_method == 'rf':
#         model = rf.Model(df_training, remove_log_p, 30, model_id)
#     elif qsar_method == 'knn':
#         model = knn.Model(df_training, remove_log_p, model_id)
#     elif qsar_method == 'xgb':
#         model = xgb.Model(df_training, remove_log_p, model_id)
#     elif qsar_method == 'svm':
#         model = svm.Model(df_training, remove_log_p, 30, model_id)
#     else:
#         # 404 NOT FOUND if requested QSAR method has not been implemented
#         abort(404, qsar_method + ' not implemented with preselected descriptors')
#
#     # Returns trained model
#     model.build_model_with_preselected_descriptors(descriptor_names_tsv)
#     return model

def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, descriptor_names_tsv):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""

    df_training = dfu.load_df(training_tsv)
    qsar_method = qsar_method.lower()
    n_jobs = 4

    if qsar_method == 'svm':
        model = mb.SVM(df_training=df_training, remove_log_p_descriptors=remove_log_p, n_jobs=n_jobs)
    elif qsar_method == 'knn':
        model = mb.KNN(df_training=df_training, remove_log_p_descriptors=remove_log_p, n_jobs=n_jobs)
    elif qsar_method == 'rf':
        model = mb.RF(df_training=df_training, remove_log_p_descriptors=remove_log_p, n_jobs=n_jobs)
    elif qsar_method == 'xgb':
        model = mb.XGB(df_training=df_training, remove_log_p_descriptors=remove_log_p, n_jobs=n_jobs)
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        print(qsar_method + ' not implemented with preselected descriptors')
        return

    # if use_grid_search == False:
    #     model.hyperparameters = None

    model.build_model(descriptor_names=descriptor_names_tsv)
    # Returns trained model:
    return model

def call_do_predictions(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""
    df_prediction = dfu.load_df(prediction_tsv)
    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])
    predictions = model.do_predictions(df_prediction)

    # print(predictions)
    # print(pred_labels)

    if predictions is None:
        return None

    # Pulls together IDs, exp vals, and predictions into JSON format
    results = pd.DataFrame(np.column_stack([pred_ids, pred_labels, predictions]), columns=['ID', 'exp', 'pred'])
    results_json = results.to_json(orient='records')
    return results_json


def get_model_details(qsar_method, model):
    """Returns detailed description of models, with version and parameter info, for each QSAR method"""
    description = model.getModelDescription()
    if description:
        return description
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, qsar_method + ' not implemented')
