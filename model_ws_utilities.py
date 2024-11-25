import pypmml

from models import df_utilities as dfu
from models import ModelBuilder as mb
import model_ws_utilities as mwu
from models import EmbeddingFromImportance as efi
from models import df_utilities as DFU

import time as time
import numpy as np
import pandas as pd
from models import GeneticOptimizer as go
from flask import abort
import requests

from sklearn2pmml.pipeline import PMMLPipeline as PMMLPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, use_pmml_pipeline,
                                                  include_standardization_in_pmml, descriptor_names_tsv=None,
                                                  n_jobs=8):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""

    df_training = dfu.load_df(training_tsv)
    qsar_method = qsar_method.lower()

    model = instantiateModel(df_training, n_jobs, qsar_method, remove_log_p, use_pmml_pipeline=use_pmml_pipeline, include_standardization_in_pmml=include_standardization_in_pmml)

    if not model:
        abort(404, qsar_method + ' not implemented')

    model.build_model(use_pmml_pipeline=use_pmml_pipeline, include_standardization_in_pmml=include_standardization_in_pmml,
                      descriptor_names=descriptor_names_tsv)
    # Returns trained model:
    return model

def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, prediction_tsv, remove_log_p, use_pmml_pipeline,
                                                  include_standardization_in_pmml, descriptor_names_tsv=None,
                                                  n_jobs=8):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""

    df_training = dfu.load_df(training_tsv)
    print('training shape=', df_training.shape)
    df_prediction = DFU.load_df(prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)
    print('training shape after removing bad descriptors in both sets=', df_training.shape)

    qsar_method = qsar_method.lower()

    model = instantiateModel(df_training, n_jobs, qsar_method, remove_log_p, use_pmml_pipeline=use_pmml_pipeline, include_standardization_in_pmml=include_standardization_in_pmml)

    if not model:
        abort(404, qsar_method + ' not implemented')

    model.build_model(use_pmml_pipeline=use_pmml_pipeline, include_standardization_in_pmml=include_standardization_in_pmml,
                      descriptor_names=descriptor_names_tsv)
    # Returns trained model:
    return model
def call_cross_validate(qsar_method, cv_training_tsv, cv_prediction_tsv, descriptor_names_tsv, use_pmml_pipeline,
                        remove_log_p=False, hyperparameters={}, n_jobs=8):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    # print(qsar_method, remove_log_p, params, n_jobs, descriptor_names_tsv)

    # print(cv_training_tsv)
    # print('\n')
    # print(cv_prediction_tsv)


    df_training = dfu.load_df(cv_training_tsv)
    print('training shape=', df_training.shape)
    df_prediction = DFU.load_df(cv_prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)


    qsar_method = qsar_method.lower()

    model = instantiateModel(df_training, n_jobs, qsar_method, remove_log_p, use_pmml_pipeline=use_pmml_pipeline)

    if not model:
        abort(404, qsar_method + ' not implemented')

    # Returns trained model
    model.build_cv_model(use_pmml_pipeline=use_pmml_pipeline, descriptor_names=descriptor_names_tsv,
                         params=hyperparameters)

    df_prediction = dfu.load_df(cv_prediction_tsv)
    predictions = model.do_predictions(df_prediction)

    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])

    if predictions is None:
        return None

    # Pulls together IDs, exp vals, and predictions into JSON format
    results = pd.DataFrame(np.column_stack([pred_ids, pred_labels, predictions]), columns=['id', 'exp', 'pred'])
    results_json = results.to_json(orient='records')
    return results_json


def instantiateModel(df_training, n_jobs, qsar_method, remove_log_p, use_pmml_pipeline=False,
                     include_standardization_in_pmml=True):
    print('Instantiating ' + qsar_method.upper() + ' model in model builder, num_jobs=' + str(
        n_jobs) + ', remove_log_p=' + str(remove_log_p))

    model = None

    if qsar_method == 'svm':
        model = mb.SVM(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'rf':
        model = mb.RF(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'xgb':
        model = mb.XGB(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'knn':
        model = mb.KNN(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'reg':
        model = mb.REG(df_training, remove_log_p, n_jobs)
    elif qsar_method == 'las':
        model = mb.LAS(df_training, remove_log_p, n_jobs)
    # elif qsar_method == 'dnn':
    #     model = dnn.Model(df_training, remove_log_p)
    else:
        pass
        # 404 NOT FOUND if requested QSAR method has not been implemented
    model.is_binary = DFU.isBinary(df_training)

    obj = mb.model_registry_model_obj(qsar_method, model.is_binary)

    if use_pmml_pipeline is False or include_standardization_in_pmml:
        model.model_obj = PMMLPipeline([('standardizer', StandardScaler()), ('estimator', obj)])
    else:
        model.model_obj = PMMLPipeline([('estimator', obj)])

    # print(model.get_model_description())
    return model



def instantiateModelForPrediction(qsar_method, is_binary, pmml_file_path, use_sklearn2pmml):
    print('instantiateModel2 ' + qsar_method.upper() + ' model in model builder')

    model = None

    if qsar_method == 'svm':
        model = mb.SVM()
    elif qsar_method == 'rf':
        model = mb.RF()
    elif qsar_method == 'xgb':
        model = mb.XGB()
    elif qsar_method == 'knn':
        model = mb.KNN()
    elif qsar_method == 'reg':
        model = mb.REG()
    # elif qsar_method == 'dnn':
    #     model = dnn.Model(df_training, remove_log_p)
    else:
        pass
        # 404 NOT FOUND if requested QSAR method has not been implemented
    model.is_binary = is_binary

    print('Setting model from pmml file')
    t1 = time.time()

    if use_sklearn2pmml:
        model.set_model_obj_pmml_for_prediction(qsar_method=qsar_method, pmml_file_path=pmml_file_path)
    else:
        model.model_obj = pypmml.Model.fromFile(pmml_file_path)
    t2 = time.time()

    print('Done in ', (t2 - t1), ' secs')

    # import pickle
    # with open('model.pickle', 'wb') as handle:
    #     pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # t3 = time.time()
    # with open('model.pickle', 'rb') as handle:
    #     model2 = pickle.load(handle)
    # t4 = time.time()
    # print('Loaded from pickle file in ', (t4 - t3), ' secs')

    print(model.get_model_description())
    return model


def call_build_embedding_ga(qsar_method, training_tsv, prediction_tsv, remove_log_p,
                            num_generations, num_optimizers, num_jobs, n_threads, descriptor_coefficient, max_length,
                            threshold,
                            use_wards,run_rfe):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""
    df_training = DFU.load_df(training_tsv)
    df_prediction = DFU.load_df(prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)
    print('training shape=', df_training.shape)

    qsar_method = qsar_method.lower()

    ga_model = mwu.instantiateModel(df_training=df_training, n_jobs=n_threads, qsar_method=qsar_method,
                                 remove_log_p=remove_log_p, use_pmml_pipeline=False)

    go.NUM_GENERATIONS = num_generations

    go.NUM_OPTIMIZERS = num_optimizers
    go.NUM_JOBS = num_jobs
    go.DESCRIPTOR_COEFFICIENT = descriptor_coefficient
    go.MAXIMUM_LENGTH = max_length
    go.THRESHOLD = threshold

    t1 = time.time()
    descriptor_names = go.runGA(df_training=df_training, model=ga_model, use_wards=use_wards,
                                remove_log_p_descriptors=remove_log_p)


    if run_rfe:
        descriptor_names, time2 = remove_descriptors_rfe(qsar_method=qsar_method,df_training=df_training,
                                   n_threads=n_threads,descriptor_names=descriptor_names)

    # embedding = json.dumps(descriptor_names)
    # print('embedding='+embedding)

    t2 = time.time()
    timeMin = (t2 - t1) / 60
    return descriptor_names, timeMin


def remove_descriptors_rfe(qsar_method, df_training, n_threads, descriptor_names):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""

    t1 = time.time()
    qsar_method = qsar_method.lower()
    model = mwu.instantiateModel(df_training=df_training, n_jobs=n_threads, qsar_method=qsar_method,
                                 remove_log_p=False, use_pmml_pipeline=False)

    # efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,n_steps=1)
    # print('After RFE, ', len(model.embedding), "descriptors", model.embedding)
    model.embedding = descriptor_names
    embedding_old = descriptor_names
    print('before rfe', embedding_old)

    while True:  # need to get more aggressive (remove 2 at a time) since first RFE didnt remove enough
        efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,
                                                  n_steps=1)
        print('After RFE iteration, ', len(model.embedding), "descriptors", model.embedding)
        if len(model.embedding) == len(embedding_old):
            break
        embedding_old = model.embedding

    descriptor_names = model.embedding

    # embedding = json.dumps(descriptor_names)
    # print('embedding='+embedding)

    t2 = time.time()
    timeMin = (t2 - t1) / 60
    return descriptor_names, timeMin


def call_build_embedding_importance(qsar_method, training_tsv, prediction_tsv, remove_log_p_descriptors, n_threads,
                                    num_generations, use_permutative, run_rfe, fraction_of_max_importance,
                                    min_descriptor_count, max_descriptor_count, use_wards):
    """Generates importance based embedding"""

    df_training = DFU.load_df(training_tsv)

    print('in call_build_embedding_importance, df_training.shape',df_training.shape)


    df_prediction = DFU.load_df(prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)

    print(df_training.shape)

    model = mwu.instantiateModel(df_training=df_training, n_jobs=n_threads, qsar_method=qsar_method,
                                 remove_log_p=remove_log_p_descriptors, use_pmml_pipeline=False)

    t1 = time.time()

    efi.generateEmbedding(model, df_training, df_prediction, remove_log_p_descriptors=remove_log_p_descriptors,
                          num_generations=num_generations, n_threads=n_threads, use_permutative=use_permutative,
                          fraction_of_max_importance=fraction_of_max_importance,
                          min_descriptor_count=min_descriptor_count, max_descriptor_count=max_descriptor_count,
                          use_wards=use_wards)

    if run_rfe:
        # efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,n_steps=1)
        # print('After RFE, ', len(model.embedding), "descriptors", model.embedding)

        embedding_old = model.embedding
        while True:  # need to get more aggressive (remove 2 at a time) since first RFE didnt remove enough
            efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,
                                                      n_steps=1)
            print('After RFE iteration, ', len(model.embedding), "descriptors", model.embedding)
            if len(model.embedding) == len(embedding_old):
                break
            embedding_old = model.embedding

    # Fit final model using final embedding:
    train_ids, train_labels, train_features, train_column_names = \
        DFU.prepare_instances2(df_training, model.embedding, False)
    model.model_obj.fit(train_features, train_labels)  # train the model
    # model.embedding = train_column_names # just in case the order changed but shouldnt matte

    # Run calcs on test set to see how well embedding did:

    if df_prediction.shape[0] != 0:
        print("Final results for embedded model:")
        score = model.do_predictions(df_prediction, return_score=True)

    t2 = time.time()

    timeMin = (t2 - t1) / 60

    descriptor_names = model.embedding

    # Returns embedding results:
    return descriptor_names, timeMin

def call_build_embedding_lasso(qsar_method, training_tsv, prediction_tsv, remove_log_p_descriptors, n_threads,run_rfe):
    """Generates importance based embedding"""

    print('enter call_build_embedding_lasso')
    df_training = DFU.load_df(training_tsv)
    print('in call_build_embedding_importance, df_training.shape',df_training.shape)

    df_prediction = DFU.load_df(prediction_tsv)
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)
    # print(df_training.shape)

    t1 = time.time()

    model = mwu.call_build_model_with_preselected_descriptors(qsar_method=qsar_method,
                                                              training_tsv=training_tsv, prediction_tsv=prediction_tsv,
                                                              remove_log_p=remove_log_p_descriptors,
                                                              use_pmml_pipeline=False,
                                                              include_standardization_in_pmml=True,
                                                              descriptor_names_tsv=None,
                                                              n_jobs=n_threads)

    clf = model.model_obj.steps[1][1]
    coef = clf.coef_
    desc = model.model_obj.feature_names_in_
    res = pd.DataFrame(np.column_stack([desc, coef]), columns=['desc', 'coef'])
    res2 = res.loc[(res['coef'] != 0.0)]

    # print(res)
    # print(res.shape)
    # print('')
    # print(res2)
    # print(res2.shape)
    # res2_list = ', '.join(res2['desc'].astype(str))
    # embedding=res2_list
    model.embedding = list(res2['desc'])

    print('before rfe, embedding size=',len(model.embedding))

    if run_rfe:
        # efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,n_steps=1)
        # print('After RFE, ', len(model.embedding), "descriptors", model.embedding)

        embedding_old = model.embedding
        while True:  # need to get more aggressive (remove 2 at a time) since first RFE didnt remove enough
            efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,
                                                      n_steps=1)
            print('After RFE iteration, ', len(model.embedding), "descriptors", model.embedding)
            if len(model.embedding) == len(embedding_old):
                break
            embedding_old = model.embedding


    # Fit final model using final embedding:
    train_ids, train_labels, train_features, train_column_names = \
        DFU.prepare_instances2(df_training, model.embedding, False)
    model.model_obj.fit(train_features, train_labels)  # train the model

    clf = model.model_obj.steps[1][1]
    coef = clf.coef_

    desc = model.model_obj.feature_names_in_
    res = pd.DataFrame(np.column_stack([desc, coef]), columns=['desc', 'coef'])
    res2 = res.loc[(res['coef'] != 0.0)]

    # res2_list = ', '.join(res2['desc'].astype(str))
    # embedding=res2_list
    model.embedding = list(res2['desc'])

    # Run calcs on test set to see how well embedding did:
    # if df_prediction.shape[0] != 0:
    #     print("Final results for embedded model:")
    #     score = model.do_predictions(df_prediction, return_score=True)

    t2 = time.time()
    timeMin = (t2 - t1) / 60
    descriptor_names = model.embedding
    # Returns embedding results:
    return descriptor_names, timeMin

def api_call_init(qsar_method, model_string, details, url_host, model_id):
    url = url_host + 'models/' + qsar_method + '/init'

    # print(model_string)
    # print(len(bytes(model_string, 'utf-8')))

    data = details
    data['model_id'] = model_id

    print(data)

    data['model'] = model_string  # store pmml string model in the payload

    # payload = {'request': json.dumps(data)}
    # print(data)
    # files = {'model': ('model.pmml', bytes(model_string, 'utf-8'))}
    # r = requests.post(url=url, data=data, files=files, timeout=999999)

    r = requests.post(url=url, json=data, timeout=999999)

    return r.text


def api_call_info(qsar_method, url_host):
    url = url_host + 'models/' + qsar_method + '/info'
    # print(url)
    r = requests.get(url=url, timeout=999999)
    # print(r.text)
    return r.text


def api_call_details(url_host, model_id):
    url = url_host + 'models/' + str(model_id)
    # print(url)
    r = requests.get(url=url, timeout=999999)
    # print(r.text)
    return r.text


def api_call_predict(url_host, prediction_tsv, model_id):
    url = url_host + 'models/predict'
    # print(url)

    data = {'model_id': model_id,
            'prediction_tsv': prediction_tsv}

    r = requests.post(url=url, data=data, timeout=999999)
    # print(r.text)
    return r.text


def api_call_build_embedding_ga(qsar_method, training_tsv, prediction_tsv, remove_log_p, n_threads, num_generations,
                                num_optimizers,
                                num_jobs, descriptor_coefficient, max_length, threshold, use_wards, urlHost):
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
            'num_jobs': num_jobs,
            'use_wards': use_wards}

    # print(data)

    url = urlHost + 'models/' + qsar_method + '/embedding'

    # print(url)
    # sending post request and saving response as response object
    r = requests.post(url=url, data=data, timeout=999999)
    # print(r.text)
    return r.text


def api_call_build_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, num_jobs, embedding_tsv,
                                                model_id, url_host):
    data = {'training_tsv': training_tsv,
            'remove_log_p': remove_log_p,
            'num_jobs': num_jobs,
            'embedding_tsv': embedding_tsv,
            'model_id': model_id}

    # print(data)
    url = url_host + 'models/' + qsar_method + '/train'

    # print(url)
    # sending
    # post request and saving response as response object
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


def call_do_predictions(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""

    # print('prediction_tsv',prediction_tsv)

    df_prediction = dfu.load_df(prediction_tsv)

    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])
    predictions = model.do_predictions(df_prediction)

    # print(predictions)
    # print(pred_labels)

    if predictions is None:
        return None

    # Pulls together IDs, exp vals, and predictions into JSON format

    # print('pred_ids',pred_ids)

    results = pd.DataFrame(np.column_stack([pred_ids, pred_labels, predictions]), columns=['id', 'exp', 'pred'])
    results_json = results.to_json(orient='records')
    return results_json


def call_do_predictions_to_df(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""
    df_prediction = dfu.load_df(prediction_tsv)
    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])
    predictions = model.do_predictions(df_prediction)
    if predictions is None:
        return None
    results = pd.DataFrame(np.column_stack([pred_ids, pred_labels, predictions]), columns=['id', 'exp', 'pred'])
    return results


def get_model_details(model):
    """Returns detailed description of models, with version and parameter info, for each QSAR method"""
    description = model.get_model_description()
    if description:
        return description
    else:
        # 404 NOT FOUND if requested QSAR method has not been implemented
        abort(404, 'details for model not available')
