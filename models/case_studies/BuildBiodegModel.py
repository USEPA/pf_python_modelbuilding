import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression

import model_ws_utilities
import model_ws_utilities as MWU
from models import df_utilities as DFU, ModelBuilder
import numpy as np
import math


def runBiodeg():
    
    # TODO to use run_model_building code

    # Fragments for Reg method
    # ['tvc', 'xvc3', 'eim', 'iddem', 'SdssNp', 'nX', 'ATS4v', 'nN', '-O- [sulfur attach]', '>C< [aliphatic attach]', 'SdssC_acnt', 'ATS2m']

    mainFolder="C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)\Comptox/0000 biodegradation OPPT/biodegradation/biowin update/datasets/"
    training_tsv_path = mainFolder + 'training-2d.txt'
    prediction_tsv_path = mainFolder + 'test-2d.txt'


    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    qsar_method = 'knn'
    # qsar_method = 'rf'
    # qsar_method = 'xgb'
    # qsar_method = 'reg'
    # qsar_method = 'las'

    # qsar_method_subset = 'xgb'
    # qsar_method_subset = 'rf'
    qsar_method_subset = qsar_method

    generateSubset = False

    if generateSubset:
        descriptor_names = generateModelSubset(training_tsv_path, prediction_tsv_path, qsar_method_subset)
    else:
        descriptor_names = None

        if qsar_method == 'reg':
            # For RBIODEG:
            descriptor_names = ['tvc', 'xvc3', 'eim', 'iddem', 'SdssNp', 'nX', 'ATS4v', 'nN', '-O- [sulfur attach]',
                                '>C< [aliphatic attach]','SdssC_acnt', 'ATS2m']

            # For FHM dataset
            # descriptor_names=['k2', 'MATS5m', 'GATS2p', 'XLOGP', 'nH', 'ATS1m', 'SsF', '-O- [aliphatic attach]', 'SsssCH_acnt',
            #  '-NH2 [nitrogen attach]', 'BEHm8']
            # R2 for Test data = 0.6810642403689915
            

            # descriptor_names,time=model_ws_utilities.remove_descriptors_rfe(qsar_method=qsar_method, df_training=df_training,
            #                                                                 n_threads=8, descriptor_names=descriptor_names)


        elif qsar_method == 'knn':
            descriptor_names = ['BEHm6', 'nS', 'MDEC14', '-NH2 [aliphatic attach]', 'nN', '-OH [aliphatic attach]',
                                'SdssNp', 'AMW', 'SsCl', 'Ms', 'xc3', 'BEHm1']
        print(descriptor_names)

    print("Executing",qsar_method)
    model = MWU.instantiateModel(df_training, n_jobs=8, qsar_method=qsar_method,remove_log_p=False, use_pmml_pipeline=False, include_standardization_in_pmml=False)
    model.build_model(use_pmml_pipeline=False, include_standardization_in_pmml=False, descriptor_names=descriptor_names)  # Note we now handle using an embedding by passing a descriptor_names list. By default it is a None type -- this will use all descriptors in df

    # printModelVariables(descriptor_names, model)
    predictions = model.do_predictions(df_prediction,return_score=False)
    # print(predictions)
    # model.getOriginalRegressionCoefficients()
    # redoRegression(descriptor_names, pred_df, training_df)
    plot_results(df_prediction, predictions)


def plot_results(df_prediction, predictions):
    df_preds = pd.DataFrame(predictions, columns=['Prediction'])
    df_pred = df_prediction[['ID', 'Property']]
    df_pred = pd.merge(df_pred, df_preds, how='left', left_index=True, right_index=True)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    df_pred.plot(kind='scatter', x='Property', y='Prediction', color='black',ax=ax)
    df_pred.plot(kind='scatter', x='Property', y='Property', color='red',ax=ax)
    plt.show()


def printModelVariables(descriptor_names, model):
    reg = model.model_obj.steps[1][1]
    coeffs = list(reg.coef_)
    intercept = reg.intercept_
    coeffs.append(intercept)
    descriptor_names.append('intercept')
    list_zip = list(zip(descriptor_names, coeffs))
    df = pd.DataFrame(list_zip, columns=['Name', 'Value'])
    print(df)


# def getOriginalRegressionCoefficients(model):
#
#     reg = model.model_obj.steps[1][1]
#     scale = model.model_obj.steps[0][1]
#
#     # Get the scaled coefficients and intercept
#     beta_scaled = reg.coef_
#     intercept_scaled = reg.intercept_
#
#     # Get the means and standard deviations used by the StandardScaler
#     means = scale.mean_
#     stds = scale.scale_
#
#     # Transform the coefficients to the unscaled version
#     beta_unscaled = beta_scaled / stds
#
#     # Transform the intercept to the unscaled version
#     intercept_unscaled = intercept_scaled - np.sum((means * beta_scaled) / stds)
#
#     # Report the unscaled coefficients and intercept
#     # print("Intercept (unscaled):", intercept_unscaled)
#     # print("Coefficients (unscaled):", beta_unscaled)
#
#     coefficients_dict = dict(zip(model['embedding'], beta_unscaled))
#
#     # Add the intercept to the dictionary
#     coefficients_dict['Intercept'] = intercept_unscaled
#
#     return coefficients_dict


def redoRegression(descriptor_names, pred_df, training_df):
    # filepathcsv=mainFolder+qsar_method+' training.csv'
    train_ids, train_labels, train_features, train_column_names, is_binary = \
        DFU.prepare_instances_with_preselected_descriptors(training_df, "training", descriptor_names)
    # training_df2 = train_features
    # training_df2['CAS'] = train_ids
    # training_df2['RBIODEG'] = train_labels
    # training_df2.to_csv(filepathcsv, encoding='utf-8', index=False)
    clf = LogisticRegression()
    clf.fit(train_features, train_labels)
    coeffs = list(clf.coef_[0])
    intercept = clf.intercept_[0]
    for coeff in coeffs:
        print(coeff)
    print(intercept)
    test_ids, test_labels, test_features, test_column_names, is_binary = \
        DFU.prepare_instances_with_preselected_descriptors(pred_df, "test", descriptor_names)
    # filepathcsv=mainFolder+qsar_method+' test.csv'
    # test_df2 = test_features
    # test_df2['CAS'] = test_ids
    # test_df2['RBIODEG'] = test_labels
    # test_df2.to_csv(filepathcsv, encoding='utf-8', index=False)
    # result=clf.predict_proba(test_features)[:,1]
    result = clf.predict(test_features)
    print(result)


def runBiodegNonLinearEpisuiteFrag():

    mainFolder="C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)\Comptox/0000 biodegradation OPPT/biodegradation/biowin update/datasets/"
    training_tsv_path = mainFolder + 'test-frag-epi-omit-bad.txt'
    prediction_tsv_path = mainFolder + 'test-frag-epi.txt'

    training_df = DFU.load_df_from_file(training_tsv_path, sep='\t')
    pred_df = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # use_pmml_pipeline = False
    # include_standardization_in_pmml = False
    #
    # qsar_method = 'reg'
    # print("Executing", qsar_method)
    # model = MWU.instantiateModel(training_df, n_jobs=8, qsar_method=qsar_method, remove_log_p=False,
    #                              use_pmml_pipeline=use_pmml_pipeline, include_standardization_in_pmml=include_standardization_in_pmml)
    # descriptor_names = None
    # model.build_model(use_pmml_pipeline=use_pmml_pipeline,
    #                   include_standardization_in_pmml=include_standardization_in_pmml,
    #                   descriptor_names=descriptor_names)  # Note we now handle using an embedding by passing a descriptor_names list. By default it is a None type -- this will use all descriptors in df
    # test_score = model.do_predictions(pred_df,return_score=True)
    #
    #
    # coef = model.model_obj.named_steps['estimator'].coef_
    # print(coef)
    # print(model.model_obj.named_steps['estimator'].intercept_)

    train_ids, train_labels, train_features, train_column_names, is_binary = \
    DFU.prepare_instances(training_df, "training", False, True)

    # print(train_column_names)
    test_ids, test_labels, test_features = DFU.prepare_prediction_instances(pred_df, train_column_names)

    # Add a column of 1's to get intercept:
    # train_features.loc[:]['intercept'] = 1.0
    # test_features.loc[:]['intercept'] = 1.0

    train_features['intercept'] = 1.0
    test_features['intercept'] = 1.0

    # Objective function for curve_fit:
    def func(X, *params):
        f = np.hstack(params).dot(X)
        f2 = []
        for pred in f:
            f2.append(math.exp(pred) / (1.0 + math.exp(pred)))
        return f2

    # def func_linear(X, *params):
    #     return np.hstack(params).dot(X)

    p = train_features.shape[1]
    # print(p)

    popt, pcov = curve_fit(func, train_features.T, train_labels, p0=np.random.randn(p)*0.001)

    # results = pd.DataFrame(np.column_stack([train_features.columns, popt]), columns=['variable', 'value'])
    # results_json = results.to_json(orient='records')
    # print(results_json)

    import json
    list_zip = list(zip(train_features.columns,popt))
    # print(json.dumps(list_zip,sort_keys=True, indent=4))
    # print(list_zip)

    # print(np.random.randn(p)*0.01)
    # print(popt)
    preds = func(test_features.T, popt)
    # print(preds)

    preds2 = []
    for pred in preds:
        if pred >= 0.5:
            preds2.append(1)
        else:
            preds2.append(0)

    # print(preds2)

    # print(preds_df)
    # print(test_labels)

    score = ModelBuilder.balanced_accuracy_score(test_labels, preds2)
    print('BA', score)

def generateModelSubset( training_tsv_path,prediction_tsv_path, qsar_method):


    if qsar_method == 'rf':
        fraction_of_max_importance = 0.25;
    elif qsar_method == 'xgb':
        fraction_of_max_importance = 0.03;


    training_tsv = DFU.read_file_to_string(training_tsv_path)
    prediction_tsv = DFU.read_file_to_string(prediction_tsv_path)
    min_descriptor_count = 20
    max_descriptor_count = 30

    if qsar_method == 'rf' or qsar_method =='xgb':

        descriptor_names,time_min = MWU.call_build_embedding_importance(qsar_method=qsar_method,
                                                                          training_tsv=training_tsv,
                                                                          prediction_tsv=prediction_tsv,
                                                                          remove_log_p_descriptors=False,
                                                                          n_threads=8, num_generations=1,
                                                                          use_permutative=True, run_rfe=True,
                                                                          fraction_of_max_importance=fraction_of_max_importance,
                                                                          min_descriptor_count=min_descriptor_count,
                                                                          max_descriptor_count=max_descriptor_count,
                                                                          use_wards=False)

    elif qsar_method == 'reg' or qsar_method == 'knn':
        descriptor_names, time_min = model_ws_utilities.call_build_embedding_ga(qsar_method=qsar_method,
                                                                      training_tsv=training_tsv,
                                                                      prediction_tsv=prediction_tsv,
                                                                      remove_log_p=False,
                                                                      num_generations=100, num_optimizers=10,
                                                                      num_jobs=2, n_threads=2,
                                                                      descriptor_coefficient=0.002, max_length=36,
                                                                      threshold=1,
                                                                      use_wards=False,run_rfe=True)


    # print(descriptor_names)
    return descriptor_names


if __name__ == "__main__":
    runBiodeg()
    # runBiodegNonLinearEpisuiteFrag()
