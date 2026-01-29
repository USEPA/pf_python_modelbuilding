'''
Created on Dec 4, 2024

@author: TMARTI02
'''


from models.run_model_building import save_json_file
from models.run_model_building import model_parameters
from models.run_model_building import getEmbedding
from models.run_model_building import buildModel
from models.run_model_building import lookAtResults


import pandas as pd
import numpy as np
import pickle
import json

import model_ws_utilities as mwu
from models import df_utilities as dfu
import models.results_utilities as ru


def a_runCaseStudiesTTR_Binding_CV():
    """
    This code uses text files for each fold (doesnt merge splitting on the fly)
    :return:
    """
    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/TTR_Binding_challenge/'

    descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    # descriptor_set = 'mordred'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)
    # mp.useEmbeddings = False
    mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    sort_embeddings = False

    df_results_all = None

    embeddings = []

    for fold in range(1, 6):
        training_file_name = "TTR_Binding_" + descriptor_set + "_train_CV" + str(fold) + ".tsv";
        prediction_file_name = "TTR_Binding_" + descriptor_set + "_test_CV" + str(fold) + ".tsv";
        training_tsv_path = inputFolder + training_file_name
        prediction_tsv_path = inputFolder + prediction_file_name
        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        if mp.useEmbeddings:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)
        else:
            embedding = None

        model = buildModel(embedding, mp, training_tsv, prediction_tsv)
        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        exp = df_results['exp'].to_numpy()
        pred = df_results['pred'].to_numpy()
        rmse = ru.calc_rmse(pred, exp)
        r2 = ru.calc_pearson_r2(pred, exp)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = df_results_all.append(df_results, ignore_index=True)

    # print(df_results_all)
    fileOut = inputFolder + "results/" + "TTR_Binding_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp)

    resultsFolder = inputFolder + "results/"
    lookAtResults(resultsFolder)


def a_runCaseStudiesTTR_Binding_CV_merge_on_fly():
    useMean = True
    # useAQC = True
    useAQC = False

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    descriptor_set = 'mordred'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)

    mp.n_threads = 4

    mp.useEmbeddings = False
    # mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    sort_embeddings = False

    df_results_all = None

    embeddings = []
    rmses = []
    r2s = []

    if useAQC:
        split_file = inputFolder + 'TTR training 5 fold splitting file with AQC 0719_2024.csv'
    else:
        split_file = inputFolder + 'TTR training 5 fold splitting file.csv'

    descriptor_file = inputFolder + 'TTR ' + descriptor_set + ' descriptors.csv'

    df_splits = dfu.load_df_from_file(split_file)
    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    df_all = pd.merge(df_splits, df_descriptors, on='QsarSmiles')

    # print(df_splits.shape)
    # print(df_descriptors.shape)
    # print(df_all.shape)

    for fold in range(1, 6):
        df_train = df_all.loc[df_all['Fold'] != 'Fold' + str(fold)]
        df_train = df_train.drop('Fold', axis=1)

        df_pred = df_all.loc[df_all['Fold'] == 'Fold' + str(fold)]
        df_pred = df_pred.drop('Fold', axis=1)

        # df_pred.to_csv(inputFolder+"pred"+str(fold)+".csv")

        # df_pred.agg(['min', 'max']).to_csv(inputFolder+str(fold)+"predMinMax.csv")

        # print( df_train[df_train.columns[0]])
        # print(fold, df_train.shape, df_pred.shape)

        prediction_tsv = df_pred.to_csv(sep='\t', index=False)
        training_tsv = df_train.to_csv(sep='\t', index=False)

        # fout=inputFolder+'training'+str(fold)+'.csv'
        # df_train.to_csv(fout, sep=',')

        # print(prediction_tsv)

        if mp.useEmbeddings:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)
        else:
            embedding = None

        model = buildModel(embedding, mp, training_tsv, prediction_tsv)
        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        print(df_results)

        exp = df_results['exp'].to_numpy()
        pred = df_results['pred'].to_numpy()
        rmse = ru.calc_rmse(pred, exp)
        r2 = ru.calc_pearson_r2(pred, exp)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)

        r2s.append(r2)
        rmses.append(rmse)

    # print(df_results_all)

    if useAQC:
        resultsFolder = inputFolder + "resultsAQC/"
    else:
        resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    results = save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp, r2s=r2s,
                             rmses=rmses)

    lookAtResults(resultsFolder, useMean=useMean)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot(property_name='TTR_Binding', title=title, exp=exp, pred=pred)


def a_runCaseStudiesTTR_Binding_CV_merge_on_fly_with_leaderboard():
    useMean = True
    # useAQC = True
    useAQC = False

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    descriptor_set = 'mordred'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)

    mp.n_threads = 4

    mp.useEmbeddings = False
    # mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    sort_embeddings = False

    df_results_all = None

    embeddings = []
    rmses = []
    r2s = []

    if useAQC:
        print('todo for useAQC')
        return
        # split_file = inputFolder + 'TTR training 5 fold splitting file with AQC 0719_2024.csv'
    else:
        split_file = inputFolder + 'TTR training + leaderboard 5 fold splitting file.csv'

    descriptor_file = inputFolder + 'TTR ' + descriptor_set + ' descriptors.csv'

    df_splits = dfu.load_df_from_file(split_file)
    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    df_all = pd.merge(df_splits, df_descriptors, on='QsarSmiles')

    # print(df_splits.shape)
    # print(df_descriptors.shape)
    # print(df_all.shape)

    for fold in range(1, 6):
        df_train = df_all.loc[df_all['Fold'] != 'Fold' + str(fold)]
        df_train = df_train.drop('Fold', axis=1)

        df_pred = df_all.loc[df_all['Fold'] == 'Fold' + str(fold)]
        df_pred = df_pred.drop('Fold', axis=1)

        # df_pred.to_csv(inputFolder+"pred"+str(fold)+".csv")

        # df_pred.agg(['min', 'max']).to_csv(inputFolder+str(fold)+"predMinMax.csv")

        # print( df_train[df_train.columns[0]])
        # print(fold, df_train.shape, df_pred.shape)

        prediction_tsv = df_pred.to_csv(sep='\t', index=False)
        training_tsv = df_train.to_csv(sep='\t', index=False)

        # fout=inputFolder+'training'+str(fold)+'.csv'
        # df_train.to_csv(fout, sep=',')

        # print(prediction_tsv)

        if mp.useEmbeddings:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)
        else:
            embedding = None

        model = buildModel(embedding, mp, training_tsv, prediction_tsv)
        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        print(df_results)

        exp = df_results['exp'].to_numpy()
        pred = df_results['pred'].to_numpy()
        rmse = ru.calc_rmse(pred, exp)
        r2 = ru.calc_pearson_r2(pred, exp)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)

        r2s.append(r2)
        rmses.append(rmse)

    # print(df_results_all)

    if useAQC:
        resultsFolder = inputFolder + "resultsAQC/"
    else:
        resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_with_leaderboard_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    results = save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp, r2s=r2s,
                             rmses=rmses)

    lookAtResults(resultsFolder, useMean=useMean)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot(fileOut=fileOut, property_name='TTR_Binding', title=title, exp=exp, pred=pred)


def a_runCaseStudiesTTR_Binding_training_prediction():
    useMean = True
    # useAQC = True
    useAQC = False

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)

    mp.n_threads = 4

    mp.useEmbeddings = False
    # mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    if useAQC:
        split_file = inputFolder + 'TTR training 5 fold splitting file with AQC 0719_2024.csv'
    else:
        split_file = inputFolder + 'TTR training 5 fold splitting file.csv'

    descriptor_file = inputFolder + 'TTR ' + descriptor_set + ' descriptors.csv'
    predictions_file = inputFolder + 'TTR predictions.csv'

    df_splits = dfu.load_df_from_file(split_file)
    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    # print(df_predictions)

    df_train = pd.merge(df_splits, df_descriptors, on='QsarSmiles')
    df_train = df_train.drop('Fold', axis=1)
    df_train.rename(columns={"QsarSmiles": "id", "median_activity_%": "Property"}, inplace=True)

    df_predictions = dfu.load_df_from_file(predictions_file)
    df_pred = pd.merge(df_predictions, df_descriptors, on='QsarSmiles')
    df_pred = df_pred.drop('dataset', axis=1).drop('QsarSmiles', axis=1)
    df_pred.insert(1, "Property", -9999)
    df_pred.rename(columns={"DTXSID": "id"}, inplace=True)

    # print(df_pred)
    # df_train.to_csv(inputFolder+'training.csv', index=False)
    # df_pred.to_csv(inputFolder + 'prediction.csv', index=False)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)
    training_tsv = df_train.to_csv(sep='\t', index=False)

    # print(training_tsv)

    if mp.useEmbeddings:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
        print('embedding', embedding)
        print('timeMin', timeMin)
    else:
        embedding = None

    model = buildModel(embedding, mp, training_tsv, prediction_tsv)

    file = open(inputFolder + 'ttr_binding_full_model_' + mp.qsar_method + '_' + descriptor_set + '.p', 'wb')
    pickle.dump(model, file)
    file.close()

    # df_results_tr = mwu.call_do_predictions_to_df(training_tsv, model)
    # print(df_results_tr)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)
    df_results.rename(columns={"id": "DTXSID"}, inplace=True)
    df_results.drop('exp', axis=1, inplace=True)
    # print (df_results)

    df_results = pd.merge(df_results, df_predictions, on='DTXSID')

    if useAQC:
        resultsFolder = inputFolder + "resultsAQC/"
    else:
        resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + "_blind_predictions.csv"

    df_results.to_csv(fileOut, index=False)
    # print(df_results)

    # import pickle
    # pickle.dumps(model)


def a_runCaseStudiesTTR_Binding_training_prediction_with_leaderboard():
    # useAQC = True
    useAQC = False

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)
    
    

    mp.n_threads = 4

    mp.useEmbeddings = False
    # mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    # if useAQC:
    #     split_file = inputFolder + 'TTR training 5 fold splitting file with AQC 0719_2024.csv'
    # else:
    #     split_file = inputFolder + 'TTR training 5 fold splitting file.csv'

    descriptor_file = inputFolder + 'TTR ' + descriptor_set + ' descriptors.csv'

    training_file = inputFolder + 'TTR training with leaderboard.csv'
    # predictions_file = inputFolder + 'TTR blind.csv'
    predictions_file = inputFolder + 'TTR predictions.csv'  # blind and leaderboard, no special chemicals (493)

    df_training = dfu.load_df_from_file(training_file)

    # df_splits = dfu.load_df_from_file(split_file)
    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    # print(df_predictions)

    df_train = pd.merge(df_training, df_descriptors, on='QsarSmiles')
    df_train.rename(columns={"QsarSmiles": "id", "median_activity_%": "Property"}, inplace=True)

    print(df_train)

    df_predictions = dfu.load_df_from_file(predictions_file)
    df_pred = pd.merge(df_predictions, df_descriptors, on='QsarSmiles')
    df_pred = df_pred.drop('dataset', axis=1).drop('QsarSmiles', axis=1)
    df_pred.insert(1, "Property", -9999)
    df_pred.rename(columns={"DTXSID": "id"}, inplace=True)

    # print(df_pred)
    # df_train.to_csv(inputFolder+'training.csv', index=False)
    # df_pred.to_csv(inputFolder + 'prediction.csv', index=False)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)
    training_tsv = df_train.to_csv(sep='\t', index=False)

    # print(training_tsv)

    if mp.useEmbeddings:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
        print('embedding', embedding)
        print('timeMin', timeMin)
    else:
        embedding = None

    model = buildModel(embedding, mp, training_tsv, prediction_tsv)

    file = open(inputFolder + 'ttr_binding_full_model_w_leaderboard_' + mp.qsar_method + '_' + descriptor_set + '.p',
                'wb')
    pickle.dump(model, file)
    file.close()

    # df_results_tr = mwu.call_do_predictions_to_df(training_tsv, model)
    # print(df_results_tr)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)
    df_results.rename(columns={"id": "DTXSID"}, inplace=True)
    df_results.drop('exp', axis=1, inplace=True)
    # print (df_results)

    df_results = pd.merge(df_results, df_predictions, on='DTXSID')

    if useAQC:
        resultsFolder = inputFolder + "resultsAQC/"
    else:
        resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_full_model_w_leaderboard_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + "_blind_and_leaderboard.csv"

    df_results.to_csv(fileOut, index=False)
    # print(df_results)

    # import pickle
    # pickle.dumps(model)


def a_runCaseStudiesTTR_Binding_special_chemicals():
    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)

    mp.n_threads = 4

    predictions_file = inputFolder + 'special chemicals.tsv'
    df_pred = dfu.load_df_from_file(predictions_file)

    # df_pred = df_pred.drop('dataset', axis=1).drop('QsarSmiles', axis=1)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)

    # print(prediction_tsv)

    model_filepath = inputFolder + 'ttr_binding_full_model_' + mp.qsar_method + '_' + descriptor_set + '.p'
    with open(model_filepath, "rb") as input_file:
        model = pickle.load(input_file)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    df_results.to_csv(inputFolder + 'special chemicals results.csv', index=False)

    print(df_results)

    # df_results.rename(columns={"id": "DTXSID"}, inplace=True)
    # df_results.drop('exp', axis=1, inplace=True)
    # # print (df_results)
    #
    # df_results = pd.merge(df_results, df_predictions, on='DTXSID')


def a_runCaseStudiesTTR_Binding_special_chemicals2():
    set = "blind"
    # set = "training"

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)
    mp.qsar_method = 'rf'
    # mp.qsar_method = 'svm'

    mp.useEmbeddings = False

    mp.n_threads = 4

    predictions_file = inputFolder + 'special chemicals ' + set + '.tsv'
    df_pred = dfu.load_df_from_file(predictions_file)

    df_pred = df_pred.drop('QsarSmiles', axis=1)
    # df_pred = df_pred.drop('dataset', axis=1).drop('QsarSmiles', axis=1)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)

    # print(prediction_tsv)

    model_filepath = inputFolder + 'ttr_binding_full_model_' + mp.qsar_method + '_' + descriptor_set + '.p'
    with open(model_filepath, "rb") as input_file:
        model = pickle.load(input_file)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    dict_results = {}

    for index, row in df_results.iterrows():
        print(row['id'], row['pred'])

        if row['id'] in dict_results:

            vals = dict_results[row['id']]
            vals.append(row['pred'])

        else:
            vals = []
            vals.append(row['pred'])
            dict_results[row['id']] = vals

    print(json.dumps(dict_results, sort_keys=False, indent=4))

    dtxsids = []
    preds = []
    import statistics
    for key in dict_results:
        vals = dict_results[key]
        mean = statistics.mean(vals)
        dtxsids.append(key)
        preds.append(mean)
        # print(key,mean)

    results = {"DTXSID": dtxsids, "pred": preds}
    df_results2 = pd.DataFrame(results)

    print(df_results2)

    resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + "_" + set + '_special.csv'

    df_results2.to_csv(fileOut, index=False)


def a_runCaseStudiesTTR_Binding_special_chemicals2_with_leaderboard():
    set = "blind"
    # set = "training"

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)
    mp.qsar_method = 'rf'
    # mp.qsar_method = 'svm'

    mp.useEmbeddings = False

    mp.n_threads = 4

    predictions_file = inputFolder + 'special chemicals ' + set + '.tsv'
    df_pred = dfu.load_df_from_file(predictions_file)

    df_pred = df_pred.drop('QsarSmiles', axis=1)
    # df_pred = df_pred.drop('dataset', axis=1).drop('QsarSmiles', axis=1)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)

    # print(prediction_tsv)

    model_filepath = inputFolder + 'ttr_binding_full_model_w_leaderboard_' + mp.qsar_method + '_' + descriptor_set + '.p'
    with open(model_filepath, "rb") as input_file:
        model = pickle.load(input_file)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    dict_results = {}

    for index, row in df_results.iterrows():
        print(row['id'], row['pred'])

        if row['id'] in dict_results:

            vals = dict_results[row['id']]
            vals.append(row['pred'])

        else:
            vals = []
            vals.append(row['pred'])
            dict_results[row['id']] = vals

    print(json.dumps(dict_results, sort_keys=False, indent=4))

    dtxsids = []
    preds = []
    import statistics
    for key in dict_results:
        vals = dict_results[key]
        mean = statistics.mean(vals)
        dtxsids.append(key)
        preds.append(mean)
        # print(key,mean)

    results = {"DTXSID": dtxsids, "pred": preds}
    df_results2 = pd.DataFrame(results)

    print(df_results2)

    resultsFolder = inputFolder + "results/"

    fileOut = resultsFolder + "TTR_Binding_full_model_w_leaderboard_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + "_" + set + '_special.csv'

    df_results2.to_csv(fileOut, index=False)


def a_runCaseStudiesTTR_Binding_CV_merge_on_fly_predict_AQC_fails():
    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/'
    descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    # descriptor_set = 'mordred'

    mp = model_parameters(property_name="median TTR Binding activity", property_units='% activity',
                          descriptor_set=descriptor_set)
    mp.useEmbeddings = False
    # mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'

    sort_embeddings = False

    embeddings = []

    split_file = inputFolder + 'TTR training 5 fold splitting file with AQC.csv'
    descriptor_file = inputFolder + 'TTR ' + descriptor_set + ' descriptors.csv'

    df_splits = dfu.load_df_from_file(split_file)
    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_all = pd.merge(df_splits, df_descriptors, on='QsarSmiles')

    df_train = df_all.drop('Fold', axis=1)
    print('training shape', df_train.shape)

    pred_file = inputFolder + 'TTR training fails AQC.csv'
    df_pred = dfu.load_df_from_file(pred_file)
    df_pred = pd.merge(df_pred, df_descriptors)
    print('test shape', df_pred.shape)

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)
    training_tsv = df_train.to_csv(sep='\t', index=False)

    # fout=inputFolder+'training'+str(fold)+'.csv'
    # df_train.to_csv(fout, sep=',')
    # print(prediction_tsv)

    if mp.useEmbeddings:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

        if sort_embeddings:
            embedding.sort()

        print('embedding', embedding)
        print('timeMin', timeMin)
        embeddings.append(embedding)
    else:
        embedding = None

    model = buildModel(embedding, mp, training_tsv, prediction_tsv)
    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()
    rmse = ru.calc_rmse(pred, exp)
    r2 = ru.calc_pearson_r2(pred, exp)

    print('RMSE=' + str(rmse) + ', R2=' + str(r2))

    # print(df_results_all)

    resultsFolder = inputFolder + "resultsAQC/"

    fileOut = resultsFolder + "TTR_Binding_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + "_training_fails_AQC.csv"

    df_results.to_csv(fileOut, index=False)

    save_json_file(df_results_all=df_results, embeddings=embeddings, fileOut=fileOut, mp=mp)



if __name__ == '__main__':
    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly()
    # a_runCaseStudiesTTR_Binding_training_prediction()
    # a_runCaseStudiesTTR_Binding_special_chemicals()
    #
    # a_runCaseStudiesTTR_Binding_CV()
    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly_predict_AQC_fails()
    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly_with_leaderboard()
    #
    # resultsFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/results/'
    # lookAtResults(resultsFolder,useMean=True)
    # lookAtEmbeddings(resultsFolder)
    
    a_runCaseStudiesTTR_Binding_training_prediction_with_leaderboard()
    a_runCaseStudiesTTR_Binding_special_chemicals2_with_leaderboard()
    
