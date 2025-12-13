'''
Created on Dec 4, 2024
@author: TMARTI02
'''

import statistics as stats

# To avoid having to type rmb. for each method
# alternatively could have made this file extend the same class 

from models.run_model_building import save_json_file
from models.run_model_building import model_parameters
from models.run_model_building import getEmbedding
from models.run_model_building import buildModel
from models.run_model_building import lookAtResults
from models.run_model_building import calc_stats
from models.run_model_building import EmbeddingImporter
from models.run_model_building import get_descriptors_dataframe



import pandas as pd
import numpy as np
import pickle
import csv
import model_ws_utilities as mwu
from models import df_utilities as dfu
import applicability_domain.applicability_domain_utilities as adu
import models.results_utilities as ru



def a_runCaseStudiesInhalationCV_merge_on_fly(adMeasure=adu.strTESTApplicabilityDomainEmbeddingEuclidean):
    units = 'ppm'
    # units = 'mgL'

    useModelFiles = False
    doAD = True
    # adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean
    # adMeasure = adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    # adMeasure=adu.strTESTApplicabilityDomainEmbeddingCosine
    # adMeasure = adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    # adMeasure = adu.strOPERA_global_index #leverage
    # adMeasure = adu.strOPERA_local_index
    # adMeasure = adu.strKernelDensity

    logging.debug('adMeasure=', adMeasure)
    logging.debug('doAD=', doAD)

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    # descriptor_set = 'morgan'
    descriptor_set = 'mordred'
    # descriptor_set = 'morgan'
    # descriptor_set = 'webtest_opera'

    property_name = "4 hour rat inhalation LC50"

    embeddingCharlie = ['TopoPSA', 'ATS4m', 'ATS5m', 'nHBAcc', 'ATS1m', 'IC0', 'TopoPSA(NO)',
                        'ATS6m', 'ATS6dv',
                        'ATS0p', 'WPath',
                        'ETA_beta']

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)

    mp.n_threads = 16

    # mp.useEmbeddings = False
    mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'
    # mp.qsar_method = 'las'

    sort_embeddings = False

    df_results_all = None

    embeddings = []

    rmses = []
    r2s = []

    rmses_inside = []
    r2s_inside = []

    rmses_outside = []
    r2s_outside = []
    covs = []
    descriptors = []

    split_file = inputFolder + 'LC50_Tr_modeling_set_all_folds.csv'

    descriptor_file = inputFolder + 'LC50_tr_descriptors_' + descriptor_set + '.tsv'

    df_splits = dfu.load_df_from_file(split_file)

    if units == 'ppm':
        df_splits = df_splits.drop('4_hr_value_mgL', axis=1)
    else:
        df_splits = df_splits.drop('4_hr_value_ppm', axis=1)

    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    if 'webtest' in descriptor_set:
        df_descriptors = df_descriptors[df_descriptors.x0 != 'Error']  # remove bad rows

    # print('number rows descriptors=',df_descriptors.shape[0])
    # print('number rows splits=', df_splits.shape[0])

    df_all = pd.merge(df_splits, df_descriptors, on='QSAR_READY_SMILES')

    logging.debug('number rows all=', df_all.shape[0])

    df_all.drop_duplicates(inplace=True)
    logging.debug('number rows all,remove duplicates=', df_all.shape[0])

    # print(df_splits.shape)
    # print(df_descriptors.shape)
    # print(df_all.shape)

    for fold in range(1, 6):

        print('******************************************************************')
        print('Fold=' + str(fold))

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

        # print(prediction_tsv)

        # fout=inputFolder+'training'+str(fold)+'.csv'
        # df_train.to_csv(fout, sep=',')

        # print(prediction_tsv)

        if mp.useEmbeddings and not useModelFiles:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)

        else:
            embedding = None

        # picklepath = 'model'+str(fold)+'.pickle'

        picklepath = inputFolder + 'model' + str(
            fold) + '_' + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
            mp.useEmbeddings) + ".pickle"

        print(picklepath)

        if useModelFiles:
            with open(picklepath, 'rb') as handle:
                model = pickle.load(handle)

        else:
            model = buildModel(embedding, mp, training_tsv, prediction_tsv)
            with open(picklepath, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        descriptors.append(len(model.embedding))

        print('number of model descriptors', len(model.embedding))

        embeddingAD = model.embedding
        # embeddingAD = embeddingCharlie

        if doAD:
            df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                                   test_tsv=prediction_tsv,
                                                                                   remove_log_p=False,
                                                                                   embedding=embeddingAD,
                                                                                   applicability_domain=adMeasure)
            df_AD.rename(columns={"idTest": "id"}, inplace=True)
            # print(df_AD.shape)
            # df_results.to_csv('adresults' + str(fold) + '.csv')
            df_results = pd.merge(df_results, df_AD, on='id')

            df_results_inside = df_results.loc[df_results['AD'] == True]
            df_results_outside = df_results.loc[df_results['AD'] == False]
            # print('inside shape=', df_results_inside.shape)

            if (df_results_inside.shape[0] > 1):
                r2_inside, rmse_inside = calc_stats(df_results_inside)
                r2s_inside.append(r2_inside)
                rmses_inside.append(rmse_inside)
            else:
                print('none inside AD')

            print('fold=' + str(fold), 'RMSE_inside=' + str(rmse_inside) + ', R2_inside=' + str(r2_inside),
                  df_results_inside.shape[0])

            if (df_results_outside.shape[0] > 1):
                r2_outside, rmse_outside = calc_stats(df_results_outside)
                r2s_outside.append(r2_outside)
                rmses_outside.append(rmse_outside)
                print('fold=' + str(fold), 'RMSE_outside=' + str(rmse_outside) + ', R2_outside=' + str(r2_outside),
                      df_results_outside.shape[0])
            else:
                print('fold=' + str(fold), 'RMSE_outside=' + "N/A" + ', R2_outside=' + "N/A",
                      df_results_outside.shape[0])

            cov = df_results_inside.shape[0] / df_results.shape[0]
            print(df_results_inside.shape[0], df_results.shape[0], cov)

            covs.append(cov)

        # print(df_results.shape)

        # print(df_results)

        r2, rmse = calc_stats(df_results)

        r2s.append(r2)
        rmses.append(rmse)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))
        print('\n')

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)

    # print(df_results_all)

    resultsFolder = inputFolder + "results/"

    fileOut = inputFolder + "results/LC50_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp, r2s=r2s, rmses=rmses)

    rmse_mean = stats.mean(rmses)

    if doAD:
        r2_mean_inside = stats.mean(r2s_inside)
        rmse_mean_inside = stats.mean(rmses_inside)

        r2_mean_outside = stats.mean(r2s_outside)
        rmse_mean_outside = stats.mean(rmses_outside)

        cov_mean = stats.mean(covs)
        descriptors_mean = stats.mean(descriptors)

        print('AD measure', adMeasure)
        print('rmse_mean', rmse_mean)
        print('rmse_mean_inside', rmse_mean_inside)
        print('rmse_mean_outside', rmse_mean_outside)
        print('cov_mean', cov_mean)
        print('descriptors_mean', descriptors_mean)

        print('\n*** AD Stats')
        print(adMeasure, rmse_mean, rmse_mean_inside, rmse_mean_outside, cov_mean, descriptors_mean)

        with open(resultsFolder + '/resultsAD.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            resultsList = [mp.descriptor_set, adMeasure, rmse_mean, rmse_mean_inside, rmse_mean_outside, cov_mean,
                           descriptors_mean]
            writer.writerow(resultsList)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot2(fileOut=fileOut, property_name='LC50', title=title, exp=exp, pred=pred)

    lookAtResults(resultsFolder, useMean=True)


def a_runCaseStudiesInhalationCV_merge_on_fly_charlie(adMeasure=adu.strTESTApplicabilityDomainEmbeddingEuclidean):
    units = 'ppm'
    # units = 'mgL'

    useModelFiles = False
    doAD = True
    # adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean
    # adMeasure = adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    # adMeasure=adu.strTESTApplicabilityDomainEmbeddingCosine
    # adMeasure = adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    # adMeasure = adu.strOPERA_global_index #leverage
    # adMeasure = adu.strOPERA_local_index
    # adMeasure = adu.strKernelDensity

    print('adMeasure=', adMeasure)
    print('doAD=', doAD)

    ei = EmbeddingImporter(None)
    # embeddingCharlie = ei.getInhalation_12_descriptors()
    # embeddingCharlie = ei.getInhalation_25_descriptors()
    embeddingCharlie = ei.getInhalation_all_descriptors()

    # embeddingCharlie = ['TopoPSA', 'ATS4m', 'ATS5m', 'nHBAcc', 'ATS1m', 'IC0', 'TopoPSA(NO)',
    #                'ATS6m', 'ATS6dv','ATS0p', 'WPath','ETA_beta']

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'morgan'
    # descriptor_set = 'webtest_opera'

    property_name = "4 hour rat inhalation LC50"

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)

    mp.n_threads = 16

    # mp.useEmbeddings = False
    mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'
    # mp.qsar_method = 'las'

    sort_embeddings = False

    df_results_all = None

    embeddings = []

    rmses = []
    r2s = []

    rmses_inside = []
    r2s_inside = []

    rmses_outside = []
    r2s_outside = []
    covs = []
    descriptors = []

    split_file = inputFolder + 'LC50_Tr_modeling_set_all_folds.csv'

    descriptor_file = inputFolder + 'LC50_tr_descriptors_' + descriptor_set + '.tsv'

    df_splits = dfu.load_df_from_file(split_file)

    if units == 'ppm':
        df_splits = df_splits.drop('4_hr_value_mgL', axis=1)
    else:
        df_splits = df_splits.drop('4_hr_value_ppm', axis=1)

    df_descriptors = dfu.load_df_from_file(descriptor_file)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    if 'webtest' in descriptor_set:
        df_descriptors = df_descriptors[df_descriptors.x0 != 'Error']  # remove bad rows

    # print('number rows descriptors=',df_descriptors.shape[0])
    # print('number rows splits=', df_splits.shape[0])

    df_all = pd.merge(df_splits, df_descriptors, on='QSAR_READY_SMILES')

    print('number rows all=', df_all.shape[0])

    df_all.drop_duplicates(inplace=True)
    print('number rows all,remove duplicates=', df_all.shape[0])

    # print(df_splits.shape)
    # print(df_descriptors.shape)
    # print(df_all.shape)

    for fold in range(1, 6):

        print('******************************************************************')
        print('Fold=' + str(fold))

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

        embedding = embeddingCharlie

        # picklepath = 'model'+str(fold)+'.pickle'

        picklepath = inputFolder + 'modelCharlie' + str(
            fold) + '_' + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
            mp.useEmbeddings) + ".pickle"

        print(picklepath)

        if useModelFiles:
            with open(picklepath, 'rb') as handle:
                model = pickle.load(handle)

        else:
            model = buildModel(embedding, mp, training_tsv, prediction_tsv)
            with open(picklepath, 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        descriptors.append(len(model.embedding))

        print('number of model descriptors', len(model.embedding))

        embeddingAD = model.embedding

        if doAD:
            df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                                   test_tsv=prediction_tsv,
                                                                                   remove_log_p=False,
                                                                                   embedding=embeddingAD,
                                                                                   applicability_domain=adMeasure)
            df_AD.rename(columns={"idTest": "id"}, inplace=True)
            # print(df_AD.shape)
            # df_results.to_csv('adresults' + str(fold) + '.csv')
            df_results = pd.merge(df_results, df_AD, on='id')

            df_results_inside = df_results.loc[df_results['AD'] == True]
            df_results_outside = df_results.loc[df_results['AD'] == False]
            # print('inside shape=', df_results_inside.shape)

            if (df_results_inside.shape[0] > 1):
                r2_inside, rmse_inside = calc_stats(df_results_inside)
                r2s_inside.append(r2_inside)
                rmses_inside.append(rmse_inside)
            else:
                print('none inside AD')

            print('fold=' + str(fold), 'RMSE_inside=' + str(rmse_inside) + ', R2_inside=' + str(r2_inside),
                  df_results_inside.shape[0])

            if (df_results_outside.shape[0] > 1):
                r2_outside, rmse_outside = calc_stats(df_results_outside)
                r2s_outside.append(r2_outside)
                rmses_outside.append(rmse_outside)
                print('fold=' + str(fold), 'RMSE_outside=' + str(rmse_outside) + ', R2_outside=' + str(r2_outside),
                      df_results_outside.shape[0])
            else:
                print('fold=' + str(fold), 'RMSE_outside=' + "N/A" + ', R2_outside=' + "N/A",
                      df_results_outside.shape[0])

            cov = df_results_inside.shape[0] / df_results.shape[0]
            print(df_results_inside.shape[0], df_results.shape[0], cov)

            covs.append(cov)

        # print(df_results.shape)

        # print(df_results)

        r2, rmse = calc_stats(df_results)

        r2s.append(r2)
        rmses.append(rmse)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))
        print('\n')

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)

    # print(df_results_all)

    resultsFolder = inputFolder + "results/"

    fileOut = inputFolder + "results/LC50_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp, r2s=r2s, rmses=rmses)

    rmse_mean = stats.mean(rmses)

    if doAD:
        r2_mean_inside = stats.mean(r2s_inside)
        rmse_mean_inside = stats.mean(rmses_inside)

        r2_mean_outside = stats.mean(r2s_outside)
        rmse_mean_outside = stats.mean(rmses_outside)

        cov_mean = stats.mean(covs)
        descriptors_mean = stats.mean(descriptors)

        print('AD measure', adMeasure)
        print('rmse_mean', rmse_mean)
        print('rmse_mean_inside', rmse_mean_inside)
        print('rmse_mean_outside', rmse_mean_outside)
        print('cov_mean', cov_mean)
        print('descriptors_mean', descriptors_mean)

        print('\n*** AD Stats')
        print(adMeasure, rmse_mean, rmse_mean_inside, rmse_mean_outside, cov_mean, descriptors_mean)

        with open(resultsFolder + '/resultsAD.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            resultsList = [mp.descriptor_set, adMeasure, rmse_mean, rmse_mean_inside, rmse_mean_outside, cov_mean,
                           descriptors_mean]
            writer.writerow(resultsList)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot2(fileOut=fileOut, property_name='LC50', title=title, exp=exp, pred=pred)

    lookAtResults(resultsFolder, useMean=True)


def a_runCaseStudiesInhalationCV_merge_on_flyFinalModel(adMeasure=adu.strTESTApplicabilityDomainEmbeddingEuclidean):
    units = 'ppm'
    # units = 'mgL'
    columnForSpecialSmiles = 'QSAR_READY_SMILES_SDE_LONGEST'

    useModelFiles = True
    doAD = True
    # adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean
    # adMeasure = adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    # adMeasure=adu.strTESTApplicabilityDomainEmbeddingCosine
    # adMeasure = adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    # adMeasure = adu.strOPERA_global_index #leverage
    # adMeasure = adu.strOPERA_local_index  # leverage
    # adMeasure = adu.strKernelDensity

    print('adMeasure=', adMeasure)
    # print('doAD=', doAD)

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'morgan'
    # descriptor_set = 'webtest_opera'

    property_name = "4 hour rat inhalation LC50"

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)

    mp.n_threads = 16

    # mp.useEmbeddings = False
    mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'
    # mp.qsar_method = 'las'

    sort_embeddings = False
    embeddings = []

    useCharlieEmbedding = True
    ei = EmbeddingImporter(None)
    # embeddingCharlie = ei.getInhalation_12_descriptors()
    # embeddingCharlie = ei.getInhalation_25_descriptors()
    embeddingCharlie = ei.getInhalation_all_descriptors()
    filterColumnsInBothSets = False

    # print(type(embeddingCharlie))
    # print(len(embeddingCharlie))

    # ************************************************************************************************************
    # prediction_file = inputFolder+'PredictionSet.csv'
    # prediction_file2 = inputFolder+'LC50_Prediction_descriptors_edit_smiles.tsv'
    #
    # df_prediction = dfu.load_df_from_file(prediction_file)
    # df_prediction = pd.DataFrame().assign(QSAR_READY_SMILES=df_prediction['QSAR_READY_SMILES'],Toxicity=-9999)
    #
    # df_prediction2 = dfu.load_df_from_file(prediction_file2)
    # df_prediction2 = pd.DataFrame().assign(QSAR_READY_SMILES=df_prediction2[columnForSpecialSmiles], Toxicity=-9999)
    #
    # df_prediction = pd.concat([df_prediction, df_prediction2])
    # df_prediction.drop_duplicates(inplace=True)
    #
    # df_prediction.to_csv(inputFolder+'PredictionSet_'+columnForSpecialSmiles+'+.csv',index=False)
    # ************************************************************************************************************
    prediction_file = inputFolder + 'PredictionSet longest sde smiles.csv'
    df_prediction = dfu.load_df_from_file(prediction_file)
    df_prediction = pd.DataFrame().assign(QSAR_READY_SMILES=df_prediction['QSAR_READY_SMILES'], Toxicity=-9999)

    # Get rid of rows that have null model descriptors:
    df_prediction = df_prediction[df_prediction.QSAR_READY_SMILES != 'C']
    df_prediction = df_prediction[df_prediction.QSAR_READY_SMILES != '[C]']

    # print(df_prediction.shape)
    # df_prediction.drop_duplicates(inplace=True)
    # print(df_prediction.shape)
    # ************************************************************************************************************

    descriptor_file_training = inputFolder + 'LC50_tr_descriptors_' + descriptor_set + '.tsv'
    df_descriptors = dfu.load_df_from_file(descriptor_file_training)

    descriptor_file_prediction = inputFolder + 'LC50_Prediction_descriptors_' + descriptor_set + '.tsv'
    df_descriptors_prediction = dfu.load_df_from_file(descriptor_file_prediction)

    # Merge the descriptor files
    df_descriptors = pd.concat([df_descriptors, df_descriptors_prediction])
    print('df_descriptors.shape', df_descriptors.shape)
    df_descriptors.drop_duplicates(inplace=True)
    print('df_descriptors.shape', df_descriptors.shape)

    # df_descriptors.to_csv('desc.csv',index=False)

    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)

    if 'webtest' in descriptor_set:
        df_descriptors = df_descriptors[df_descriptors.x0 != 'Error']  # remove bad rows

    # *******************************************************************************************
    split_file = inputFolder + 'LC50_Tr_modeling_set_all_folds.csv'
    df_splits = dfu.load_df_from_file(split_file)
    if units == 'ppm':
        df_splits = df_splits.drop('4_hr_value_mgL', axis=1)
    else:
        df_splits = df_splits.drop('4_hr_value_ppm', axis=1)

    df_splits.rename(columns={df_splits.columns[1]: "Toxicity"}, inplace=True)
    df_splits = df_splits.drop('Fold', axis=1)
    # print(df_splits)
    # *******************************************************************************************

    df_train = pd.merge(df_splits, df_descriptors, on='QSAR_READY_SMILES')
    df_pred = pd.merge(df_prediction, df_descriptors, on='QSAR_READY_SMILES')
    df_pred.drop_duplicates(inplace=True)

    # df_pred.to_csv('pred.csv',index=False)

    # if True:
    #     return

    # df_labels = df_train[df_train.columns[1]]
    # print(df_labels)

    print('shape train=', df_train.shape)
    print('shape pred=', df_pred.shape)
    # print(df_train.columns)
    # print(df_pred.columns)

    # print(df_train.loc['Toxicity'])
    # print(type(df_train.columns[1]))
    # print(df_pred.loc['Toxicity'])

    # if True:
    #     return

    print('******************************************************************')

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)
    training_tsv = df_train.to_csv(sep='\t', index=False)
    print('Done converting to tsv')

    # print('num lines prediction_tsv',len(prediction_tsv.splitlines()))

    # fout=inputFolder+'training'+str(fold)+'.csv'
    # df_train.to_csv(fout, sep=',')

    # print(prediction_tsv)

    if useCharlieEmbedding:
        embedding = embeddingCharlie
    else:
        if mp.useEmbeddings and not useModelFiles:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)

        else:
            embedding = None

    # picklepath = 'model'+str(fold)+'.pickle'

    picklepath = inputFolder + 'modelFinal' + '_' + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".pickle"

    print(picklepath)

    if useModelFiles:
        with open(picklepath, 'rb') as handle:
            model = pickle.load(handle)

    else:
        model = buildModel(embedding, mp, training_tsv, prediction_tsv, filterColumnsInBothSets=filterColumnsInBothSets)
        with open(picklepath, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    print('number of model descriptors', len(model.embedding))

    if doAD:
        df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                               test_tsv=prediction_tsv,
                                                                               remove_log_p=False,
                                                                               embedding=model.embedding,
                                                                               applicability_domain=adMeasure,
                                                                               filterColumnsInBothSets=filterColumnsInBothSets)

        df_AD.rename(columns={"idTest": "id"}, inplace=True)
        df_AD.drop_duplicates(inplace=True)
        # print(df_AD.shape)
        # df_results.to_csv('adresults' + str(fold) + '.csv')
        df_results = pd.merge(df_results, df_AD, on='id')

        # df_results_inside = df_results.loc[df_results['AD'] == True]
        # df_results_outside = df_results.loc[df_results['AD'] == False]
        # print('inside shape=', df_results_inside.shape)
        # cov = df_results_inside.shape[0]/df_results.shape[0]
        # print('cov=',cov)
        # print(df_results.shape)
        # print(df_results)

    # print(df_results_all)

    fileOut = inputFolder + "results/LC50_Prediction_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results.to_csv(fileOut, index=False)  # writes results with AD


def a_runCaseStudiesInhalationCV_merge_on_flyFinalModel2(adMeasure=adu.strTESTApplicabilityDomainEmbeddingEuclidean):
    """This version runs all the records in the predictions descriptor file and then merges with prediction set file at the end"""

    units = 'ppm'
    # units = 'mgL'
    columnForSpecialSmiles = 'QSAR_READY_SMILES_SDE_LONGEST'

    useModelFiles = False
    doAD = True
    # adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean
    # adMeasure = adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    # adMeasure=adu.strTESTApplicabilityDomainEmbeddingCosine
    # adMeasure = adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    # adMeasure = adu.strOPERA_global_index #leverage
    # adMeasure = adu.strOPERA_local_index  # leverage
    # adMeasure = adu.strKernelDensity

    print('adMeasure=', adMeasure)
    # print('doAD=', doAD)

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'morgan'
    # descriptor_set = 'webtest_opera'

    property_name = "4 hour rat inhalation LC50"

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)
    mp.n_threads = 16
    # mp.useEmbeddings = False
    mp.useEmbeddings = True
    mp.qsar_method = 'rf'
    sort_embeddings = False
    embeddings = []

    useCharlieEmbedding = True
    ei = EmbeddingImporter(None)
    # embeddingCharlie = ei.getInhalation_12_descriptors()
    # embeddingCharlie = ei.getInhalation_25_descriptors()
    embeddingCharlie = ei.getInhalation_all_descriptors()
    filterColumnsInBothSets = False

    # ************************************************************************************************************
    prediction_file = inputFolder + 'PredictionSet longest sde smiles.csv'
    df_prediction_set = dfu.load_df_from_file(prediction_file)
    # print(df_prediction_set.columns)

    # ************************************************************************************************************
    descriptor_file_training = inputFolder + 'LC50_tr_descriptors_' + descriptor_set + '.tsv'
    df_descriptors_training = get_descriptors_dataframe(descriptor_file_training, descriptor_set)
    # ************************************************************************************************************
    descriptor_file_prediction = inputFolder + 'LC50_Prediction_descriptors_' + descriptor_set + '.tsv'
    df_pred = get_descriptors_dataframe(descriptor_file_prediction, descriptor_set)
    df_pred.insert(1, 'Toxicity', -9999)
    df_pred.drop_duplicates(inplace=True)

    # ************************************************************************************************************
    # Get rid of rows that cause AD to fail. Ideally the AD code should figure this out and
    # exclude the rows with null embedding descriptors

    bad_smiles = ['C', '[C]']
    for smiles in bad_smiles:
        df_pred = df_pred[df_pred.QSAR_READY_SMILES != smiles]

    # Salts will have null embedding descriptors:
    filter = df_pred['QSAR_READY_SMILES'].str.contains('\\.')  # note need escape character for .
    df_pred = df_pred[~filter]

    print(df_pred.shape[0])
    # print(df_descriptors_prediction.columns)
    # df_pred.to_csv('pred.csv',index=False)
    # *******************************************************************************************
    split_file = inputFolder + 'LC50_Tr_modeling_set_all_folds.csv'
    df_splits = dfu.load_df_from_file(split_file)
    if units == 'ppm':
        df_splits = df_splits.drop('4_hr_value_mgL', axis=1)
    else:
        df_splits = df_splits.drop('4_hr_value_ppm', axis=1)
    df_splits.rename(columns={df_splits.columns[1]: "Toxicity"}, inplace=True)
    df_splits = df_splits.drop('Fold', axis=1)
    df_train = pd.merge(df_splits, df_descriptors_training, on='QSAR_READY_SMILES')
    # *******************************************************************************************
    print('shape train=', df_train.shape)
    print('shape pred=', df_pred.shape)
    # if True:
    #     return
    print('******************************************************************')

    prediction_tsv = df_pred.to_csv(sep='\t', index=False)
    training_tsv = df_train.to_csv(sep='\t', index=False)
    print('Done converting to tsv')

    # print('num lines prediction_tsv',len(prediction_tsv.splitlines()))

    # fout=inputFolder+'training'+str(fold)+'.csv'
    # df_train.to_csv(fout, sep=',')

    # print(prediction_tsv)

    if useCharlieEmbedding:
        embedding = embeddingCharlie
    else:
        if mp.useEmbeddings and not useModelFiles:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)

        else:
            embedding = None

    # picklepath = 'model'+str(fold)+'.pickle'

    picklepath = inputFolder + 'modelFinal' + '_' + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".pickle"

    print(picklepath)

    if useModelFiles:
        with open(picklepath, 'rb') as handle:
            model = pickle.load(handle)

    else:
        model = buildModel(embedding, mp, training_tsv, prediction_tsv, filterColumnsInBothSets=filterColumnsInBothSets)
        with open(picklepath, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)
    df_results.rename(columns={"id": "QSAR_READY_SMILES"}, inplace=True)

    print('number of model descriptors', len(model.embedding))

    if doAD:
        df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                               test_tsv=prediction_tsv,
                                                                               remove_log_p=False,
                                                                               embedding=model.embedding,
                                                                               applicability_domain=adMeasure,
                                                                               filterColumnsInBothSets=filterColumnsInBothSets)

        df_AD.rename(columns={"idTest": "QSAR_READY_SMILES"}, inplace=True)
        df_AD.drop_duplicates(inplace=True)
        # print(df_AD.shape)
        # df_results.to_csv('adresults' + str(fold) + '.csv')
        df_results = pd.merge(df_results, df_AD, on='QSAR_READY_SMILES')

        # df_results_inside = df_results.loc[df_results['AD'] == True]
        # df_results_outside = df_results.loc[df_results['AD'] == False]
        # print('inside shape=', df_results_inside.shape)
        # cov = df_results_inside.shape[0]/df_results.shape[0]
        # print('cov=',cov)
        # print(df_results.shape)
        # print(df_results)

    # print(df_results_all)

    df_results = pd.merge(df_prediction_set, df_results, on='QSAR_READY_SMILES')

    fileOut = inputFolder + "results/LC50_Prediction_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results.to_csv(fileOut, index=False)  # writes results with AD


def a_runCaseStudiesInhalationCV_merge_on_flyFinalModelTrain(adMeasure=adu.strTESTApplicabilityDomainEmbeddingEuclidean):
    """This version runs all the records in the predictions descriptor file and then merges with prediction set file at the end"""

    units = 'ppm'
    # units = 'mgL'

    useModelFiles = False
    doAD = True
    # adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean
    # adMeasure = adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    # adMeasure=adu.strTESTApplicabilityDomainEmbeddingCosine
    # adMeasure = adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    # adMeasure = adu.strOPERA_global_index #leverage
    # adMeasure = adu.strOPERA_local_index  # leverage
    # adMeasure = adu.strKernelDensity

    print('adMeasure=', adMeasure)
    # print('doAD=', doAD)

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
    descriptor_set = 'mordred'
    # descriptor_set = 'morgan'
    # descriptor_set = 'webtest_opera'

    property_name = "4 hour rat inhalation LC50"

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)
    mp.n_threads = 16
    # mp.useEmbeddings = False
    mp.useEmbeddings = True
    mp.qsar_method = 'rf'
    sort_embeddings = False
    embeddings = []

    useCharlieEmbedding = True
    ei = EmbeddingImporter(None)
    # embeddingCharlie = ei.getInhalation_12_descriptors()
    # embeddingCharlie = ei.getInhalation_25_descriptors()
    embeddingCharlie = ei.getInhalation_all_descriptors()
    filterColumnsInBothSets = False

    # ************************************************************************************************************
    descriptor_file_training = inputFolder + 'LC50_tr_descriptors_' + descriptor_set + '.tsv'
    df_descriptors_training = get_descriptors_dataframe(descriptor_file_training, descriptor_set)
    # ************************************************************************************************************

    # print(df_descriptors_prediction.columns)
    # df_pred.to_csv('pred.csv',index=False)
    # *******************************************************************************************
    split_file = inputFolder + 'LC50_Tr_modeling_set_all_folds.csv'
    df_splits = dfu.load_df_from_file(split_file)
    if units == 'ppm':
        df_splits = df_splits.drop('4_hr_value_mgL', axis=1)
    else:
        df_splits = df_splits.drop('4_hr_value_ppm', axis=1)
    df_splits.rename(columns={df_splits.columns[1]: "Toxicity"}, inplace=True)
    df_splits = df_splits.drop('Fold', axis=1)
    df_train = pd.merge(df_splits, df_descriptors_training, on='QSAR_READY_SMILES')
    # *******************************************************************************************
    print('shape train=', df_train.shape)
    # if True:
    #     return
    print('******************************************************************')

    training_tsv = df_train.to_csv(sep='\t', index=False)
    prediction_tsv = training_tsv  #we are running resubstitution of training set

    print('Done converting to tsv')

    # print('num lines prediction_tsv',len(prediction_tsv.splitlines()))

    # fout=inputFolder+'training'+str(fold)+'.csv'
    # df_train.to_csv(fout, sep=',')

    # print(prediction_tsv)

    if useCharlieEmbedding:
        embedding = embeddingCharlie
    else:
        if mp.useEmbeddings and not useModelFiles:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)

            if sort_embeddings:
                embedding.sort()

            print('embedding', embedding)
            print('timeMin', timeMin)
            embeddings.append(embedding)

        else:
            embedding = None

    # picklepath = 'model'+str(fold)+'.pickle'

    picklepath = inputFolder + 'modelFinal' + '_' + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".pickle"

    print(picklepath)

    if useModelFiles:
        with open(picklepath, 'rb') as handle:
            model = pickle.load(handle)

    else:
        model = buildModel(embedding, mp, training_tsv, prediction_tsv, filterColumnsInBothSets=filterColumnsInBothSets)
        with open(picklepath, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)
    df_results.rename(columns={"id": "QSAR_READY_SMILES"}, inplace=True)

    print('number of model descriptors', len(model.embedding))

    if doAD:
        df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                               test_tsv=prediction_tsv,
                                                                               remove_log_p=False,
                                                                               embedding=model.embedding,
                                                                               applicability_domain=adMeasure,
                                                                               filterColumnsInBothSets=filterColumnsInBothSets,
                                                                            returnTrainingAD=True)




        df_AD.rename(columns={"idTrain": "QSAR_READY_SMILES"}, inplace=True)
        df_AD.drop_duplicates(inplace=True)

        # print(df_AD.shape)
        # df_results.to_csv('adresults' + str(fold) + '.csv')
        df_results = pd.merge(df_results, df_AD, on='QSAR_READY_SMILES')

        df_results_inside = df_results.loc[df_results['AD'] == True]
        df_results_outside = df_results.loc[df_results['AD'] == False]

        print('inside shape=', df_results_inside.shape)
        print('outside shape=', df_results_outside.shape)

        cov = df_results_inside.shape[0]/df_results.shape[0]
        print('cov=',cov)
        # print(df_results.shape)
        # print(df_results)

    # print(df_results_all)

    # df_results = pd.merge(df_prediction_set, df_results, on='QSAR_READY_SMILES')

    fileOut = inputFolder + "results/LC50_Training_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results.to_csv(fileOut, index=False)  # writes results with AD


def a_runCaseStudiesRatInhalationLC50():
    # descriptor_set = 'webtest'
    # descriptor_set = 'padel'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'mordred'
    descriptor_set = 'mordred_opera'

    # units = 'mgL'
    units = 'ppm'
    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    property_name = "4 hour rat inhalation LC50"
    if units == 'ppm':
        property_units = 'log10 ppm'
    elif units == 'mgL':
        property_units = 'log10 mg/L'

    mp = model_parameters(property_name, property_units, descriptor_set)
    # mp.useEmbeddings = False
    mp.useEmbeddings = True

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'reg'

    df_results_all = None
    embeddings = []

    for fold in range(1, 6):

        print('\nfold=' + str(fold))
        training_file_name = "LC50_tr_log10_" + units + "_" + descriptor_set + "_train_CV" + str(fold) + ".tsv";
        prediction_file_name = "LC50_tr_log10_" + units + "_" + descriptor_set + "_test_CV" + str(fold) + ".tsv";
        training_tsv_path = inputFolder + training_file_name
        prediction_tsv_path = inputFolder + prediction_file_name
        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        if mp.useEmbeddings:
            embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
            embeddings.append(embedding)
        else:
            embedding = None

        model = buildModel(embedding, mp, training_tsv, prediction_tsv)

        df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

        exp = df_results['exp'].to_numpy()
        pred = df_results['pred'].to_numpy()
        rmse = ru.calc_rmse(pred, exp)
        print('fold=' + str(fold), 'RMSE=' + str(rmse))

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = df_results_all.append(df_results, ignore_index=True)

    # print(df_results_all)
    fileOut = inputFolder + "results/LC50_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"
    df_results_all.to_csv(fileOut, index=False)
    save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp)



if __name__ == '__main__':
    # a_runCaseStudiesRatInhalationLC50()
    # a_runCaseStudiesInhalationCV_merge_on_fly()
    # a_runCaseStudiesInhalationCV_merge_on_fly_charlie()
    # a_runCaseStudiesInhalationCV_merge_on_flyFinalModel()
    # a_runCaseStudiesInhalationCV_merge_on_flyFinalModel2()
    a_runCaseStudiesInhalationCV_merge_on_flyFinalModelTrain()

    # measures = [adu.strTESTApplicabilityDomainEmbeddingEuclidean,
    #             adu.strTESTApplicabilityDomainAllDescriptorsEuclideanDistance
    #     , adu.strTESTApplicabilityDomainEmbeddingCosine
    #     , adu.strTESTApplicabilityDomainAlLDescriptorsCosine
    #     , adu.strOPERA_global_index
    #     , adu.strOPERA_local_index]
    #
    # for measure in measures:
    #     print(measure)
    #     a_runCaseStudiesInhalationCV_merge_on_fly(adMeasure=measure)

    # folder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/results/'
    # lookAtResults(folder,useMean=True)