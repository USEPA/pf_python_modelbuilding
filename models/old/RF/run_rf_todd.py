# import math
import os

from numpy import array
from sklearn.feature_selection import RFECV

import utils
from models.RF import rf_model_1_3 as rf1_3, rf_model_1_4 as rf1_4, rf_model_1_1 as rf1_1, rf_model_1_2 as rf1_2
from models import df_utilities as DFU
from models import GeneticOptimizer as go
import model_ws_utilities as mwu
from models import EmbeddingFromImportance as efi

import models.results_utilities as ru



import operator

import pandas as pd
import time
import scipy
import matplotlib.pyplot as plt

import numpy as np

from os.path import exists

strWaterSolubility = 'Water solubility'
strVaporPressure = 'Vapor pressure'
strHenrysLawConstant = 'Henry\'s law constant'

splittingPFASOnly = 'T=PFAS only, P=PFAS'
# splittingAll = 'T=all, P=PFAS'
splittingAllButPFAS = 'T=all but PFAS, P=PFAS'
splittingRepresentative = 'RND_REPRESENTATIVE'

def loadEmbedding(filepath):
    f = open(filepath, 'r')
    import json
    data = json.load(f)
    embedding = '\t'.join(data['embedding'])
    print(embedding)
    return embedding


def runCaseStudyPFAS():
    '''
    Runs PFAS modeling case studies where we can decide which chemicals to use in training set and then predict for PFAS chemicals
    '''

    datasetName = 'HLC from exp_prop and chemprop'
    # datasetName = 'WS from exp_prop and chemprop'
    # datasetName = 'VP from exp_prop and chemprop'
    # datasetName = 'LogP from exp_prop and chemprop'
    # datasetName = 'MP from exp_prop and chemprop'
    # datasetName = 'BP from exp_prop and chemprop'
    # datasetName = 'ExpProp BCF Fish_TMM'

    splittingName = splittingPFASOnly
    # splittingName = splittingAll
    # splittingName = splittingAllButPFAS

    caseStudyPFAS(datasetName, splittingName, True)


def runCaseStudiesPFAS():
    '''
    Runs PFAS modeling case studies where we can decide which chemicals to use in training set and then predict for PFAS chemicals
    '''

    datasetNames = []
    datasetNames.append('HLC from exp_prop and chemprop')
    datasetNames.append('WS from exp_prop and chemprop')
    datasetNames.append('VP from exp_prop and chemprop')
    datasetNames.append('LogP from exp_prop and chemprop')
    datasetNames.append('MP from exp_prop and chemprop')
    datasetNames.append('BP from exp_prop and chemprop')
    # datasetNames.append('ExpProp BCF Fish_TMM')

    splittingNames = []
    splittingNames.append(splittingPFASOnly)
    splittingNames.append(splittingRepresentative)
    splittingNames.append(splittingAllButPFAS)

    folder = '../datasets/'

    with open(folder + 'pfasresults.txt', 'w') as f:

        for splittingName in splittingNames:
            for datasetName in datasetNames:
                result = caseStudyPFAS(datasetName, splittingName, False)
                f.write(result + "\n")
                f.flush()

    f.close()


def caseStudyPFAS(datasetName, splittingName, showPlot):
    '''
    Parameterized version
    '''

    useEmbeddings = False
    print('\n\n**************\n', datasetName, splittingName)

    n_threads = 20
    remove_log_p_descriptors = False
    version = 1.3
    descriptorSetName = 'WebTEST-default'
    folder = '../datasets/' + datasetName + '/PFAS/'

    num_generations = 10
    num_optimizers = 10
    qsar_method = 'knn'
    n_threads = 16  # threads to use with RF- TODO
    num_jobs = 4  # jobs to use for GA search for embedding- using 16 has marginal benefit over 4 or 8
    descriptor_coefficient = 0.002
    max_length = 24
    threshold = 1  # TMM attempt to get more descriptors from stage 1

    training_file_name = datasetName + "_" + descriptorSetName + "_" + splittingName + "_training.tsv";
    prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    import os
    absolute_path = os.path.abspath(training_tsv_path)
    print("Full path: " + absolute_path)

    if not exists(training_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    if not exists(prediction_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    descriptor_names = 'desc1,desc2,desc3'  # set descriptor names manually or call webservice to figure it out

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    training_tsv = DFU.read_file_to_string(training_tsv_path)
    prediction_tsv = DFU.read_file_to_string(prediction_tsv_path)

    # training_tsv=df_training.to_csv(index=False)
    # prediction_tsv = df_prediction.to_csv(index=False)

    if useEmbeddings:
        descriptor_names, timeGA = mwu.call_build_embedding_ga(qsar_method=qsar_method,
                                                               training_tsv=training_tsv, prediction_tsv=prediction_tsv,
                                                               remove_log_p=False, n_threads=n_threads,
                                                               num_generations=num_generations,
                                                               num_optimizers=num_optimizers, num_jobs=num_jobs,
                                                               descriptor_coefficient=descriptor_coefficient,
                                                               max_length=max_length, threshold=threshold, model_id=1)

    model = buildModel(descriptor_names, df_training, n_threads, remove_log_p_descriptors, useEmbeddings, version)
    # predictions = model.do_predictions(df_prediction)

    predictions = model.do_predictions(df_prediction)
    score = model.do_predictions2(df_prediction)

    r2 = ru.calcStats(predictions, df_prediction, None)

    strScore = str("{:.3f}".format(score))

    # result = datasetName+'\t'+splittingName+'\t'+strScore+"\t"+r2
    result = datasetName + '\t' + splittingName + '\t' + r2

    print('***', result, '***')
    # print('timeGA', timeGA)

    if showPlot:
        datasetNameShort = datasetName.replace(" from exp_prop and chemprop", "")
        title = datasetNameShort + '\t' + splittingName
        # TODO fix next line:
        # ru.generatePlot(propertyName=datasetNameShort, title=title, df_prediction=df_prediction, predictions=predictions)

    return result


def caseStudyExpProp():
    '''
    Runs PFAS modeling case studies where we can decide which chemicals to use in training set and then predict for PFAS chemicals
    :return:
    '''

    datasetName = 'HLC from exp_prop and chemprop'
    # datasetName = 'LogP from exp_prop and chemprop'
    # datasetName = 'BP from exp_prop and chemprop'
    # datasetName = 'MP from exp_prop and chemprop'
    # datasetName = 'VP from exp_prop and chemprop'
    # datasetName = 'WS from exp_prop and chemprop'
    # datasetName = 'ExpProp BCF Fish_TMM'
    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False
    version = 1.1
    descriptor_software = 'WebTEST-default'
    folder = '../datasets/' + datasetName + '/'

    # splitting = 'RND_REPRESENTATIVE'
    # splitting = 'OPERA'
    # splitting = 'NOT_IN_OPERA_PREDICTION_SET'
    splitting = 'NOT_IN_OPERA_TRAINING_SET'

    splittingOPERA = 'OPERA'

    useEmbeddings = False

    print(datasetName, splitting)

    training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
    prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

    excelPath = folder + datasetName + '_' + descriptor_software + '_' + splitting + "_prediction.xlsx"

    if splitting == 'NOT_IN_OPERA_PREDICTION_SET':  # still need to use OPERA prediction set:
        prediction_file_name = datasetName + '_' + descriptor_software + '_' + splittingOPERA + "_prediction.tsv"

    if splitting == 'NOT_IN_OPERA_TRAINING_SET':  # still need to use OPERA prediction set:
        training_file_name = datasetName + '_' + descriptor_software + '_' + splittingOPERA + "_training.tsv"

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    # import os
    # absolute_path = os.path.abspath(training_tsv_path)
    # print("Full path: " + absolute_path)

    if not exists(training_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    if not exists(prediction_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    descriptor_names = 'desc1,desc2,desc3'  # set descriptor names manually or call webservice to figure it out

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    print("Ntraining", "Nprediction")
    print(df_training.shape[0], df_prediction.shape[0])

    if splitting == 'NOT_IN_OPERA_PREDICTION_SET' and (
            'HLC' in datasetName or 'WS' in datasetName):  # Opera uses alternate sign for units
        df_prediction['Property'] = df_prediction['Property'].apply(lambda x: x * -1)

    if splitting == 'NOT_IN_OPERA_TRAINING_SET' and (
            'HLC' in datasetName or 'WS' in datasetName):  # Opera uses alternate sign for units
        df_training['Property'] = df_training['Property'].apply(lambda x: x * -1)

    if splitting == 'OPERA' and ('HLC' in datasetName or 'WS' in datasetName):  # Opera uses alternate sign for units
        df_prediction['Property'] = df_prediction['Property'].apply(lambda x: x * -1)
        df_training['Property'] = df_training['Property'].apply(lambda x: x * -1)

    model = buildModel(descriptor_names, df_training, n_threads, remove_log_p_descriptors, useEmbeddings, version)
    # predictions = model.do_predictions(df_prediction)

    predictions = model.do_predictions(df_prediction)
    score = model.do_predictions2(df_prediction)

    datasetNameShort = datasetName.replace(" from exp_prop and chemprop", "")

    if splitting != 'NOT_IN_OPERA_TRAINING_SET':
        title = datasetNameShort + ', TR=' + splitting
    else:
        title = datasetNameShort + ', P=' + splitting

    generatePlot(datasetNameShort, title, df_prediction, None, splitting, predictions)








def buildModel(descriptor_names, df_training, n_threads, remove_log_p_descriptors, useEmbeddings, version):
    if version == 1.1:  # doesnt use max samples
        model = rf1_1.Model(df_training, remove_log_p_descriptors, n_threads)
        model.build_model()

    elif version == 1.2:  # doesnt use max samples
        model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
        model.build_model()
    elif version == 1.3:  # uses max samples
        model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
        if useEmbeddings:
            model.build_model_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model()
    elif version == 1.4:
        model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
        if useEmbeddings:
            model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model_with_grid_search()
    return model


def lookAtOperaResults():
    '''
    Runs PFAS modeling case studies where we can decide which chemicals to use in training set and then predict for PFAS chemicals
    :return:
    '''

    datasetName = 'HLC from exp_prop and chemprop'
    # datasetName = 'LogP from exp_prop and chemprop'
    # datasetName = 'BP from exp_prop and chemprop'
    # datasetName = 'MP from exp_prop and chemprop'
    # datasetName = 'VP from exp_prop and chemprop'
    # datasetName = 'WS from exp_prop and chemprop'
    # datasetName = 'ExpProp BCF Fish_TMM'
    # Parameters needed to build model:

    descriptor_software = 'WebTEST-default'
    folder = '../datasets/' + datasetName + '/'

    splitting = 'NOT_IN_OPERA_TRAINING_SET'
    splittingOPERA = 'OPERA'

    print(datasetName, splitting)

    excelPath = folder + datasetName + '_' + descriptor_software + '_' + splitting + "_prediction.xlsx"

    training_file_name = datasetName + '_' + descriptor_software + '_' + splittingOPERA + "_training.tsv"
    training_tsv_path = folder + training_file_name

    prediction_file_name = datasetName + '_' + descriptor_software + '_NOT_IN_OPERA_TRAINING_SET_prediction.tsv'
    prediction_tsv_path = folder + prediction_file_name

    # import os
    # absolute_path = os.path.abspath(training_tsv_path)
    # print("Full path: " + absolute_path)

    if not exists(training_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    if not exists(prediction_tsv_path):
        print(training_tsv_path + ' doesnt exist')
        return

    # print(prediction_file_name)

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    if splitting == 'NOT_IN_OPERA_TRAINING_SET' and (
            'HLC' in datasetName or 'WS' in datasetName):  # Opera uses alternate sign for units
        df_prediction['Property'] = df_prediction['Property'].apply(lambda x: x * -1)

    # print(df_prediction)

    print("Ntraining", "Nprediction")
    print(df_training.shape[0], df_prediction.shape[0])

    filepathOpera = folder + datasetName + '_exp_prop_prediction-smi_OPERA2.9Pred.csv'
    print(filepathOpera)

    df_opera = pd.read_csv(filepathOpera)

    print(datasetName + '-smi_OPERA2.9Pred.csv')

    yValues = {}
    xValues = {}

    for ind in df_opera.index:
        ID = df_opera['MoleculeID'][ind]
        Pred = df_opera[df_opera.columns[1]][ind]
        print(ID, Pred)
        yValues[ID] = Pred

    for ind in df_prediction.index:
        Exp = df_prediction[df_prediction.columns[1]][ind]
        key = 'C' + str(ind + 1)
        if key in yValues.keys():
            xValues[key] = Exp

    xvals = []
    yvals = []

    for key, value in xValues.items():
        xvals.append(value)
        yvals.append(yValues[key])
        print(value, yValues[key])

    # print(xvals)
    # print(yvals)

    df_preds = pd.DataFrame(yvals, columns=['Prediction'])
    df_exps = pd.DataFrame(xvals, columns=['Property'])

    # df_pred = df_prediction[['ID', 'Property']]
    df_pred = pd.merge(df_exps, df_preds, how='left', left_index=True, right_index=True)

    # print(df_pred)

    # df_pred.to_excel(excelPath, index=False)
    # TODO add graph to excel ?

    # with open(writePath, 'a') as f:
    #     dfAsString = df_pred.to_string(header=True, index=False)
    #     f.write(dfAsString)

    import matplotlib.pyplot as plt
    # a scatter plot comparing num_children and num_pets
    # myplot=df_pred.plot(kind='scatter', x='Property', y='Prediction', color='black')

    m, b, r_value, p_value, std_err = scipy.stats.linregress(df_pred['Property'], df_pred['Prediction'])
    strR2 = '$r^2$=' + str("{:.2f}".format(r_value ** 2))

    fig, ax = plt.subplots()

    datasetNameShort = datasetName.replace(' from exp_prop and chemprop', '')

    if splitting != 'NOT_IN_OPERA_TRAINING_SET':
        plt.title(datasetNameShort + ', TR=' + splitting + ' (' + strR2 + ')')
    else:
        plt.title(datasetNameShort + ', P=' + splitting + ' (' + strR2 + ')')

    plt.xlabel('experimental ' + datasetNameShort)
    plt.ylabel('predicted ' + datasetNameShort)
    ax.scatter(df_pred['Property'], df_pred['Prediction'])
    ax.plot(df_pred['Property'], m * df_pred['Property'] + b)

    # ax.annotate('R2: ' + str("{:.2f}".format(r_value ** 2)), xy=(7, 3))
    # ax.annotate('formula: ' + str("{:.2f}".format(m)) + 'x + ' + str("{:.2f}".format(b)), xy=(7,4))

    fig.show()
    plt.show()


def caseStudyPOD():
    """
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    """

    # endpoint = 'Octanol water partition coefficient'
    endpoint = 'pod'
    # endpoint = strVaporPressure
    # endpoint = strHenrysLawConstant

    version = 1.2
    useEmbeddings = False
    useEmbeddingsJson = False
    generateEmbedding = True

    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    descriptor_software = 'PaDEL_OPERA'

    directory = "../../datasets/pod/"

    training_file_name = endpoint + ' training set.txt'
    prediction_file_name = endpoint + ' test set.txt'

    training_tsv_path = directory + training_file_name
    prediction_tsv_path = directory + prediction_file_name

    if useEmbeddings:
        if generateEmbedding:
            print('todo')
        else:
            descriptor_names = 'xp7 Hmax x0 x2 XLOGP xv0 MDEO11 SsOH Qv SsF SdO GATS5m'
            # print(descriptor_names)

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    print(training_tsv_path)

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    if version == 1.2:  # doesnt use max samples
        model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
        model.build_model()
    elif version == 1.3:  # uses max samples
        model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
        if useEmbeddings:
            model.build_model_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model()
    elif version == 1.4:
        model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
        if useEmbeddings:
            model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model_with_grid_search()

    # TODO store final hyperparameters in the model description...
    # TODO set optimizer to null before returning model to database

    predictions = model.do_predictions(df_prediction)

    predictions = model.do_predictions(df_prediction)

    df_preds = pd.DataFrame(predictions, columns=['Prediction'])
    df_pred = df_prediction[['ID', 'Property']]
    df_pred = pd.merge(df_pred, df_preds, how='left', left_index=True, right_index=True)

    writePath = directory + endpoint + '_RFversion' + str(version) + '.xlsx';
    df_pred.to_excel(writePath, index=False)

    # with open(writePath, 'a') as f:
    #     dfAsString = df_pred.to_string(header=True, index=False)
    #     f.write(dfAsString)

    import matplotlib.pyplot as plt
    # a scatter plot comparing num_children and num_pets
    # myplot=df_pred.plot(kind='scatter', x='Property', y='Prediction', color='black')

    m, b, r_value, p_value, std_err = scipy.stats.linregress(df_pred['Property'], df_pred['Prediction'])

    fig, ax = plt.subplots()
    ax.scatter(df_pred['Property'], df_pred['Prediction'])
    ax.plot(df_pred['Property'], m * df_pred['Property'] + b)
    ax.annotate('R2: ' + str("{:.2f}".format(r_value ** 2)), xy=(7, 3))
    # ax.annotate('formula: ' + str("{:.2f}".format(m)) + 'x + ' + str("{:.2f}".format(b)), xy=(7,4))

    fig.show()
    plt.show()


def caseStudyOPERA():
    '''
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    '''

    # endpoint = 'Octanol water partition coefficient'
    endpoint = strWaterSolubility
    # endpoint = strVaporPressure
    # endpoint = strHenrysLawConstant

    version = 1.3
    useEmbeddings = False
    useEmbeddingsJson = False
    generateEmbedding = True

    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    descriptor_software = 'PaDEL_OPERA'

    folder = '../../datasets/caseStudyOpera'
    training_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_training.tsv'
    prediction_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_prediction.tsv'
    prediction_file_name2 = 'Data from Standard ' + endpoint + ' from exp_prop external to ' + endpoint + ' OPERA_' + descriptor_software + '_full.tsv'

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name
    prediction_tsv_path2 = folder + prediction_file_name2

    if useEmbeddings:

        if generateEmbedding:
            print('todo')
        else:
            if useEmbeddingsJson:
                descriptor_names = loadEmbedding(folder + endpoint + '_embedding.json')
            else:
                if endpoint == strWaterSolubility:
                    descriptor_names = 'SdCH2 k2 BELv5 BELm7 AN BELm2 BELv6 ATS5p SssssC_acnt xc3 BEHm1 BEHm7'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'XLogP	apol	minHsOH	naasC	minHBa	MLFER_A	nHBAcc	maxdNH	MLFER_E	mindNH	MDEO-11'  # OPERA's

                elif endpoint == strVaporPressure:
                    descriptor_names = 'xp7 Hmax x0 x2 XLOGP xv0 MDEO11 SsOH Qv SsF SdO GATS5m'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'MLFER_S	nHBAcc_Lipinski	piPC7	EE_Dt	SHBd	TopoPSA	nHBDon	MLFER_L	MLFER_E	AATSC0v	nssO	MDEC-23'

                elif endpoint == strHenrysLawConstant:
                    descriptor_names = 'Hmax nN SdO Qv Gmax MATS1v xc4 MATS2m Qs SdssC BEHm1 Hmin'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'nHBDon	MLFER_S	GATS1e	ndssC	ATS3m	nHBint6	nHBAcc2	AATSC0i	SpAD_Dzm'

    # print(descriptor_names)

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    if version == 1.2:  # doesnt use max samples
        model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
        model.build_model()
    elif version == 1.3:  # uses max samples
        model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
        if useEmbeddings:
            model.build_model_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model()
    elif version == 1.4:
        model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
        if useEmbeddings:
            model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model_with_grid_search()

    # TODO store final hyperparameters in the model description...
    # TODO set optimizer to null before returning model to database

    predictions = model.do_predictions(df_prediction)

    import os
    isFile = os.path.isfile(prediction_tsv_path2)

    if not os.path.isfile(prediction_tsv_path2):
        print('external pred file missing')
        return

    df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')
    df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera
    predictions2 = model.do_predictions(df_prediction2)

    df_preds2 = pd.DataFrame(predictions2, columns=['Prediction'])
    df_pred2 = df_prediction2[['ID', 'Property']]
    df_pred2 = pd.merge(df_pred2, df_preds2, how='left', left_index=True, right_index=True)

    # a scatter plot comparing num_children and num_pets
    # df_pred2.plot(kind='scatter', x='Property', y='Prediction', color='black')
    # plt.show()

    # print(df_pred2)



def caseStudyOPERA():
    '''
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    '''

    # endpoint = 'Octanol water partition coefficient'
    endpoint = strWaterSolubility
    # endpoint = strVaporPressure
    # endpoint = strHenrysLawConstant

    version = 1.3
    useEmbeddings = False
    useEmbeddingsJson = False
    generateEmbedding = True

    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    descriptor_software = 'PaDEL_OPERA'

    folder = '../../datasets/caseStudyOpera'
    training_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_training.tsv'
    prediction_file_name = endpoint + ' OPERA_' + descriptor_software + '_OPERA_prediction.tsv'
    prediction_file_name2 = 'Data from Standard ' + endpoint + ' from exp_prop external to ' + endpoint + ' OPERA_' + descriptor_software + '_full.tsv'

    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name
    prediction_tsv_path2 = folder + prediction_file_name2

    if useEmbeddings:

        if generateEmbedding:
            print('todo')
        else:
            if useEmbeddingsJson:
                descriptor_names = loadEmbedding(folder + endpoint + '_embedding.json')
            else:
                if endpoint == strWaterSolubility:
                    descriptor_names = 'SdCH2 k2 BELv5 BELm7 AN BELm2 BELv6 ATS5p SssssC_acnt xc3 BEHm1 BEHm7'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'XLogP	apol	minHsOH	naasC	minHBa	MLFER_A	nHBAcc	maxdNH	MLFER_E	mindNH	MDEO-11'  # OPERA's

                elif endpoint == strVaporPressure:
                    descriptor_names = 'xp7 Hmax x0 x2 XLOGP xv0 MDEO11 SsOH Qv SsF SdO GATS5m'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'MLFER_S	nHBAcc_Lipinski	piPC7	EE_Dt	SHBd	TopoPSA	nHBDon	MLFER_L	MLFER_E	AATSC0v	nssO	MDEC-23'

                elif endpoint == strHenrysLawConstant:
                    descriptor_names = 'Hmax nN SdO Qv Gmax MATS1v xc4 MATS2m Qs SdssC BEHm1 Hmin'

                    if descriptor_software == 'PaDEL_OPERA':
                        descriptor_names = 'nHBDon	MLFER_S	GATS1e	ndssC	ATS3m	nHBint6	nHBAcc2	AATSC0i	SpAD_Dzm'

    # print(descriptor_names)

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    if version == 1.2:  # doesnt use max samples
        model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
        model.build_model()
    elif version == 1.3:  # uses max samples
        model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
        if useEmbeddings:
            model.build_model_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model()
    elif version == 1.4:
        model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
        if useEmbeddings:
            model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
        else:
            model.build_model_with_grid_search()

    # TODO store final hyperparameters in the model description...
    # TODO set optimizer to null before returning model to database

    predictions = model.do_predictions(df_prediction)

    import os
    isFile = os.path.isfile(prediction_tsv_path2)

    if not os.path.isfile(prediction_tsv_path2):
        print('external pred file missing')
        return

    df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')
    df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera
    predictions2 = model.do_predictions(df_prediction2)

    df_preds2 = pd.DataFrame(predictions2, columns=['Prediction'])
    df_pred2 = df_prediction2[['ID', 'Property']]
    df_pred2 = pd.merge(df_pred2, df_preds2, how='left', left_index=True, right_index=True)

    # a scatter plot comparing num_children and num_pets
    # df_pred2.plot(kind='scatter', x='Property', y='Prediction', color='black')
    # plt.show()

    # print(df_pred2)


class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        self.embedding_df = pd.read_csv(embedding_filepath, delimiter=",")

    def get_embedding(self, endpoint):
        return eval(str(self.embedding_df.loc[self.embedding_df['Property'] == endpoint]['embedding'].iloc[0]))


def caseStudyOPERA_knn_embedding():
    '''
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    '''

    ei = EmbeddingImporter(
        "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 python\pf_python_modelbuilding/data/knn_ga_embeddings.csv")

    # endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
    #                   "Vapor pressure", "Water solubility", "Boiling point",
    #                   "Melting point", "Octanol water partition coefficient"]
    # endpointsOPERA = ["LogKoa"]
    endpointsOPERA = ["Octanol water partition coefficient"]

    # endpointsOPERA = ["Water solubility"]
    # endpointsOPERA = ["Henry's law constant"]
    # endpointsOPERA = ["Vapor pressure"]

    IDENTIFIER = 'ID'
    PROPERTY = 'Property'

    version = 1.3
    useEmbeddings = True
    useGridSearch = False

    filepath_out = '../data/OPERA rf using knn embeddings.txt'
    f = open(filepath_out, "w")

    mainFolder = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/"

    for endpoint in endpointsOPERA:

        descriptor_software = 'T.E.S.T. 5.1'
        # descriptor_software = 'PaDEL-default'
        # descriptor_software = 'PaDEL_OPERA'

        # folder = '../datasets/caseStudyOpera'

        folder = mainFolder + endpoint + ' OPERA/'

        training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
        prediction_file_name2 = 'Data from Standard ' + endpoint + ' from exp_prop external to ' + endpoint + ' OPERA_' + descriptor_software + '_full.tsv'

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        prediction_tsv_path2 = folder + prediction_file_name2

        if useEmbeddings:
            descriptor_names = ei.get_embedding(endpoint)

            if descriptor_names is None:
                print("missing embedding")
                continue

        # Parameters needed to build model:
        n_threads = 20
        remove_log_p_descriptors = False

        # print (training_tsv_path)

        df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
        df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

        if version == 1.2:  # doesnt use max samples
            model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
            model.build_model()
        elif version == 1.3:  # uses max samples
            model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)

            if useEmbeddings:
                print('1.3 with embeddings')
                print(descriptor_names)
                model.build_model_with_preselected_descriptors(descriptor_names)
            else:
                model.build_model()
        elif version == 1.4:
            model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)

            if useGridSearch:
                if useEmbeddings:
                    model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
                else:
                    model.build_model_with_grid_search()
            else:
                model.build_model()

        # TODO store final hyperparameters in the model description...
        # TODO set optimizer to null before returning model to database

        score = model.do_predictions2(df_prediction)
        f.write(endpoint + "\t" + str(score) + '\n')
        f.flush()

        print('zz Score for Test data = ', score)

        import os
        isFile = os.path.isfile(prediction_tsv_path2)

        if not os.path.isfile(prediction_tsv_path2):
            print('external pred file missing')
            return

        df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')
        df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera
        predictions2 = model.do_predictions(df_prediction2)

        df_preds2 = pd.DataFrame(predictions2, columns=['Prediction'])
        df_pred2 = df_prediction2[['ID', 'Property']]
        df_pred2 = pd.merge(df_pred2, df_preds2, how='left', left_index=True, right_index=True)

        import matplotlib.pyplot as plt
        # a scatter plot comparing num_children and num_pets
        df_pred2.plot(kind='scatter', x='Property', y='Prediction', color='black')
        plt.show()

        # print(df_pred2)
    f.close()


def runOperaSetsWithImportance():
    qsar_method = 'rf'
    # qsar_method = 'xgb'

    # endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
    #                   "Vapor pressure", "Water solubility", "Boiling point",
    #                   "Melting point", "Octanol water partition coefficient"]
    # endpointsOPERA = ["Octanol water partition coefficient"]
    endpointsOPERA = ["Henry's law constant"]
    # endpointsOPERA = ["Water solubility"]


    # descriptor_software = 'T.E.S.T. 5.1'
    descriptor_software = 'WebTEST-default'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    print()

    filename = os.path.join(utils.get_project_root(), 'datasets_benchmark','importance',qsar_method + '.txt')

    # filename = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 python/modeling services/pf_python_modelbuilding/datasets_benchmark/importance/'


    f = open(filename, "w")

    f.write('endpoint\tscore\tlen(embedding)\tembedding)\telapsed_time_seconds\n')

    num_generations = 1
    use_permutative = True
    perform_rfe = True
    min_descriptor_count = 20  # Minimum descriptors we keep from a generation
    max_descriptor_count = 30  # Minimum descriptors we keep from a generation
    n_threads = 20
    use_wards = False


    if qsar_method == 'rf':
        fraction_of_max_importance = 0.25
    elif qsar_method == 'xgb':
        fraction_of_max_importance = 0.03
    else:
        print('invalid method')
        return

    # fraction_of_max_importance = 0.5 # force to hit min desc

    for endpoint in endpointsOPERA:
        start_time = time.time()

        if endpoint == 'Octanol water partition coefficient':
            remove_log_p_descriptors = True
        else:
            remove_log_p_descriptors = False

        df_training, df_prediction = loadOperaDataset(endpoint=endpoint, descriptor_software=descriptor_software,
                                                      remove_log_p_descriptors=remove_log_p_descriptors)

        score, embedding = runSetWithImportance(endpoint=endpoint, df_training=df_training,
                                                df_prediction=df_prediction,
                                                qsar_method=qsar_method,
                                                num_generations=num_generations, use_permutative=use_permutative,
                                                perform_rfe=perform_rfe, n_threads=n_threads,
                                                remove_log_p_descriptors=remove_log_p_descriptors,
                                                min_descriptor_count=min_descriptor_count,
                                                max_descriptor_count=max_descriptor_count,
                                                fraction_of_max_importance=fraction_of_max_importance,use_wards=use_wards)

        elapsed_time = time.time() - start_time

        f.write(endpoint + '\t' + str(score) + '\t' + str(len(embedding)) + '\t' + str(
            embedding) + '\t' + str(elapsed_time) + '\n')

        f.flush()

    f.close()



def runExppropSetsWithImportance():
    qsar_method = 'rf'
    # qsar_method = 'xgb'
    # qsar_method='svm'

    # datasets = ["HLC v1 res_qsar"]
    # datasets = ["VP v1 res_qsar"]
    # datasets = ["WS v1 res_qsar"]

    datasets = ["HLC v1 res_qsar", "WS v1 res_qsar", "VP v1 res_qsar","BP v1 res_qsar","LogP v1 res_qsar","MP v1 res_qsar"]


    # descriptor_software = 'T.E.S.T. 5.1'
    descriptor_software = 'WebTEST-default'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    # splitting = splittingPFASOnly
    # splitting = splittingAllButPFAS
    splitting = splittingRepresentative

    print()

    filename = os.path.join(utils.get_project_root(), 'datasets','importance',qsar_method+'_'+splitting + '.txt')


    f = open(filename, "w")

    f.write('endpoint\tscore\tlen(embedding)\tembedding)\telapsed_time_seconds\n')

    num_generations = 1
    use_permutative = True
    perform_rfe = True
    max_descriptor_count = 30  # Minimum descriptors we keep from a generation
    n_threads = 20
    use_wards = False


    if qsar_method == 'rf':
        fraction_of_max_importance = 0.25
    elif qsar_method == 'xgb':
        fraction_of_max_importance = 0.03
    else:
        fraction_of_max_importance = 0.05
        print('invalid method')
        # return

    # fraction_of_max_importance = 0.5 # force to hit min desc

    for dataset in datasets:


        start_time = time.time()

        if dataset == 'Octanol water partition coefficient':
            remove_log_p_descriptors = True
        else:
            remove_log_p_descriptors = False

        df_training, df_prediction = loadExpPropDataset(dataset=dataset,
                                                        descriptor_software=descriptor_software,
                                                        splitting=splitting)
        n = df_training.shape[0]

        min_descriptor_count = 20  # Minimum descriptors we keep from a generation
        min_descriptor_count_2 = int(n/10)

        if min_descriptor_count_2 < min_descriptor_count:
            min_descriptor_count = min_descriptor_count_2
            # print('Small data set, using min_descriptor_count =',min_descriptor_count_2)


        # if df_training.shape[0] < 100:
        #     max_descriptor_count = int(n/10)
        #     min_descriptor_count = int(max_descriptor_count/2)
        #     print(min_descriptor_count, max_descriptor_count)


        score, embedding = runSetWithImportance(endpoint=dataset, df_training=df_training,
                                                df_prediction=df_prediction,
                                                qsar_method=qsar_method,
                                                num_generations=num_generations, use_permutative=use_permutative,
                                                perform_rfe=perform_rfe, n_threads=n_threads,
                                                remove_log_p_descriptors=remove_log_p_descriptors,
                                                min_descriptor_count=min_descriptor_count,
                                                max_descriptor_count=max_descriptor_count,
                                                fraction_of_max_importance=fraction_of_max_importance,use_wards=use_wards)

        elapsed_time = time.time() - start_time

        f.write(dataset + '\t' + str(score) + '\t' + str(len(embedding)) + '\t' + str(
            embedding) + '\t' + str(elapsed_time) + '\n')

        f.flush()

    f.close()

def runOperaSet():
    qsar_method = 'rf'
    # qsar_method = 'xgb'

    endpoint = "LogBCF"

    if endpoint == 'Octanol water partition coefficient':
        remove_log_p_descriptors = True
    else:
        remove_log_p_descriptors = False

    num_generations = 1
    use_permutative = True
    perform_rfe = True
    min_descriptor_count = 8  # Minimum descriptors we keep from a generation

    start_time = time.time()

    if qsar_method == 'rf':
        fraction_of_max_importance = 0.25
    elif qsar_method == 'xgb':
        fraction_of_max_importance = 0.02
    else:
        print('invalid method')
        return

    fraction_of_max_importance = 0.01

    score, embedding = runSetWithImportance(endpoint=endpoint, qsar_method=qsar_method,
                                            num_generations=num_generations, use_permutative=use_permutative,
                                            perform_rfe=perform_rfe,
                                            remove_log_p_descriptors=remove_log_p_descriptors,
                                            min_descriptor_count=min_descriptor_count,
                                            fraction_of_max_importance=fraction_of_max_importance)

    elapsed_time = time.time() - start_time


def loadOperaDataset(endpoint, descriptor_software, remove_log_p_descriptors):

    folder = os.path.join(utils.get_project_root(), 'datasets_benchmark',endpoint + ' OPERA')

    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

    training_tsv_path = os.path.join(folder, training_file_name)
    prediction_tsv_path = os.path.join(folder, prediction_file_name)

    # print (training_tsv_path)
    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)
    return df_training, df_prediction


def loadExpPropDataset(dataset, descriptor_software, splitting):

    folder = os.path.join(utils.get_project_root(), 'datasets', dataset, 'PFAS')

    training_file_name = dataset + '_' + descriptor_software + '_' + splitting + '_training.tsv'
    prediction_file_name = dataset + '_' + descriptor_software + '_' + splitting + '_prediction.tsv'

    training_tsv_path = os.path.join(folder, training_file_name)
    prediction_tsv_path = os.path.join(folder, prediction_file_name)

    # print (training_tsv_path)
    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    df_training = DFU.filter_columns_in_both_sets(df_training, df_prediction)
    return df_training, df_prediction


def runSetWithImportance(endpoint, df_training, df_prediction, qsar_method, num_generations, use_permutative,
                         perform_rfe, n_threads,
                         remove_log_p_descriptors, min_descriptor_count, max_descriptor_count,
                         fraction_of_max_importance,use_wards):
    '''
    Use importance and RFE to get embedding for RF and XGB mnodels
    :return:
    '''

    print('\n*** endpoint', endpoint)

    vars = {}
    vars['num_generations'] = num_generations
    vars['use_permutative'] = use_permutative
    vars['perform_rfe '] = perform_rfe
    vars['min_descriptor_count'] = min_descriptor_count
    vars['max_descriptor_count'] = max_descriptor_count
    vars['n_threads'] = n_threads
    vars['use_wards'] = use_wards
    print(vars)

    # ***********************************************************************************************
    model = mwu.instantiateModel(df_training, n_threads, qsar_method, remove_log_p_descriptors, False)


    # min_descriptor_count = int(5 * math.log(len(df_training.index)) - 20) # make descriptors a function of dataset size- not needed if use min and max descriptors and RFE
    # if min_descriptor_count < 5:
    #     min_descriptor_count = 5
    # print('min_descriptor_count',min_descriptor_count)


    efi.generateEmbedding(model, df_training, df_prediction, remove_log_p_descriptors=remove_log_p_descriptors,
                          num_generations=num_generations, n_threads=n_threads, use_permutative=use_permutative,
                          fraction_of_max_importance=fraction_of_max_importance,
                          min_descriptor_count=min_descriptor_count, max_descriptor_count=max_descriptor_count, use_wards=use_wards)

    print('After importance calculations, ', len(model.embedding), "descriptors", model.embedding)


    if perform_rfe:
        # efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,n_steps=1)
        # print('After RFE, ', len(model.embedding), "descriptors", model.embedding)

        embedding_old = model.embedding
        while True:  # need to get more agressive (remove 2 at a time) since first RFE didnt remove enough
            efi.perform_recursive_feature_elimination(model=model, df_training=df_training, n_threads=n_threads,
                                                      n_steps=1)
            print('After RFE iteration, ', len(model.embedding), "descriptors", model.embedding)
            if len(model.embedding) == len(embedding_old):
                break
            embedding_old = model.embedding

    # Fit final model using final embedding:
    train_ids, train_labels, train_features, train_column_names = \
        DFU.prepare_instances2(df_training, model.embedding, False)
    model.model_obj.fit(train_features, train_labels)

    model.embedding = train_column_names

    # Run calcs on test set to see how well embedding did:
    print('prediction set results for embedded model:')
    score = model.do_predictions(df_prediction, return_score=True)

    # from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
    # pmml_pipeline = make_pmml_pipeline(model.model_obj)
    # sklearn2pmml(pmml_pipeline, "model.pmml")






    # print (model.get_model_description())

    return score, model.embedding




def caseStudyTEST_knn_embedding():
    '''
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    '''

    # endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'DevTox', 'LLNA', 'Mutagenicity']
    endpointsTEST = ['LC50']
    version = 1.2
    useEmbeddings = False

    dataFolder = '../data/'

    embedding_filename = dataFolder + 'TEST knn results NUM_GENERATIONS=100 NUM_OPTIMIZERS=10 descriptor_coefficient=0.002 max_length=24.txt'

    # filepath_output = dataFolder + 'RF' + str(
    #     version) + ' using knn embeddings TEST sets descriptor_coefficient=0.002 max_length=24.txt'

    filepath_output = dataFolder + 'RF_2024_03_11.txt'


    # embedding_filename = dataFolder+'TEST knn results NUM_GENERATIONS=100 NUM_OPTIMIZERS=10 descriptor_coefficient=0.001 max_length=40.txt'
    # filepath_output = dataFolder+'RF'+str(version) + ' using knn embeddings TEST sets descriptor_coefficient=0.001 max_length=40.txt'

    # embedding_filename = dataFolder+'TEST knn results NUM_GENERATIONS=100 NUM_OPTIMIZERS=10 descriptor_coefficient=0.001 max_length=50.txt'
    # filepath_output = dataFolder+'RF'+str(version) + ' using knn embeddings TEST sets descriptor_coefficient=0.001 max_length=50.txt'

    # embedding_filename = dataFolder+'TEST knn results NUM_GENERATIONS=100 NUM_OPTIMIZERS=10 descriptor_coefficient=0.000 max_length=40.txt'
    # filepath_output = dataFolder+'RF'+str(version) + ' using knn embeddings TEST sets descriptor_coefficient=0.000 max_length=40.txt'

    # embedding_filename = None
    # filepath_output = dataFolder+'RF'+str(version) + ' using knn embeddings TEST sets all descriptors.txt'

    f = open(filepath_output, "w")

    mainFolder = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/"

    f.write('Endpoint\tScore\n')

    # Parameters needed to build model:
    n_threads = 10

    for endpoint in endpointsTEST:

        remove_log_p_descriptors = False
        training_tsv_path = mainFolder + endpoint + '/' + endpoint + '_training_set-2d.csv'
        prediction_tsv_path = mainFolder + endpoint + '/' + endpoint + '_prediction_set-2d.csv'

        if useEmbeddings:
            descriptor_names = getEmbeddingFromGA_text_file(endpoint, embedding_filename)

            if descriptor_names is None:
                print("missing embedding in " + embedding_filename)
                continue
            else:
                print(descriptor_names)

        # print (training_tsv_path)

        df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
        df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

        if version == 1.2:  # doesnt use max samples
            model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
            model.build_model()

        elif version == 1.3:  # uses max samples
            model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)

            if useEmbeddings:
                print('1.3 with embeddings')
                print(descriptor_names)
                model.build_model_with_preselected_descriptors(descriptor_names)
            else:
                model.build_model()
        elif version == 1.4:
            model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
            if useEmbeddings:
                model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
            else:
                model.build_model_with_grid_search()
        else:
            print("Version" + str(version) + " not implemented")
            return

        score = model.do_predictions2(df_prediction)
        f.write(endpoint + "\t" + str(score) + '\n')
        f.flush()

        print('zz Score for Test data = ', score)

        # print(df_pred2)
    f.close()


def caseStudyOPERA_rf_embedding():
    '''
    Train model using OPERA prediction set and predict using OPERA's prediction set and then external set from our postgres
    :return:
    '''

    endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
                      "Vapor pressure", "Water solubility", "Boiling point",
                      "Melting point", "Octanol water partition coefficient"]
    # endpointsOPERA = ["LogKoa"]
    # endpointsOPERA = ["Melting point", "Octanol water partition coefficient"]

    IDENTIFIER = 'ID'
    PROPERTY = 'Property'
    n_threads = 16
    version = 1.3

    go.NUM_GENERATIONS = 10
    go.NUM_OPTIMIZERS = 10

    filename = '../../data/opera RF (RF embedding) results NUM_GENERATIONS=' + str(go.NUM_GENERATIONS) + \
               ' NUM_OPTIMIZERS=' + str(go.NUM_OPTIMIZERS) + '.txt'

    f = open(filename, "w")

    if version == 1.3:  # uses max samples
        dummy_model = rf1_3.Model(None, False, n_threads)
        model_desc = rf1_3.ModelDescription(dummy_model)
    elif version == 1.4:  # uses max samples
        dummy_model = rf1_4.Model(None, False, n_threads, 1)
        model_desc = rf1_4.ModelDescription(dummy_model)
    f.write(model_desc.to_json() + '\n\n')

    f.write('ENDPOINT\tscore\tscore_embed\tlen(features)\tfeatures\tTime(min)\n')
    f.flush()

    folder_datasets = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/"

    for endpoint in endpointsOPERA:

        descriptor_software = 'T.E.S.T. 5.1'
        # descriptor_software = 'PaDEL-default'
        # descriptor_software = 'PaDEL_OPERA'

        # folder = '../datasets/caseStudyOpera'

        folder = folder_datasets + endpoint + ' OPERA/'

        training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
        prediction_file_name2 = 'Data from Standard ' + endpoint + ' from exp_prop external to ' + endpoint + ' OPERA_' + descriptor_software + '_full.tsv'

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        prediction_tsv_path2 = folder + prediction_file_name2

        # Parameters needed to build model:
        remove_log_p_descriptors = False

        # print (training_tsv_path)

        df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
        df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

        if version == 1.2:  # doesnt use max samples
            model = rf1_2.Model(df_training, remove_log_p_descriptors, n_threads)
            model.build_model()

        elif version == 1.3:  # uses max samples
            model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
            ga_model = rf1_3.Model(df_training, False, n_threads)
            ga_model.is_binary = DFU.isBinary(df_training)
            t1 = time.time()
            descriptor_names = go.runGA(df_training, ga_model.getModel())
            t2 = time.time()

            model.build_model_with_preselected_descriptors(descriptor_names)
            score_embed = model.do_predictions2(df_prediction)

            model = rf1_3.Model(df_training, remove_log_p_descriptors, n_threads)
            model.build_model()
            score = model.do_predictions2(df_prediction)  # results with no embedding used


        elif version == 1.4:  # uses grid search
            model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)

            ga_model = rf1_4.Model(df_training, False)
            ga_model.is_binary = DFU.isBinary(df_training)
            t1 = time.time()
            descriptor_names = go.runGA(df_training, ga_model.getModel())
            t2 = time.time()
            model.build_model_with_grid_search_with_preselected_descriptors(descriptor_names)
            score_embed = model.do_predictions2(df_prediction)

            model = rf1_4.Model(df_training, remove_log_p_descriptors, n_threads, 1)
            model.build_model()
            score = model.do_predictions2(df_prediction)  # results with no embedding used

        timeMins = (t2 - t1) / 60

        # TODO store final hyperparameters in the model description...
        # TODO set optimizer to null before returning model to database

        print(endpoint + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(descriptor_names)) + '\t' + str(
            descriptor_names) + '\t' + str(timeMins) + '\n')

        f.write(endpoint + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(descriptor_names)) + '\t' + str(
            descriptor_names) + '\t' + str(timeMins) + '\n')

        f.flush()

        # import os
        # isFile = os.path.isfile(prediction_tsv_path2)
        #
        # if not os.path.isfile(prediction_tsv_path2):
        #     print('external pred file missing')
        #     return
        #
        df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')
        df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera
        predictions2 = model.do_predictions(df_prediction2)

        df_preds2 = pd.DataFrame(predictions2, columns=['Prediction'])
        df_pred2 = df_prediction2[['ID', 'Property']]
        df_pred2 = pd.merge(df_pred2, df_preds2, how='left', left_index=True, right_index=True)

        # a scatter plot comparing num_children and num_pets
        df_pred2.plot(kind='scatter', x='Property', y='Prediction', color='black')
        plt.show()

        # print(df_pred2)
    f.close()


def getEmbeddingFromGA_text_file(endpoint, filename):
    file1 = open(filename, 'r')
    file1.readline()
    file1.readline()
    file1.readline()
    Lines = file1.readlines()
    for line in Lines:
        values = line.split('\t')
        endpointi = values[0]
        features = values[4]

        if endpoint == endpointi:
            descriptor_names = features
            break
    return descriptor_names


if __name__ == "__main__":
    # caseStudyOPERA()

    # loadEmbedding('../datasets/Water solubility_embedding.json')
    # caseStudyOPERA_knn_embedding()
    caseStudyTEST_knn_embedding()
    # caseStudyOPERA_rf_embedding()
    # caseStudyPOD()
    # caseStudyExpProp()
    # runCaseStudyPFAS()
    # runCaseStudiesPFAS()
    # lookAtOperaResults()

    # runOperaSetsWithImportance()
    # runExppropSetsWithImportance()
    # runOperaSet()

    # runOperaSetWithImportance(strHenrysLawConstant,'rf')
