import json
import os
import pandas as pd
import numpy as np

import applicability_domain.applicability_domain_utilities as adu
from models import df_utilities as DFU
import model_ws_utilities as mwu
import df_utilities as dfu

import models.results_utilities as ru
from models import ModelBuilder as mb

import pickle
import csv
import statistics as stats

import os
from models import EmbeddingFromImportance as efi

import matplotlib.pyplot as plt

num_jobs = 4

np.random.seed(seed=42)  # makes results the same each time


class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        if embedding_filepath:
            self.df = pd.read_excel(embedding_filepath, engine='openpyxl')

    def get_embedding(self, dataset_name, num_generations, splitting_name):
        # https://note.nkmk.me/en/python-pandas-multiple-conditions/
        df_and = self.df[(self.df['dataset_name'] == dataset_name)]  # Note need parentheses or doesnt work!
        df_and = df_and[(df_and['splitting_name'] == splitting_name)]  # Note
        df_and = df_and[(df_and['num_generations'] == num_generations)]
        # print(df_and)

        embedding_tsv = str(df_and['embedding_tsv'].iloc[0])

        embedding = embedding_tsv.split('\t')

        return embedding

    def getInhalation_12_descriptors(self):
        return ['TopoPSA', 'ATS4m', 'ATS5m', 'nHBAcc', 'ATS1m', 'IC0', 'TopoPSA(NO)',
                'ATS6m', 'ATS6dv', 'ATS0p', 'WPath', 'ETA_beta']

    def getInhalation_25_descriptors(self):
        return ['TopoPSA', 'ATS4m', 'ATS5m', 'nHBAcc', 'ATS1m', 'IC0', 'TopoPSA(NO)',
                'ATS6m', 'ATS6dv', 'ATS0p', 'WPath', 'ETA_beta', 'MIC1', 'AMW', 'AMID_X',
                'ATSC0m', 'VR2_A', 'GATS1Z', 'MIC0', 'ATS5dv', 'BCUTi-1h', 'AATS0p', 'Sm', 'ATS0m', 'Mp']

    def getInhalation_all_descriptors(self):
        return ['nBase', 'SpMax_A', 'SpMAD_A', 'VR2_A', 'nHetero', 'nH', 'nN', 'nO', 'nS', 'nX', 'ATS1dv',
                'ATS4dv', 'ATS5dv', 'ATS6dv', 'ATS8dv', 'ATS8d', 'ATS1s', 'ATS3s', 'ATS4s', 'ATS5s', 'ATS6s',
                'ATS8s', 'ATS0m', 'ATS1m', 'ATS2m', 'ATS3m', 'ATS4m', 'ATS5m', 'ATS6m', 'ATS7m', 'ATS8m',
                'ATS0p', 'ATS4p', 'ATS4i', 'ATS6i', 'ATS8i', 'AATS0d', 'AATS0s', 'AATS1s', 'AATS0m',
                'AATS1m', 'AATS0v', 'AATS1are', 'AATS0p', 'AATS1p', 'AATS0i', 'AATS1i', 'ATSC0dv', 'ATSC1dv',
                'ATSC2dv', 'ATSC3dv', 'ATSC4dv', 'ATSC5dv', 'ATSC6dv', 'ATSC7dv', 'ATSC8dv', 'ATSC0d',
                'ATSC1d', 'ATSC2d', 'ATSC3d', 'ATSC4d', 'ATSC5d', 'ATSC6d', 'ATSC7d', 'ATSC8d', 'ATSC0s',
                'ATSC1s', 'ATSC2s', 'ATSC3s', 'ATSC4s', 'ATSC5s', 'ATSC6s', 'ATSC7s', 'ATSC8s', 'ATSC7Z',
                'ATSC0m', 'ATSC1m', 'ATSC2m', 'ATSC3m', 'ATSC4m', 'ATSC5m', 'ATSC6m', 'ATSC8m', 'ATSC0v',
                'ATSC1v', 'ATSC2v', 'ATSC3v', 'ATSC4v', 'ATSC5v', 'ATSC6v', 'ATSC7v', 'ATSC8v', 'ATSC2se',
                'ATSC5se', 'ATSC6se', 'ATSC7se', 'ATSC8se', 'ATSC1pe', 'ATSC3pe', 'ATSC4pe', 'ATSC0are',
                'ATSC0p', 'ATSC1p', 'ATSC2p', 'ATSC3p', 'ATSC4p', 'ATSC5p', 'ATSC6p', 'ATSC7p', 'ATSC8p',
                'ATSC0i', 'ATSC1i', 'ATSC2i', 'ATSC3i', 'ATSC4i', 'ATSC5i', 'ATSC6i', 'ATSC7i', 'ATSC8i',
                'AATSC0dv', 'AATSC1dv', 'AATSC0d', 'AATSC1d', 'AATSC0s', 'AATSC1s', 'AATSC0m', 'AATSC1m',
                'AATSC0v', 'AATSC1v', 'AATSC0are', 'AATSC1are', 'AATSC0p', 'AATSC1p', 'AATSC0i', 'AATSC1i',
                'MATS1Z', 'MATS1v', 'MATS1se', 'MATS1are', 'MATS1p', 'MATS1i', 'GATS1Z', 'GATS1v', 'GATS1se',
                'GATS1are', 'GATS1p', 'GATS1i', 'BCUTdv-1h', 'BCUTdv-1l', 'BCUTd-1h', 'BCUTd-1l', 'BCUTs-1h',
                'BCUTs-1l', 'BCUTZ-1h', 'BCUTm-1l', 'BCUTv-1h', 'BCUTv-1l', 'BCUTse-1l', 'BCUTpe-1h',
                'BCUTare-1l', 'BCUTp-1h', 'BCUTi-1h', 'BCUTi-1l', 'BalabanJ', 'SpMAD_DzZ', 'SM1_Dzm',
                'SpDiam_Dzp', 'SpMAD_Dzp', 'SM1_Dzp', 'SpMAD_Dzi', 'SM1_Dzi', 'nBondsS', 'nBondsD', 'C1SP2',
                'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3', 'C3SP3', 'FCSP3', 'Xch-6d', 'Xch-7d', 'Xch-6dv',
                'Xch-7dv', 'Xc-3d', 'Xc-4d', 'Xc-5d', 'Xc-3dv', 'Xc-4dv', 'Xc-5dv', 'Xc-6dv', 'Xpc-4d',
                'Xpc-6d', 'Xpc-4dv', 'Xpc-6dv', 'AXp-0d', 'AXp-1d', 'Xp-2dv', 'Xp-3dv', 'Xp-4dv', 'Xp-5dv',
                'Xp-6dv', 'AXp-0dv', 'AXp-1dv', 'Sm', 'Mp', 'NssCH2', 'NdsCH', 'NsssCH', 'NdssC', 'NaasC',
                'NssssC', 'NssNH', 'NsOH', 'NdO', 'NssO', 'NdsssP', 'NdS', 'NssS', 'SsCH3', 'SdCH2',
                'SssCH2', 'SsssCH', 'SdssC', 'SaasC', 'SssssC', 'SsssN', 'SsF', 'SsCl', 'AETA_alpha',
                'ETA_shape_p', 'ETA_shape_y', 'ETA_shape_x', 'ETA_beta', 'AETA_beta_s', 'ETA_beta_ns_d',
                'AETA_eta', 'ETA_eta_L', 'AETA_eta_L', 'ETA_epsilon_5', 'ETA_dEpsilon_B', 'ETA_dEpsilon_D',
                'ETA_dBeta', 'AETA_dBeta', 'ETA_psi_1', 'ETA_dPsi_A', 'fragCpx', 'fMF', 'nHBAcc', 'nHBDon',
                'IC0', 'IC1', 'IC2', 'TIC2', 'SIC0', 'SIC1', 'SIC2', 'BIC3', 'CIC0', 'CIC1', 'CIC2', 'CIC5',
                'MIC0', 'MIC1', 'ZMIC5', 'GhoseFilter', 'FilterItLogS', 'PEOE_VSA1', 'PEOE_VSA2',
                'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'PEOE_VSA10',
                'SMR_VSA1', 'SMR_VSA3', 'SMR_VSA5', 'SMR_VSA6', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA5',
                'SlogP_VSA6', 'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
                'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10', 'VSA_EState1',
                'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
                'VSA_EState8', 'VSA_EState9', 'AMID_h', 'MID_C', 'AMID_N', 'AMID_O', 'AMID_X', 'MPC10',
                'piPC3', 'piPC4', 'piPC5', 'piPC7', 'piPC8', 'piPC10', 'TpiPC10', 'bpol', 'nRing', 'n5Ring',
                'n6Ring', 'nHRing', 'n5HRing', 'n6HRing', 'naRing', 'naHRing', 'nARing', 'nAHRing', 'nFRing',
                'nRot', 'RotRatio', 'SLogP', 'TopoPSA(NO)', 'TopoPSA', 'GGI1', 'GGI4', 'GGI5', 'GGI6',
                'GGI7', 'GGI8', 'JGI1', 'JGI2', 'JGI3', 'JGI4', 'JGI5', 'JGI6', 'JGI7', 'JGI8', 'JGI9',
                'JGT10', 'Radius', 'TopoShapeIndex', 'SRW05', 'TSRW10', 'AMW', 'WPath', 'mZagreb1']


def a_runCaseStudiesExpPropPFAS():
    inputFolder = '../datasets/'

    useEmbeddings = False
    num_generations = 100

    qsar_method = 'rf'
    # qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'

    splitting = 'RND_REPRESENTATIVE'
    # splitting = 'T=PFAS only, P=PFAS'
    # splitting = 'T=all, P=PFAS'
    # splitting = 'T=all but PFAS, P=PFAS'

    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    datasetNames.append("WS from exp_prop and chemprop")
    datasetNames.append("VP from exp_prop and chemprop")
    # datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    # ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

    fileOut = inputFolder + 'results2/' + qsar_method + '_' + splitting + '_useEmbeddings=' + str(
        useEmbeddings) + '.txt'
    print('output file=', fileOut)

    f = open(fileOut, "w")
    f.write('dataset\tR2\tMAE\n')

    for datasetName in datasetNames:

        if 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
            num_generations = 10

        run_dataset(datasetName, descriptor_software, f, inputFolder, num_generations, qsar_method, splitting,
                    useEmbeddings)

    f.close()


def runTrainingPredictionExample():
    descriptor_software = 'WebTEST-default'
    # datasetName = 'exp_prop_96HR_RT_LC50_v4 modeling'
    # datasetName = 'exp_prop_96HR_FHM_LC50_v4 modeling'
    # datasetName = 'exp_prop_96HR_BG_LC50_v4 modeling'
    # datasetName = 'exp_prop_96HR_BG_LC50_v4 modeling'
    datasetName = 'BP v1 modeling'

    np.random.seed(seed=42)  # makes results the same each time

    mp = model_parameters(property_name=datasetName, property_units="N/A", descriptor_set=descriptor_software)

    # mp.useEmbeddings = True
    mp.useEmbeddings = False

    mp.n_threads = 8

    mp.qsar_method = 'rf'
    # mp.qsar_method = 'las'
    # mp.qsar_method = 'knn'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'svm'
    # mp.qsar_method = 'reg'
    # mp.qsar_method = 'las'

    mp.remove_log_p = False
    if 'LogP' in datasetName:
        mp.remove_log_p = True

    inputFolder = '../datasets_exp_prop/'
    training_file_name = datasetName + "_" + descriptor_software + "_RND_REPRESENTATIVE_training.tsv"
    prediction_file_name = datasetName + "_" + descriptor_software + "_RND_REPRESENTATIVE_prediction.tsv"

    training_tsv_path = inputFolder + datasetName + '/' + training_file_name
    prediction_tsv_path = inputFolder + datasetName + '/' + prediction_file_name

    training_tsv = dfu.read_file_to_string(training_tsv_path)
    prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

    embeddings = []

    if mp.useEmbeddings:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
        print('embedding', embedding)
        print('timeMin', timeMin)
        embeddings.append(embedding)
    else:
        embedding = None

    model = buildModel(embedding, mp, training_tsv, prediction_tsv)

    print('best parameters', model.hyperparameters)

    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    df_results_training = mwu.call_do_predictions_to_df(training_tsv, model)
    exp_training = df_results_training['exp'].to_numpy()
    pred_training = df_results_training['pred'].to_numpy()

    df_results_prediction = mwu.call_do_predictions_to_df(prediction_tsv, model)
    exp_prediction = df_results_prediction['exp'].to_numpy()
    pred_prediction = df_results_prediction['pred'].to_numpy()

    # resultsFolder = inputFolder + "results/"
    # fileOut = resultsFolder + datasetName + "_" + mp.qsar_method + "_embedding=" + str(
    # mp.useEmbeddings) + ".csv"

    fileOut = inputFolder + datasetName + "/" + datasetName + "_" + mp.descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_prediction.to_csv(fileOut, index=False)

    results = save_json_file(df_results_all=df_results_prediction, embeddings=embeddings, fileOut=fileOut, mp=mp,
                             hyperparameters=model.hyperparameters, hyperparameter_grid=model.hyperparameter_grid)

    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()

    MAE = ru.calc_MAE(pred, exp)
    strMAE = str("{:.3f}".format(MAE))

    R2 = ru.calc_pearson_r2(pred, exp)
    strR2 = str("{:.3f}".format(R2))

    print('*****************************\n' + datasetName + '\tR2=' + strR2 + '\tMAE=' + strMAE + '\n')

    figtitle = datasetName + "_" + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    title = descriptor_software + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generateTrainingPredictionPlot(fileOut=fileOut, property_name=datasetName, title=title, figtitle=figtitle,
                                      exp_training=list(exp_training), pred_training=list(pred_training),
                                      exp_prediction=list(exp_prediction), pred_prediction=list(pred_prediction))

    lookAtResults(inputFolder + datasetName)


def run_dataset(datasetName, descriptor_software, f, inputFolder, num_generations, qsar_method, splitting,
                useEmbeddings):
    # print (splitting)

    training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
    prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

    if 'PFAS' in splitting:
        training_tsv_path = inputFolder + datasetName + '/PFAS/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/PFAS/' + prediction_file_name
    else:
        training_tsv_path = inputFolder + datasetName + '/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/' + prediction_file_name

    training_tsv = dfu.read_file_to_string(training_tsv_path)
    prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

    mp = model_parameters(property_name=datasetName, property_units="N/A", descriptor_set=descriptor_software)
    mp.useEmbeddings = useEmbeddings
    mp.num_generations_ga = num_generations
    mp.remove_log_p = False
    if 'LogP' in datasetName:
        mp.remove_log_p = True

    if mp.useEmbeddings:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
        print('embedding', embedding)
        print('timeMin', timeMin)
    else:
        embedding = None

    model = buildModel(embedding, mp, training_tsv, prediction_tsv)
    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()

    MAE = ru.calc_MAE(pred, exp)
    strMAE = str("{:.3f}".format(MAE))

    R2 = ru.calc_pearson_r2(pred, exp)
    strR2 = str("{:.3f}".format(R2))

    # title = descriptor_software + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    # ru.generatePlot(property_name=datasetName, title=title, exp=list(exp), pred=list(pred))

    print('*****' + datasetName + '\t' + strR2 + '\t' + strMAE + '\n')
    f.write(datasetName + '\t' + strR2 + '\t' + strMAE + '\n')
    f.flush()


def calcStats(predictions, df_prediction, excelPath):
    df_preds = pd.DataFrame(predictions, columns=['Prediction'])
    df_pred = df_prediction[['ID', 'Property']]
    df_pred = pd.merge(df_pred, df_preds, how='left', left_index=True, right_index=True)

    if excelPath:
        df_pred.to_excel(excelPath, index=False)

    # a scatter plot comparing num_children and num_pets
    # myplot=df_pred.plot(kind='scatter', x='Property', y='Prediction', color='black')
    m, b, r_value, p_value, std_err = scipy.stats.linregress(df_pred['Property'], df_pred['Prediction'])
    strR2 = str("{:.3f}".format(r_value ** 2))

    y_true, predictions = np.array(df_pred['Property']), np.array(df_pred['Prediction'])

    # print (y_true)
    # print(predictions)

    MAE = np.mean(np.abs(y_true - predictions))
    strMAE = str("{:.3f}".format(MAE))
    # print(strR2,MAE)

    return strR2, strMAE


def compare_spaces():
    import numpy as np
    gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]  # [0.01]

    gamma_space2 = list([10 ** x for x in range(-3, 4)])
    print('gamma_space', gamma_space)
    print('gamma_space2', gamma_space2)

    c_space = np.arange(-3, 4, 0.5)  # wrong has negative values
    c_space2 = list([10 ** x for x in range(-3, 4)])
    c_space3 = np.logspace(-1, 1, 10)

    # print('c_space',c_space)
    print('c_space2', c_space2)
    print('c_space3', c_space3)


def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, descriptor_names_tsv,
                                                  n_jobs):
    """Loads TSV training data into a pandas DF and calls the appropriate training method"""

    df_training = dfu.load_df(training_tsv)
    qsar_method = qsar_method.lower()

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

    # Returns trained model

    model.build_model(descriptor_names=descriptor_names_tsv)
    return model


def assembleResults():
    splittings = ['T=PFAS only, P=PFAS', 'T=all, P=PFAS', 'T=all but PFAS, P=PFAS', 'T=all, P=all']
    useEmbeddingsArray = [False, True]
    # qsar_method = 'rf'
    # qsar_method = 'knn'
    # qsar_method = 'xgb'
    qsar_method = 'svm'
    inputFolder = '../datasets/'
    resultsName = 'results'

    import os.path

    dfnew = pd.DataFrame()

    for useEmbeddings in useEmbeddingsArray:
        for splitting in splittings:
            filepath = inputFolder + resultsName + '/' + qsar_method + '_' + splitting + '_useEmbeddings=' + str(
                useEmbeddings) + '.txt'

            if not os.path.isfile(filepath):
                print('missing:' + filepath)
                continue

            df = pd.read_csv(filepath, delimiter='\t')

            if 'datasetName' not in dfnew:
                dfnew['dataset'] = df['dataset']

            newName = splitting + '_useEmbeddings=' + str(useEmbeddings)
            dfnew[newName] = df['MAE']

    print(dfnew)

    dfnew.to_csv(inputFolder + resultsName + '/' + qsar_method + '.csv', index=False)


class model_parameters:
    def __init__(self, property_name, property_units, descriptor_set):
        self.property_name = property_name
        self.property_units = property_units
        self.descriptor_set = descriptor_set

        self.dataset_name = None
        self.is_binary = False

        self.qsar_method = 'rf'
        self.useEmbeddings = True
        # self.useEmbeddings = False

        self.use_wards = False
        self.use_pmml = False
        self.include_standardization_in_pmml = True

        # Permutative importance params:
        self.use_permutative = True
        self.run_rfe = True
        self.min_descriptor_count = 20
        self.max_descriptor_count = 30
        self.n_threads = 20
        self.num_generations_pi = 1

        # Genetic algorithm params:
        self.num_generations_ga = 100
        self.num_optimizers = 10
        self.max_length = 24
        self.descriptor_coefficient = 0.002
        self.threshold = 1
        self.remove_log_p = False


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


def save_json_file(df_results_all, embeddings, fileOut, mp, hyperparameters=None, hyperparameter_grid=None, r2s=None,
                   rmses=None, statistics_AD=None):
    exp = df_results_all['exp'].to_numpy()
    pred = df_results_all['pred'].to_numpy()

    results = {}

    if mp.is_binary:
        # BA = ru.calc_BA(pred2,exp2)
        print('\nentire prediction set:')
        BA = ru.calc_BA(pred, exp)
        strBA = str("{:.3f}".format(BA))

        print(mp.dataset_name + '_' + mp.descriptor_set + '_' + mp.qsar_method + '_' + 'embedding=' + str(
            mp.useEmbeddings) + '\t' + strBA + '\n')

        statistics = {"BA": BA}

    else:
        RMSE = ru.calc_rmse(pred, exp)
        strRMSE = str("{:.3f}".format(RMSE))

        MAE = ru.calc_MAE(pred, exp)
        strMAE = str("{:.3f}".format(MAE))

        R2 = ru.calc_pearson_r2(pred, exp)
        strR2 = str("{:.3f}".format(R2))

        statistics = {"rmse": RMSE, "r2": R2, "MAE": MAE}

        print(mp.dataset_name + '_' + mp.descriptor_set + '_' + mp.qsar_method + '_' + 'embedding=' + str(
            mp.useEmbeddings) + '\t' + strR2 + '\t' + strMAE + '\n')

    results["model_parameters"] = mp.__dict__
    results["statistics"] = statistics
    results["embeddings"] = embeddings

    if hyperparameter_grid is not None:
        results["hyperparameter_grid"] = hyperparameter_grid

    if hyperparameters is not None:
        results["hyperparameters"] = hyperparameters

    if r2s is not None:
        statistics["r2s"] = r2s
        statistics["rmses"] = rmses

        statistics["r2_mean"] = stats.mean(r2s)
        statistics["rmse_mean"] = stats.mean(rmses)

    if statistics_AD is not None:
        statistics['statistics_AD'] = statistics_AD

    fileOutJson = fileOut.replace(".csv", ".json")
    print(fileOutJson)
    with open(fileOutJson, 'w') as myfile:
        myfile.write(json.dumps(results, indent=4))

    print(json.dumps(results))
    print(statistics)

    return results


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

    # mp.qsar_method = 'rf'
    # mp.qsar_method = 'xgb'
    # mp.qsar_method = 'knn'
    mp.qsar_method = 'svm'
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
        mp.useEmbeddings) + "_blind.csv"

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

    # mp.qsar_method = 'rf'
    mp.qsar_method = 'svm'

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


def buildModel(embedding, mp, training_tsv, prediction_tsv, filterColumnsInBothSets=True):
    model = mwu.call_build_model_with_preselected_descriptors(qsar_method=mp.qsar_method,
                                                              training_tsv=training_tsv, prediction_tsv=prediction_tsv,
                                                              remove_log_p=mp.remove_log_p,
                                                              use_pmml_pipeline=mp.use_pmml,
                                                              include_standardization_in_pmml=mp.include_standardization_in_pmml,
                                                              descriptor_names_tsv=embedding,
                                                              n_jobs=mp.n_threads,
                                                              filterColumnsInBothSets=filterColumnsInBothSets)
    return model


def getEmbedding(mp, prediction_tsv, training_tsv):
    if mp.qsar_method == 'rf' or mp.qsar_method == 'xgb':
        if mp.qsar_method == 'rf':
            fraction_of_max_importance = 0.25;
        elif mp.qsar_method == 'xgb':
            fraction_of_max_importance = 0.03;

        embedding, timeMin = mwu.call_build_embedding_importance(qsar_method=mp.qsar_method,
                                                                 training_tsv=training_tsv,
                                                                 prediction_tsv=prediction_tsv,
                                                                 remove_log_p_descriptors=mp.remove_log_p,
                                                                 n_threads=mp.n_threads,
                                                                 num_generations=mp.num_generations_pi,
                                                                 use_permutative=mp.use_permutative,
                                                                 run_rfe=mp.run_rfe,
                                                                 fraction_of_max_importance=fraction_of_max_importance,
                                                                 min_descriptor_count=mp.min_descriptor_count,
                                                                 max_descriptor_count=mp.max_descriptor_count,
                                                                 use_wards=mp.use_wards)


    elif mp.qsar_method == 'knn' or mp.qsar_method == 'reg':

        embedding, timeMin = mwu.call_build_embedding_ga(qsar_method=mp.qsar_method,
                                                         training_tsv=training_tsv,
                                                         prediction_tsv=prediction_tsv,
                                                         remove_log_p=mp.remove_log_p,
                                                         num_generations=mp.num_generations_ga,
                                                         num_optimizers=mp.num_optimizers,
                                                         num_jobs=num_jobs, n_threads=mp.n_threads,
                                                         descriptor_coefficient=mp.descriptor_coefficient,
                                                         max_length=mp.max_length,
                                                         threshold=mp.threshold,
                                                         use_wards=mp.use_wards,
                                                         run_rfe=False)
    elif mp.qsar_method == 'las':

        embedding, timeMin = mwu.call_build_embedding_lasso(qsar_method=mp.qsar_method,
                                                            training_tsv=training_tsv,
                                                            prediction_tsv=prediction_tsv,
                                                            remove_log_p_descriptors=mp.remove_log_p,
                                                            n_threads=mp.n_threads, run_rfe=True)

    return embedding, timeMin


def lookAtResults(folder, useMean=False, displayAD=False, isBinary=False):
    if isBinary:
        header = ["dataset_name", "property_units", "descriptor_set", "qsar_method", "useEmbeddings", "BA"]
    else:
        if useMean:
            header = ["dataset_name", "property_units", "descriptor_set", "qsar_method", "useEmbeddings", "r2_mean",
                      "rmse_mean"]
        else:
            header = ["dataset_name", "property_units", "descriptor_set", "qsar_method", "useEmbeddings", "r2_pooled",
                      "rmse_pooled"]

    if displayAD:
        if isBinary:
            header.append('BA_inside')
            header.append('BA_outside')
        else:
            header.append('rmse_inside')
            header.append('rmse_outside')
        header.append('cov')

    print('\n')
    print(header)

    with open(folder + '/results.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(header)

        bestRow = None

        if isBinary:
            bestStat = 0
        else:
            bestStat = 99999999

        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            # checking if it is a file
            if ".json" not in filename:
                continue

            f = open(filepath)
            results = json.load(f)
            # addMissingMetadataJsonFile(filename, filepath, results)
            # print (filename, results)
            # print (filename, results)
            bestRow, bestStat, resultRow = get_stats_from_json(bestStat, bestRow, displayAD, isBinary, results, useMean)

            print(resultRow, sep="\t")
            writer.writerow(resultRow)

        print('\nBest:')
        print(bestRow, sep="\t")

        best = ["Best:"]
        writer.writerow([])
        writer.writerow(best)
        writer.writerow(bestRow)


def get_stats_from_json(bestStat, bestRow, displayAD, isBinary, results, useMean):
    mp = results["model_parameters"]
    stats = results["statistics"]

    resultRow = [mp["dataset_name"], mp["property_units"], mp["descriptor_set"], mp["qsar_method"], mp["useEmbeddings"]]

    if useMean:
        if "r2_mean" not in stats:
            print("missing r2_mean for", mp["property_units"], mp["descriptor_set"], mp["qsar_method"],
                  mp["useEmbeddings"])
            resultRow.append('missing')
            resultRow.append('missing')
        else:
            resultRow.append(round(stats["r2_mean"], 3))
            resultRow.append(round(stats["rmse_mean"], 3))

            if stats["rmse_mean"] < bestStat:
                bestStat = stats["rmse_mean"]
                bestRow = resultRow
    else:

        if isBinary:
            resultRow.append(round(stats["BA"], 3))
        else:
            resultRow.append(round(stats["r2"], 3))
            resultRow.append(round(stats["rmse"], 3))

        if displayAD:
            statsAD = stats["statistics_AD"]

            if isBinary:
                resultRow.append(round(statsAD['BA_inside'], 3))
                resultRow.append(round(statsAD['BA_outside'], 3))
            else:
                resultRow.append(round(statsAD['rmse_inside'], 3))
                resultRow.append(round(statsAD['rmse_outside'], 3))

            resultRow.append(round(statsAD['cov'], 3))

    # print (bestStat,stats["rmse"])

    if not isBinary and stats["rmse"] < bestStat:
        bestRow = resultRow
        bestStat = stats["rmse"]
    elif isBinary and stats["BA"] > bestStat:
        bestRow = resultRow
        bestStat = stats["BA"]

    return bestRow, bestStat, resultRow


def lookAtResults2(dataset, folder):
    # print (folder)

    # print("\n\nproperty_units", "descriptor_set", "qsar_method", "useEmbeddings", "pearson_r2", "rmse", sep="\t")

    header = ["property_units", "descriptor_set", "qsar_method", "useEmbeddings", "r2", "rmse"]

    property_units = []
    descriptor_set = []
    qsar_method = []
    useEmbeddings = []
    r2 = []
    rmse = []
    datasets = []
    composite = []

    # print(dataset)

    for filename in os.listdir(folder):

        # print(type(filename))
        filepath = folder + '/' + filename

        if ".json" not in filename:
            continue

        # print('\t',filename)

        # print(filepath)
        # print(str(filename))

        f = open(filepath)
        results = json.load(f)
        # addMissingMetadataJsonFile(filename, filepath, results)
        # print (filename, results)

        # print (filename, results)
        mp = results["model_parameters"]
        stats = results["statistics"]

        datasets.append(dataset)
        property_units.append(mp["property_units"])
        descriptor_set.append(mp["descriptor_set"])
        qsar_method.append(mp["qsar_method"])
        useEmbeddings.append(mp["useEmbeddings"])
        r2.append(stats["r2"])
        rmse.append(round(stats["rmse"], 3))
        composite.append(str(mp['descriptor_set']) + '_' + str(mp['qsar_method']) + '_' + str(mp['useEmbeddings']))

        # if results["embeddings"] is not None:
        #     print(filename, mp["qsar_method"], stats['r2'],len(results["embeddings"][0].split(',')))

        if len(results["embeddings"]) > 0 and mp["qsar_method"] == 'las':
            print(filename, mp["qsar_method"], stats['r2'], len(results["embeddings"]))

    dict = {'dataset': datasets, 'property_units': property_units, 'descriptor_set': descriptor_set,
            'qsar_method': qsar_method,
            'useEmbeddings': useEmbeddings, 'r2': r2, 'rmse': rmse, 'composite': composite}

    df = pd.DataFrame(dict)
    df = df[df['dataset'].str.contains('v4', regex=True)]

    return df


def lookAtEmbeddings(folder):
    print("\n\nproperty_units", "descriptor_set", "qsar_method", "embeddingLength", sep="\t")

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)
        # checking if it is a file

        if ".json" not in filename:
            continue

        f = open(filepath)
        results = json.load(f)

        # print (filename, results)
        mp = results["model_parameters"]

        if not mp["useEmbeddings"]:
            continue

        len1 = len(results['embeddings'][0])
        len2 = len(results['embeddings'][1])
        len3 = len(results['embeddings'][2])
        len4 = len(results['embeddings'][3])
        len5 = len(results['embeddings'][4])
        lenAvg = (len1 + len2 + len3 + len4 + len5) / 5

        print(mp["property_units"], mp["descriptor_set"], mp["qsar_method"], lenAvg, sep="\t")


def addMissingMetadataJsonFile(filename, filepath, results):
    mp = results["model_parameters"]
    # Add property name and units to the json file for files that dont have them:
    mp["property_name"] = "4 hour rat inhalation LC50"
    if "ppm" in filename:
        mp["property_units"] = "4 hour rat inhalation LC50"
    if 'ppm' in filename:
        mp["property_units"] = 'log10 ppm'
    elif 'mgL' in filename:
        mp["property_units"] = 'log10 mg/L'
    if "webtest_opera" in filename:
        mp["descriptor_set"] = "webtest_opera"
    elif "webtest" in filename:
        mp["descriptor_set"] = "webtest"
    else:
        print("missing descriptor set in " + filename)
    with open(filepath, 'w') as myfile:
        myfile.write(json.dumps(results))


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

    print('adMeasure=', adMeasure)
    print('doAD=', doAD)

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'padel'
    # descriptor_set = 'webtest'
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

def get_descriptors_dataframe(descriptor_file_training, descriptor_set):
    df_descriptors = dfu.load_df_from_file(descriptor_file_training)
    df_descriptors = df_descriptors.replace('null', np.nan).replace('{}', np.nan)
    if 'webtest' in descriptor_set:
        df_descriptors = df_descriptors[df_descriptors.x0 != 'Error']  # remove bad rows

    df_descriptors.drop_duplicates(inplace=True)

    return df_descriptors


def calc_stats(df_results):
    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()
    rmse = ru.calc_rmse(pred, exp)
    r2 = ru.calc_pearson_r2(pred, exp)
    return r2, rmse


def calc_binary_stats(df_results):
    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()
    BA = ru.calc_BA(pred, exp)
    return BA


def ttr_binding():
    """
    runs all the calculations for ttr_binding
    :return:
    """
    a_runCaseStudiesTTR_Binding_CV_merge_on_fly()
    # a_runCaseStudiesTTR_Binding_training_prediction()
    # a_runCaseStudiesTTR_Binding_special_chemicals()
    # a_runCaseStudiesTTR_Binding_special_chemicals2()

    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly_with_leaderboard()
    # a_runCaseStudiesTTR_Binding_training_prediction_with_leaderboard()
    # a_runCaseStudiesTTR_Binding_special_chemicals2_with_leaderboard()
    # a_runCaseStudiesTTR_Binding_special_chemicals()

    # a_runCaseStudiesTTR_Binding_CV()
    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly_predict_AQC_fails()

    resultsFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/results/'
    # lookAtResults(resultsFolder,useMean=True)
    # lookAtEmbeddings(resultsFolder)


def inhalation_lc50():
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


def runModelOptions():
    np.random.seed(seed=42)
    property_name = "LC50"
    useModelFiles = False

    # datasetName = 'BP v1 modeling'
    # datasetName = 'HLC v1 modeling'
    datasetName = 'exp_prop_96HR_FHM_LC50_v4 modeling'
    # datasetName = 'exp_prop_96HR_BG_LC50_v4 modeling'
    # datasetName = 'exp_prop_96HR_RT_LC50_v4 modeling'
    # datasetName = 'exp_prop_48HR_DM_LC50_v4 modeling'
    # datasetName = 'LD50 TEST'
    is_binary = False

    # datasetName = 'Mutagenicity TEST'
    # datasetName = 'LLNA TEST'
    # is_binary = True

    useAD = True
    adMeasure = adu.strTESTApplicabilityDomainEmbeddingEuclidean

    units = '-log(M)'

    # descriptor_sets = ['WebTEST-default', 'Mordred-default', 'PaDel-default']
    # descriptor_sets = ['WebTEST-default', 'ToxPrints-default','RDKit-default', 'PaDel-default']
    descriptor_sets = ['ToxPrints-default']
    # descriptor_sets = ['WebTEST-default']

    # qsar_methods = ['rf', 'xgb']
    qsar_methods = ['rf', 'knn']
    # qsar_methods = ['las']
    # qsar_methods = ['rf']

    useEmbeddings = ['True', 'False']
    # useEmbeddings = ['True']
    # useEmbeddings = ['False']

    # inputFolder = 'C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets_exp_prop/'
    # inputFolder = 'C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets/'
    # inputFolder = 'C:/Users/tmarti02/OneDrive - Environmental Protection Agency (EPA)/0 Python/modeling services/pf_python_modelbuilding/datasets/'
    # inputFolder = '../datasets/' #relative path should work

    if 'exp_prop' in datasetName:
        inputFolder = '../datasets_exp_prop/'  # relative path should work
    else:
        inputFolder = '../datasets_benchmark_TEST/'  # relative path should work

    # resultsFolder = inputFolder + datasetName + "/results/"
    # resultsFolder = inputFolder + datasetName + "/results_expanded_grid/"
    # resultsFolder = inputFolder + datasetName + "/las_results_linspace/"
    # resultsFolder = inputFolder + datasetName + "/results_expanded_las_rounded/"
    resultsFolder = inputFolder + datasetName + "/results_setac_2024/"
    # resultsFolder = inputFolder + datasetName + "/results_las_optimized_tol/"

    if not os.path.exists(resultsFolder):
        os.mkdir(resultsFolder)

    for descriptor_set in descriptor_sets:

        if 'datasets_benchmark_TEST' in inputFolder:
            training_file_name = datasetName + " " + descriptor_set + " training.tsv"
            prediction_file_name = datasetName + " " + descriptor_set + " prediction.tsv"
        else:
            training_file_name = datasetName + "_" + descriptor_set + "_RND_REPRESENTATIVE_training.tsv"
            prediction_file_name = datasetName + "_" + descriptor_set + "_RND_REPRESENTATIVE_prediction.tsv"

        training_tsv_path = inputFolder + datasetName + '/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/' + prediction_file_name

        print(training_tsv_path)

        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        for qsar_method in qsar_methods:

            mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)
            mp.qsar_method = qsar_method
            mp.n_threads = 16  # lower this number if have high CPU usage
            mp.dataset_name = datasetName
            mp.is_binary = is_binary

            for strUseEmbeddings in useEmbeddings:

                if (qsar_method == 'reg') and (strUseEmbeddings == 'False'):
                    continue
                if (qsar_method == 'svm') and (strUseEmbeddings == 'True'):
                    continue
                # if (qsar_method == 'las') and (z == 'True'):
                #     continue

                if strUseEmbeddings == 'True':
                    mp.useEmbeddings = True
                else:
                    mp.useEmbeddings = False

                run_set_of_options(mp=mp, prediction_tsv=prediction_tsv, training_tsv=training_tsv,
                                   inputFolder=inputFolder, resultsFolder=resultsFolder,
                                   useModelFiles=useModelFiles, useAD=useAD, adMeasure=adMeasure)

    lookAtResults(folder=resultsFolder, useMean=False, displayAD=useAD, isBinary=is_binary)


def run_set_of_options(mp, prediction_tsv, training_tsv, inputFolder, resultsFolder, useModelFiles, useAD, adMeasure):
    if mp.useEmbeddings and not useModelFiles:
        embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
        print('embedding', embedding)
        print('timeMin', timeMin)
    else:
        embedding = None

    # picklepath = 'model'+str(fold)+'.pickle'

    picklepath = inputFolder + mp.dataset_name + '/' + 'model' + "_" + mp.descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".pickle"
    print('model path', picklepath)

    if useModelFiles:
        with open(picklepath, 'rb') as handle:
            model = pickle.load(handle)
    else:
        model = buildModel(embedding, mp, training_tsv, prediction_tsv)
        with open(picklepath, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Simpler code w/o saving model to file:
    # if mp.useEmbeddings:
    #     embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
    #     print('embedding', embedding)
    #     print('timeMin', timeMin)
    #     # embeddings.append(embedding)
    # else:
    #     embedding = None
    #
    # # print('rfe embedding=',embedding)
    # model = buildModel(embedding, mp, training_tsv, prediction_tsv)

    # print('final embedding',model.embedding)
    save_las_coeffs(model, mp, resultsFolder)

    save_prediction_results('training', training_tsv, training_tsv, embedding, model, mp, resultsFolder, useAD,
                            adMeasure)
    save_prediction_results('prediction', training_tsv, prediction_tsv, embedding, model, mp, resultsFolder, useAD,
                            adMeasure)


def save_prediction_results(set, training_tsv, prediction_tsv, embedding, model, mp, resultsFolder, doAD, adMeasure):
    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    fileOut = resultsFolder + mp.dataset_name + '_' + mp.descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + '_' + set + ".csv"

    if set == 'prediction':
        if doAD:
            df_results, statistics_AD = runAD(mp.is_binary, adMeasure, df_results, model, training_tsv, prediction_tsv)

        df_results.to_csv(fileOut, index=False)

        results = save_json_file(df_results_all=df_results, embeddings=embedding, fileOut=fileOut, mp=mp,
                                 hyperparameters=model.hyperparameters,
                                 hyperparameter_grid=model.hyperparameter_grid, r2s=None, rmses=None,
                                 statistics_AD=statistics_AD)

    if not mp.is_binary:
        title = mp.descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
        if set == 'prediction':
            title = title + ' (P)'
        else:
            title = title + ' (T)'

        exp = list(df_results['exp'].to_numpy())
        pred = list(df_results['pred'].to_numpy())
        ru.generatePlot2(fileOut=fileOut, property_name=mp.dataset_name, title=title, exp=exp, pred=pred)

    return df_results


def runAD(isBinary, adMeasure, df_results, model, training_tsv, prediction_tsv):
    df_AD = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                           test_tsv=prediction_tsv,
                                                                           remove_log_p=False,
                                                                           embedding=model.embedding,
                                                                           applicability_domain=adMeasure)
    df_AD.rename(columns={"idTest": "id"}, inplace=True)
    # print(df_AD.shape)
    # df_results.to_csv('adresults' + str(fold) + '.csv')
    df_results = pd.merge(df_results, df_AD, on='id')
    df_results_inside = df_results.loc[df_results['AD'] == True]
    df_results_outside = df_results.loc[df_results['AD'] == False]
    # print('inside shape=', df_results_inside.shape)

    if (df_results_inside.shape[0] > 1):

        if isBinary:
            print('\nInside AD:')
            BA_inside = calc_binary_stats(df_results_inside)
        else:
            r2_inside, rmse_inside = calc_stats(df_results_inside)
            print('RMSE_inside=' + str(rmse_inside) + ', R2_inside=' + str(r2_inside), df_results_inside.shape[0])

    else:
        print('none inside AD')

    if (df_results_outside.shape[0] > 1):

        if isBinary:
            print('\nOutside AD:')
            BA_outside = calc_binary_stats(df_results_outside)
        else:
            r2_outside, rmse_outside = calc_stats(df_results_outside)
            print('RMSE_outside=' + str(rmse_outside) + ', R2_outside=' + str(r2_outside),
                  df_results_outside.shape[0])
    else:
        print('RMSE_outside=' + "N/A" + ', R2_outside=' + "N/A", df_results_outside.shape[0])

    cov = df_results_inside.shape[0] / df_results.shape[0]

    # print(df_results_inside.shape[0], df_results.shape[0], cov)

    statistics_AD = {}

    if isBinary:
        statistics_AD['BA_inside'] = BA_inside
        statistics_AD['BA_outside'] = BA_outside
    else:
        statistics_AD['rmse_inside'] = rmse_inside
        statistics_AD['rmse_outside'] = rmse_outside
        statistics_AD['r2_inside'] = r2_inside
        statistics_AD['r2_outside'] = r2_outside

    statistics_AD['cov'] = cov

    return df_results, statistics_AD


def save_las_coeffs(model, mp, resultsFolder):
    if mp.qsar_method == 'las' and mp.useEmbeddings:
        clf = model.model_obj.steps[1][1]
        coef = clf.coef_
        desc = model.model_obj.feature_names_in_
        res = pd.DataFrame(np.column_stack([desc, coef]), columns=['desc', 'coef'])
        res2 = res.loc[(res['coef'] != 0.0)]
        res2 = res2.reindex(res2['coef'].abs().sort_values(ascending=False).index)

        fileOutCoeffs = resultsFolder + mp.dataset_name + '_' + mp.descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
            mp.useEmbeddings) + "_coeffs.csv"
        res2.to_csv(fileOutCoeffs)
        # print(res2)


def getAllResults():
    folderMain = "C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets_exp_prop"

    dfResults = None

    for file in os.listdir(folderMain):

        if os.path.isfile(folderMain + "/" + file):
            continue

        # filepath = folderMain+"/"+file+"/results"
        # filepath = folderMain + "/" + file + "/results_expanded_grid2"
        # filepath = folderMain + "/" + file + "/results_expanded_grid"
        filepath = folderMain + "/" + file + "/results_expanded_las_rounded"
        # filepath = folderMain + "/" + file + "/results_las_optimized_tol"

        # print(filepath)
        if 'v4' in file:
            df = lookAtResults2(file, filepath)

            if dfResults is None:
                dfResults = df
            else:
                dfResults = pd.concat([dfResults, df], ignore_index=True)

            dfBest = dfResults.groupby(["composite"], as_index=False)["rmse"].mean()
            dfBest = dfBest.sort_values(by="rmse", ascending=True)

    # dfResults.to_csv(folderMain+"/allresults.csv",index=False)
    with pd.ExcelWriter(folderMain + "/allresults_expanded_las_rounded.xlsx") as writer:
        dfResults.to_excel(writer, sheet_name="all results", index=False)
        dfBest.to_excel(writer, sheet_name="best results", index=False)


def lookAtResultsForDatasets():
    # datasetNames = ['exp_prop_96HR_FHM_LC50_v4 modeling']
    # # datasetNames = ['exp_prop_96HR_FHM_LC50_v4 modeling', 'exp_prop_96HR_BG_LC50_v4 modeling',
    # #                 'exp_prop_96HR_RT_LC50_v4 modeling', 'exp_prop_48HR_DM_LC50_v4 modeling']

    # *********************************************************************************************************
    datasetNames = ['exp_prop_96HR_FHM_LC50_v4 modeling', 'LLNA TEST', 'LD50 TEST', 'Mutagenicity TEST']

    for datasetName in datasetNames:

        if 'exp_prop' in datasetName:
            inputFolder = '../datasets_exp_prop/'
        else:
            inputFolder = '../datasets_benchmark_TEST/'

        if datasetName in ['LLNA TEST', 'Mutagenicity TEST']:
            isBinary = True
        else:
            isBinary = False

        resultsFolder = inputFolder + datasetName + "/results_setac_2024/"
        lookAtResults(resultsFolder, useMean=False, displayAD=True, isBinary=isBinary)


if __name__ == "__main__":
    # ttr_binding()
    # inhalation_lc50()
    # runTrainingPredictionExample()
    runModelOptions()

    # print(np.logspace(-4, 0, num=35))

    # lookAtResultsForDatasets()

    # for v in range(1, 5):
    #     dataset="exp_prop_96HR_BG_LC50_v"+str(v)+" modeling"
    #     runTrainingPredictionExample(dataset)

    # runTrainingPredictionExample()

    # assembleResults()
    # a_runCaseStudiesExpPropPFAS()
    # runTrainingPredictionExample()
    # compare_spaces()
    # getAllResults()
