import json
import os
import pandas as pd
import numpy as np
from models import df_utilities as DFU
import model_ws_utilities as mwu
import df_utilities as dfu

import models.results_utilities as ru
from models import ModelBuilder as mb

import pickle
import csv
import os
import csv
import os
from models import EmbeddingFromImportance as efi

import matplotlib.pyplot as plt

num_jobs = 4

np.random.seed(seed=42) #makes results the same each time

class EmbeddingImporter:
    def __init__(self, embedding_filepath):
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

    fileOut = inputFolder + 'results2/' + qsar_method + '_' + splitting + '_useEmbeddings=' + str(useEmbeddings) + '.txt'
    print('output file=', fileOut)

    f = open(fileOut,"w")
    f.write('dataset\tR2\tMAE\n')

    for datasetName in datasetNames:

        if 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
            num_generations = 10

        run_dataset(datasetName, descriptor_software, f, inputFolder, num_generations, qsar_method, splitting,
                    useEmbeddings)

    f.close()

def runTrainingPredictionExample():

    descriptor_software = 'WebTEST-default'
    datasetName = 'exp_prop_96HR_BG_LC50_v4 modeling'

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

    mp.remove_log_p = False
    if 'LogP' in datasetName:
        mp.remove_log_p = True

    inputFolder = '../datasets_exp_prop/'
    training_file_name = datasetName+"_"+descriptor_software+"_RND_REPRESENTATIVE_training.tsv"
    prediction_file_name = datasetName+"_"+descriptor_software+ "_RND_REPRESENTATIVE_prediction.tsv"

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

    if mp.qsar_method == 'las':
        clf = model.model_obj.steps[1][1]
        coef = clf.coef_
        desc = model.model_obj.feature_names_in_
        res = pd.DataFrame(np.column_stack([desc, coef]), columns=['desc', 'coef'])
        res2 = res.loc[(res['coef'] != 0.0)]
        print(res2)
        res2_list = ', '.join(res2['desc'].astype(str))
        embeddings.append(res2_list)

        #plt.bar(res2['desc'],res2['coef'])
        #plt.xticks(rotation=90)
        #plt.grid()
        #plt.title(datasetName + " Feature Selection Based on Lasso")
        #plt.xlabel("Features")
        #plt.ylabel("Importance")
        #plt.ylim(0, 0.15)
        #plt.show()

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

    fileOut = inputFolder + datasetName+"/"+datasetName+"_" + mp.descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(mp.useEmbeddings) + ".csv"

    df_results_prediction.to_csv(fileOut, index=False)

    results = save_json_file(df_results_all=df_results_prediction, embeddings=embeddings, fileOut=fileOut, mp=mp, hyperparameters=model.hyperparameters, hyperparameter_grid=model.hyperparameter_grid)

    exp = df_results['exp'].to_numpy()
    pred = df_results['pred'].to_numpy()

    MAE = ru.calc_MAE(pred, exp)
    strMAE = str("{:.3f}".format(MAE))

    R2 = ru.calc_pearson_r2(pred, exp)
    strR2 = str("{:.3f}".format(R2))

    print('*****************************\n' + datasetName + '\tR2=' + strR2 + '\tMAE=' + strMAE + '\n')

    figtitle = datasetName +"_"+ mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    title = descriptor_software + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generateTrainingPredictionPlot(fileOut=fileOut, property_name=datasetName, title=title, figtitle=figtitle,
                                      exp_training=list(exp_training),pred_training=list(pred_training),
                                      exp_prediction=list(exp_prediction), pred_prediction=list(pred_prediction))


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

    MAE = ru.calc_MAE(pred,exp)
    strMAE = str("{:.3f}".format(MAE))

    R2 = ru.calc_pearson_r2(pred,exp)
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


def save_json_file(df_results_all, embeddings, fileOut, mp, hyperparameters=None, hyperparameter_grid=None, r2s=None, rmses=None):
    exp = df_results_all['exp'].to_numpy()
    pred = df_results_all['pred'].to_numpy()
    results = {}
    statistics = {"rmse": ru.calc_rmse(pred, exp), "r2": ru.calc_pearson_r2(pred, exp)}

    results["model_parameters"] = mp.__dict__
    results["statistics"] = statistics
    results["embeddings"] = embeddings

    if hyperparameter_grid is not None:
        results["hyperparameter_grid"] = hyperparameter_grid

    if hyperparameters is not None:
        results["hyperparameters"] = hyperparameters

    import statistics as stats

    if r2s is not None:
        statistics["r2s"] = r2s
        statistics["rmses"] = rmses

        statistics["r2_mean"] = stats.mean(r2s)
        statistics["rmse_mean"] = stats.mean(rmses)

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

    lookAtResults(resultsFolder,useMean=useMean)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot(property_name='TTR_Binding', title=title, exp=exp, pred=pred)

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

    file = open(inputFolder+'ttr_binding_full_model.p', 'wb')
    pickle.dump(model,file)
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

    df_results.to_csv(fileOut,index=False)
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

    with open(inputFolder+'ttr_binding_full_model.p', "rb") as input_file:
        model = pickle.load(input_file)


    df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

    df_results.to_csv(inputFolder+'special chemicals results.csv',index=False)

    print(df_results)

    # df_results.rename(columns={"id": "DTXSID"}, inplace=True)
    # df_results.drop('exp', axis=1, inplace=True)
    # # print (df_results)
    #
    # df_results = pd.merge(df_results, df_predictions, on='DTXSID')



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


def buildModel(embedding, mp, training_tsv, prediction_tsv):
    model = mwu.call_build_model_with_preselected_descriptors(qsar_method=mp.qsar_method,
                                                              training_tsv=training_tsv, prediction_tsv=prediction_tsv,
                                                              remove_log_p=mp.remove_log_p,
                                                              use_pmml_pipeline=mp.use_pmml,
                                                              include_standardization_in_pmml=mp.include_standardization_in_pmml,
                                                              descriptor_names_tsv=embedding,
                                                              n_jobs=mp.n_threads)
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




def lookAtResults(folder, useMean=False):
    print("\n\nproperty_units", "descriptor_set", "qsar_method", "useEmbeddings", "pearson_r2", "rmse", sep="\t")

    header=["property_units","descriptor_set","qsar_method","useEmbeddings","r2","rmse"]

    with open(folder + '/results.csv', 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(header)

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
            mp = results["model_parameters"]
            stats = results["statistics"]

            if useMean:

                if "r2_mean" not in stats:
                    print("missing r2_mean for",mp["property_units"], mp["descriptor_set"], mp["qsar_method"], mp["useEmbeddings"])
                else:

                    print(mp["property_units"], mp["descriptor_set"], mp["qsar_method"], mp["useEmbeddings"],
                          round(stats["r2_mean"], 3), round(stats["rmse_mean"], 3), sep="\t")
                    csvfile.write(mp["property_units"], mp["descriptor_set"], mp["qsar_method"], mp["useEmbeddings"],
                          round(stats["r2_mean"], 3), round(stats["rmse_mean"], 3), sep="\t")

            else:
                resultRow=[mp["property_units"], mp["descriptor_set"], mp["qsar_method"], mp["useEmbeddings"],round(stats["r2"], 3), round(stats["rmse"], 3)]

                print(resultRow, sep="\t")

                writer.writerow(resultRow)


def lookAtResults2(dataset, folder):

    # print (folder)

    # print("\n\nproperty_units", "descriptor_set", "qsar_method", "useEmbeddings", "pearson_r2", "rmse", sep="\t")

    header=["property_units","descriptor_set","qsar_method","useEmbeddings","r2","rmse"]

    property_units = []
    descriptor_set = []
    qsar_method = []
    useEmbeddings = []
    r2 = []
    rmse = []
    datasets=[]
    composite=[]

    # print(dataset)

    for filename in os.listdir(folder):

        # print(type(filename))
        filepath = folder+'/'+filename

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

        if len(results["embeddings"])>0 and mp["qsar_method"]=='las':
            print(filename, mp["qsar_method"], stats['r2'],len(results["embeddings"]))


    dict = {'dataset': datasets, 'property_units': property_units, 'descriptor_set': descriptor_set, 'qsar_method': qsar_method,
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
        lenAvg=(len1+len2+len3+len4+len5)/5

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




def a_runCaseStudiesInhalationCV_merge_on_fly():
    units = 'ppm'
    # units = 'mgL'

    inputFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/'

    # descriptor_set = 'webtest'
    # descriptor_set = 'webtest_opera'
    # descriptor_set = 'padel'
    # descriptor_set = 'morgan'
    descriptor_set = 'mordred'

    property_name = "4 hour rat inhalation LC50"

    mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)

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
        df_descriptors = df_descriptors[df_descriptors.x0 != 'Error'] #remove bad rows

    df_all = pd.merge(df_splits, df_descriptors, on='QSAR_READY_SMILES')

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

        exp = df_results['exp'].to_numpy()
        pred = df_results['pred'].to_numpy()



        rmse = ru.calc_rmse(pred, exp)
        r2 = ru.calc_pearson_r2(pred, exp)

        r2s.append(r2)
        rmses.append(rmse)

        print('fold=' + str(fold), 'RMSE=' + str(rmse) + ', R2=' + str(r2))

        if fold == 1:
            df_results_all = df_results
        else:
            df_results_all = pd.concat([df_results_all, df_results], ignore_index=True)



        r2s.append(r2)
        rmses.append(rmse)

    # print(df_results_all)

    resultsFolder = inputFolder + "results/"

    fileOut = inputFolder + "results/LC50_" + units + "_" + descriptor_set + "_" + mp.qsar_method + "_embedding=" + str(
        mp.useEmbeddings) + ".csv"

    df_results_all.to_csv(fileOut, index=False)

    save_json_file(df_results_all=df_results_all, embeddings=embeddings, fileOut=fileOut, mp=mp, r2s=r2s, rmses=rmses)

    exp = list(df_results_all['exp'].to_numpy())
    pred = list(df_results_all['pred'].to_numpy())
    title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)
    ru.generatePlot(property_name='LC50', title=title, exp=exp, pred=pred)


    lookAtResults(resultsFolder)


def ttr_binding():
    """
    runs all the calculations for ttr_binding
    :return:
    """
    a_runCaseStudiesTTR_Binding_CV_merge_on_fly()
    # a_runCaseStudiesTTR_Binding_training_prediction()
    # a_runCaseStudiesTTR_Binding_special_chemicals()

    # a_runCaseStudiesTTR_Binding_CV()
    # a_runCaseStudiesTTR_Binding_CV_merge_on_fly_predict_AQC_fails()

    resultsFolder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/000 Papers/2024 tetko challenge/modeling/results/'
    # lookAtResults(resultsFolder,useMean=True)
    # lookAtEmbeddings(resultsFolder)


def inhalation_lc50():
    # a_runCaseStudiesRatInhalationLC50()
    a_runCaseStudiesInhalationCV_merge_on_fly()


    # folder = 'C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/0 model_management/hibernate_qsar_model_building/data/modeling/CoMPAIT/results/'
    # lookAtResults(folder)

def runModelOptions():
    np.random.seed(seed=42)
    property_name = "96 hour rainbow trout LC50"

    #datasetName = "MP v1 modeling"
    datasetName = 'exp_prop_96HR_RT_LC50_v5 modeling'

    units = '-log10(M)'
    #units = 'TBA'

    #descriptor_sets = ['WebTEST-default', 'Mordred-default', 'PaDel-default', 'ToxPrints-default']
    descriptor_sets = ['ToxPrints-default']

    #qsar_methods = ['knn', 'svm']
    qsar_methods = ['las']
    #qsar_methods = ['xgb']

    #useEmbeddings = ['True', 'False']
    useEmbeddings = ['False']

    #inputFolder = '../datasets/' #relative path should work
    #inputFolder = '../datasets_exp_prop/'
    inputFolder = 'C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets_exp_prop/'
    #resultsFolder = inputFolder + datasetName + "/results/"
    resultsFolder = inputFolder + datasetName + "/results_expanded_grid/"
    # resultsFolder = inputFolder + datasetName + "/las_results_linspace/"
    #resultsFolder = inputFolder + datasetName + "/results_expanded_las_rounded/"
    #resultsFolder = inputFolder + datasetName + "/results_las_optimized_tol/"

    if not os.path.exists(resultsFolder):
        os.mkdir(resultsFolder)


    for descriptor_set in descriptor_sets:
        training_file_name = datasetName + "_" + descriptor_set + "_RND_REPRESENTATIVE_training.tsv"
        prediction_file_name = datasetName + "_" + descriptor_set + "_RND_REPRESENTATIVE_prediction.tsv"

        training_tsv_path = inputFolder + datasetName + '/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/' + prediction_file_name

        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        for qsar_method in qsar_methods:

            mp = model_parameters(property_name=property_name, property_units=units, descriptor_set=descriptor_set)
            mp.qsar_method=qsar_method
            mp.n_threads = 4# lower this number if have high CPU usage

            for z in useEmbeddings:

                if (qsar_method == 'reg') and (z == 'False'):
                    continue
                if (qsar_method == 'svm') and (z == 'True'):
                    continue
                #if (qsar_method == 'las') and (z == 'True'):
                #    continue

                if z == 'True':
                    mp.useEmbeddings=True
                    embedding, timeMin = getEmbedding(mp, prediction_tsv, training_tsv)
                    print('embedding', embedding)
                    print('timeMin', timeMin)
                    # embeddings.append(embedding)
                else:
                    mp.useEmbeddings=False
                    embedding = None

                # print('rfe embedding=',embedding)
                model = buildModel(embedding, mp, training_tsv, prediction_tsv)
                # print('final embedding',model.embedding)

                #if mp.qsar_method == 'las':
                    #clf = model.model_obj.steps[1][1]
                    #coef = clf.coef_
                    #desc = model.model_obj.feature_names_in_
                    #res = pd.DataFrame(np.column_stack([desc, coef]), columns=['desc', 'coef'])
                    #res2 = res.loc[(res['coef'] != 0.0)]
                    #print(res2)
                    # res2_list = ', '.join(res2['desc'].astype(str))
                    # embedding=res2_list
                    #embedding = list(res2['desc'])

                df_results = mwu.call_do_predictions_to_df(prediction_tsv, model)

                df_results_training = mwu.call_do_predictions_to_df(training_tsv, model)
                exp_training = df_results_training['exp'].to_numpy()
                pred_training = df_results_training['pred'].to_numpy()

                df_results_prediction = mwu.call_do_predictions_to_df(prediction_tsv, model)
                exp_prediction = df_results_prediction['exp'].to_numpy()
                pred_prediction = df_results_prediction['pred'].to_numpy()

                exp = df_results['exp'].to_numpy()
                pred = df_results['pred'].to_numpy()

                MAE = ru.calc_MAE(pred, exp)
                strMAE = str("{:.3f}".format(MAE))

                R2 = ru.calc_pearson_r2(pred, exp)
                strR2 = str("{:.3f}".format(R2))

                fileOut = resultsFolder + datasetName + '_' + descriptor_set + "_" + qsar_method + "_embedding=" + str(z) + ".csv"

                df_results.to_csv(fileOut, index=False)

                results = save_json_file(df_results_all=df_results, embeddings=embedding, fileOut=fileOut, mp=mp, hyperparameters=model.hyperparameters, hyperparameter_grid=model.hyperparameter_grid)

                print(datasetName + '_' + descriptor_set + '_' + qsar_method + '_' + 'embedding=' + z + '\t' + strR2 + '\t' + strMAE + '\n')

                figtitle = datasetName + "_" + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)

                title = descriptor_set + ' ' + mp.qsar_method + " useEmbeddings=" + str(mp.useEmbeddings)

                ru.generateTrainingPredictionPlot(fileOut=fileOut, property_name=datasetName, title=title,
                                                   figtitle=figtitle,
                                                   exp_training=list(exp_training), pred_training=list(pred_training),
                                                   exp_prediction=list(exp_prediction),
                                                   pred_prediction=list(pred_prediction),showPlot=False)

    lookAtResults(folder = resultsFolder)

def getAllResults():

    folderMain="C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets_exp_prop"
    #folderMain = '../datasets/'
    dfResults=None

    for file in os.listdir(folderMain):

        if os.path.isfile(folderMain+"/"+file):
            continue

        #filepath = folderMain+"/"+file+"/results"
        # filepath = folderMain + "/" + file + "/results_expanded_grid2"
        #filepath = folderMain + "/" + file + "/results_expanded_grid"
        filepath = folderMain + "/" + file + "/results_expanded_las_rounded"
        #filepath = folderMain + "/" + file + "/results_las_optimized_tol"

        # print(filepath)
        if 'v4' in file:
            df=lookAtResults2(file, filepath)

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


if __name__ == "__main__":
    # ttr_binding()
    # inhalation_lc50()
    # runTrainingPredictionExample()
    runModelOptions()

    # resultsFolder= 'C:/Users/lbatts/OneDrive - Environmental Protection Agency (EPA)/0 Python/pf_python_modelbuilding/datasets/results/'
    # lookAtResults(resultsFolder)

    # for v in range(1, 5):
    #     dataset="exp_prop_96HR_BG_LC50_v"+str(v)+" modeling"
    #     runTrainingPredictionExample(dataset)
    # runTrainingPredictionExample()

    # assembleResults()
    # a_runCaseStudiesExpPropPFAS()
    # runTrainingPredictionExample()
    # compare_spaces()
    #getAllResults()