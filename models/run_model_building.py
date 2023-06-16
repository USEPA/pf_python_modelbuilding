
import pandas as pd
import numpy as np

import model_ws_utilities as mwu
import df_utilities as dfu
from models.RF import run_rf_todd as run_rf_todd


# from models_refactor import knn_model
# from models_refactor.qsar_model import QSAR_Model
# from models_refactor.svm_model import SVM_Model
# from models_refactor.knn_model import KNN_Model

from models import ModelBuilder as mb

num_jobs = 4


class EmbeddingImporter:
    def __init__(self, embedding_filepath):
        self.df = pd.read_excel(embedding_filepath, engine='openpyxl')

    def get_embedding(self, dataset_name, num_generations, splitting_name):
        # https://note.nkmk.me/en/python-pandas-multiple-conditions/
        df_and = self.df[(self.df['dataset_name'] == dataset_name)]  # Note need parentheses or doesnt work!
        df_and = df_and[(df_and['splitting_name'] == splitting_name)]  # Note
        df_and = df_and[(df_and['num_generations'] == num_generations)]
        # print(df_and)

        embedding_tsv=str(df_and['embedding_tsv'].iloc[0])

        embedding = embedding_tsv.split('\t')

        return embedding


def caseStudyTEST_RunGA():
    """
    Loops through TEST toxicity data sets :
    1. Generates GA embedding using chosen QSAR method
    2. Builds qsar model with and without embedding
    3. Reports results for test set
    4. Exports results to text file
    """

    # change following to folder where TEST sample sets are stored:
    # test_directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/DataSetsBenchmarkTEST_Toxicity/"
    test_directory = "../datasets_benchmark_TEST/"

    # endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'DevTox', 'LLNA', 'Mutagenicity']
    # endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']
    # endpointsTEST = ['LLNA']
    endpointsTEST = ['Mutagenicity']

    # qsar_method = 'rf'
    qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    # *****************************************************************************************************************
    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'RDKit-default'
    # descriptor_software = 'ToxPrints-default'
    descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************


    for ENDPOINT in endpointsTEST:
        remove_log_p = False

        directory = test_directory + ENDPOINT + ' TEST/'

        print(ENDPOINT, descriptor_software)

        # training_file_name = ENDPOINT + '_training_set-2d.csv'
        # prediction_file_name = ENDPOINT + '_prediction_set-2d.csv'

        training_file_name = ENDPOINT + ' TEST ' + descriptor_software + ' training.tsv'
        prediction_file_name = ENDPOINT + ' TEST ' + descriptor_software + ' prediction.tsv'

        folder = directory

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        # print(training_tsv_path)

        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        embedding_tsv = None


        model = call_build_model_with_preselected_descriptors(qsar_method=qsar_method,training_tsv=training_tsv,remove_log_p=remove_log_p,
                                                      descriptor_names_tsv=embedding_tsv,n_jobs=1)

        predictions = call_do_predictions(prediction_tsv, model)

        for pred in predictions:
            print(pred)

        # print(predictions)


    # f.close()

def a_runCaseStudiesExpPropPFAS():
    inputFolder = '../datasets/'

    useEmbeddings = False
    num_generations = 100

    # qsar_method = 'rf'
    qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'

    splitting = 'T=PFAS only, P=PFAS'
    # splitting = 'T=all, P=PFAS'
    # splitting = 'T=all but PFAS, P=PFAS'


    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    # datasetNames.append("WS from exp_prop and chemprop")
    # datasetNames.append("VP from exp_prop and chemprop")
    # datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")


    ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

    f = open(inputFolder+'results2/'+qsar_method+'_'+splitting+'_useEmbeddings='+str(useEmbeddings)+'.txt', "w")
    f.write('dataset\tR2\tMAE\n')

    for datasetName in datasetNames:
        run_dataset(datasetName, descriptor_software, ei, f, inputFolder, num_generations, qsar_method, splitting,
                    useEmbeddings)


    f.close()

    def a_runCaseStudiesExpPropPFAS():
        inputFolder = '../datasets/'

        useEmbeddings = True

        num_generations = 100
        # qsar_method = 'rf'
        qsar_method = 'knn'
        # qsar_method = 'xgb'
        # qsar_method = 'svm'

        descriptor_software = 'WebTEST-default'

        splitting = 'T=PFAS only, P=PFAS'
        # splitting = 'T=all, P=PFAS'
        # splitting = 'T=all but PFAS, P=PFAS'

        datasetNames = []
        # datasetNames.append("HLC from exp_prop and chemprop")
        # datasetNames.append("WS from exp_prop and chemprop")
        # datasetNames.append("VP from exp_prop and chemprop")
        # datasetNames.append("LogP from exp_prop and chemprop")
        # datasetNames.append("MP from exp_prop and chemprop")
        datasetNames.append("BP from exp_prop and chemprop")

        if useEmbeddings:
            ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

        f = 'bob'  # TODO make it a filewriter

        for datasetName in datasetNames:
            training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
            prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

            training_tsv_path = inputFolder + datasetName + '/PFAS/' + training_file_name
            prediction_tsv_path = inputFolder + datasetName + '/PFAS/' + prediction_file_name

            remove_log_p = False
            if 'LogP' in datasetName:
                remove_log_p = True

            training_tsv = dfu.read_file_to_string(training_tsv_path)
            prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

            embedding_tsv=None

            if useEmbeddings:
                if splitting == 'T=PFAS only, P=PFAS':
                    num_generations = 100
                elif 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
                    num_generations = 10

                embedding_tsv = ei.get_embedding(dataset_name=datasetName, num_generations=num_generations,
                                                 splitting_name=splitting)
                print(embedding_tsv)


            model = call_build_model_with_preselected_descriptors(qsar_method=qsar_method,
                                                                       training_tsv=training_tsv,
                                                                       remove_log_p=remove_log_p,
                                                                       descriptor_names_tsv=embedding_tsv,n_jobs=4)

            r2,MAE=call_do_predictions(prediction_tsv, model)

            print('*****',datasetName, r2, MAE)
            # print(results_json)


def a_runCaseStudiesExpPropPFAS_all():
    inputFolder = '../datasets/'



    num_generations = 100

    # qsar_method = 'rf'
    qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'


    splittings = ['T=PFAS only, P=PFAS','T=all, P=PFAS','T=all but PFAS, P=PFAS']
    useEmbeddingsArray =[False, True]

    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    datasetNames.append("WS from exp_prop and chemprop")
    datasetNames.append("VP from exp_prop and chemprop")
    datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    ei = EmbeddingImporter('../embeddings/embeddings.xlsx')


    for useEmbeddings in useEmbeddingsArray:

        for splitting in splittings:

            f = open(inputFolder + 'results2/' + qsar_method + '_' + splitting + '_useEmbeddings=' +
                     str(useEmbeddings) + '.txt',"w")
            f.write('dataset\tR2\tMAE\n')

            for datasetName in datasetNames:
                run_dataset(datasetName, descriptor_software, ei, f, inputFolder, num_generations, qsar_method, splitting,
                            useEmbeddings)
            f.close()


def run_dataset(datasetName, descriptor_software, ei, f, inputFolder, num_generations, qsar_method, splitting,
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


    remove_log_p = False
    if 'LogP' in datasetName:
        remove_log_p = True
    training_tsv = dfu.read_file_to_string(training_tsv_path)
    prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)
    embedding_tsv = None

    if useEmbeddings:
        if splitting == 'T=PFAS only, P=PFAS':
            num_generations = 100
        elif 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
            num_generations = 10

        if splitting == 'RND_REPRESENTATIVE':
            splitting ='T=all, P=PFAS'

        # print (datasetName, num_generations,splitting)


        embedding_tsv = ei.get_embedding(dataset_name=datasetName, num_generations=num_generations,
                                         splitting_name=splitting)
    model = call_build_model_with_preselected_descriptors(qsar_method=qsar_method, training_tsv=training_tsv,
                                                          remove_log_p=remove_log_p,
                                                          descriptor_names_tsv=embedding_tsv,
                                                          n_jobs=num_jobs)
    predictions = call_do_predictions(prediction_tsv, model)

    # print(predictions)

    df_prediction = dfu.load_df(prediction_tsv)
    strR2, strMAE = run_rf_todd.calcStats(predictions, df_prediction, None)
    print('*****' + datasetName + '\t' + strR2 + '\t' + strMAE + '\n')
    f.write(datasetName + '\t' + strR2 + '\t' + strMAE + '\n')
    f.flush()


def call_do_predictions(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""
    df_prediction = dfu.load_df(prediction_tsv)
    predictions = model.do_predictions(df_prediction)

    return predictions


def a_runCaseStudiesExpProp():

    inputFolder = '../datasets/'

    useEmbeddings = True

    num_generations = 100

    # qsar_method = 'rf'
    qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'
    splitting = 'RND_REPRESENTATIVE'

    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    # datasetNames.append("WS from exp_prop and chemprop")
    # datasetNames.append("VP from exp_prop and chemprop")
    # datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")


    ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

    f = open(inputFolder+'results2/'+qsar_method+'_T=all, P=all_useEmbeddings='+str(useEmbeddings)+'.txt', "w")
    f.write('dataset\tR2\tMAE\n')

    for datasetName in datasetNames:
        run_dataset(datasetName, descriptor_software, ei, f, inputFolder, num_generations, qsar_method, splitting,
                    useEmbeddings)

    f.close()



def compare_spaces():

    import numpy as np
    gamma_space = [np.power(2, i) / 1000.0 for i in range(0, 10, 2)]  # [0.01]

    gamma_space2=list([10 ** x for x in range(-3, 4)])
    print('gamma_space',gamma_space)
    print('gamma_space2',gamma_space2)

    c_space = np.arange(-3, 4, 0.5)  # wrong has negative values
    c_space2 = list([10 ** x for x in range(-3, 4)])
    c_space3 = np.logspace(-1, 1, 10)

    # print('c_space',c_space)
    print('c_space2',c_space2)
    print('c_space3',c_space3)


def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, descriptor_names_tsv,n_jobs):
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
    splittings = ['T=PFAS only, P=PFAS', 'T=all, P=PFAS', 'T=all but PFAS, P=PFAS','T=all, P=all']
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
            filepath=inputFolder + resultsName+'/' + qsar_method + '_' + splitting + '_useEmbeddings='+str(useEmbeddings) + '.txt'

            if not os.path.isfile(filepath):
                print('missing:'+filepath)
                continue

            df = pd.read_csv(filepath, delimiter='\t')

            if 'datasetName' not in dfnew:
                dfnew['dataset']=df['dataset']

            newName = splitting + '_useEmbeddings=' + str(useEmbeddings)
            dfnew[newName] = df['MAE']



    print(dfnew)

    dfnew.to_csv(inputFolder + resultsName+'/'+qsar_method+'.csv',index=False)




if __name__ == "__main__":
    # assembleResults()
    a_runCaseStudiesExpPropPFAS()
    # a_runCaseStudiesExpPropPFAS_all()
    # a_runCaseStudiesExpProp()
    # compare_spaces()
    # caseStudyTEST_RunGA()
