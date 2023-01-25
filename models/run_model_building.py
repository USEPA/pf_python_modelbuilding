
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

def a_runCaseStudiesExpPropPFAS():
    inputFolder = '../datasets/'

    useEmbeddings = True
    useGridSearch = True
    num_generations = 100

    # qsar_method = 'rf'
    # qsar_method = 'knn'
    # qsar_method = 'xgb'
    qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'

    # splitting = 'T=PFAS only, P=PFAS'
    # splitting = 'T=all, P=PFAS'
    splitting = 'T=all but PFAS, P=PFAS'


    datasetNames = []
    # datasetNames.append("HLC from exp_prop and chemprop")
    # datasetNames.append("WS from exp_prop and chemprop")
    # datasetNames.append("VP from exp_prop and chemprop")
    datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    if useEmbeddings:
        ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

    f = open(inputFolder+'results/'+splitting+'_useEmbeddings='+str(useEmbeddings)+'.txt', "w")

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

        embedding_tsv = None

        if useEmbeddings:
            if splitting == 'T=PFAS only, P=PFAS':
                num_generations = 100
            elif 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
                num_generations = 10
            embedding_tsv = ei.get_embedding(dataset_name=datasetName, num_generations=num_generations,
                                             splitting_name=splitting)

        model = call_build_model_with_preselected_descriptors(qsar_method=qsar_method,training_tsv=training_tsv,remove_log_p=remove_log_p,
                                                      descriptor_names_tsv=embedding_tsv,use_grid_search=useGridSearch,model_id=1)

        R2, MAE = call_do_predictions(prediction_tsv, model)

        print('*****' + datasetName + '\t' + R2 + '\t' + MAE + '\n')

        f.write(datasetName+'\t'+R2+'\t'+MAE+'\n')
        f.flush()

    f.close()

    def a_runCaseStudiesExpPropPFAS():
        inputFolder = '../datasets/'

        useEmbeddings = True
        useGridSearch = True
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

            if useEmbeddings:

                if splitting == 'T=PFAS only, P=PFAS':
                    num_generations = 100
                elif 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
                    num_generations = 10

                embedding_tsv = ei.get_embedding(dataset_name=datasetName, num_generations=num_generations,
                                                 splitting_name=splitting)
                print(embedding_tsv)
                model = mwu.call_build_model_with_preselected_descriptors2(qsar_method=qsar_method,
                                                                           training_tsv=training_tsv,
                                                                           remove_log_p=remove_log_p,
                                                                           descriptor_names_tsv=embedding_tsv,
                                                                           model_id=1,
                                                                           use_grid_search=useGridSearch)
            else:
                model = mwu.call_build_model2(qsar_method=qsar_method, training_tsv=training_tsv,
                                              remove_log_p=remove_log_p,
                                              model_id=1, doGridSearch=useGridSearch)

            r2,MAE=call_do_predictions(prediction_tsv, model)

            print('*****',datasetName, r2, MAE)
            # print(results_json)


def call_do_predictions(prediction_tsv, model):
    """Loads TSV prediction data into a pandas DF, stores IDs and exp vals,
    and calls the appropriate prediction method"""
    df_prediction = dfu.load_df(prediction_tsv)
    pred_ids = np.array(df_prediction[df_prediction.columns[0]])
    pred_labels = np.array(df_prediction[df_prediction.columns[1]])
    predictions = model.do_predictions(df_prediction)

    return run_rf_todd.calcStats(predictions, df_prediction, None)


def a_runCaseStudiesExpProp():
    inputFolder = '../datasets/'

    useEmbeddings = True
    useGridSearch = True
    num_generations = 100
    # qsar_method = 'rf'
    qsar_method = 'knn'
    # qsar_method = 'xgb'
    # qsar_method = 'svm'

    descriptor_software = 'WebTEST-default'

    splitting = 'RND_REPRESENTATIVE'


    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    datasetNames.append("WS from exp_prop and chemprop")
    datasetNames.append("VP from exp_prop and chemprop")
    datasetNames.append("LogP from exp_prop and chemprop")
    datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    if useEmbeddings:
        ei = EmbeddingImporter('../embeddings/embeddings.xlsx')

    f='bob' #TODO make it a filewriter

    for datasetName in datasetNames:
        training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
        prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

        training_tsv_path = inputFolder + datasetName+'/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName+'/' + prediction_file_name

        remove_log_p = False
        if 'LogP' in datasetName:
            remove_log_p = True

        training_tsv = dfu.read_file_to_string(training_tsv_path)
        prediction_tsv = dfu.read_file_to_string(prediction_tsv_path)

        if useEmbeddings:

            if splitting == 'T=PFAS only, P=PFAS':
                num_generations = 100
            elif 'MP' in datasetName or splitting == 'T=all but PFAS, P=PFAS':
                num_generations = 10


            embedding_tsv = ei.get_embedding(dataset_name=datasetName, num_generations=num_generations,
                                             splitting_name='T=all, P=PFAS')
            print(embedding_tsv)
            model = mwu.call_build_model_with_preselected_descriptors2(qsar_method=qsar_method,training_tsv=training_tsv,
                                                                       remove_log_p=remove_log_p,descriptor_names_tsv=embedding_tsv,
                                                                       model_id=1, use_grid_search=useGridSearch)
        else:
            model = mwu.call_build_model2(qsar_method=qsar_method, training_tsv=training_tsv, remove_log_p=remove_log_p,
                                         model_id=1,doGridSearch=useGridSearch)

        results_json = mwu.call_do_predictions(prediction_tsv, model)




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


def call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p, descriptor_names_tsv,
                                                  model_id, use_grid_search):
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

    # Returns trained model

    if use_grid_search == False:
        model.hyperparameters = None

    model.build_model(descriptor_names=descriptor_names_tsv)



    return model

if __name__ == "__main__":
    a_runCaseStudiesExpPropPFAS()
    # a_runCaseStudiesExpProp()
    # compare_spaces()

