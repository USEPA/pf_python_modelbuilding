import time
from models import df_utilities as DFU
import model_ws_utilities as mwu
import json
from models import rf_model as rf
from models import svm_model as svm
# from models import dnn_model as dnn
from models import xgb_model as xgb
from models import knn_model as knn

# Set GA hyperparameters:
num_generations = 1
num_optimizers = 10
qsar_method = 'knn'
n_threads = 16  # threads to use with RF- TODO
num_jobs = 4  # jobs to use for GA search for embedding- using 16 has marginal benefit over 4 or 8
descriptor_coefficient = 0.002
max_length = 24
# threshold = 2  # Nate's value
threshold = 1  # TMM attempt to get more descriptors from stage 1

# descriptor_coefficient = 0.0
# max_length = 40

urlHost = 'http://localhost:5004/'
# urlHost = 'http://v2626umcth819.rtord.epa.gov:5004/'
useAPI = False  # whether or not to use API call to run GA calcs

# *****************************************************************************************************************



def get_ga_description(descriptor_coefficient, max_length, n_threads, num_generations, num_optimizers, num_jobs,
                       qsar_method , threshold):
    dictGA = {}
    dictGA['num_generations'] = num_generations
    dictGA['num_optimizers'] = num_optimizers
    dictGA['qsar_method'] = qsar_method
    dictGA['n_threads'] = n_threads
    dictGA['num_jobs'] = num_jobs
    dictGA['descriptor_coefficient'] = descriptor_coefficient
    dictGA['max_length'] = max_length
    dictGA['threshold'] = threshold
    return dictGA


def getModelDescription(qsar_method):
    model = None
    if qsar_method == 'svm':
        model = svm.Model(None, False, 1)
        return svm.ModelDescription(model).to_json()
    elif qsar_method == 'rf':
        model = rf.Model(None, False, 1)
        return rf.ModelDescription(model).to_json()
    elif qsar_method == 'xgb':
        model = xgb.Model(None, False)
        return xgb.ModelDescription(model).to_json()
    elif qsar_method == 'knn':
        model = knn.Model(None, False, 1)
        model.getModel()
        return knn.ModelDescription(model).to_json()


def caseStudyOPERA_RunGA():
    directoryOPERA = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/"

    endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
                      "Vapor pressure", "Water solubility", "Boiling point",
                      "Melting point", "Octanol water partition coefficient"]

    # endpointsOPERA = ["LogBCF"]
    # endpointsOPERA = ["Octanol water partition coefficient"]

    # *****************************************************************************************************************
    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'ToxPrints-default'
    descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************

    # output filepath:
    filename = '../data/opera ' + qsar_method + ' results NUM_GENERATIONS=' + str(num_generations) + \
               ' NUM_OPTIMIZERS=' + str(num_optimizers) + '_' + str(round(time.time() * 1000)) + '.txt'
    f = open(filename, "w")

    f.write(getModelDescription(qsar_method) + '\n')

    dictGA = get_ga_description(descriptor_coefficient=descriptor_coefficient, max_length=max_length,
                                n_threads=n_threads, num_optimizers=num_optimizers,
                                num_generations=num_generations, num_jobs=num_jobs,
                                qsar_method=qsar_method, threshold=threshold)
    print(json.dumps(dictGA) + '\n')
    f.write(json.dumps(dictGA) + '\n')

    f.write('ENDPOINT\tscore\tscore_embed\tlen(features)\tfeatures\tTime(min)\n')
    f.flush()

    for ENDPOINT in endpointsOPERA:
        if ENDPOINT == 'Octanol water partition coefficient':
            remove_log_p = True
        else:
            remove_log_p = False

        print (ENDPOINT, descriptor_software)
        directory = directoryOPERA + ENDPOINT + ' OPERA/'

        training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
        folder = directory

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        # print(training_tsv_path)

        run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path)

    f.close()


def run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path):

    training_tsv = DFU.read_file_to_string(training_tsv_path)
    prediction_tsv = DFU.read_file_to_string(prediction_tsv_path)
    df_prediction = DFU.load_df(prediction_tsv)

    if useAPI:
        # Run from API call:
        str_result = mwu.api_call_build_embedding_ga(qsar_method=qsar_method, training_tsv=training_tsv,
                                                     remove_log_p=remove_log_p, n_threads=n_threads,
                                                     num_generations=num_generations, num_optimizers=num_optimizers,
                                                     num_jobs=num_jobs,
                                                     descriptor_coefficient=descriptor_coefficient,
                                                     max_length=max_length, threshold=threshold, urlHost=urlHost)
        dict_result = json.loads(str_result)
        features = dict_result['embedding']
        if type(features) == 'str':
            features = json.loads(features)
        # features = dict_result['embedding']
        timeGA = dict_result['timeMin']

    else:
        # Run from method
        features, timeGA = mwu.call_build_embedding_ga(qsar_method=qsar_method,
                                                       training_tsv=training_tsv, prediction_tsv=prediction_tsv,
                                                       remove_log_p=remove_log_p, n_threads=n_threads,
                                                       num_generations=num_generations,
                                                       num_optimizers=num_optimizers, num_jobs=num_jobs,
                                                       descriptor_coefficient=descriptor_coefficient,
                                                       max_length=max_length, threshold=threshold, model_id=1)
    print('embedding = ', features)
    print('Time to run ga  = ', timeGA, 'mins')
    # **************************************************************************************
    # Build model based on embedded descriptors: TODO use api to run this code
    embed_model = mwu.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p,
                                                                    features, 1)
    score_embed = embed_model.do_predictions_score(df_prediction)
    # **************************************************************************************
    # Build model based on all descriptors (except correlated and constant ones) as baseline prediction:
    full_model = mwu.call_build_model(qsar_method, training_tsv, remove_log_p, 1)
    score = full_model.do_predictions_score(df_prediction)
    print(ENDPOINT + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(features)) + '\t' + str(
        features) + '\t' + str(timeGA) + '\n')
    f.write(ENDPOINT + '\t' + str(score) + '\t' + str(score_embed) + '\t' + str(len(features)) + '\t' + str(
        features) + '\t' + str(timeGA) + '\n')
    # f.write(ENDPOINT + '\t' + str(score) + '\n')
    f.flush()


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
    test_directory = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark_TEST/"

    # endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'DevTox', 'LLNA', 'Mutagenicity']
    endpointsTEST = ['LC50DM', 'LC50', 'LD50', 'IGC50', 'LLNA', 'Mutagenicity']
    # endpointsTEST = ['Mutagenicity']

    # *****************************************************************************************************************
    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'
    descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'RDKit-default'
    # descriptor_software = 'ToxPrints-default'
    # descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************

    # output filepath:
    filename = '../data/TEST ' + qsar_method + ' results NUM_GENERATIONS=' + str(num_generations) + \
               ' NUM_OPTIMIZERS=' + str(num_optimizers) + '_' + str(round(time.time() * 1000)) + '.txt'
    f = open(filename, "w")

    f.write(getModelDescription(qsar_method) + '\n')
    # f.write("num_generations="+str(num_generations)+"\t"+"num_optimizers="+str(num_optimizers)+"\t"+"num_jobs="+str(num_jobs) + '\n')

    dictGA = get_ga_description(descriptor_coefficient=descriptor_coefficient, max_length=max_length,
                                n_threads=n_threads, num_optimizers=num_optimizers,
                                num_generations=num_generations, num_jobs=num_jobs,
                                qsar_method=qsar_method,threshold=threshold)
    print(json.dumps(dictGA) + '\n')
    f.write(json.dumps(dictGA) + '\n')

    f.write('ENDPOINT\tscore\tscore_embed\tlen(features)\tfeatures\tTime(min)\n')
    f.flush()

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

        run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path)

    f.close()


def caseStudyPOD():
    """
    Loops through TEST toxicity data sets :
    1. Generates GA embedding using chosen QSAR method
    2. Builds qsar model with and without embedding
    3. Reports results for test set
    4. Exports results to text file
    """

    # change following to folder where TEST sample sets are stored:
    directory = "../datasets/pod/"
    endpoint = 'POD'

    # *****************************************************************************************************************
    # Set GA hyperparameters:
    num_generations = 100  # 100 is recommended for storing in database, use 1 for quick testing (100X faster)
    num_optimizers = 10  # 10 is recommended for storing in database
    qsar_method = 'knn'
    n_threads = 16  # threads to use with RF- TODO
    num_jobs = 4  # jobs to use for GA search for embedding- using 16 has marginal benefit over 4 or 8

    # descriptor_coefficient = 0.002 # Nate's default value
    # max_length = 24 # Nate's default value

    descriptor_coefficient = 0.001
    max_length = 50

    urlHost = 'http://localhost:5004/'
    # urlHost = 'http://v2626umcth819.rtord.epa.gov:5004/'
    useAPI = False  # whether or not to use API call to run GA calcs
    # *****************************************************************************************************************

    # output filepath:
    filename = '../datasets/pod/' + qsar_method + ' results NUM_GENERATIONS=' + str(num_generations) + \
               ' NUM_OPTIMIZERS=' + str(num_optimizers) + '_' + str(round(time.time() * 1000)) + '.txt'
    f = open(filename, "w")

    f.write(getModelDescription(qsar_method) + '\n')
    # f.write("num_generations="+str(num_generations)+"\t"+"num_optimizers="+str(num_optimizers)+"\t"+"num_jobs="+str(num_jobs) + '\n')

    dictGA = get_ga_description(descriptor_coefficient=descriptor_coefficient, max_length=max_length,
                                n_threads=n_threads, num_optimizers=num_optimizers,
                                num_generations=num_generations, num_jobs=num_jobs,
                                qsar_method=qsar_method)
    print(json.dumps(dictGA) + '\n')
    f.write(json.dumps(dictGA) + '\n')

    f.write('ENDPOINT\tscore\tscore_embed\tlen(features)\tfeatures\tTime(min)\n')
    f.flush()

    remove_log_p = False

    training_file_name = endpoint + ' training set.txt'
    prediction_file_name = endpoint + ' test set.txt'

    training_tsv_path = directory + training_file_name
    prediction_tsv_path = directory + prediction_file_name
    # print(training_tsv_path)

    run_endpoint(endpoint, f, remove_log_p, training_tsv_path, prediction_tsv_path)

    f.close()


def bob():
    training_tsv_path = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/LogKOA OPERA/LogKOA OPERA PaDEL-default training.tsv"
    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    print(df_training.shape)


if __name__ == "__main__":
    caseStudyOPERA_RunGA()
    # caseStudyTEST_RunGA()
    # caseStudyPOD()
    # bob()
