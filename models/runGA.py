"""
@author: Todd Martin
2022
"""
import time
from models import df_utilities as DFU
import model_ws_utilities as mwu
import json
# from models import dnn_model as dnn
from models_old import knn_model as knn, svm_model as svm, rf_model as rf, xgb_model as xgb

# Set GA hyperparameters:
num_generations = 100
num_optimizers = 10
qsar_method = 'knn'
n_threads = 16  # threads to use with RF- TODO
num_jobs = 2  # jobs to use for GA search for embedding- using 16 has marginal benefit over 4 or 8
descriptor_coefficient = 0.002
max_length = 24
# threshold = 2  # Nate's value
threshold = 1  # TMM attempt to get more descriptors from stage 1
lanId = 'tmarti02'

# descriptor_coefficient = 0.0
# max_length = 40

urlHost = 'http://localhost:5004/'
# urlHost = 'http://v2626umcth819.rtord.epa.gov:5004/'
useAPI = False  # whether or not to use API call to run GA calcs


# *****************************************************************************************************************


def get_ga_description(descriptor_coefficient, max_length, n_threads, num_generations, num_optimizers, num_jobs,
                       qsar_method, threshold):
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
    # directoryOPERA = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/"
    directoryOPERA = "../datasets_benchmark/"

    # endpointsOPERA = ["LogKoa", "LogKmHL", "Henry's law constant", "LogBCF", "LogOH", "LogKOC",
    #                   "Vapor pressure", "Water solubility", "Boiling point",
    #                   "Melting point", "Octanol water partition coefficient"]

    # endpointsOPERA = ["LogBCF"]
    endpointsOPERA = ["Octanol water partition coefficient"]
    # endpointsOPERA = ["Melting point", "Octanol water partition coefficient"]
    # endpointsOPERA = ["Melting point"]
    # *****************************************************************************************************************
    # descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'
    descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'ToxPrints-default'
    # descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************

    # output filepath:
    filename = '../data/opera ' + descriptor_software + ' ' + qsar_method + ' results NUM_GENERATIONS=' + str(
        num_generations) + \
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

        print(ENDPOINT, descriptor_software)
        directory = directoryOPERA + ENDPOINT + ' OPERA/'

        training_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' training.tsv'
        prediction_file_name = ENDPOINT + ' OPERA ' + descriptor_software + ' prediction.tsv'
        folder = directory

        training_tsv_path = folder + training_file_name
        prediction_tsv_path = folder + prediction_file_name
        # print(training_tsv_path)

        a_run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path)

    f.close()


def a_runCaseStudiesExpProp():
    inputFolder = '../datasets/'
    outputFolder = '../datasets/GA/'

    # output filepath:
    filename = outputFolder + 'exp_prop_gen=' + str(num_generations) + '_opt=' + str(num_optimizers) + '_threshold=' + str(
        threshold) + '_' + str(round(time.time() * 1000)) + '.json'

    datasetNames = []
    # datasetNames.append("HLC from exp_prop and chemprop")
    # datasetNames.append("WS from exp_prop and chemprop")
    # datasetNames.append("VP from exp_prop and chemprop")
    datasetNames.append("LogP from exp_prop and chemprop")
    # datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    # *****************************************************************************************************************
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'ToxPrints-default'
    descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************
    splitting = 'RND_REPRESENTATIVE'

    f = open(filename, "w")
    f.write(getModelDescription(qsar_method) + '\n')
    f.flush()

    for datasetName in datasetNames:
        remove_log_p = False

        if 'LogP' in datasetName:
            remove_log_p = True

        ci = CalculationInfo(descriptor_coefficient=descriptor_coefficient, max_length=max_length,
                             n_threads=n_threads, num_optimizers=num_optimizers,
                             num_generations=num_generations, num_jobs=num_jobs, threshold=threshold,
                             datasetName=datasetName, descriptorSetName=descriptor_software, splittingName=splitting,
                             remove_log_p=remove_log_p)

        print(json.dumps(ci, indent=4))
        print(datasetName)

        training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
        prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

        training_tsv_path = inputFolder + datasetName + '/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/' + prediction_file_name

        a_run_dataset(f, training_tsv_path, prediction_tsv_path, ci)

    f.close()

def a_runCaseStudiesExpPropPFAS():
    inputFolder = '../datasets/'
    outputFolder = '../datasets/GA/'

    # splitting = 'T=PFAS only, P=PFAS'
    ## splitting = 'T=all, P=PFAS'  #dont need to run
    splitting = 'T=all but PFAS, P=PFAS'


    # output filepath:
    filename = outputFolder + splitting+'_gen=' + str(num_generations) + '_opt=' + str(num_optimizers) + '_threshold=' + str(
        threshold) + '_' + str(round(time.time() * 1000)) + '.json'

    datasetNames = []
    datasetNames.append("HLC from exp_prop and chemprop")
    datasetNames.append("WS from exp_prop and chemprop")
    datasetNames.append("VP from exp_prop and chemprop")
    datasetNames.append("LogP from exp_prop and chemprop")
    datasetNames.append("MP from exp_prop and chemprop")
    # datasetNames.append("BP from exp_prop and chemprop")

    # *****************************************************************************************************************
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'
    # descriptor_software = 'ToxPrints-default'
    descriptor_software = 'WebTEST-default'
    # *****************************************************************************************************************

    f = open(filename, "w")
    f.write(getModelDescription(qsar_method) + '\n')
    f.flush()

    for datasetName in datasetNames:

        remove_log_p = False
        if 'LogP' in datasetName:
            remove_log_p = True

        ci = CalculationInfo(descriptor_coefficient=descriptor_coefficient, max_length=max_length,
                             n_threads=n_threads, num_optimizers=num_optimizers,
                             num_generations=num_generations, num_jobs=num_jobs, threshold=threshold,
                             datasetName=datasetName, descriptorSetName=descriptor_software, splittingName=splitting,
                             remove_log_p=remove_log_p)

        print(json.dumps(ci, indent=4))
        print(datasetName)

        training_file_name = datasetName + '_' + descriptor_software + '_' + splitting + "_training.tsv"
        prediction_file_name = training_file_name.replace('training.tsv', 'prediction.tsv')

        training_tsv_path = inputFolder + datasetName + '/PFAS/' + training_file_name
        prediction_tsv_path = inputFolder + datasetName + '/PFAS/' + prediction_file_name

        a_run_dataset(f, training_tsv_path, prediction_tsv_path, ci)

    f.close()


def a_run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path):
    training_tsv = DFU.read_file_to_string(training_tsv_path)
    prediction_tsv = DFU.read_file_to_string(prediction_tsv_path)
    df_prediction = DFU.load_df(prediction_tsv)

    # features=  ['nN', 'SpMAD_Dt', 'ALogP', 'AATS3e', 'ETA_EtaP_L', 'ETA_Alpha', 'minHsOH', 'ATSC1v', 'nBondsD2']
    # features= ['minHsOH', 'ATSC3c', 'nO', 'SpMAD_Dt', 'AATS3e', 'nN', 'ATSC1v', 'ATSC1e', 'nHCsats', 'ETA_Alpha']
    # timeGA=1

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


class CalculationInfo(dict):

    def __init__(self, descriptor_coefficient, max_length, n_threads, num_optimizers, num_generations, num_jobs,
                 threshold, datasetName, descriptorSetName, splittingName, remove_log_p):

        # Add dictionary so can just output the fields we want in the correct order in the json:
        # TODO add remove_log_p to the dictionary
        dict.__init__(self, num_generations=num_generations, num_optimizers=num_optimizers,
                      num_jobs=num_jobs, n_threads=n_threads, max_length=max_length,
                      descriptor_coefficient=descriptor_coefficient,
                      threshold=threshold)  # dont need things like datasetName in the dictionary

        self.descriptor_coefficient = descriptor_coefficient
        self.max_length = max_length
        self.n_threads = n_threads
        self.num_optimizers = num_optimizers
        self.num_generations = num_generations
        self.num_jobs = num_jobs
        self.threshold = threshold
        self.datasetName = datasetName
        self.descriptorSetName = descriptorSetName
        self.splittingName = splittingName
        self.remove_log_p = remove_log_p


class DescriptorEmbedding(dict):

    def __init__(self, createdBy, description, descriptorSetName, embeddingTsv, name, datasetName, importanceTsv,
                 qsarMethod, splittingName, score, scoreEmbed, timeGA):
        dict.__init__(self, createdBy=createdBy, description=description, descriptorSetName=descriptorSetName,
                      embeddingTsv=embeddingTsv, name=name, datasetName=datasetName, importanceTsv=importanceTsv,
                      qsarMethod=qsarMethod, splittingName=splittingName, score=score, scoreEmbed=scoreEmbed,
                      timeGA=timeGA)

        self.createdBy = createdBy
        self.description = description
        self.descriptorSetName = descriptorSetName
        self.embeddingTsv = embeddingTsv
        self.name = name
        self.datasetName = datasetName
        self.importanceTsv = importanceTsv
        self.qsarMethod = qsarMethod
        self.splittingName = splittingName
        self.score = score
        self.scoreEmbed = scoreEmbed
        self.timeGA = timeGA


def a_run_dataset(f, training_tsv_path, prediction_tsv_path, ci):
    training_tsv = DFU.read_file_to_string(training_tsv_path)
    prediction_tsv = DFU.read_file_to_string(prediction_tsv_path)
    df_prediction = DFU.load_df(prediction_tsv)

    if useAPI:
        # Run from API call:
        str_result = mwu.api_call_build_embedding_ga(qsar_method=qsar_method, training_tsv=training_tsv,
                                                     remove_log_p=ci.remove_log_p, n_threads=n_threads,
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
                                                       remove_log_p=ci.remove_log_p, n_threads=n_threads,
                                                       num_generations=num_generations,
                                                       num_optimizers=num_optimizers, num_jobs=num_jobs,
                                                       descriptor_coefficient=descriptor_coefficient,
                                                       max_length=max_length, threshold=threshold, model_id=1)
    print('embedding = ', features)
    print('Time to run ga  = ', timeGA, 'mins')
    # **************************************************************************************
    # Build model based on embedded descriptors: TODO use api to run this code
    embed_model = mwu.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, ci.remove_log_p,
                                                                    features, 1)
    score_embed = embed_model.do_predictions(df_prediction, return_score=True)
    # **************************************************************************************
    # Build model based on all descriptors (except correlated and constant ones) as baseline prediction:
    full_model = mwu.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, ci.remove_log_p,
                                                                    None, 1)
    score = full_model.do_predictions(df_prediction, return_score=True)

    name = ci.datasetName + "_" + ci.descriptorSetName + "_" + str(int(time.time()))

    embeddingTsv = '\t'.join(features)
    description = json.dumps(ci)

    de = DescriptorEmbedding(createdBy=lanId, description=description, descriptorSetName=ci.descriptorSetName,
                             embeddingTsv=embeddingTsv, name=name, datasetName=ci.datasetName, importanceTsv='N/A',
                             qsarMethod=qsar_method, splittingName=ci.splittingName, score=score,
                             scoreEmbed=score_embed,
                             timeGA=timeGA)

    print(json.dumps(de, indent=4))
    f.write(json.dumps(de)+'\n')
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
    test_directory = "../datasets_benchmark_TEST/"

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
                                qsar_method=qsar_method, threshold=threshold)
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

        a_run_endpoint(ENDPOINT, f, remove_log_p, training_tsv_path, prediction_tsv_path)

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

    a_run_endpoint(endpoint, f, remove_log_p, training_tsv_path, prediction_tsv_path)

    f.close()


def bob():
    training_tsv_path = "C:/Users/TMARTI02/OneDrive - Environmental Protection Agency (EPA)/0 java/QSAR_Model_Building/data/datasets_benchmark/LogKOA OPERA/LogKOA OPERA PaDEL-default training.tsv"
    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    print(df_training.shape)


if __name__ == "__main__":
    # a_runCaseStudiesExpProp()
    # a_runCaseStudiesExpPropPFAS()
    # caseStudyOPERA_RunGA()
    # caseStudyTEST_RunGA()
    # caseStudyPOD()
    # bob()


