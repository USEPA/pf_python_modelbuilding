import scipy

# from models import rf_model_1_1 as rf1_1
# from models import rf_model_1_2 as rf1_2
# from models import rf_model_1_3 as rf1_3
# from models import rf_model_1_4 as rf1_4
from models import df_utilities as DFU

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

import GeneticOptimizer as go
from os.path import exists

strWaterSolubility = 'Water solubility'
strVaporPressure = 'Vapor pressure'
strHenrysLawConstant = 'Henry\'s law constant'
strKMHL='LogKmHL'


def loadEmbedding(filepath):
    f = open(filepath, 'r')
    import json
    data = json.load(f)
    embedding = '\t'.join(data['embedding'])
    print(embedding)
    return embedding





def caseStudyGA():

    # ENDPOINT = "LogKmHL"
    ENDPOINT = "Henry's law constant"

    endpointsOPERA = ["Water solubility", "LogKmHL", "LogKOA", "LogKOC", "LogBCF", "Vapor pressure", "Boiling point",
                      "Melting point","Henry's law constant"]
    endpointsTEST = ['LC50', 'LC50DM', 'IGC50', 'LD50']

    if ENDPOINT in endpointsOPERA:
        IDENTIFIER = 'ID'
        PROPERTY = 'Property'
        DELIMITER = '\t'
        directory = r"C:\\Users\Weeb\\Documents\\QSARmod\\data\\DataSetsBenchmark\\" + ENDPOINT + " OPERA\\" + ENDPOINT + " OPERA T.E.S.T. 5.1"
        trainPath = "training.tsv"
        testPath = "prediction.tsv"
    elif ENDPOINT in endpointsTEST:
        IDENTIFIER = 'CAS'
        PROPERTY = 'Tox'
        DELIMITER = ','
        directory = r"C:\\Users\Weeb\\Documents\\QSARmod\\dataDataSetsBenchmarkTEST_Toxicity\\" + ENDPOINT + r"\\" + ENDPOINT
        trainPath = "_training_set-2d.csv"
        testPath = "_prediction_set-2d.csv"


    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'PaDEL-default'
    # descriptor_software = 'PaDEL_OPERA'

    folder = '../datasets/caseStudyOpera/'
    training_file_name = ENDPOINT + ' OPERA_' + descriptor_software + '_OPERA_training.tsv'
    prediction_file_name = ENDPOINT + ' OPERA_' + descriptor_software + '_OPERA_prediction.tsv'
    prediction_file_name2 = 'Data from Standard ' + ENDPOINT + ' from exp_prop external to ' + ENDPOINT + ' OPERA_' + descriptor_software + '_full.tsv'

    training_tsv_path = folder + training_file_name

    print(training_tsv_path)

    prediction_tsv_path = folder + prediction_file_name
    prediction_tsv_path2 = folder + prediction_file_name2

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # df_training = df_training.loc[:, (df_training != 0).any(axis=0)]

    # Parameters needed to build model:
    n_threads = 20
    remove_log_p_descriptors = False

    model = Pipeline([('standardizer', StandardScaler()), ('estimator', KNeighborsRegressor())])




    features = go.runGA(df_training, IDENTIFIER, PROPERTY, model)

    print(features)


    # df_prediction2 = DFU.load_df_from_file(prediction_tsv_path2, sep='\t')
    # df_prediction2.Property = df_prediction2.Property * (-1)  # fix units to match opera
    # predictions2 = model.do_predictions(df_prediction2)
    #
    # df_preds2 = pd.DataFrame(predictions2, columns=['Prediction'])
    # df_pred2 = df_prediction2[['ID', 'Property']]
    # df_pred2 = pd.merge(df_pred2, df_preds2, how='left', left_index=True, right_index=True)


if __name__ == "__main__":
    # caseStudyOPERA()
    # caseStudyPFAS()

    # loadEmbedding('../datasets/Water solubility_embedding.json')
    caseStudyGA()