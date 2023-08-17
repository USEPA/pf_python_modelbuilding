import os

from xgboost import XGBRegressor

import pypmml
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn_pmml_model.ensemble import PMMLForestClassifier, PMMLForestRegressor
from sklearn_pmml_model.neighbors import PMMLKNeighborsRegressor

from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import make_pmml_pipeline, sklearn2pmml

from pypmml import Model as Model_pypmml


import numpy as np
import utils
from models import df_utilities as DFU
import model_ws_utilities as mwu
import pickle


import json
import pandas as pd
from sklearn.preprocessing import StandardScaler




from sklearn.pipeline import Pipeline
from scipy import stats


urlHost='http://localhost:5004/'

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



def runTestSet():

    # qsar_method = 'rf'
    qsar_method = 'xgb'

    n_threads = 20
    # use_wards = False

    endpoint = "LLNA"
    # endpoint = "Mutagenicity"

    descriptor_software = 'WebTEST-default'

    if endpoint == 'Octanol water partition coefficient':
        remove_log_p_descriptors = True
    else:
        remove_log_p_descriptors = False

    embedding = None


    folder = os.path.join(utils.get_project_root(), 'datasets_benchmark_TEST',endpoint + ' TEST')
    training_file_name = endpoint + ' TEST ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' TEST ' + descriptor_software + ' prediction.tsv'

    training_tsv_path = os.path.join(folder, training_file_name)
    prediction_tsv_path = os.path.join(folder, prediction_file_name)

    print(prediction_tsv_path)

    runSet(embedding, n_threads,  training_tsv_path,prediction_tsv_path, qsar_method, remove_log_p_descriptors,True)

def runOperaSet():

    # qsar_method = 'rf'
    # qsar_method = 'xgb'
    qsar_method = 'svm'
    # qsar_method = 'knn'

    n_threads = 20
    # use_wards = False

    # endpoint = "Henry's law constant"
    # endpoint = "Water solubility"
    endpoint = 'LogKOA'

    descriptor_software = 'WebTEST-default'

    if endpoint == 'Octanol water partition coefficient':
        remove_log_p_descriptors = True
    else:
        remove_log_p_descriptors = False

    embedding = None

    folder = os.path.join(utils.get_project_root(), 'datasets_benchmark',endpoint + ' OPERA')
    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'

    training_tsv_path = os.path.join(folder, training_file_name)
    prediction_tsv_path = os.path.join(folder, prediction_file_name)

    print(prediction_tsv_path)

    runSetWithAPI(embedding, n_threads,  training_tsv_path,prediction_tsv_path, qsar_method, remove_log_p_descriptors,False)
    # runSet(embedding, n_threads, training_tsv_path, prediction_tsv_path, qsar_method, remove_log_p_descriptors, False)
    # runSetKnn(embedding, n_threads, training_tsv_path, prediction_tsv_path, qsar_method, remove_log_p_descriptors, False)

    # from sklearn2pmml import make_pmml_pipeline, sklearn2pmml
    # pmml_pipeline = make_pmml_pipeline(model.model_obj)
    # sklearn2pmml(pmml_pipeline, "model.pmml")

    # fileObj2 = open('model.obj', 'rb')
    # pickleModel = pickle.load(fileObj2)
    # predictionsPickle = pickleModel.do_predictions(df_prediction, return_score=False)


def runSetWithAPI(embedding, n_threads,training_tsv_path, prediction_tsv_path, qsar_method, remove_log_p_descriptors,is_categorical):
    model_id=1;
    # info=mwu.api_info(qsar_method=qsar_method,url_host=urlHost)
    # print(info)

    with open(training_tsv_path, 'r') as file:
        training_tsv = file.read()

    with open(prediction_tsv_path, 'r') as file:
        prediction_tsv = file.read()

    # pmml_path=os.path.join(utils.get_project_root(), 'model_api.pmml')
    # with open(pmml_path, 'r') as file:
    #     pmml_string = file.read()



    pmml_string = mwu.api_call_build_with_preselected_descriptors(qsar_method, training_tsv,
                                                            remove_log_p=remove_log_p_descriptors,
                                                            embedding_tsv=embedding, num_jobs=n_threads,
                                                           url_host=urlHost,model_id=model_id)

    # print(type(pmml_string))

    details = mwu.api_call_details(url_host=urlHost,model_id=model_id)
    print('details1', details)

    predictions = mwu.api_call_predict(url_host=urlHost,prediction_tsv=prediction_tsv, model_id=model_id)
    # print('predictions1', predictions)


    details = json.loads(details)

    # for key in details:
    #     details[key] = json.dumps(details[key]) # make flat
        # print(key, details[key])

    result = mwu.api_call_init(qsar_method=qsar_method, model_string=pmml_string, url_host=urlHost,
                               model_id=model_id, details=details)

    # print(result)

    details = mwu.api_call_details(url_host=urlHost,model_id=model_id)
    print('details2', details)


    predictions = mwu.api_call_predict(url_host=urlHost,prediction_tsv=prediction_tsv, model_id=model_id)

    # print(pmml)


def runSet(embedding, n_threads,training_tsv_path, prediction_tsv_path, qsar_method, remove_log_p_descriptors,is_categorical):

    # # with open(training_tsv_path, 'r') as file:
    # #     training_tsv = file.read()
    # #
    # #
    # # model = mwu.call_build_model_with_preselected_descriptors(qsar_method=qsar_method, training_tsv=training_tsv, remove_log_p=remove_log_p_descriptors, descriptor_names_tsv=embedding,
    # #                                               n_jobs=n_threads)
    # #
    # #
    # df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')

    # # print(df_prediction.shape)
    # predictions = model.do_predictions(df_prediction, return_score=False)


    # print(model.embedding)
    # fileObj = open('model.obj', 'wb')
    # pickle.dump(model,fileObj)
    # fileObj.close()
    # pmml_pipeline.compact=True

    # print('storing file to pmml:')
    # sklearn2pmml(model.model_obj, "model.pmml")


    # predictions = model.predict(test_df)
    # print(predictions)


    # print('loading model from pmml:')
    # modelFromPMML = mwu.instantiateModel2(qsar_method=qsar_method,is_categorical=is_categorical)  # init from model_ws should take care of this when doing from java
    # modelFromPMML.model_obj = Model_pypmml.fromFile(pmml_path)
    #
    # modelFromPMML.embedding = modelFromPMML.model_obj.dataDictionary.fieldNames
    # modelFromPMML.embedding.remove('Property')
    # print(modelFromPMML.embedding)
    #
    # print('running predictions for pmml model:')
    # modelFromPMML.do_predictions(df_prediction, return_score=False)

    # print(pmml_path)

    # modelFromPMML = mwu.instantiateModel2(qsar_method=qsar_method,is_categorical=is_categorical)  # init from model_ws should take care of this when doing from java
    # modelFromPMML.set_model_obj(pmml_path, qsar_method=qsar_method)

    # print(type(modelFromPMML))
    # print('here1', type(modelFromPMML.model_obj))

    # clf = PMMLKNeighborsRegressor(pmml=pmml_path)
    # modelFromPMML.embedding = []
    # for key in modelFromPMML.model_obj.field_mapping.keys():
    #     if 'standardScaler' not in str(key) and str(key)!='Property':
    #         # print (key)
    #         modelFromPMML.embedding.append(key)

    # modelFromPMML.embedding = modelFromPMML.model_obj.dataDictionary.fieldNames
    # modelFromPMML.embedding.remove('Property')

    # print(clf.field_mapping.keys())

    # embedding = list(df_prediction.columns)
    # embedding.remove("Property")
    # embedding.remove("ID")

    # print(len(embedding))

    # pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, modelFromPMML.embedding)

    # predictions = modelFromPMML.model_obj.predict(pred_features)

    # predictions=modelFromPMML.do_predictions(df_prediction=df_prediction)

    # print(predictions)
    # print(clf.score(pred_features, pred_labels))

    # import pandas as pd
    # pmml_path = os.path.join(utils.get_project_root(), 'model_api.pmml')
    # test_df = pd.read_csv(prediction_tsv_path)
    # model = PMMLForestRegressor(pmml=pmml_path)
    # predictions = model.predict(test_df)

    # ************************************************************************************************

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn2pmml.pipeline import PMMLPipeline
    from sklearn.pipeline import Pipeline
    from scipy import stats

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    train_ids, train_labels, train_features, train_column_names, is_categorical = \
        DFU.prepare_instances(df_training, "training", remove_log_p_descriptors, False)
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)

    # model_obj = PMMLPipeline([('standardizer', StandardScaler()), ('estimator', RandomForestRegressor())])
    model_obj = PMMLPipeline([('estimator', RandomForestRegressor())])
    model_obj.fit(train_features, train_labels)
    sklearn2pmml(model_obj,'bob.pmml')

    # predictions = model_obj.predict(pred_features)
    # print (predictions)
    # score = stats.pearsonr(predictions, pred_labels)[0]
    # score = score * score
    # print(r'R2 for Test data = {score}'.format(score=score))

    # ************************************************************************************************

    # df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    # pred_features = df_prediction
    #
    model_obj2 = PMMLForestRegressor(pmml='bob.pmml')
    predictions = model_obj2.predict(pred_features)
    predictions = np.array(predictions)
    # print (predictions)
    score = stats.pearsonr(predictions, pred_labels)[0]
    score = score * score
    print(r'sklearn2pmml R2 for Test data = {score}'.format(score=score))

    model_obj3 = Model_pypmml.fromFile('bob.pmml')
    predictions = model_obj3.predict(pred_features)
    predictions = np.array(predictions[predictions.columns[0]])
    # print (predictions)
    score = stats.pearsonr(predictions, pred_labels)[0]
    score = score * score
    print(r'pypmml R2 for Test data = {score}'.format(score=score))

    # ************************************************************************************************


def runSetKnn(embedding, n_threads,training_tsv_path, prediction_tsv_path, qsar_method, remove_log_p_descriptors,is_categorical):

    use_scaling = True

    df_training = DFU.load_df_from_file(training_tsv_path, sep='\t')
    train_ids, train_labels, train_features, train_column_names, is_categorical = \
        DFU.prepare_instances(df_training, "training", remove_log_p_descriptors, False)
    df_prediction = DFU.load_df_from_file(prediction_tsv_path, sep='\t')
    pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)

    if use_scaling:
        ss = StandardScaler()
        train_features = pd.DataFrame(ss.fit_transform(train_features), columns=train_features.columns)
        pred_features = pd.DataFrame(ss.transform(pred_features), columns=pred_features.columns)

    # model_obj = PMMLPipeline([('estimator', RandomForestRegressor())])
    model_obj = PMMLPipeline([('estimator', KNeighborsRegressor())])
    model_obj.fit(train_features, train_labels)

    predictions = model_obj.predict(pred_features)
    predictions = np.array(predictions)
    score = stats.pearsonr(predictions, pred_labels)[0]
    score = score * score
    print(r'R2 for Test data = {score}'.format(score=score))

    sklearn2pmml(model_obj,'bob.pmml')

    # model_obj2 = PMMLForestRegressor(pmml='bob.pmml')
    model_obj2 = PMMLKNeighborsRegressor(pmml='bob.pmml')
    predictions = model_obj2.predict(pred_features)
    predictions = np.array(predictions)
    # print (predictions)
    score = stats.pearsonr(predictions, pred_labels)[0]
    score = score * score
    print(r'sklearn2pmml R2 for Test data = {score}'.format(score=score))

    # model_obj3 = Model_pypmml.fromFile('bob.pmml')
    # predictions = model_obj3.predict(pred_features)
    # predictions = np.array(predictions[predictions.columns[0]])
    # # print (predictions)
    # score = stats.pearsonr(predictions, pred_labels)[0]
    # score = score * score
    # print(r'pypmml R2 for Test data = {score}'.format(score=score))

    # ************************************************************************************************

def runNateExample():
    file_path = r"O:\Public\CharlieLowe\PMML_test\{filename}"

    # model_name = r"LogP_random_rf_model.pmml"
    model_name = r"sklearn_rf.pmml"
    test_data = r"LogP_random_test_Transformed.csv"

    model_path = file_path.format(filename=model_name)

    test_df = pd.read_csv(file_path.format(filename=test_data))
    model = PMMLForestRegressor(pmml=model_path)

    predictions = model.predict(test_df)
    print (predictions)

    model_obj3 = Model_pypmml.fromFile(model_path)
    predictions = model_obj3.predict(test_df)
    print (predictions)


def runNateExample2():
    '''
    Build using Charlie's data set
    :return:
    '''
    file_path = r"O:\Public\CharlieLowe\PMML_test\{filename}"
    model_path = 'logp.pmml'
    training_csv_path = file_path.format(filename='LogP_random_train_Transformed.csv')
    prediction_csv_path = file_path.format(filename='LogP_random_test_Transformed.csv')

    df_training = DFU.load_df_from_file(training_csv_path, sep='\t')
    df_prediction = DFU.load_df_from_file(prediction_csv_path, sep='\t')

    # train_ids, train_labels, train_features, train_column_names, is_categorical = \
    #     DFU.prepare_instances(df_training, "training", False, False)
    # pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)

    train_features = df_training[df_training.columns.difference(["DTXSID", "LogP"],sort=False)]
    train_labels = df_training["LogP"]

    pred_features = df_prediction[train_features.columns]
    pred_labels = df_prediction["LogP"]

    # model_obj = Pipeline([('standardizer', StandardScaler()), ('estimator', RandomForestRegressor())])
    model_obj = PMMLPipeline([('standardizer', StandardScaler()), ('estimator', RandomForestRegressor())])
    # model_obj = PMMLPipeline([('standardizer', StandardScaler()), ('estimator', XGBRegressor())])



    # model_obj = make_pmml_pipeline(model_obj)

    # model_obj = Pipeline([('standardizer', StandardScaler()), ('estimator', RandomForestRegressor())])
    # model_obj = make_pmml_pipeline(model_obj)

    # model_obj = PMMLPipeline([('estimator', RandomForestRegressor())])
    # model_obj = PMMLPipeline([('estimator', KNeighborsRegressor())])

    # print(train_features)

    # ss = StandardScaler()
    # train_features = pd.DataFrame(ss.fit_transform(train_features), columns=train_features.columns)
    # pred_features = pd.DataFrame(ss.fit_transform(pred_features), columns=pred_features.columns)

        # print(train_features)

    # print(train_features.shape)
    # print(pred_features.shape)

    # train_labels = pd.Series(train_labels, name="Property")



    model_obj.fit(train_features, train_labels)


    print(model_obj.get_params())


    sklearn2pmml(model_obj, model_path)
    predictions = model_obj.predict(pred_features)

    print(predictions)

    # print(train_labels)

    print("")


    # model_from_file = PMMLForestRegressor(pmml=model_path)
    model_from_file = pypmml.Model.fromFile(model_path)


    # model_from_file = PMMLKNeighborsRegressor(pmml=model_path)
    predictions = model_from_file.predict(pred_features)

    print("Preds from reload")
    print(predictions)

    # score = stats.pearsonr(predictions, pred_labels)[0]
    # score = score * score
    # print(r'R2 for Test data = {score}'.format(score=score))

    # model_obj3 = Model_pypmml.fromFile(model_path)
    # predictions = model_obj3.predict(test_df)
    # print(predictions)


if __name__ == "__main__":
    # runOperaSet()
    # runTestSet()
    # runNateExample()
    runNateExample2()

    # from sklearn_pmml_model.ensemble import PMMLForestRegressor
    # import pandas as pd
    #
    # file_path = r"O:\Public\CharlieLowe\PMML_test\{filename}"
    # model_name = r"LogP_random_rf_model.pmml"
    # test_data = r"LogP_random_test_Transformed.csv"
    #
    # test_df = pd.read_csv(file_path.format(filename=test_data))
    # model = PMMLForestRegressor(pmml=file_path.format(filename=model_name))
    # predictions = model.predict(test_df)
    # print(predictions)
    #
    # model = Model_pypmml.fromFile(file_path.format(filename=model_name))
    # predictions = model.predict(test_df)
    # print(predictions)
