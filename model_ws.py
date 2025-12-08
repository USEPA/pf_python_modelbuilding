import json

from flask import Flask, request, abort
import logging
import pickle
from dotenv import load_dotenv

load_dotenv()
import model_ws_utilities as mwu
import model_ws_db_utilities as mwdu
from applicability_domain import applicability_domain_utilities as adu

from sklearn2pmml import sklearn2pmml

app = Flask(__name__)
# Limit logging output for easier readability
log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

# use_pmml_pipeline_during_model_building = True # if true use PMMLPipeline with standardizing happening separate during model building
# use_sklearn2pmml = True # if false uses pypmml to load the file. Note: pypmml doesnt handle knn predictions the same way...


"""
Flask webservice to build QSAR models with a variety of modeling strategies (RF, SVM, DNN, XGB...more to come?)
Run with Python 3.9 to avoid problems with parallelizing RF (bug in older versions of joblib backing sklearn)
@author: TMARTI02 (Todd Martin) - RF, base webservice code
@author: GSincl01 (Gabriel Sinclair) - SVM (based on work by CRupakhe), XGB, refactored webservice code
@author: cramslan (Christian Ramsland) - DNN
Repository created 05/21/2021
"""


@app.route('/models/<string:qsar_method>/info', methods=['GET'])
def info(qsar_method):
    """Returns a short, generic description of the QSAR method"""
    return mwu.get_model_info(qsar_method), 200


@app.route('/models/<string:qsar_method>/train', methods=['POST'])
def train(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""

    print('enter train')

    obj = request.form
    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the training data as a TSV

    if obj.get('use_pmml'):
        use_pmml = obj.get('use_pmml', '').lower() == 'true'
    else:
        abort(400, 'missing use_pmml')

    # TODO we might want to have option to not use standardization at all- not needed for RF or XGB (only need for kNN)- standardization causes interoperability problems when loading pmml

    if obj.get('include_standardization_in_pmml'):
        include_standardization_in_pmml = obj.get('include_standardization_in_pmml', '').lower() == 'true'
    else:
        abort(400, 'missing include_standardization_in_pmml')

    if obj.get('save_to_database'):  # Sets boolean remove_log_p from string
        save_to_database = obj.get('save_to_database', '').lower() == 'true'
    else:
        save_to_database = False

    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    # Can't train a model without data
    if training_tsv is None:
        abort(400, 'missing training tsv')

    model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if obj.get('num_jobs'):
        n_jobs = int(obj.get('num_jobs'))
    else:
        n_jobs = 8

    embedding = get_embedding(obj)
    print("embedding = ***\t", embedding, '\t***')

    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    model = mwu.call_build_model_with_preselected_descriptors(qsar_method=qsar_method,
                                                              training_tsv=training_tsv,
                                                              prediction_tsv=prediction_tsv,
                                                              remove_log_p=remove_log_p,
                                                              use_pmml_pipeline=use_pmml,
                                                              include_standardization_in_pmml=include_standardization_in_pmml,
                                                              descriptor_names_tsv=embedding,
                                                              n_jobs=n_jobs, filterColumnsInBothSets=True)

    if model is None:
        abort(500, 'unknown model training error')

    # Sets status 200 OK
    status = 200

    # If model number provided for storage, stores the model and sets status 201 CREATED instead
    if model_id.strip():
        mwu.models[model_id] = model
        status = 201

    if save_to_database:
        mwdu.saveModelToDatabase(model, model_id)
        return 'model bytes saved to database', 202
    else:
        # Returns model bytes
        if use_pmml:

            pmml_file = 'model.pmml'

            sklearn2pmml(model.model_obj,
                         pmml_file)  # write pmml to harddrive temporarily- TODO will this cause problems in docker???

            with open(pmml_file, 'r') as file:
                return bytes(file.read(), 'utf-8'), status  # return pmml as string, todo compress it?

        else:
            return pickle.dumps(model), status


@app.route('/models/prediction_applicability_domain', methods=['POST'])
def prediction_applicability_domain():
    """Generates applicability domain values"""

    obj = request.form

    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    test_tsv = obj.get('test_tsv')  # Retrieves the training data as a TSV

    # print(embedding_tsv)

    applicability_domain = obj.get('applicability_domain')

    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')

    if test_tsv is None:
        test_tsv = request.files.get('test_tsv').read().decode('UTF-8')

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    # Can't train a model without data
    if training_tsv is None:
        abort(400, 'missing training tsv')

    # Need test set to run AD on:
    if test_tsv is None:
        abort(400, 'missing test tsv')

    embedding = get_embedding(obj)

    print("embedding = ***\t", embedding, '\t***')

    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    output = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                            test_tsv=test_tsv,
                                                                            remove_log_p=remove_log_p,
                                                                            embedding=embedding,
                                                                            applicability_domain=applicability_domain,
                                                                            filterColumnsInBothSets=True)
    result = output.to_json(orient='records', lines=True)
    # print(result)
    return result


def get_embedding(obj):
    embedding_tsv = obj.get('embedding_tsv')

    if embedding_tsv is None:
        embedding_tsv_obj = request.files.get('embedding_tsv')  # try  reading from file
        if embedding_tsv_obj is not None:
            embedding_tsv = embedding_tsv_obj.read().decode('UTF-8')

    if embedding_tsv is None:
        return None

    if len(embedding_tsv) == 0:
        embedding = None
    else:
        embedding = []
        if "\t" in embedding_tsv:
            embedding = embedding_tsv.split("\t")
        else:
            return 'error'

    return embedding


@app.route('/models/<string:qsar_method>/embedding', methods=['POST'])
def train_embedding_ga(qsar_method):
    """Post method that trains GA embedding for the specified QSAR method on provided data"""

    print('Enter train_embedding (method to make GA based embedding)')

    obj = request.form

    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if training_tsv is None:
        abort(400, 'missing training tsv')

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the training data as a TSV
    if prediction_tsv is None:
        print('prediction_tsv is none!')
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')

    # if obj.get('save_to_database'):  # Sets boolean remove_log_p from string
    #     save_to_database = obj.get('save_to_database', '').lower() == 'true'
    # else:
    #     save_to_database = False

    # model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if obj.get('use_wards'):  # Sets boolean remove_log_p from string
        use_wards = obj.get('use_wards', '').lower() == 'true'
    else:
        use_wards = False

    num_generations = int(obj.get('num_generations'))
    num_optimizers = int(obj.get('num_optimizers'))
    num_jobs = int(obj.get('num_jobs'))

    max_length = int(obj.get('max_length'))
    threshold = int(obj.get('threshold'))
    descriptor_coefficient = float(obj.get('descriptor_coefficient'))
    n_threads = int(obj.get('n_threads'))

    # print('use_wards = ',use_wards)
    # print('use_wards2 = ', obj.get('use_wards'))
    # print(num_generations)

    embedding, timeMin = mwu.call_build_embedding_ga(qsar_method=qsar_method,
                                                     training_tsv=training_tsv,
                                                     prediction_tsv=prediction_tsv,
                                                     remove_log_p=remove_log_p,
                                                     num_generations=num_generations,
                                                     num_optimizers=num_optimizers,
                                                     num_jobs=num_jobs, n_threads=n_threads,
                                                     descriptor_coefficient=descriptor_coefficient,
                                                     max_length=max_length,
                                                     threshold=threshold,
                                                     use_wards=use_wards,
                                                     run_rfe=False)

    result_obj = {}
    result_obj['embedding'] = embedding
    result_obj['timeMin'] = timeMin
    result_str = json.dumps(result_obj)

    print('result_str=' + result_str)
    return result_str


@app.route('/models/<string:qsar_method>/embedding_importance', methods=['POST'])
def train_embedding_importance(qsar_method):
    """Post method that trains importance based embedding for the specified QSAR method on provided data"""

    print('Enter train_embedding_importance')

    obj = request.form

    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if training_tsv is None:
        abort(400, 'missing training tsv')

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the training data as a TSV
    if prediction_tsv is None:
        print('prediction_tsv is none!')
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')

    # if obj.get('save_to_database'):  # Sets boolean remove_log_p from string
    #     save_to_database = obj.get('save_to_database', '').lower() == 'true'
    # else:
    #     save_to_database = False

    # model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if obj.get('use_wards'):  # Sets boolean remove_log_p from string
        use_wards = obj.get('use_wards', '').lower() == 'true'
    else:
        use_wards = False

    if obj.get('run_rfe'):  # Sets boolean remove_log_p from string
        run_rfe = obj.get('run_rfe', '').lower() == 'true'
    else:
        run_rfe = False

    if obj.get('use_permutative'):  # Sets boolean remove_log_p from string
        use_permutative = obj.get('use_permutative', '').lower() == 'true'
    else:
        use_permutative = False

    # TODO add importance_type for when not using permutative importance

    num_generations = int(obj.get('num_generations'))
    fraction_of_max_importance = float(obj.get('fraction_of_max_importance'))
    min_descriptor_count = int(obj.get('min_descriptor_count'))
    max_descriptor_count = int(obj.get('max_descriptor_count'))
    n_threads = int(obj.get('n_threads'))

    embedding, timeMin = mwu.call_build_embedding_importance(qsar_method=qsar_method,
                                                             training_tsv=training_tsv,
                                                             prediction_tsv=prediction_tsv,
                                                             remove_log_p_descriptors=remove_log_p,
                                                             n_threads=n_threads,
                                                             num_generations=num_generations,
                                                             use_permutative=use_permutative,
                                                             run_rfe=run_rfe,
                                                             fraction_of_max_importance=fraction_of_max_importance,
                                                             min_descriptor_count=min_descriptor_count,
                                                             max_descriptor_count=max_descriptor_count,
                                                             use_wards=use_wards)

    result_obj = {}
    result_obj['embedding'] = embedding
    result_obj['timeMin'] = timeMin
    result_str = json.dumps(result_obj)

    print('result_str=' + result_str)
    return result_str


@app.route('/models/<string:qsar_method>/embedding_lasso', methods=['POST'])
def train_embedding_lasso(qsar_method):
    """Post method that trains importance based embedding for the specified QSAR method on provided data"""

    print('Enter train_embedding_importance')

    obj = request.form

    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if training_tsv is None:
        abort(400, 'missing training tsv')

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the training data as a TSV
    if prediction_tsv is None:
        print('prediction_tsv is none!')
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if obj.get('run_rfe'):  # Sets boolean remove_log_p from string
        run_rfe = obj.get('run_rfe', '').lower() == 'true'
    else:
        run_rfe = False

    n_threads = int(obj.get('n_threads'))

    embedding, timeMin = mwu.call_build_embedding_lasso(qsar_method=qsar_method,
                                                        training_tsv=training_tsv,
                                                        prediction_tsv=prediction_tsv,
                                                        remove_log_p_descriptors=remove_log_p,
                                                        n_threads=n_threads,
                                                        run_rfe=run_rfe)

    result_obj = {}
    result_obj['embedding'] = embedding
    result_obj['timeMin'] = timeMin
    result_str = json.dumps(result_obj)

    print('result_str=' + result_str)
    return result_str


@app.route('/models/<string:qsar_method>/cross_validate', methods=['POST'])
def cross_validate_fold(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""
    print('\n********************************************************************************************************')
    print('run_cross_validate_fold')

    obj = request.form

    if obj.get('use_pmml'):
        use_pmml = obj.get('use_pmml', '').lower() == 'true'
    else:
        abort(400, 'missing use_pmml')

    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if training_tsv is None:
        abort(400, 'missing training tsv')

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the training data as a TSV
    if prediction_tsv is None:
        print('prediction_tsv is none!')
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')

    # print('prediction_tsv',prediction_tsv)

    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if obj.get('num_jobs'):
        n_jobs = int(obj.get('num_jobs'))
    else:
        n_jobs = 8

    embedding = get_embedding(obj)

    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    print("embedding = ***\t", embedding, '\t***')

    hyperparameters = obj.get('hyperparameters')
    hyperparameters = json.loads(hyperparameters)  # convert to dictionary

    return mwu.call_cross_validate(qsar_method=qsar_method,
                                   cv_training_tsv=training_tsv, cv_prediction_tsv=prediction_tsv,
                                   descriptor_names_tsv=embedding,
                                   use_pmml_pipeline=use_pmml,
                                   remove_log_p=remove_log_p,
                                   hyperparameters=hyperparameters, n_jobs=n_jobs)


#
# @app.route('/models/<string:qsar_method>/predictsa', methods=['POST'])
# def predictpythonstorage(qsar_method):
#     """Makes predictions for a stored model on provided data"""
#     obj = request.form
#     model_id = obj.get('model_id')  # Retrieves the model number to use
#
#     prediction_tsv = obj.get('prediction_tsv')  # Retrieves the prediction data as a TSV
#     if prediction_tsv is None:
#         prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')
#
#     # Can't make predictions without data
#     if prediction_tsv is None:
#         abort(400, 'missing prediction tsv')
#     # Can't make predictions without a model
#     if model_id is None:
#         abort(400, 'missing model id')
#
#     # Gets stored model using model number
#     model = None
#     if mwu.models[model_id] is not None:
#         model = mwu.models[model_id]
#     else:
#         model = loadModelFromDatabase(model_id)
#
#     # 404 NOT FOUND if no model stored under provided number
#     if model is None:
#         abort(404, 'no stored model with id ' + model_id)
#
#     # Calls the appropriate prediction method and returns the results
#     return mwu.call_do_predictions(prediction_tsv, model), 200


@app.route('/models/predictDB', methods=['POST', 'GET'])
def predictDB():
    """Automates prediction and AD for single smiles using model in database"""
    if request.method == 'POST':
        obj = request.form
        smiles = obj.get('smiles')  # Retrieves the model number to use
        model_id = obj.get('model_id')
    elif request.method == 'GET':
        # Handle GET request here
        smiles = request.args.get('smiles')  # Retrieves the model number to use
        model_id = request.args.get('model_id')

    if not smiles or not model_id:
        return "Both smiles and model_id are required", 400
    return mwdu.predictFromDB(model_id, smiles, mwu)


@app.route('/models/predict', methods=['POST'])
def predict():
    """Makes predictions for a stored model on provided data"""
    obj = request.form
    model_id = obj.get('model_id')  # Retrieves the model number to use

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the prediction data as a TSV
    if prediction_tsv is None:
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')

    # Can't make predictions without data
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')
    # Can't make predictions without a model
    if model_id is None:
        abort(400, 'missing model id')

    if mwu.models[model_id] is not None:
        # Gets stored model using model number
        model = mwu.models[model_id]
    else:
        abort(400,'Need to init model or use predictDB API call instead')

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Calls the appropriate prediction method and returns the results
    return mwu.call_do_predictions(prediction_tsv, model), 200



@app.route('/models/plot', methods=['POST'])
def generate_plot():
    """Makes predictions for a stored model on provided data"""
    obj = request.form
    model_id = obj.get('model_id')  # Retrieves the model number to use

    model_name = obj.get('model_name')  # Retrieves the model number to use

    plot_type = obj.get('plot_type')

    training_tsv = obj.get('training_tsv')  # Retrieves the prediction data as a TSV
    if training_tsv is None:
        training_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')

    prediction_tsv = obj.get('prediction_tsv')  # Retrieves the prediction data as a TSV
    if prediction_tsv is None:
        prediction_tsv = request.files.get('prediction_tsv').read().decode('UTF-8')

    # Can't make predictions without data
    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')
    # Can't make predictions without a model
    if model_id is None:
        abort(400, 'missing model id')

    if mwu.models[model_id] is not None:
        # Gets stored model using model number
        model = mwu.models[model_id]
    else:
        abort(400,'Need to init model first')

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Calls the appropriate prediction method and returns the results
    return mwu.call_generate_plot(training_tsv,prediction_tsv, model, model_name, plot_type), 200


@app.route('/models/initPMML', methods=['POST'])
def initPMML():
    """Loads a model and stores it under the provided number"""

    # print('enter initPMML')

    # form_obj = request.form
    # files_obj = request.files  # Retrieves the files attached to the request

    form_obj = request.get_json()
    model_id = form_obj.get('model_id')  # Retrieves the model number to use for persistent storage
    # print('form_obj',form_obj)

    # print('model_id=',model_id)

    # Can't store a model unless number is specified
    if model_id is None:
        abort(400, 'missing model id')

    if model_id in mwu.models:
        print('already have model in memory')
        model = mwu.models[model_id]
        return model.get_model_description(), 201

    # if mwu.models[model_id] is not None:
    #     model = mwu.models[model_id]
    #     print('Already have model loaded, description=:',model.get_model_description)
    #     return model.get_model_description(), 201

    # Retrieves the model file from the request files
    # model_file = files_obj['model']
    model_file = form_obj['model']

    # print(model_file)
    # return ""

    print('use_sklearn2mml in form_obj:', form_obj.get('use_sklearn2pmml'))

    if form_obj.get('use_sklearn2pmml') is None:
        abort(400, 'missing use_sklearn2pmml')

    if isinstance(form_obj.get('use_sklearn2pmml'), str):
        use_sklearn2pmml = form_obj.get('use_sklearn2pmml', '').lower() == 'true'
    else:
        use_sklearn2pmml = form_obj.get('use_sklearn2pmml')

    print('use_sklearn2mml variable', form_obj.get('use_sklearn2pmml'))

    model = None

    # print (files_obj)

    if model_file is None:
        print('Missing model bytes')
        # Can't store a model if none provided
        abort(400, 'missing model bytes')

    print('have model file, type = ', type(model_file))
    pmml_file_path = 'model_api.pmml'

    # model_file.save(pmml_file_path, buffer_size=16384)  # save to hard drive so can load it

    f = open(pmml_file_path, "w")
    f.write(model_file)
    f.close()

    print('wrote pmmlfile to harddrive')

    if isinstance(form_obj['is_binary'], bool):
        is_binary = form_obj['is_binary']
    else:
        is_binary = form_obj['is_binary'].lower == 'true'

    # print('is_categorical', is_categorical)
    model = mwu.instantiateModelForPrediction(qsar_method=form_obj['qsar_method'],
                                              is_binary=is_binary, pmml_file_path=pmml_file_path,
                                              use_sklearn2pmml=use_sklearn2pmml)  # init from model_ws should take care of this when doing from java
    model.set_details(details=form_obj)

    # print(model.model_obj)
    # model.embedding = model.model_obj.dataDictionary.fieldNames
    # model.embedding.remove('Property')

    # Stores model under provided number
    mwu.models[model_id] = model

    print('After init model_description =', model.get_model_description())

    # 400 BAD REQUEST if something is wrong with the loaded bytes
    if model is None:
        print('Model is none')
        abort(400, 'unknown model initialization error')

    # Return storage ID and 201 CREATED
    return model.get_model_description(), 201


@app.route('/models/initPickle', methods=['POST'])
def initPickle():
    """Loads a model and stores it under the provided number"""
    print('enter initPickle')

    form_obj = request.form
    # print('form_obj',form_obj)
    files_obj = request.files  # Retrieves the files attached to the request

    model_id = form_obj.get('model_id')  # Retrieves the model number to use for persistent storage

    # Can't store a model unless number is specified
    if model_id is None:
        abort(400, 'missing model id')

    # Retrieves the model file from the request files
    model_file = files_obj['model']

    # print (files_obj)

    if model_file is not None:

        print('have model file, type = ', type(model_file))

        # print('is_categorical', is_categorical)
        model = pickle.loads(model_file.read())

        # printEqn(model)

        if not hasattr(model, "is_binary"):
            print('model.is_binary is none, setting to false')
            model.is_binary = False

            # if form_obj['is_binary']:
            #     if isinstance(form_obj['is_binary'], bool):
            #         model.is_binary = form_obj['is_binary']
            #     else:
            #         model.is_binary = form_obj['is_binary'].lower == 'true'
            #     print(model.is_binary)

        # Stores model under provided number
        mwu.models[model_id] = model

        print('After init model_description =', model.get_model_description())
        return model.get_model_description(), 201

    else:
        # Can't store a model if none provided
        abort(400, 'missing model bytes')


@app.route('/models/<string:model_id>', methods=['GET'])
def details(model_id):
    """Returns a detailed description of the QSAR model with version and parameter information (also inits the model if needed)"""

    # model = mwu.models[model_id]
    model = mwdu.init_model(model_id, mwu)

    # print('details3', model.get_model_description())

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Retrieves details from specified model
    model_details = mwu.get_model_details(model)
    if model_details is None:
        # 404 NOT FOUND if model has no detail information
        abort(404, 'no details for stored model with id ' + model_id)

    # Return description and 200 OK
    return model_details, 200



@app.route('/models', methods=['GET'])
def available_models():
    """Returns a detailed description of the QSAR model with version and parameter information (also inits the model if needed)"""

    # model = mwu.models[model_id]
    models = mwdu.get_available_models()

    # Return description and 200 OK
    return models, 200

@app.route('/models/reg_coeff/<string:model_id>', methods=['GET'])
def model_coeffs(model_id):
    """Returns a detailed description of the QSAR model with version and parameter information"""
    model = mwdu.init_model(model_id, mwu)

    # print('details3', model.get_model_description())

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    model.getOriginalRegressionCoefficients()

    # TODO add code to return the coefficients instead of just printing to python io

    return "OK", 200


def printEqn(model):
    estimator = model.model_obj.steps[1][1]
    coefficients = estimator.coef_
    intercept = estimator.intercept_
    print('coefficients', coefficients)
    print('intercept', intercept)

    # # Construct the equation string
    # equation = "log(p/1-p) = "
    # if intercept:
    #     equation += f"{intercept[0]:.2f} + "
    # for i, coef in enumerate(coefficients[0]):
    #     equation += f"{coef:.2f} * {feature_names[i]} + "
    #
    # # Remove the trailing " + "
    # equation = equation[:-3]
    #
    # print(equation)


@app.route('/models/<string:model_id>/object', methods=['GET'])
def model_obj(model_id):
    """Returns model object"""

    model = mwu.models[model_id]

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    if model.model_obj is None:
        # 404 NOT FOUND if model has no detail information
        abort(404, 'no model object for id ' + model_id)

    # Return model_obj
    return model.model_obj, 200




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
