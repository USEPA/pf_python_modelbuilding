import json

from flask import Flask, request, abort
import logging
import pickle
from dotenv import load_dotenv
load_dotenv()
import model_ws_utilities
from qsar_models import ModelByte, Model
import model_ws_db_utilities as mwdu
from applicability_domain import applicability_domain_utilities as adu


app = Flask(__name__)
# Limit logging output for easier readability
log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

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
    return model_ws_utilities.get_model_info(qsar_method), 200


@app.route('/models/<string:qsar_method>/train', methods=['POST'])
def train(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""

    obj = request.form
    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV

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

    if obj.get('n_jobs'):
        n_jobs = int(obj.get('n_jobs'))
    else:
        n_jobs = 8

    embedding = get_embedding(obj)
    print("embedding = ***\t", embedding, '\t***')

    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    model = model_ws_utilities.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, remove_log_p,
                                                                             embedding, n_jobs=n_jobs)

    if model is None:
        abort(500, 'unknown model training error')

    # Sets status 200 OK
    status = 200

    # If model number provided for storage, stores the model and sets status 201 CREATED instead
    if model_id.strip():
        model_ws_utilities.models[model_id] = model
        status = 201

    if save_to_database:
        mwdu.saveModelToDatabase(model, model_id)
        return 'model bytes saved to database', 202
    else:
        # Returns model bytes
        return pickle.dumps(model), status


@app.route('/models/prediction_applicability_domain', methods=['POST'])
def prediction_applicability_domain():
    """Trains a model for the specified QSAR method on provided data"""

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

    print ("embedding = ***\t",embedding,'\t***')


    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    output = adu.generate_applicability_domain_with_preselected_descriptors(training_tsv=training_tsv,
                                                                              test_tsv=test_tsv,
                                                                              remove_log_p=remove_log_p,
                                                                              embedding=embedding,
                                                                              applicability_domain=applicability_domain)

    # Sets status 200 OK
    status = 200

    # If model number provided for storage, stores the model and sets status 201 CREATED instead

    result = output.to_json(orient='records', lines=True)
    # print(result)
    return result


def get_embedding(obj):

    embedding_tsv = obj.get('embedding_tsv')

    if embedding_tsv is None:
        embedding_tsv_obj = request.files.get('embedding_tsv') # try  reading from file
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
def train_embedding(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""

    print('Enter train_embedding')

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


    model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage
    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    # Can't train a model without data

    num_generations = int(obj.get('num_generations'))
    num_optimizers = int(obj.get('num_optimizers'))
    num_jobs = int(obj.get('num_jobs'))

    max_length = int(obj.get('max_length'))
    threshold = int(obj.get('threshold'))
    descriptor_coefficient = float(obj.get('descriptor_coefficient'))

    # print(num_generations)


    n_threads = obj.get('n_threads')

    embedding, timeMin = model_ws_utilities.call_build_embedding_ga(qsar_method=qsar_method,
                                                                    training_tsv=training_tsv,prediction_tsv=prediction_tsv,
                                                                    remove_log_p=remove_log_p, n_threads=n_threads,
                                                                    num_generations=num_generations,
                                                                    num_optimizers=num_optimizers,
                                                                    num_jobs=num_jobs,
                                                                    descriptor_coefficient=descriptor_coefficient,
                                                                    max_length=max_length,
                                                                    threshold=threshold,
                                                                    model_id=model_id)

    result_obj = {}
    result_obj['embedding'] = embedding
    result_obj['timeMin'] = timeMin
    result_str = json.dumps(result_obj)

    print('result_str=' + result_str)
    return result_str


@app.route('/models/<string:qsar_method>/cross_validate', methods=['POST'])
def cross_validate_fold(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""
    print('run_cross_validate_fold')

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

    if obj.get('num_jobs'):
        n_jobs = int(obj.get('num_jobs'))
    else:
        n_jobs = 8


    embedding = get_embedding(obj)

    if embedding and embedding == 'error':
        abort(400, 'non blank embedding and dont have tab character')

    print ("embedding = ***\t",embedding,'\t***')

    params = obj.get('params')
    params = json.loads(params)  # convert to dictionary

    return model_ws_utilities.call_cross_validate(qsar_method=qsar_method,
                                                  cv_training_tsv=training_tsv, cv_prediction_tsv=prediction_tsv,
                                                  descriptor_names_tsv=embedding, remove_log_p=remove_log_p,
                                                  params=params, n_jobs=n_jobs)

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
#     if model_ws_utilities.models[model_id] is not None:
#         model = model_ws_utilities.models[model_id]
#     else:
#         model = loadModelFromDatabase(model_id)
#
#     # 404 NOT FOUND if no model stored under provided number
#     if model is None:
#         abort(404, 'no stored model with id ' + model_id)
#
#     # Calls the appropriate prediction method and returns the results
#     return model_ws_utilities.call_do_predictions(prediction_tsv, model), 200


def loadModelFromDatabase(model_id):
    session = mwdu.getDatabaseSession()
    query = session.query(ModelByte).filter_by(fk_model_id=model_id).one()
    bytes = query.bytes
    model = pickle.loads(bytes)
    session.flush() #do we need this?
    session.commit() #do we need this?
    return model


@app.route('/models/<string:qsar_method>/predict', methods=['POST'])
def predict(qsar_method):
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


    if model_ws_utilities.models[model_id] is not None:
        # Gets stored model using model number
        model = model_ws_utilities.models[model_id]
    else:
        model = loadModelFromDatabase(model_id)

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Calls the appropriate prediction method and returns the results
    return model_ws_utilities.call_do_predictions(prediction_tsv, model), 200


@app.route('/models/<string:qsar_method>/init', methods=['POST'])
def init(qsar_method):
    """Loads a model and stores it under the provided number"""
    form_obj = request.form
    files_obj = request.files  # Retrieves the files attached to the request
    model_id = form_obj.get('model_id')  # Retrieves the model number to use for persistent storage

    # Can't store a model unless number is specified
    if model_id is None:
        abort(400, 'missing model id')

    # Retrieves the model file from the request files
    model_file = files_obj['model']
    model = None
    if model_file is not None:
        # Loads model bytes
        model = pickle.loads(model_file.read())
        # Stores model under provided number
        model_ws_utilities.models[model_id] = model
    else:
        # Can't store a model if none provided
        abort(400, 'missing model bytes')

    # 400 BAD REQUEST if something is wrong with the loaded bytes
    if model is None:
        abort(400, 'unknown model initialization error')

    # Return storage ID and 201 CREATED
    return model_id, 201



@app.route('/models/<string:qsar_method>/<string:model_id>', methods=['GET'])
def details(qsar_method, model_id):
    """Returns a detailed description of the QSAR model with version and parameter information"""
    model = model_ws_utilities.models[model_id]

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Retrieves details from specified model
    model_details = model_ws_utilities.get_model_details(qsar_method, model)
    if model_details is None:
        # 404 NOT FOUND if model has no detail information
        abort(404, 'no details for stored model with id ' + model_id)

    # Return description and 200 OK
    return model_details, 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
