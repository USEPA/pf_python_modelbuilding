from flask import Flask, request, abort
import logging
import pickle
from dotenv import load_dotenv
load_dotenv()
import model_ws_utilities
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from qsar_models import ModelByte, Model



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
    embedding_tsv = obj.get('embedding_tsv')

    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if embedding_tsv is None:
        embedding_tsv_obj = request.files.get('embedding_tsv')
        if embedding_tsv_obj is not None:
            embedding_tsv = embedding_tsv_obj.read().decode('UTF-8')

    model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage
    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    # Can't train a model without data
    if training_tsv is None:
        abort(400, 'missing training tsv')

    if embedding_tsv is None or len(embedding_tsv)==0:
        # Calls the appropriate model training method, throwing 500 SERVER ERROR if it does not give back a good model
        model = model_ws_utilities.call_build_model(qsar_method, training_tsv, remove_log_p)
        if model is None:
            abort(500, 'unknown model training error')
    else:
        embedding = []
        if "," in embedding_tsv:
            embedding = embedding_tsv.split(",")
        elif "\t" in embedding_tsv:
            embedding = embedding_tsv.split("\t")
        model = model_ws_utilities.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, embedding)
        if model is None:
            abort(500, 'unknown model training error')
            
    # Sets status 200 OK
    status = 200
    # If model number provided for storage, stores the model and sets status 201 CREATED instead
    if model_id.strip():
        model_ws_utilities.models[model_id] = model
        status = 201

    # Returns model bytes
    return pickle.dumps(model), status

@app.route('/models/<string:qsar_method>/trainpythonstorage', methods=['POST'])
def trainpythonstorage(qsar_method):
    """Trains a model for the specified QSAR method on provided data"""
    obj = request.form
    training_tsv = obj.get('training_tsv')  # Retrieves the training data as a TSV
    embedding_tsv = obj.get('embedding_tsv')

    if training_tsv is None:
        training_tsv = request.files.get('training_tsv').read().decode('UTF-8')
    if embedding_tsv is None:
        embedding_tsv_obj = request.files.get('embedding_tsv')
        print(training_tsv)
        if embedding_tsv_obj is not None:
            embedding_tsv = embedding_tsv_obj.read().decode('UTF-8')

    model_id = obj.get('model_id')  # Retrieves the model number to use for persistent storage
    if obj.get('remove_log_p'):  # Sets boolean remove_log_p from string
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    # Can't train a model without data
    if training_tsv is None:
        abort(400, 'missing training tsv')

    if embedding_tsv is None or len(embedding_tsv)==0:
        # Calls the appropriate model training method, throwing 500 SERVER ERROR if it does not give back a good model
        model = model_ws_utilities.call_build_model(qsar_method, training_tsv, remove_log_p)
        if model is None:
            print('model is None')
            abort(500, 'unknown model training error')
        else:
            print("postgresql://" + os.getenv("DEV_QSAR_USER") + ":" + os.getenv("DEV_QSAR_PASS") + "@" + os.getenv("DEV_QSAR_HOST") + ":" + os.getenv("DEV_QSAR_PORT") + "/" + os.getenv("DEV_QSAR_DATABASE"))
            engine = create_engine("postgresql://" + os.getenv("DEV_QSAR_USER") + ":" + os.getenv("DEV_QSAR_PASS") + "@" + os.getenv("DEV_QSAR_HOST") + ":" + os.getenv("DEV_QSAR_PORT") + "/" + os.getenv("DEV_QSAR_DATABASE"), echo=True)
            Session = sessionmaker(bind = engine)
            session = Session()
            genericModel = session.query(Model).filter_by(id=model_id).first()
            
            modelBytes = ModelByte(created_at = func.now(),
                                   created_by = os.getenv("LAN_ID"),
                                   updated_at = func.now(),
                                   updated_by = None,
                                   bytes = pickle.dumps(model),
                                   fk_model = genericModel)
            session.add(modelBytes)
            session.flush()
            session.commit()

    else:
        embedding = []
        if "," in embedding_tsv:
            embedding = embedding_tsv.split(",")
        elif "\t" in embedding_tsv:
            embedding = embedding_tsv.split("\t")
        model = model_ws_utilities.call_build_model_with_preselected_descriptors(qsar_method, training_tsv, embedding)
        if model is None:
            abort(500, 'unknown model training error')
    
    # Sets status 200 OK
    status = 200
    
    
    # If model number provided for storage, stores the model and sets status 201 CREATED instead
    if model_id.strip():
        model_ws_utilities.models[model_id] = model
        status = 201
    
    
    # Returns model bytes
    return 'worked', status

@app.route('/models/<string:qsar_method>/predictsa', methods=['POST'])
def predictpythonstorage(qsar_method):
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

    # Gets stored model using model number
    model = None
    if model_ws_utilities.models[model_id] is not None:
        model = model_ws_utilities.models[model_id]
    else:
        engine = create_engine("postgresql://" + os.getenv("DEV_QSAR_USER") + ":" + os.getenv("DEV_QSAR_PASS") + "@" + os.getenv("DEV_QSAR_HOST") + ":" + os.getenv("DEV_QSAR_PORT") + "/" + os.getenv("DEV_QSAR_DATABASE"), echo=True)
        Session = sessionmaker(bind = engine)
        session = Session()
        query = session.query(ModelByte).filter_by(fk_model_id=model_id).one()
        bytes = query.bytes
        model = pickle.loads(bytes)
        session.flush()
        session.commit()

    # 404 NOT FOUND if no model stored under provided number
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    # Calls the appropriate prediction method and returns the results
    return model_ws_utilities.call_do_predictions(prediction_tsv, model), 200


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

    # Gets stored model using model number
    model = model_ws_utilities.models[model_id]
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
