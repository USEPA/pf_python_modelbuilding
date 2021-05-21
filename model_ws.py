from flask import Flask, request, abort
import logging
import pickle
import dill

import model_ws_utilities

app = Flask(__name__)
# Limit logging output for easier readability
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


@app.route('/models/<string:qsar_method>/info', methods=['GET'])
def info(qsar_method):
    return model_ws_utilities.get_model_info(qsar_method), 200


@app.route('/models/<string:qsar_method>/train', methods=['POST'])
def train(qsar_method):
    obj = request.form
    training_tsv = obj.get('training_tsv')
    model_id = obj.get('model_id')
    if obj.get('remove_log_p'):
        remove_log_p = obj.get('remove_log_p', '').lower() == 'true'
    else:
        remove_log_p = False

    if training_tsv is None:
        abort(400, 'missing training tsv')

    model = model_ws_utilities.call_build_model(qsar_method, training_tsv, remove_log_p)
    if model is None:
        abort(500, 'unknown model training error')

    status = 200
    if model_id.strip():
        model_ws_utilities.models[model_id] = model
        status = 201

    if qsar_method.lower() == 'dnn':
        return dill.dumps(model), status
    else:
        return pickle.dumps(model), status


@app.route('/models/<string:qsar_method>/predict', methods=['POST'])
def predict(qsar_method):
    obj = request.form
    prediction_tsv = obj.get('prediction_tsv')
    model_id = obj.get('model_id')

    if prediction_tsv is None:
        abort(400, 'missing prediction tsv')
    if model_id is None:
        abort(400, 'missing model id')

    model = model_ws_utilities.models[model_id]
    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    return model_ws_utilities.call_do_predictions(prediction_tsv, model), 200


@app.route('/models/<string:qsar_method>/init', methods=['POST'])
def init(qsar_method):
    form_obj = request.form
    files_obj = request.files
    model_id = form_obj.get('model_id')

    if model_id is None:
        abort(400, 'missing model id')

    model_file = files_obj['model']
    model = None
    if model_file is not None:
        if qsar_method.lower() == 'dnn':
            model = dill.loads(model_file.read())
        else:
            model = pickle.loads(model_file.read())
        model_ws_utilities.models[model_id] = model
    else:
        abort(400, 'missing model bytes')

    if model is None:
        abort(400, 'unknown model initialization error')

    return model_id, 201


@app.route('/models/<string:qsar_method>/<string:model_id>', methods=['GET'])
def details(qsar_method, model_id):
    model = model_ws_utilities.models[model_id]

    if model is None:
        abort(404, 'no stored model with id ' + model_id)

    model_details = model_ws_utilities.get_model_details(qsar_method, model)
    if model_details is None:
        abort(404, 'no details for stored model with id ' + model_id)

    return model_details, 200


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
