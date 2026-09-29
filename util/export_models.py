'''
Created on Jul 28, 2026

@author: TMARTI02
'''

import pickle
import json
import os
from dotenv import load_dotenv
load_dotenv('../personal.env')

def load_model_by_id(model_id):
    """
    Loads a model from pickle file
    Add following to ModelInitializer.init_model():
        model = load_model_by_id(model_id)
        if model is not None:
            models[model_id] = model
            logging.info(f'modelId {model_id} loaded from pickle file')
            return model
    """
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        raise ValueError("PROJECT_ROOT environment variable is not set.")

    pickle_path = os.path.join(project_root, "model export", f"{model_id}.pkl")
    
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
            return model
    else:
        return None

def export_models(model_ids):

    import model_ws_db_utilities as mwdu
    # print (json.dumps(mwdu.dict_missing_dsstox_records,indent=4))
    
    mi = mwdu.ModelInitializer()
    project_root = os.environ["PROJECT_ROOT"]
    print(project_root)
    
    for model_id in model_ids:
        model = mi._init_model_from_postgres(model_id)
        pickle_path = os.path.join(project_root,"model export", f"{model_id}.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_models(model_ids):
    for model_id in model_ids:
        model = load_model_by_id(model_id)
        print(model.modelId, model.propertyName, model.qsar_method, "loaded")


if __name__ == '__main__':
    model_ids = [1763, 1754, 1756, 1757, 1758]
    # export_models(model_ids)
    load_models(model_ids)
