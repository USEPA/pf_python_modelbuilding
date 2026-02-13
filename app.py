
"""
Uvicorn webservice to build QSAR models with a variety of modeling strategies (RF, SVM, DNN, XGB...more to come?)
Run with Python 3.12
@author: TMARTI02 (Todd Martin) - RF, base webservice code, predictions for new chemicals and reports
@author: GSincl01 (Gabriel Sinclair), XGB, refactored webservice code
@author: cramslan (Christian Ramsland) - DNN
Repository created 05/21/2021
"""

import json
import logging
import io
from logging import INFO, DEBUG
from model_ws_db_utilities import ModelPredictor
from report_creator_dict import ReportCreator
import util.get_model_file as gmf

from dotenv import load_dotenv
load_dotenv()

import coloredlogs
import connexion
from connexion.middleware import MiddlewarePosition
from connexion.options import SwaggerUIOptions
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, Response, JSONResponse, StreamingResponse
from urllib.parse import quote


coloredlogs.install(level=DEBUG, milliseconds=True,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

options = SwaggerUIOptions(spec_path="/api/predictor_models/swagger.yaml",
                           swagger_ui_path="/api/predictor_models/swagger")
app = connexion.AsyncApp(__name__, swagger_ui_options=options)
app.add_middleware(
    CORSMiddleware,
    position=MiddlewarePosition.BEFORE_EXCEPTION,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_api('swagger.yaml', swagger_ui_options=options)


app_flask = None

def get_version():
    try:
        from build_info import BUILD_TIMESTAMP, BUILD_NUMBER
    except ImportError:
        BUILD_TIMESTAMP = None
        BUILD_NUMBER = None

    return dict(name="predictor_models",
                title="EPA/Models",
                version="1.0.0",
                compiled=BUILD_TIMESTAMP,
                build_id=BUILD_NUMBER)


def get_metadata():
    return dict(
        version=get_version()
    )
    

async def get_file(type_id: int, model_id: int):
    """
    Gets model files (png plots, pdf QMRFs, xlsx summaries etc)
    """
    
    try:
        raw_bytes, file_name, mime_type = gmf.fetch_model_file(model_id=model_id, type_id=type_id)
    except FileNotFoundError as e:
        return {"detail": str(e)}, 404
    except ValueError as e:
        return {"detail": str(e)}, 400
    except Exception as e:
        return {"detail": f"Database error: {e}"}, 500

    as_attachment = (type_id == 2)  # your rule
    disposition = "attachment" if as_attachment else "inline"
    cd = f'{disposition}; filename="{file_name}"; filename*=UTF-8\'\'{quote(str(file_name))}'

    bio = io.BytesIO(raw_bytes)
    bio.seek(0)
    headers = {
        "Content-Disposition": cd,
        "Cache-Control": "no-cache, no-store, max-age=0",
        "Pragma": "no-cache",
    }
    return StreamingResponse(bio, media_type=mime_type, headers=headers)



def predict_identifier(identifier, model_id, report_format):
    """Automates prediction and AD for single identifier using model in database
    """    
    
    if report_format not in ['json', 'html']:
        report_format = 'json'
    
    from API_Utilities import SearchAPI
    import os
    serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
    
    chemicals, code = SearchAPI.call_resolver_get(serverAPIs, identifier)
    
    # print(chemicals, code)
    
    if code != 200:
        return {"error": 404, "message": "not found"}, 404
    
    if len(chemicals)>0:    
        smiles = chemicals[0]["chemical"]["smiles"]
    else:
        return {"error": 404, "message": "not found"}, 404
    
    
    mp = ModelPredictor()
    modelResultsJson = mp.predictFromDB(model_id, smiles)

    if report_format == "html":
        rc=ReportCreator()
        
        if "invalid" in modelResultsJson.lower():
            return HTMLResponse(content=modelResultsJson)
        
        modelResultsHtml = rc.create_html_report_from_json(modelResultsJson)
        return HTMLResponse(content=modelResultsHtml)    
    else:
        # if just return a Response then can skip the json.loads step (takes time)
        return Response(content=modelResultsJson, media_type="application/json") #by using this return type it wont try to serialize the json again    




def predictDB_POST(body):
    return predictDB(body['smiles'], body['model_id'], "json")


# @app.route('/api/predictor_models/predict', methods=['POST', 'GET'])  # old flask route
def predictDB(smiles, model_id, report_format):
    """Automates prediction and AD for single smiles using model in database"""
        
    report_format = report_format.lower()
    if report_format not in ['json', 'html']:
        report_format = 'json'
        
    mp = ModelPredictor()
    
    # TODO: should we just return a JSON array in either case?
    
    if isinstance(smiles, list):
        modelResultsArray = []

        #Following just runs each one at a time, will need to dive into code to ran true batch calculations
        # With current code just better off pinging API one at a time since then wont need to deserialize 
        for current_smiles in smiles:  
            logging.debug("Running " +current_smiles)
            modelResultsJson =  mp.predictFromDB(model_id, current_smiles)
            modelResults = json.loads(modelResultsJson) #deserialize so can add to array
            modelResultsArray.append(modelResults)
        return JSONResponse(content=modelResultsArray) #this return type will automatically serialize the array    
    else:
    
        modelResultsJson = mp.predictFromDB(model_id, smiles)
    
        if report_format == "html":
            rc=ReportCreator()
            
            if "invalid" in modelResultsJson.lower():
                return HTMLResponse(content=modelResultsJson)
            
            modelResultsHtml = rc.create_html_report_from_json(modelResultsJson)
            return HTMLResponse(content=modelResultsHtml)    
        else:
            # if just return a Response then can skip the json.loads step (takes time)
            return Response(content=modelResultsJson, media_type="application/json") #by using this return type it wont try to serialize the json again    



