"""
Uvicorn webservice to build QSAR models with a variety of modeling strategies (RF, SVM, DNN, XGB...more to come?)
Run with Python 3.12
@author: TMARTI02 (Todd Martin) - RF, base webservice code, predictions for new chemicals and reports
@author: GSincl01 (Gabriel Sinclair), XGB, refactored webservice code
@author: cramslan (Christian Ramsland) - DNN
Repository created 05/21/2021
"""
import io
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
import util.get_model_file as gmf
from logging import DEBUG

import coloredlogs
import connexion
from dotenv import load_dotenv
from connexion.middleware import MiddlewarePosition
from connexion.options import SwaggerUIOptions
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, Response, JSONResponse, StreamingResponse, FileResponse

from model_ws_db_utilities import ModelPredictor
from report_creator_dict import ReportCreator

_PROCESS_PREDICTOR = None

load_dotenv()

coloredlogs.install(level=DEBUG, milliseconds=True,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
logging.basicConfig(level=logging.INFO)

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


def get_file(type_id: int=None, model_id: int=None):
    if type_id is None or model_id is None:
        return JSONResponse(
            {"error": "Missing required query params: type_id and model_id"},
            status_code=400,
        )

    try:
        type_id = int(type_id)
        model_id = int(model_id)
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": "type_id and model_id must be integers"},
            status_code=400,
        )

    try:
        raw_bytes, file_name, mime_type = gmf.fetch_model_file(model_id=model_id, type_id=type_id)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Database error: {e}"}, status_code=500)

    disposition = "attachment" if type_id == 2 else "inline"

    bio = io.BytesIO(raw_bytes)
    bio.seek(0)

    headers = {
        "Content-Disposition": f'{disposition}; filename="{file_name}"',
        "Cache-Control": "no-store",
    }

    return StreamingResponse(
        bio,
        media_type=mime_type or "application/octet-stream",
        headers=headers,
    )


def predict_identifier(identifier: str, model_id: int, report_format: str="json"):
    """Automates prediction and AD for single identifier using model in database"""

    # normalize report_format
    report_format = (report_format or "json").lower()
    if report_format not in ("json", "html"):
        report_format = "json"

    # model_id -> int
    try:
        model_id = int(model_id)
    except (TypeError, ValueError):
        return JSONResponse({"error": "bad_request", "message": "model_id must be integer"}, status_code=400)

    if not identifier:
        return JSONResponse({"error": "bad_request", "message": "identifier is required"}, status_code=400)

    # Resolve identifier -> SMILES
    from API_Utilities import SearchAPI
    import os

    serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
    chemicals, code = SearchAPI.call_resolver_get(serverAPIs, identifier)

    if code != 200 or not chemicals:
        return JSONResponse(
            {"error": "not_found", "message": f"Could not find {identifier}"},
            status_code=404,
        )

    smiles = (chemicals[0].get("chemical") or {}).get("smiles")
    if not smiles:
        return JSONResponse(
            {"error": "not_found", "message": f"Could not find {identifier}"},
            status_code=404,
        )

    # Predict
    mp = ModelPredictor()
    modelResultsJson = mp.predictFromDB(model_id, smiles)

    if isinstance(modelResultsJson, str) and "invalid" in modelResultsJson.lower():
        return JSONResponse({"error": "invalid", "message": modelResultsJson}, status_code=400)

    if report_format == "html":
        rc = ReportCreator()
        html = rc.create_html_report_from_json(modelResultsJson)
        return HTMLResponse(html, status_code=200)

    return Response(content=_to_json_str(modelResultsJson), media_type="application/json")


def _to_obj(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (str, bytes, bytearray)):
        return json.loads(x)
    raise TypeError(f"Unsupported prediction type: {type(x)}")


def _to_json_str(x):
    if isinstance(x, (dict, list)):
        return json.dumps(x)
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    raise TypeError(f"Unsupported prediction type: {type(x)}")


def _init_process_predictor():
    global _PROCESS_PREDICTOR
    _PROCESS_PREDICTOR = ModelPredictor()


def _predict_smiles_in_process(args):
    model_id, current_smiles = args
    predictor = _PROCESS_PREDICTOR
    if predictor is None:
        _init_process_predictor()
        predictor = _PROCESS_PREDICTOR
    if predictor is None:
        raise RuntimeError("Failed to initialize process predictor")
    pred = predictor.predictFromDB(model_id, current_smiles)
    return _to_obj(pred)


def predictDB_POST(body):
    """Automates prediction and AD for batch smiles using model in database"""
    max_workers = int(os.getenv("PREDICT_BATCH_WORKERS", os.cpu_count() or 1))
    max_workers = max(1, min(max_workers, len(body["smiles"])))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_process_predictor) as executor:
        modelResultsArray = list(executor.map(_predict_smiles_in_process, ((body["model_id"], s) for s in body["smiles"])))

    return JSONResponse(content=modelResultsArray)


def predictDB(smiles, model_id, report_format):
    """Automates prediction and AD for single smiles using model in database"""

    report_format = (report_format or "json").lower()
    if report_format not in ("json", "html"):
        report_format = "json"

    mp = ModelPredictor()
    pred = mp.predictFromDB(model_id, smiles)

    if report_format == "html":
        rc = ReportCreator()
        modelResultsHtml = rc.create_html_report_from_json(_to_json_str(pred))
        return HTMLResponse(content=modelResultsHtml)

    return Response(content=_to_json_str(pred), media_type="application/json")


if __name__ == '__main__':
    log = logging.getLogger('pymongo.topology')
    log.setLevel(logging.INFO)
    app.run(host='0.0.0.0', port=5004)
