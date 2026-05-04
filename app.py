"""
Uvicorn webservice to build QSAR models with a variety of modeling strategies (RF, SVM, DNN, XGB...more to come?)
Run with Python 3.12
@author: TMARTI02 (Todd Martin) - RF, base webservice code, predictions for new chemicals and reports
@author: GSincl01 (Gabriel Sinclair), XGB, refactored webservice code
@author: cramslan (Christian Ramsland) - DNN
Repository created 05/21/2021
"""
import io
import logging
import os

os.environ.setdefault("PREDICTOR_MODEL_ARTIFACT_CACHE_ENABLED", "true")
os.environ.setdefault("MODEL_ARTIFACT_CACHE_ENABLED", "true")
os.environ.setdefault("PREDICTOR_MODEL_POSTGRES_FALLBACK_ENABLED", "false")
os.environ.setdefault("MODEL_POSTGRES_FALLBACK_ENABLED", "false")
os.environ.setdefault("MONGO_CACHE_ENABLED", "true")

import coloredlogs
import connexion
from connexion.middleware import MiddlewarePosition
from connexion.options import SwaggerUIOptions
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse

import util.get_model_file as gmf
from util.helpers import (
    collect_model_details_for_metadata,
    make_predictdb_post_response,
    make_predictdb_response,
)


def _get_log_level(env_var: str, default: str) -> int:
    value = os.getenv(env_var, default).strip().upper()
    level = getattr(logging, value, None)
    if isinstance(level, int):
        return level

    logging.warning("Invalid log level %r for %s; using %s", value, env_var, default.upper())
    return getattr(logging, default.upper())


def _configure_logging():
    app_level = _get_log_level("APP_LOG_LEVEL", "INFO")
    coloredlogs.install(
        level=app_level,
        milliseconds=True,
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)',
    )
    logging.basicConfig(level=app_level)

    logger_levels = {
        "connexion": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.operations.openapi3": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.validators.parameter": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.middleware.validation": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.middleware.security": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.middleware.abstract": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "connexion.middleware.swagger_ui": _get_log_level("CONNEXION_LOG_LEVEL", "WARNING"),
        "pymongo": _get_log_level("PYMONGO_LOG_LEVEL", "WARNING"),
        "pymongo.topology": _get_log_level("PYMONGO_LOG_LEVEL", "WARNING"),
        "pymongo.connection": _get_log_level("PYMONGO_LOG_LEVEL", "WARNING"),
        "pymongo.command": _get_log_level("PYMONGO_LOG_LEVEL", "WARNING"),
        "pymongo.serverSelection": _get_log_level("PYMONGO_LOG_LEVEL", "WARNING"),
        "sqlalchemy": _get_log_level("SQLALCHEMY_LOG_LEVEL", "ERROR"),
        "uvicorn": _get_log_level("UVICORN_LOG_LEVEL", "INFO"),
        "uvicorn.access": _get_log_level("UVICORN_ACCESS_LOG_LEVEL", "INFO"),
        "uvicorn.error": _get_log_level("UVICORN_ERROR_LOG_LEVEL", "INFO"),
    }

    for logger_name, level in logger_levels.items():
        logging.getLogger(logger_name).setLevel(level)


_configure_logging()

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

    return dict(id="predictor_models",
                name="WebTEST v2",
                description="""These models are based on additional curation and expansion of the WebTEST v1 data sets including
                 aggregation from public websites (e.g., as described in the publication
                 <a href="https://doi.org/10.1021/acs.chemrestox.2c00379" target="_blank">Transparency in Modeling through Careful Application of OECD’s QSAR/QSPR Principles via a Curated Water Solubility Data Set</a>
                 and using a single modeling approach, distinct from the multi-model consensus modeling approach for WebTEST v1""",
                version="1.0.0",
                compiled=BUILD_TIMESTAMP,
                build_id=BUILD_NUMBER)


_metadata = None


def get_metadata():
    global _metadata
    if _metadata is None:
        model_ids = list(range(1065, 1071))
        model_details_array = collect_model_details_for_metadata(model_ids)
        metadata = dict(
            version=get_version(),
            endpoints=model_details_array
        )
        if len(model_details_array) == len(model_ids):
            _metadata = metadata
        else:
            logging.warning(
                "Metadata endpoints incomplete; not caching response expected=%s actual=%s",
                len(model_ids),
                len(model_details_array),
            )
        return metadata

    return _metadata


def get_file(type_id: int = None, model_id: int = None):
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


def predictDB_POST(body):
    """Automates prediction and AD for batch smiles using model in database"""
    return make_predictdb_post_response(body)


def predictDB(model_id, smiles=None, identifier=None, report_format='json'):
    """Automates prediction and AD for single smiles using model in database"""
    return make_predictdb_response(
        model_id=model_id,
        smiles=smiles,
        identifier=identifier,
        report_format=report_format
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
