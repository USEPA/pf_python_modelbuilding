import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

from API_Utilities import SearchAPI
from model_ws_db_utilities import ModelPredictor
from report_creator_dict import ReportCreator
from starlette.responses import HTMLResponse, JSONResponse

_PROCESS_PREDICTOR = None


def _coerce_json_safe(value):
    if isinstance(value, dict):
        return {str(key): _coerce_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_coerce_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_coerce_json_safe(item) for item in value]

    if isinstance(value, set):
        return [_coerce_json_safe(item) for item in value]

    # Convert numpy/pandas scalar-like values without importing heavy dependencies here.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar_value = item_method()
        except Exception:
            scalar_value = None
        else:
            if scalar_value is not value:
                return _coerce_json_safe(scalar_value)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            list_value = tolist_method()
        except Exception:
            list_value = None
        else:
            if list_value is not value:
                return _coerce_json_safe(list_value)

    return value


def _to_obj(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, (str, bytes, bytearray)):
        return json.loads(x)
    raise TypeError(f"Unsupported prediction type: {type(x)}")


def to_json_str(x):
    if isinstance(x, (dict, list)):
        return json.dumps(_coerce_json_safe(x))
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    raise TypeError(f"Unsupported prediction type: {type(x)}")


def _to_obj_safe(x):
    try:
        return _coerce_json_safe(_to_obj(x))
    except Exception:
        try:
            return {"error": to_json_str(x)}
        except Exception:
            return {"error": str(x)}


def build_error_response(chemical_identifier, code, message, details=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = _coerce_json_safe(details)

    return _coerce_json_safe({
        "chemicalIdentifiers": chemical_identifier or "",
        "modelResults": None,
        "error": error,
    })


def build_batch_error_response(code, message, details=None, model_details=None, predictions=None):
    error = {"code": code, "message": message}
    if details is not None:
        error["details"] = _coerce_json_safe(details)

    payload = {
        "modelDetails": _coerce_json_safe(model_details),
        "predictions": _coerce_json_safe(predictions or []),
        "error": error,
    }
    return _coerce_json_safe(payload)


def _extract_chemical_identifier(payload, fallback=""):
    if isinstance(payload, str):
        return payload

    if not isinstance(payload, dict):
        return fallback or ""

    chemical = payload.get("chemicalIdentifiers")
    if isinstance(chemical, str):
        return chemical

    if isinstance(chemical, dict):
        for key in ("chemId", "smiles", "canonicalSmiles", "cid", "sid", "name"):
            value = chemical.get(key)
            if isinstance(value, str) and value:
                return value

    for key in ("smiles", "identifier", "chemicalIdentifiers"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value

    return fallback or ""


def normalize_error_payload(payload, chemical_identifier=""):
    obj = _to_obj_safe(payload)

    if isinstance(obj, dict):
        if isinstance(obj.get("error"), dict) and obj.get("modelResults") is None and "chemicalIdentifiers" in obj:
            if not isinstance(obj.get("chemicalIdentifiers"), str):
                obj["chemicalIdentifiers"] = _extract_chemical_identifier(obj, chemical_identifier)
            return _coerce_json_safe(obj)

        if "error" in obj:
            details = obj["error"] if not isinstance(obj["error"], str) else None
            message = obj["error"] if isinstance(obj["error"], str) else "Prediction request failed"
            return _coerce_json_safe(build_error_response(
                _extract_chemical_identifier(obj, chemical_identifier),
                "prediction_error",
                message,
                details,
            ))

        model_results = obj.get("modelResults")
        if isinstance(model_results, dict) and model_results.get("predictionError"):
            prediction_error = model_results.get("predictionError")
            details = prediction_error if not isinstance(prediction_error, str) else None
            message = prediction_error if isinstance(prediction_error, str) else "Prediction request failed"
            return _coerce_json_safe(build_error_response(
                _extract_chemical_identifier(obj, chemical_identifier),
                "prediction_error",
                message,
                details,
            ))

    return _coerce_json_safe(obj)


def _strip_common_model_details(predictions):
    common_model_details = None
    common_model_details_set = False
    stripped_predictions = []

    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue

        model_details = prediction.get("modelDetails")
        if model_details is None:
            continue

        if not common_model_details_set:
            common_model_details = model_details
            common_model_details_set = True
            continue

        if model_details != common_model_details:
            logging.warning("Batch predictions returned different modelDetails values; keeping them per item")
            return None, predictions

    for prediction in predictions:
        if isinstance(prediction, dict):
            prediction_copy = dict(prediction)
            prediction_copy.pop("modelDetails", None)
            stripped_predictions.append(prediction_copy)
        else:
            stripped_predictions.append(prediction)

    return common_model_details, stripped_predictions


def build_predictdb_post_payload(predictions, error=None):
    normalized_predictions = [_coerce_json_safe(_to_obj_safe(prediction)) for prediction in predictions]
    model_details, stripped_predictions = _strip_common_model_details(normalized_predictions)

    payload = {
        "modelDetails": model_details,
        "predictions": stripped_predictions,
    }
    if error is not None:
        payload["error"] = _coerce_json_safe(error)
    return _coerce_json_safe(payload)


def _error_status_code(payload, default=400):
    if not isinstance(payload, dict):
        return 200

    if payload.get("modelResults") is not None or payload.get("error") is None:
        return 200

    error_code = ((payload.get("error") or {}).get("code") or "").lower()
    if error_code == "bad_request":
        return 400
    if error_code == "not_found":
        return 404
    if error_code in {"resolver_error", "internal_error"}:
        return 500
    return default


def init_process_predictor():
    global _PROCESS_PREDICTOR
    _PROCESS_PREDICTOR = ModelPredictor()


def predict_smiles_in_process(args):
    model_id, current_smiles = args
    predictor = _PROCESS_PREDICTOR
    if predictor is None:
        init_process_predictor()
        predictor = _PROCESS_PREDICTOR
    if predictor is None:
        return build_error_response(current_smiles, "prediction_error", "Failed to initialize process predictor")

    try:
        pred = predictor.predictFromDB(model_id, current_smiles)
        return normalize_error_payload(pred, current_smiles)
    except Exception as exc:
        logging.exception("Unhandled exception in process prediction for %s", current_smiles)
        return build_error_response(current_smiles, "internal_error", "Unhandled prediction error", str(exc))


def dedupe_smiles_preserve_order(smiles_list):
    unique_smiles = []
    index_map = {}
    for idx, smiles in enumerate(smiles_list):
        if smiles not in index_map:
            unique_smiles.append(smiles)
            index_map[smiles] = [idx]
        else:
            index_map[smiles].append(idx)
    return unique_smiles, index_map


def collect_model_details_for_metadata(model_ids, smiles):
    with ProcessPoolExecutor(max_workers=6, initializer=init_process_predictor) as executor:
        return list(executor.map(predict_smiles_in_process, zip(model_ids, [smiles] * len(model_ids))))


def make_predictdb_post_response(body):
    if not isinstance(body, dict):
        return JSONResponse(
            build_batch_error_response("bad_request", "Request body must be a JSON object"),
            status_code=400,
        )

    smiles_list = body.get("smiles")
    model_id = body.get("model_id")

    if not isinstance(smiles_list, list):
        return JSONResponse(
            build_batch_error_response("bad_request", "'smiles' must be an array"),
            status_code=400,
        )

    if model_id is None:
        return JSONResponse(
            build_batch_error_response("bad_request", "'model_id' is required"),
            status_code=400,
        )

    if any(not isinstance(s, str) for s in smiles_list):
        return JSONResponse(
            build_batch_error_response("bad_request", "All 'smiles' items must be strings"),
            status_code=400,
        )

    if not smiles_list:
        return JSONResponse(content=build_predictdb_post_payload([]))

    unique_smiles, index_map = dedupe_smiles_preserve_order(smiles_list)
    batch_mode = os.getenv("PREDICT_BATCH_MODE", "thread").strip().lower()

    if batch_mode == "process":
        try:
            max_workers = int(os.getenv("PREDICT_BATCH_WORKERS", os.cpu_count() or 1))
            max_workers = max(1, min(max_workers, len(unique_smiles)))
            with ProcessPoolExecutor(max_workers=max_workers, initializer=init_process_predictor) as executor:
                unique_results = list(executor.map(predict_smiles_in_process, ((model_id, s) for s in unique_smiles)))
        except Exception as exc:
            logging.exception("Unhandled exception in batch process pool")
            return JSONResponse(
                content=build_batch_error_response(
                    "internal_error",
                    "Unhandled batch prediction error",
                    str(exc),
                ),
                status_code=500,
            )
    else:
        predictor = ModelPredictor()
        try:
            raw_results = predictor.predictFromDB(model_id, unique_smiles)
            if not isinstance(raw_results, list):
                raise TypeError(f"Batch prediction returned {type(raw_results)} instead of list")
            if len(raw_results) != len(unique_smiles):
                raise ValueError(
                    f"Batch prediction returned {len(raw_results)} results for {len(unique_smiles)} SMILES"
                )
            unique_results = [normalize_error_payload(pred, smi) for smi, pred in zip(unique_smiles, raw_results)]
        except Exception as exc:
            logging.exception("Unhandled exception in batch prediction")
            return JSONResponse(
                content=build_batch_error_response(
                    "internal_error",
                    "Unhandled batch prediction error",
                    str(exc),
                ),
                status_code=500,
            )

    model_results = [None] * len(smiles_list)
    for smiles, prediction in zip(unique_smiles, unique_results):
        for idx in index_map[smiles]:
            model_results[idx] = prediction

    return JSONResponse(content=build_predictdb_post_payload(model_results))


def make_predictdb_response(model_id, smiles=None, identifier=None, report_format="json", cim_api_server=None):
    if smiles and identifier:
        return JSONResponse(
            build_error_response(smiles, "bad_request", f"Both SMILES '{smiles}' and identifier {identifier} are provided"),
            status_code=400,
        )

    if identifier:
        try:
            chemicals, code = SearchAPI.call_resolver_get(cim_api_server, identifier)
        except Exception as exc:
            logging.exception("Resolver lookup failed for identifier=%s", identifier)
            return JSONResponse(
                build_error_response(identifier, "resolver_error", "Resolver lookup failed", str(exc)),
                status_code=500,
            )

        if code != 200 or not chemicals:
            return JSONResponse(
                build_error_response(identifier, "not_found", f"Could not find {identifier}"),
                status_code=404,
            )
        smiles = (chemicals[0].get("chemical") or {}).get("smiles")

    if not smiles:
        if not identifier:
            return JSONResponse(
                build_error_response("", "bad_request", "Either 'smiles' or 'identifier' is required"),
                status_code=400,
            )

        return JSONResponse(
            build_error_response(identifier, "not_found", f"Could not find {identifier}"),
            status_code=404,
        )

    predictor = ModelPredictor()
    try:
        pred = predictor.predictFromDB(model_id, smiles)
    except Exception as exc:
        logging.exception("Unhandled exception in single prediction for %s", smiles)
        return JSONResponse(
            build_error_response(smiles, "internal_error", "Unhandled prediction error", str(exc)),
            status_code=500,
        )

    report_format = (report_format or "json").lower()
    if report_format not in ("json", "html"):
        report_format = "json"

    normalized_pred = normalize_error_payload(pred, smiles)

    if report_format == "html":
        if isinstance(normalized_pred, dict) and normalized_pred.get("error") is not None and normalized_pred.get("modelResults") is None:
            return JSONResponse(content=normalized_pred, status_code=_error_status_code(normalized_pred))
        report_creator = ReportCreator()
        model_results_html = report_creator.create_html_report_from_json(to_json_str(normalized_pred))
        return HTMLResponse(content=model_results_html)

    return JSONResponse(content=normalized_pred, status_code=_error_status_code(normalized_pred))
