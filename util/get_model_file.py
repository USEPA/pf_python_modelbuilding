'''
Created on Feb 11, 2026

@author: TMARTI02
'''
import logging
import mimetypes
from db import model_cache
from util.database_utilities import DatabaseUtilities

cache = {}

def coerce_to_bytes(value):
    # psycopg2 may return memoryview for BYTEA; ensure bytes (probably not needed for sqlalchemy)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    try:
        return bytes(value)
    except Exception:
        return None

def _cache_key(model_id: int, type_id: int):
    # Cache key uses only (modelId, typeId)
    return (int(model_id), int(type_id))

def _file_name_for_type(model_id: int, type_id: int) -> str:
    if type_id == 1:
        return f"Model_{model_id}_QMRF.pdf"
    if type_id == 2:
        return f"Model_{model_id}_ExcelSummary.xlsx"
    if type_id == 3:
        return f"Model_{model_id}_ScatterPlot.png"
    if type_id == 4:
        return f"Model_{model_id}_Histogram.png"
    if type_id == 5:
        return f"Model_{model_id}_webpage.html"
    return f"model_{model_id}_type_{type_id}"


def fetch_model_file_from_postgres(model_id: int, type_id: int):
    """
    Fetch the file bytes, filename, and MIME type from the model_files table
    using (fk_model_id, fk_file_type_id).
    Returns (raw_bytes, file_name, mime_type).
    """
    du = DatabaseUtilities("qsar_models")  # schema if needed
    table = "model_files"

    # NOTE: ensure these column names match your table
    try:
        row = du.get_row(table=table, fk_file_type_id=type_id, fk_model_id=model_id)
        if row is None:
            raise FileNotFoundError("No matching file found")

        raw_bytes = coerce_to_bytes(getattr(row, "file", None))
        if not raw_bytes:
            raise ValueError("Could not read file bytes from 'file' column")
    finally:
        try:
            du.session.close()
        except Exception:
            logging.debug("Could not close Postgres session after model file fetch", exc_info=True)

    file_name = _file_name_for_type(model_id, type_id)
    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    return raw_bytes, file_name, mime_type


def fetch_model_file(model_id: int, type_id: int):
    """
    Fetch model files from Mongo first, then Postgres as a backfill fallback.
    Results are cached in-memory.
    """
    model_id = int(model_id)
    type_id = int(type_id)
    key = _cache_key(model_id=model_id, type_id=type_id)
    cached = cache.get(key)
    if cached is not None:
        return cached  # (raw_bytes, file_name, mime_type)

    if model_cache.model_artifact_cache_enabled():
        try:
            cached_file = model_cache.read_model_file(model_id, type_id)
        except model_cache.ModelArtifactCacheUnavailableError as exc:
            logging.warning(
                "Predictor model file cache unavailable model_id=%s type_id=%s: %s",
                model_id,
                type_id,
                exc,
            )
        else:
            if cached_file is not None:
                cache[key] = cached_file
                return cached_file

    if not model_cache.postgres_fallback_enabled():
        raise FileNotFoundError(
            f"No cached model file found for model_id={model_id} type_id={type_id}."
        )

    value = fetch_model_file_from_postgres(model_id, type_id)

    if model_cache.model_artifact_cache_enabled():
        try:
            model_cache.write_model_file(model_id, type_id, *value)
        except model_cache.ModelArtifactCacheUnavailableError as exc:
            logging.warning(
                "Could not store model file in Mongo model file cache model_id=%s type_id=%s: %s",
                model_id,
                type_id,
                exc,
            )
        except Exception:
            logging.exception(
                "Could not store model file in Mongo model file cache model_id=%s type_id=%s",
                model_id,
                type_id,
            )

    cache[key] = value  # FIX: assign into dict instead of cache.set(...)
    return value    

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    fetch_model_file(1065,3)
