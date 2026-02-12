'''
Created on Feb 11, 2026

@author: TMARTI02
'''
import mimetypes
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

def fetch_model_file(model_id: int, type_id: int):
    """
    Fetch the file bytes, filename, and MIME type from the model_files table
    using (fk_model_id, fk_file_type_id). Results are cached in-memory.
    Returns (raw_bytes, file_name, mime_type).
    """
    du = DatabaseUtilities("qsar_models")  # schema if needed
    table = "model_files"

    key = _cache_key(model_id=model_id, type_id=type_id)
    cached = cache.get(key)
    if cached is not None:
        return cached  # (raw_bytes, file_name, mime_type)

    # NOTE: ensure these column names match your table
    row = du.get_row(table=table, fk_file_type_id=type_id, fk_model_id=model_id)
    if row is None:
        raise FileNotFoundError("No matching file found")

    raw_bytes = coerce_to_bytes(getattr(row, "file", None))
    if not raw_bytes:
        raise ValueError("Could not read file bytes from 'file' column")

    # Build a filename and guess MIME type from extension
    if type_id == 1:
        file_name = f"Model_{model_id}_QMRF.pdf"
    elif type_id == 2:
        file_name = f"Model_{model_id}_ExcelSummary.xlsx"
    elif type_id == 3:
        file_name = f"Model_{model_id}_ScatterPlot.png"
    elif type_id == 4:
        file_name = f"Model_{model_id}_Histogram.png"
    else:
        # Ensure file_name is always defined
        file_name = f"model_{model_id}_type_{type_id}"

    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

    value = (raw_bytes, file_name, mime_type)
    cache[key] = value  # FIX: assign into dict instead of cache.set(...)
    return value    

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    fetch_model_file(1065,3)
