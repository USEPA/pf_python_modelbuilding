import logging
from pathlib import Path
from typing import Any, Tuple

def get_project_root() -> Path:
    return Path(__file__).parent


import time
from functools import wraps


def timer(func):
    """
    Decorator to measure the execution time of a function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"{func.__name__!r} executed in {elapsed_time:.4f}s")

        return result

    return wrapper

import numpy as np
import pandas as pd
from datetime import datetime
import json
import base64

def row_to_json(df, row=0):
    row = df.iloc[row]
    row_dict = {col: to_json_safe(val) for col, val in row.items()}
    json_str = json.dumps(row_dict, indent=4)
    return json_str
    

def print_first_row(df, row=0):
    row = df.iloc[row]
    row_dict = {col: to_json_safe(val) for col, val in row.items()}
    json_str = json.dumps(row_dict, indent=4)
    print(json_str)

def to_json_safe(
    obj: Any,
    *,
    omit_nulls: bool = False,
) -> Any:
    """
    Convert Python/pandas/numpy objects into JSON-safe values.
    - All missing values (NaN, None, pd.NA, NaT) become None.
    - If omit_nulls=True, drop None values from dicts and lists (recursively).

    Notes:
    - NaN is not valid JSON; this function ensures NaN -> None.
    - For lists/tuples/sets, elements equal to None are removed when omit_nulls=True.
    - For dicts, keys whose value is None are removed when omit_nulls=True.
    """

    # 1) Missing values first: catch NaN, None, pd.NA, NaT, etc.
    try:
        # pd.isna works for most types and won't error on scalars
        if pd.isna(obj):
            return None
    except Exception:
        # If pd.isna can't evaluate (e.g., complex custom objects), ignore
        pass

    # 2) Binary-like buffers
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            return "base64:" + base64.b64encode(b).decode("ascii")

    # 3) Numpy scalar types (not missing)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        # By now, NaN was caught above; this is a real number
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # 4) Pandas/numpy containers
    if isinstance(obj, np.ndarray):
        converted = [to_json_safe(x, omit_nulls=omit_nulls) for x in obj.tolist()]
        return [x for x in converted if x is not None] if omit_nulls else converted

    if isinstance(obj, pd.Series):
        converted = [to_json_safe(x, omit_nulls=omit_nulls) for x in obj.tolist()]
        return [x for x in converted if x is not None] if omit_nulls else converted

    if isinstance(obj, pd.DataFrame):
        # Convert to records, then recurse per-record dict
        records = obj.to_dict(orient="records")
        converted_records = [
            to_json_safe(r, omit_nulls=omit_nulls) for r in records
        ]
        # Filter out None records (unlikely unless entire row is null and omit_nulls cascades)
        return [r for r in converted_records if r is not None] if omit_nulls else converted_records

    # 5) Date/time
    if isinstance(obj, (pd.Timestamp, datetime)):
        # NaT was caught by pd.isna earlier, so valid timestamps remain
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()

    # 6) Dicts
    if isinstance(obj, dict):
        converted = {str(k): to_json_safe(v, omit_nulls=omit_nulls) for k, v in obj.items()}
        if omit_nulls:
            converted = {k: v for k, v in converted.items() if v is not None}
            # If everything was None and omitted, you may choose to return None; we keep {} by default.
        return converted

    # 7) Lists/tuples/sets
    if isinstance(obj, (list, tuple, set)):
        converted_list = [to_json_safe(x, omit_nulls=omit_nulls) for x in obj]
        if omit_nulls:
            converted_list = [x for x in converted_list if x is not None]
        return converted_list

    # 8) Fallback: return as-is
    return obj