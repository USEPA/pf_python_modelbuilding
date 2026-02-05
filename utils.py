import logging
from pathlib import Path


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

def to_json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [to_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [to_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(x) for x in obj]
    if pd.isna(obj):
        return None
    return obj