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
