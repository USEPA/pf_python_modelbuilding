"""Bounded process-local reuse of successful preprocessing, never predictions."""
from collections import OrderedDict
from copy import deepcopy
import threading
import time


class PreprocessingCache:
    def __init__(self, capacity=1024, ttl=900):
        self.capacity = capacity
        self.ttl = ttl
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires <= time.monotonic():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return deepcopy(value)

    def put(self, key, value):
        with self._lock:
            self._entries[key] = (time.monotonic() + self.ttl, deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)


standardization_cache = PreprocessingCache()
descriptor_cache = PreprocessingCache(capacity=512)
