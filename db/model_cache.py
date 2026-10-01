from __future__ import annotations

import logging
import math
import os
import pickle
import zlib
from datetime import datetime, timezone
from typing import Any

from util.serialization_compat import deserialize_model

try:
    from pymongo import ASCENDING, MongoClient, ReplaceOne
    from pymongo.errors import PyMongoError
except ImportError:
    ASCENDING = None
    MongoClient = None
    ReplaceOne = None

    class PyMongoError(Exception):
        pass


MODEL_CACHE_SCHEMA_VERSION = 1
DEFAULT_MODEL_COLLECTION_NAME = "predictor_models_model_cache"
DEFAULT_CHUNK_COLLECTION_NAME = "predictor_models_model_cache_chunks"
DEFAULT_FILE_COLLECTION_NAME = "predictor_models_model_file_cache"
DEFAULT_DETAILS_COLLECTION_NAME = "predictor_models_model_details_cache"
DEFAULT_CHUNK_SIZE_BYTES = 8 * 1024 * 1024


class ModelArtifactCacheUnavailableError(RuntimeError):
    pass


def _get_env(*names: str, default=None):
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _get_int_env(*names: str, default: int) -> int:
    value = _get_env(*names)
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        logging.warning("Invalid integer value for %s=%r; using default=%s", names[0], value, default)
        return int(default)


def _get_bool_env(*names: str, default: bool) -> bool:
    value = _get_env(*names)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logging.warning("Invalid boolean value for %s=%r; using default=%s", names[0], value, default)
    return default


def model_artifact_cache_enabled() -> bool:
    return _get_bool_env(
        "PREDICTOR_MODEL_ARTIFACT_CACHE_ENABLED",
        "MODEL_ARTIFACT_CACHE_ENABLED",
        default=True,
    )


def postgres_fallback_enabled() -> bool:
    return _get_bool_env(
        "PREDICTOR_MODEL_POSTGRES_FALLBACK_ENABLED",
        "MODEL_POSTGRES_FALLBACK_ENABLED",
        default=True,
    )


def _model_key(model_id: int | str) -> str:
    return str(int(model_id))


def _file_key(model_id: int | str, type_id: int | str) -> str:
    return f"{int(model_id)}:{int(type_id)}"


def _details_key(model_id: int | str, file_api: str | None) -> str:
    return f"{int(model_id)}:{file_api or ''}"


def _owner_key(kind: str, key: str) -> str:
    return f"{kind}:{key}"


def _chunk_size_bytes() -> int:
    return max(
        1024,
        _get_int_env(
            "PREDICTOR_MODEL_ARTIFACT_CHUNK_SIZE_BYTES",
            "MODEL_ARTIFACT_CHUNK_SIZE_BYTES",
            default=DEFAULT_CHUNK_SIZE_BYTES,
        ),
    )


def connect_mongo():
    if not model_artifact_cache_enabled():
        raise ModelArtifactCacheUnavailableError("Predictor model artifact Mongo cache disabled")
    if MongoClient is None:
        raise ModelArtifactCacheUnavailableError("pymongo is not installed")

    mongo_user = _get_env("MONGO_USER")
    mongo_password = _get_env("MONGO_PASSWORD")
    if mongo_user and not mongo_password:
        raise ModelArtifactCacheUnavailableError("MONGO_USER is set but MONGO_PASSWORD is empty")
    if mongo_password and not mongo_user:
        raise ModelArtifactCacheUnavailableError("MONGO_PASSWORD is set but MONGO_USER is empty")

    client_kwargs = dict(
        host=_get_env("MONGO_HOST", default="localhost"),
        port=_get_int_env("MONGO_PORT", default=27017),
        appname=_get_env(
            "PREDICTOR_MODEL_ARTIFACT_MONGO_APP_NAME",
            "MONGO_APP_NAME",
            default="predictor_model_artifact_cache",
        ),
        serverSelectionTimeoutMS=_get_int_env(
            "PREDICTOR_MODEL_ARTIFACT_MONGO_SERVER_SELECTION_TIMEOUT_MS",
            "MONGO_SERVER_SELECTION_TIMEOUT_MS",
            default=10000,
        ),
        connectTimeoutMS=_get_int_env(
            "PREDICTOR_MODEL_ARTIFACT_MONGO_CONNECT_TIMEOUT_MS",
            "MONGO_CONNECT_TIMEOUT_MS",
            default=10000,
        ),
        socketTimeoutMS=_get_int_env(
            "PREDICTOR_MODEL_ARTIFACT_MONGO_SOCKET_TIMEOUT_MS",
            "MONGO_SOCKET_TIMEOUT_MS",
            default=60000,
        ),
    )
    if mongo_user:
        client_kwargs["username"] = mongo_user
        client_kwargs["password"] = mongo_password
        client_kwargs["authSource"] = _get_env("MONGO_AUTH_SOURCE", default="admin")

    client = MongoClient(**client_kwargs)
    client.admin.command("ping")
    database = client[_get_env("MONGO_DATABASE", default="predictor")]
    model_collection = database[
        _get_env("PREDICTOR_MODEL_ARTIFACT_MONGO_COLLECTION", default=DEFAULT_MODEL_COLLECTION_NAME)
    ]
    chunk_collection = database[
        _get_env("PREDICTOR_MODEL_ARTIFACT_CHUNK_MONGO_COLLECTION", default=DEFAULT_CHUNK_COLLECTION_NAME)
    ]
    file_collection = database[
        _get_env("PREDICTOR_MODEL_FILE_MONGO_COLLECTION", default=DEFAULT_FILE_COLLECTION_NAME)
    ]
    return client, model_collection, chunk_collection, file_collection


def _get_database(client):
    return client[_get_env("MONGO_DATABASE", default="predictor")]


def _get_model_details_collection(client):
    return _get_database(client)[
        _get_env("PREDICTOR_MODEL_DETAILS_MONGO_COLLECTION", default=DEFAULT_DETAILS_COLLECTION_NAME)
    ]


def ensure_indexes(model_collection, chunk_collection, file_collection) -> None:
    if ASCENDING is None:
        return
    model_collection.create_index([("key", ASCENDING)], unique=True, name="key_idx")
    model_collection.create_index([("model_id", ASCENDING)], unique=True, name="model_id_idx")
    chunk_collection.create_index(
        [("owner_key", ASCENDING), ("chunk_index", ASCENDING)],
        unique=True,
        name="owner_chunk_idx",
    )
    file_collection.create_index([("key", ASCENDING)], unique=True, name="key_idx")
    file_collection.create_index(
        [("model_id", ASCENDING), ("type_id", ASCENDING)],
        unique=True,
        name="model_type_idx",
    )


def ensure_model_details_indexes(details_collection) -> None:
    if ASCENDING is None:
        return
    details_collection.create_index([("key", ASCENDING)], unique=True, name="key_idx")
    details_collection.create_index(
        [("model_id", ASCENDING), ("file_api", ASCENDING)],
        unique=True,
        name="model_file_api_idx",
    )


def _bson_safe(value):
    if isinstance(value, dict):
        return {str(key): _bson_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_bson_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_bson_safe(item) for item in value]

    if isinstance(value, set):
        return [_bson_safe(item) for item in value]

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            scalar_value = item_method()
        except Exception:
            scalar_value = None
        else:
            if scalar_value is not value:
                return _bson_safe(scalar_value)

    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            list_value = tolist_method()
        except Exception:
            list_value = None
        else:
            if list_value is not value:
                return _bson_safe(list_value)

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _blob_metadata(payload: bytes) -> tuple[bytes, int, int, int]:
    compressed = zlib.compress(payload)
    chunk_size = _chunk_size_bytes()
    chunk_count = (len(compressed) + chunk_size - 1) // chunk_size
    return compressed, len(payload), len(compressed), chunk_count


def _write_blob(chunk_collection, owner_key: str, payload: bytes) -> dict[str, int]:
    compressed, byte_count, compressed_byte_count, chunk_count = _blob_metadata(payload)
    chunk_size = _chunk_size_bytes()
    operations = []
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        stop = start + chunk_size
        operations.append(
            ReplaceOne(
                {"owner_key": owner_key, "chunk_index": chunk_index},
                {
                    "owner_key": owner_key,
                    "chunk_index": chunk_index,
                    "data": compressed[start:stop],
                },
                upsert=True,
            )
        )

    chunk_collection.delete_many({"owner_key": owner_key, "chunk_index": {"$gte": chunk_count}})
    if operations:
        chunk_collection.bulk_write(operations, ordered=True)

    return {
        "byte_count": byte_count,
        "compressed_byte_count": compressed_byte_count,
        "chunk_count": chunk_count,
    }


def _read_blob(chunk_collection, owner_key: str, chunk_count: int) -> bytes | None:
    if chunk_count <= 0:
        return b""

    chunks = list(
        chunk_collection.find(
            {"owner_key": owner_key},
            {"_id": 0, "chunk_index": 1, "data": 1},
        ).sort("chunk_index", ASCENDING)
    )
    if len(chunks) != chunk_count:
        logging.warning(
            "Mongo artifact cache chunk count mismatch owner_key=%s expected=%s actual=%s",
            owner_key,
            chunk_count,
            len(chunks),
        )
        return None

    data = b"".join(bytes(chunk["data"]) for chunk in chunks)
    return zlib.decompress(data)


def write_model_snapshot(model_id: int | str, model: Any) -> None:
    client = None
    try:
        client, model_collection, chunk_collection, file_collection = connect_mongo()
        ensure_indexes(model_collection, chunk_collection, file_collection)
        key = _model_key(model_id)
        owner_key = _owner_key("model", key)
        payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        blob_info = _write_blob(chunk_collection, owner_key, payload)
        now = datetime.now(timezone.utc)
        model_collection.replace_one(
            {"key": key},
            {
                "key": key,
                "schema_version": MODEL_CACHE_SCHEMA_VERSION,
                "model_id": int(model_id),
                "owner_key": owner_key,
                "format": "pickle+zlib",
                "updated_at": now,
                **blob_info,
            },
            upsert=True,
        )
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model artifact cache unavailable: {exc}") from exc
    finally:
        if client is not None:
            client.close()


def read_model_snapshot(model_id: int | str):
    client = None
    try:
        client, model_collection, chunk_collection, _ = connect_mongo()
        key = _model_key(model_id)
        document = model_collection.find_one(
            {"key": key, "schema_version": MODEL_CACHE_SCHEMA_VERSION}
        )
        if not document:
            return None

        payload = _read_blob(
            chunk_collection,
            document["owner_key"],
            int(document.get("chunk_count") or 0),
        )
        if payload is None:
            return None
        return deserialize_model(payload)
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model artifact cache unavailable: {exc}") from exc
    except Exception as exc:
        logging.warning("Could not restore model_id=%s from Mongo artifact cache: %s", model_id, exc)
        return None
    finally:
        if client is not None:
            client.close()


def write_model_details(model_id: int | str, file_api: str | None, payload: dict[str, Any]) -> None:
    client = None
    try:
        client, _, _, _ = connect_mongo()
        details_collection = _get_model_details_collection(client)
        ensure_model_details_indexes(details_collection)
        key = _details_key(model_id, file_api)
        now = datetime.now(timezone.utc)
        details_collection.replace_one(
            {"key": key},
            {
                "key": key,
                "schema_version": MODEL_CACHE_SCHEMA_VERSION,
                "model_id": int(model_id),
                "file_api": file_api or "",
                "format": "bson",
                "payload": _bson_safe(payload),
                "updated_at": now,
            },
            upsert=True,
        )
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model details cache unavailable: {exc}") from exc
    finally:
        if client is not None:
            client.close()


def read_model_details(model_id: int | str, file_api: str | None) -> dict[str, Any] | None:
    client = None
    try:
        client, _, _, _ = connect_mongo()
        details_collection = _get_model_details_collection(client)
        document = details_collection.find_one(
            {
                "key": _details_key(model_id, file_api),
                "schema_version": MODEL_CACHE_SCHEMA_VERSION,
            },
            {"_id": 0, "payload": 1},
        )
        if not document:
            return None
        payload = document.get("payload")
        return _bson_safe(payload) if isinstance(payload, dict) else None
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model details cache unavailable: {exc}") from exc
    except Exception as exc:
        logging.warning("Could not restore modelDetails model_id=%s from Mongo cache: %s", model_id, exc)
        return None
    finally:
        if client is not None:
            client.close()


def write_model_file(
    model_id: int | str,
    type_id: int | str,
    raw_bytes: bytes,
    file_name: str,
    mime_type: str,
) -> None:
    client = None
    try:
        client, model_collection, chunk_collection, file_collection = connect_mongo()
        ensure_indexes(model_collection, chunk_collection, file_collection)
        key = _file_key(model_id, type_id)
        owner_key = _owner_key("file", key)
        blob_info = _write_blob(chunk_collection, owner_key, bytes(raw_bytes))
        now = datetime.now(timezone.utc)
        file_collection.replace_one(
            {"key": key},
            {
                "key": key,
                "schema_version": MODEL_CACHE_SCHEMA_VERSION,
                "model_id": int(model_id),
                "type_id": int(type_id),
                "file_name": file_name,
                "mime_type": mime_type,
                "owner_key": owner_key,
                "format": "bytes+zlib",
                "updated_at": now,
                **blob_info,
            },
            upsert=True,
        )
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model file cache unavailable: {exc}") from exc
    finally:
        if client is not None:
            client.close()


def read_model_file(model_id: int | str, type_id: int | str) -> tuple[bytes, str, str] | None:
    client = None
    try:
        client, _, chunk_collection, file_collection = connect_mongo()
        key = _file_key(model_id, type_id)
        document = file_collection.find_one(
            {"key": key, "schema_version": MODEL_CACHE_SCHEMA_VERSION}
        )
        if not document:
            return None
        raw_bytes = _read_blob(
            chunk_collection,
            document["owner_key"],
            int(document.get("chunk_count") or 0),
        )
        if raw_bytes is None:
            return None
        return raw_bytes, document["file_name"], document["mime_type"]
    except PyMongoError as exc:
        raise ModelArtifactCacheUnavailableError(f"Mongo model file cache unavailable: {exc}") from exc
    except Exception as exc:
        logging.warning(
            "Could not restore model file model_id=%s type_id=%s from Mongo cache: %s",
            model_id,
            type_id,
            exc,
        )
        return None
    finally:
        if client is not None:
            client.close()


__all__ = [
    "MODEL_CACHE_SCHEMA_VERSION",
    "ModelArtifactCacheUnavailableError",
    "connect_mongo",
    "ensure_indexes",
    "model_artifact_cache_enabled",
    "postgres_fallback_enabled",
    "read_model_file",
    "read_model_details",
    "read_model_snapshot",
    "write_model_file",
    "write_model_details",
    "write_model_snapshot",
]
