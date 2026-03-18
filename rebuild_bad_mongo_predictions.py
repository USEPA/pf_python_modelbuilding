import argparse
import json
import logging
import os
import time
from functools import lru_cache
from typing import Any

import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError

from util.indigo_utils import IndigoUtils
from util.prediction_cache_key_utils import build_prediction_cache_key, normalize_inchi_key


_INDIGO_UTILS = None


def _get_indigo_utils() -> IndigoUtils:
    global _INDIGO_UTILS
    if _INDIGO_UTILS is None:
        _INDIGO_UTILS = IndigoUtils()
    return _INDIGO_UTILS


@lru_cache(maxsize=50000)
def _inchi_key_from_smiles(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    smiles_text = str(smiles).strip()
    if not smiles_text:
        return None
    try:
        return normalize_inchi_key(_get_indigo_utils().inchi_key_from_smiles(smiles_text))
    except Exception as exc:
        logging.warning("Failed to generate InChIKey for SMILES=%s: %s", smiles_text, exc)
        return None


def _build_mongo_client(
    server_selection_timeout_ms: int,
    connect_timeout_ms: int,
    socket_timeout_ms: int,
) -> MongoClient:
    return MongoClient(
        host=os.getenv("MONGO_HOST", "192.168.1.3"),
        port=int(os.getenv("MONGO_PORT", "27017")),
        username=os.getenv("MONGO_USER", "root"),
        password=os.getenv("MONGO_PASSWORD", "qqq123"),
        authSource="admin",
        serverSelectionTimeoutMS=server_selection_timeout_ms,
        connectTimeoutMS=connect_timeout_ms,
        socketTimeoutMS=socket_timeout_ms,
    )


def _is_failed_standardization_error(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower().endswith("failed standardization")


def _has_failed_standardization(prediction: Any) -> bool:
    if isinstance(prediction, dict):
        model_results = prediction.get("modelResults")

        if isinstance(model_results, dict):
            if _is_failed_standardization_error(model_results.get("predictionError")):
                return True

        if isinstance(model_results, list):
            for item in model_results:
                if isinstance(item, dict) and _is_failed_standardization_error(item.get("predictionError")):
                    return True

    return False


def _contains_na_string(value: Any) -> bool:
    if isinstance(value, str):
        return value == "N/A"

    if isinstance(value, dict):
        return any(_contains_na_string(item) for item in value.values())

    if isinstance(value, list):
        return any(_contains_na_string(item) for item in value)

    return False


def _replace_na_with_none(value: Any) -> Any:
    if isinstance(value, str):
        return None if value == "N/A" else value

    if isinstance(value, dict):
        return {key: _replace_na_with_none(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_replace_na_with_none(item) for item in value]

    return value


def _prediction_to_obj(prediction: Any) -> Any:
    if isinstance(prediction, dict):
        return prediction

    if isinstance(prediction, str):
        try:
            return json.loads(prediction)
        except Exception:
            return prediction

    return prediction


def _extract_prediction_chemical_identifiers(prediction: Any) -> dict[str, Any] | None:
    prediction_obj = _prediction_to_obj(prediction)
    if not isinstance(prediction_obj, dict):
        return None

    chemical = prediction_obj.get("chemicalIdentifiers")
    if isinstance(chemical, dict):
        return chemical

    return None


def _extract_model_id_from_key(key: Any) -> int | None:
    if not isinstance(key, str) or "-" not in key:
        return None
    _, model_id_text = key.rsplit("-", 1)
    try:
        return int(model_id_text)
    except ValueError:
        return None


def _extract_smiles_from_prediction(prediction: Any) -> str | None:
    chemical = _extract_prediction_chemical_identifiers(prediction)
    if not chemical:
        return None

    for field_name in ("canonicalSmiles", "smiles"):
        value = chemical.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_inchi_key_from_prediction(prediction: Any) -> str | None:
    chemical = _extract_prediction_chemical_identifiers(prediction)
    if not chemical:
        return None

    return normalize_inchi_key(chemical.get("inchiKey"))


def _ensure_prediction_inchi_key(prediction: Any, fallback_smiles: str | None = None) -> tuple[Any, bool]:
    prediction_obj = _prediction_to_obj(prediction)
    if not isinstance(prediction_obj, dict):
        return prediction, False

    chemical = prediction_obj.get("chemicalIdentifiers")
    if not isinstance(chemical, dict):
        return prediction_obj, False

    raw_inchi_key = chemical.get("inchiKey")
    normalized_inchi_key = normalize_inchi_key(raw_inchi_key)
    if normalized_inchi_key is not None:
        if raw_inchi_key == normalized_inchi_key:
            return prediction_obj, False

        updated_prediction = dict(prediction_obj)
        updated_chemical = dict(chemical)
        updated_chemical["inchiKey"] = normalized_inchi_key
        updated_prediction["chemicalIdentifiers"] = updated_chemical
        return updated_prediction, True

    smiles = _extract_smiles_from_prediction(prediction_obj) or fallback_smiles
    inchi_key = _inchi_key_from_smiles(smiles)
    if not inchi_key:
        return prediction_obj, False

    updated_prediction = dict(prediction_obj)
    updated_chemical = dict(chemical)
    updated_chemical["inchiKey"] = inchi_key
    updated_prediction["chemicalIdentifiers"] = updated_chemical
    return updated_prediction, True


def _should_rebuild(prediction: Any) -> bool:
    if isinstance(prediction, str):
        return True
    return _has_failed_standardization(prediction)


def _extract_smiles_and_model_id_from_key(key: Any) -> tuple[str, int] | tuple[None, None]:
    if not isinstance(key, str) or not key:
        return None, None

    if "-" not in key:
        return None, None

    smiles, model_id_text = key.rsplit("-", 1)
    if not smiles:
        return None, None

    try:
        model_id = int(model_id_text)
    except ValueError:
        return None, None

    return smiles, model_id


def _extract_smiles_and_model_id_for_rebuild(key: Any, prediction: Any) -> tuple[str | None, int | None]:
    model_id = _extract_model_id_from_key(key)
    smiles = _extract_smiles_from_prediction(prediction)

    if smiles is not None:
        return smiles, model_id

    smiles_from_key, _ = _extract_smiles_and_model_id_from_key(key)
    return smiles_from_key, model_id


def _build_migrated_cache_key(key: Any, prediction: Any) -> tuple[str | None, str | None]:
    model_id = _extract_model_id_from_key(key)
    if model_id is None:
        return None, "could not parse model_id from key"

    smiles = _extract_smiles_from_prediction(prediction)
    if not smiles:
        smiles, _ = _extract_smiles_and_model_id_from_key(key)

    migrated_key = build_prediction_cache_key(
        model_id,
        _inchi_key_from_smiles,
        smiles=smiles,
        chemical=_extract_prediction_chemical_identifiers(prediction),
    )
    if not migrated_key:
        return None, "could not determine inchiKey from prediction or smiles"

    return migrated_key, None


def _call_predict_endpoint(base_url: str, model_id: int, smiles: str, timeout: int) -> bool:
    endpoint = f"{base_url.rstrip('/')}/api/predictor_models/predict"
    try:
        response = requests.get(
            endpoint,
            params={"model_id": model_id, "smiles": smiles, "report_format": "json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        logging.warning("Predict request failed for model_id=%s smiles=%s: %s", model_id, smiles, exc)
        return False

    if not response.ok:
        logging.warning(
            "Predict request returned %s for model_id=%s smiles=%s",
            response.status_code,
            model_id,
            smiles,
        )
        return False

    return True


def _get_collection(client: MongoClient) -> Collection:
    db_name = os.getenv("MONGO_DATABASE", "predictor")
    collection_name = os.getenv("MONGO_COLLECTION", "predictor_models_cache")
    return client[db_name][collection_name]


def _fetch_next_batch(collection: Collection, last_id, batch_size: int) -> list[dict[str, Any]]:
    query = {} if last_id is None else {"_id": {"$gt": last_id}}
    cursor = (
        collection.find(query, {"key": 1, "prediction": 1})
        .sort("_id", 1)
        .limit(batch_size)
    )
    return list(cursor)


def _count_error_categories(
    collection: Collection,
    batch_size: int,
    retries: int,
    retry_delay: float,
) -> tuple[int, int, int, int, int]:
    scanned = 0
    prediction_string_count = 0
    failed_standardization_count = 0
    contains_na_count = 0
    keys_needing_migration_count = 0
    last_id = None

    while True:
        try:
            batch = _fetch_next_batch(collection, last_id, batch_size)
        except PyMongoError as exc:
            retry_ok = False
            for attempt in range(1, retries + 1):
                logging.warning(
                    "Mongo pre-scan batch read failed (attempt %d/%d): %s",
                    attempt,
                    retries,
                    exc,
                )
                time.sleep(retry_delay)
                try:
                    batch = _fetch_next_batch(collection, last_id, batch_size)
                    retry_ok = True
                    break
                except PyMongoError as retry_exc:
                    exc = retry_exc

            if not retry_ok:
                raise

        if not batch:
            break

        for doc in batch:
            scanned += 1
            prediction = doc.get("prediction")
            if isinstance(prediction, str):
                prediction_string_count += 1
            if _has_failed_standardization(prediction):
                failed_standardization_count += 1
            if _contains_na_string(prediction):
                contains_na_count += 1
            migrated_key, _ = _build_migrated_cache_key(doc.get("key"), prediction)
            if isinstance(migrated_key, str) and migrated_key != doc.get("key"):
                keys_needing_migration_count += 1
            last_id = doc["_id"]

    return scanned, prediction_string_count, failed_standardization_count, contains_na_count, keys_needing_migration_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find bad cached predictions in Mongo, delete them, and rebuild via predict endpoint."
        )
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("PREDICT_API_BASE_URL", "http://192.168.1.7:5004"),
        help="Base URL of predictor service, for example http://localhost:5004",
    )
    parser.add_argument(
        "--smiles-log-file",
        default="processed_smiles.txt",
        help="Output text file with one processed smiles per line",
    )
    parser.add_argument(
        "--enable-processing",
        action="store_true",
        help="If set, rebuild deleted cache entries via predict endpoint",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=int(os.getenv("PREDICT_REQUEST_TIMEOUT_SECONDS", "30")),
        help="Timeout for each predict request in seconds",
    )
    parser.add_argument(
        "--mongo-batch-size",
        type=int,
        default=int(os.getenv("MONGO_BATCH_SIZE", "200")),
        help="How many cache documents to read per batch",
    )
    parser.add_argument(
        "--mongo-retries",
        type=int,
        default=int(os.getenv("MONGO_RETRIES", "5")),
        help="Retry attempts for Mongo read failures",
    )
    parser.add_argument(
        "--mongo-retry-delay",
        type=float,
        default=float(os.getenv("MONGO_RETRY_DELAY_SECONDS", "2")),
        help="Delay between Mongo read retries in seconds",
    )
    parser.add_argument(
        "--mongo-server-selection-timeout-ms",
        type=int,
        default=int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "20000")),
        help="Mongo server selection timeout in milliseconds",
    )
    parser.add_argument(
        "--mongo-connect-timeout-ms",
        type=int,
        default=int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "20000")),
        help="Mongo connect timeout in milliseconds",
    )
    parser.add_argument(
        "--mongo-socket-timeout-ms",
        type=int,
        default=int(os.getenv("MONGO_SOCKET_TIMEOUT_MS", "120000")),
        help="Mongo socket timeout in milliseconds",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    processed = 0
    rebuilt_ok = 0
    deleted = 0
    scanned = 0
    deleted_errors = 0
    sanitized_na = 0
    sanitize_update_failed = 0
    migrated_keys = 0
    migration_failed = 0
    duplicate_key_collisions = 0

    client = _build_mongo_client(
        server_selection_timeout_ms=args.mongo_server_selection_timeout_ms,
        connect_timeout_ms=args.mongo_connect_timeout_ms,
        socket_timeout_ms=args.mongo_socket_timeout_ms,
    )
    try:
        client.admin.command("ping")
        collection = _get_collection(client)
        pre_scanned, pre_prediction_strings, pre_failed_standardization, pre_contains_na, pre_keys_needing_migration = _count_error_categories(
            collection=collection,
            batch_size=args.mongo_batch_size,
            retries=args.mongo_retries,
            retry_delay=args.mongo_retry_delay,
        )

        print("Pre-check summary before processing:")
        print(f"Scanned records: {pre_scanned}")
        print(f"prediction is string (instead of JSON): {pre_prediction_strings}")
        print(f"failed standardization: {pre_failed_standardization}")
        print(f'contains "N/A": {pre_contains_na}')
        print(f"keys needing migration to inchiKey-model_id: {pre_keys_needing_migration}")

        last_id = None

        with open(args.smiles_log_file, "a", encoding="utf-8") as smiles_log:
            while True:
                try:
                    batch = _fetch_next_batch(collection, last_id, args.mongo_batch_size)
                except PyMongoError as exc:
                    retry_ok = False
                    for attempt in range(1, args.mongo_retries + 1):
                        logging.warning(
                            "Mongo batch read failed (attempt %d/%d): %s",
                            attempt,
                            args.mongo_retries,
                            exc,
                        )
                        time.sleep(args.mongo_retry_delay)

                        try:
                            client.close()
                        except Exception:
                            pass

                        client = _build_mongo_client(
                            server_selection_timeout_ms=args.mongo_server_selection_timeout_ms,
                            connect_timeout_ms=args.mongo_connect_timeout_ms,
                            socket_timeout_ms=args.mongo_socket_timeout_ms,
                        )
                        collection = _get_collection(client)

                        try:
                            batch = _fetch_next_batch(collection, last_id, args.mongo_batch_size)
                            retry_ok = True
                            break
                        except PyMongoError as retry_exc:
                            exc = retry_exc

                    if not retry_ok:
                        raise

                if not batch:
                    break

                batch_scanned = 0
                batch_skipped_no_rebuild = 0
                batch_skipped_bad_key = 0
                batch_delete_failed = 0
                batch_deleted_errors = 0
                batch_processed = 0
                batch_rebuilt_ok = 0
                batch_rebuild_failed = 0
                batch_sanitized_na = 0
                batch_sanitize_failed = 0
                batch_migrated_keys = 0
                batch_migration_failed = 0
                batch_duplicate_key_collisions = 0

                for doc in batch:
                    scanned += 1
                    batch_scanned += 1
                    prediction = doc.get("prediction")
                    key = doc.get("key")
                    has_na_values = _contains_na_string(prediction)

                    if has_na_values and not _should_rebuild(prediction):
                        prediction = _replace_na_with_none(prediction)
                        has_na_values = False
                        prediction_changed = True
                    else:
                        prediction_changed = False

                    prediction, inchi_key_backfilled = _ensure_prediction_inchi_key(prediction)
                    prediction_changed = prediction_changed or inchi_key_backfilled

                    if not _should_rebuild(prediction):
                        batch_skipped_no_rebuild += 1
                        migrated_key, migration_reason = _build_migrated_cache_key(key, prediction)
                        update_fields = {}

                        if prediction_changed:
                            update_fields["prediction"] = prediction

                        if isinstance(migrated_key, str) and migrated_key != key:
                            update_fields["key"] = migrated_key

                        if update_fields:
                            try:
                                collection.update_one(
                                    {"_id": doc["_id"]},
                                    {"$set": update_fields},
                                )
                                if "prediction" in update_fields and _contains_na_string(doc.get("prediction")):
                                    sanitized_na += 1
                                    batch_sanitized_na += 1
                                if "key" in update_fields:
                                    migrated_keys += 1
                                    batch_migrated_keys += 1
                            except DuplicateKeyError:
                                duplicate_key_collisions += 1
                                batch_duplicate_key_collisions += 1
                                try:
                                    collection.delete_one({"_id": doc["_id"]})
                                except PyMongoError as exc:
                                    migration_failed += 1
                                    batch_migration_failed += 1
                                    logging.warning(
                                        "Duplicate key collision while migrating key=%s to %s and failed to delete old document: %s",
                                        key,
                                        migrated_key,
                                        exc,
                                    )
                                else:
                                    logging.info(
                                        "Deleted duplicate cache entry during key migration: old_key=%s new_key=%s",
                                        key,
                                        migrated_key,
                                    )
                            except PyMongoError as exc:
                                if "prediction" in update_fields and _contains_na_string(doc.get("prediction")):
                                    sanitize_update_failed += 1
                                    batch_sanitize_failed += 1
                                if "key" in update_fields:
                                    migration_failed += 1
                                    batch_migration_failed += 1
                                logging.warning(
                                    'Failed to update cache entry for key=%s (new_key=%s): %s',
                                    key,
                                    migrated_key,
                                    exc,
                                )

                        elif migrated_key is None and migration_reason:
                            migration_failed += 1
                            batch_migration_failed += 1
                            logging.warning("Skip key migration for key=%s: %s", key, migration_reason)

                        last_id = doc["_id"]
                        continue

                    try:
                        delete_result = collection.delete_one({"_id": doc["_id"]})
                        if delete_result.deleted_count:
                            deleted += 1
                            deleted_errors += 1
                            batch_deleted_errors += 1
                    except PyMongoError as exc:
                        batch_delete_failed += 1
                        logging.warning("Failed to delete cache entry for key=%s: %s", key, exc)
                        last_id = doc["_id"]
                        continue

                    if args.enable_processing:
                        smiles, model_id = _extract_smiles_and_model_id_for_rebuild(key, prediction)
                        if smiles is None or model_id is None:
                            batch_skipped_bad_key += 1
                            logging.warning("Skip rebuild due to unexpected key format: %s", key)
                            last_id = doc["_id"]
                            continue

                        smiles_log.write(f"{smiles}\n")
                        processed += 1
                        batch_processed += 1

                        if _call_predict_endpoint(args.api_base_url, model_id, smiles, args.request_timeout):
                            rebuilt_ok += 1
                            batch_rebuilt_ok += 1
                        else:
                            batch_rebuild_failed += 1

                    last_id = doc["_id"]

                logging.info(
                    "Batch done: scanned=%d, deleted_errors=%d, processed=%d, rebuilt_ok=%d, rebuilt_failed=%d, "
                    "sanitized_na=%d, sanitize_failed=%d, migrated_keys=%d, migration_failed=%d, duplicate_key_collisions=%d, "
                    "skipped_no_rebuild=%d, skipped_bad_key=%d, delete_failed=%d | "
                    "Totals: scanned=%d, deleted_errors=%d, processed=%d, deleted=%d, rebuilt_ok=%d, sanitized_na=%d, "
                    "sanitize_failed=%d, migrated_keys=%d, migration_failed=%d, duplicate_key_collisions=%d",
                    batch_scanned,
                    batch_deleted_errors,
                    batch_processed,
                    batch_rebuilt_ok,
                    batch_rebuild_failed,
                    batch_sanitized_na,
                    batch_sanitize_failed,
                    batch_migrated_keys,
                    batch_migration_failed,
                    batch_duplicate_key_collisions,
                    batch_skipped_no_rebuild,
                    batch_skipped_bad_key,
                    batch_delete_failed,
                    scanned,
                    deleted_errors,
                    processed,
                    deleted,
                    rebuilt_ok,
                    sanitized_na,
                    sanitize_update_failed,
                    migrated_keys,
                    migration_failed,
                    duplicate_key_collisions,
                )

    except PyMongoError as exc:
        logging.error("Mongo operation failed: %s", exc)
        raise
    finally:
        client.close()

    print(f"Processed records: {processed}")
    print(f"Deleted error records: {deleted_errors}")
    print(f"Deleted from mongo: {deleted}")
    print(f"Successfully rebuilt via predictDB: {rebuilt_ok}")
    print(f'Replaced "N/A" with null: {sanitized_na}')
    print(f'Failed "N/A" updates: {sanitize_update_failed}')
    print(f"Migrated keys to inchiKey-model_id: {migrated_keys}")
    print(f"Failed key migrations: {migration_failed}")
    print(f"Duplicate key collisions handled: {duplicate_key_collisions}")
    print(f"Scanned records: {scanned}")


if __name__ == "__main__":
    main()
