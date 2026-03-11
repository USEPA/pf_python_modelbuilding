import argparse
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


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
) -> tuple[int, int, int]:
    scanned = 0
    prediction_string_count = 0
    failed_standardization_count = 0
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
            elif _has_failed_standardization(prediction):
                failed_standardization_count += 1
            last_id = doc["_id"]

    return scanned, prediction_string_count, failed_standardization_count


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

    client = _build_mongo_client(
        server_selection_timeout_ms=args.mongo_server_selection_timeout_ms,
        connect_timeout_ms=args.mongo_connect_timeout_ms,
        socket_timeout_ms=args.mongo_socket_timeout_ms,
    )
    try:
        client.admin.command("ping")
        collection = _get_collection(client)
        pre_scanned, pre_prediction_strings, pre_failed_standardization = _count_error_categories(
            collection=collection,
            batch_size=args.mongo_batch_size,
            retries=args.mongo_retries,
            retry_delay=args.mongo_retry_delay,
        )

        print("Pre-check summary before processing:")
        print(f"Scanned records: {pre_scanned}")
        print(f"prediction is string (instead of JSON): {pre_prediction_strings}")
        print(f"failed standardization: {pre_failed_standardization}")

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

                for doc in batch:
                    scanned += 1
                    batch_scanned += 1
                    prediction = doc.get("prediction")
                    if not _should_rebuild(prediction):
                        batch_skipped_no_rebuild += 1
                        last_id = doc["_id"]
                        continue

                    key = doc.get("key")
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
                        smiles, model_id = _extract_smiles_and_model_id_from_key(key)
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
                    "skipped_no_rebuild=%d, skipped_bad_key=%d, delete_failed=%d | "
                    "Totals: scanned=%d, deleted_errors=%d, processed=%d, deleted=%d, rebuilt_ok=%d",
                    batch_scanned,
                    batch_deleted_errors,
                    batch_processed,
                    batch_rebuilt_ok,
                    batch_rebuild_failed,
                    batch_skipped_no_rebuild,
                    batch_skipped_bad_key,
                    batch_delete_failed,
                    scanned,
                    deleted_errors,
                    processed,
                    deleted,
                    rebuilt_ok,
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
    print(f"Scanned records: {scanned}")


if __name__ == "__main__":
    main()
