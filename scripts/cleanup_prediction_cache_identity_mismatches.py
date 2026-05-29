#!/usr/bin/env python3
"""
Find, delete, or recompute predictor_models_cache records whose cache-key
InChIKey does not match prediction.chemicalIdentifiers.inchiKey.

These records are unsafe because the cache key belongs to one chemical while the
stored prediction payload belongs to another. The cleanup is intentionally split
into two phases:

1. scan the whole collection and write mismatched cache keys to a file;
2. after the scan finishes, read that file in batches, delete stale cache rows,
   and optionally warm the cache by calling the prediction API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from pymongo import DeleteOne, MongoClient
from pymongo.errors import PyMongoError
from requests import RequestException, Timeout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


from util.prediction_cache_key_utils import (  # noqa: E402
    inchi_keys_match_connectivity,
    normalize_inchi_key,
    prediction_cache_key_inchi_key,
)


DEFAULT_BATCH_SIZE = 100
DEFAULT_CURSOR_BATCH_SIZE = 5000
DEFAULT_PROGRESS_EVERY = 5000
DEFAULT_PREDICTION_BATCH_SIZE = 25
DEFAULT_RESOLVER_LOOKUP_API = "https://cim-dev.sciencedataexperts.com/api/resolver/lookup"
DEFAULT_PREDICT_API = "https://cim-dev.sciencedataexperts.com/api/predictor_models/predict"
DEFAULT_PREDICTION_GATEWAY_SPLIT_THRESHOLD = 10
DEFAULT_MISMATCH_KEYS_FILE = Path("/tmp/predictor_cache_identity_mismatch_keys.txt")
KEY_RE = re.compile(r"^([A-Z]{14}-[A-Z]{10}-[A-Z])-(.+)$")
LOGGER = logging.getLogger("prediction_cache_identity_cleanup")


@dataclass(frozen=True)
class MismatchCandidate:
    doc_id: Any
    key: str
    model_id: str
    key_inchi_key: str
    chemical_inchi_key: str
    chemical_inchi_key_raw: str
    chemical_smiles: str | None
    chemical_canonical_smiles: str | None
    chemical_name: str | None
    has_standardized_chemical: bool


@dataclass
class CleanupStats:
    scanned_documents: int = 0
    candidates: int = 0
    planned_deletes: int = 0
    deleted_documents: int = 0
    resolver_batches: int = 0
    resolver_hits: int = 0
    prediction_batches: int = 0
    prediction_requests: int = 0
    prediction_failures: int = 0
    recomputed_predictions: int = 0
    skipped_missing_key_inchi: int = 0
    skipped_missing_chemical_inchi: int = 0
    skipped_without_resolver_match: int = 0


def _load_env_file_without_dotenv(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def load_env(env_file: Path | None) -> None:
    if env_file is None:
        return

    if load_dotenv is not None:
        load_dotenv(env_file)
        return

    _load_env_file_without_dotenv(env_file)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find or delete poisoned predictor_models_cache records where the "
            "InChIKey encoded in key does not match prediction.chemicalIdentifiers.inchiKey."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to an env file with Mongo credentials. Defaults to pf_python_model_building/.env.",
    )
    parser.add_argument(
        "--skip-env-file",
        action="store_true",
        help="Do not load values from --env-file.",
    )
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help="Optional full Mongo URI. If omitted, MONGO_HOST/MONGO_PORT/MONGO_USER/MONGO_PASSWORD are used.",
    )
    parser.add_argument("--database", default=None, help="Mongo database name.")
    parser.add_argument(
        "--collection",
        default="predictor_models_cache",
        help="Mongo collection name.",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=[],
        help="Exact cache key to inspect. Can be supplied more than once.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help="Only report/delete cache keys for this model id. Can be supplied more than once.",
    )
    parser.add_argument(
        "--only-without-standardized",
        action="store_true",
        help="Only flag mismatches that do not already have prediction.standardizedChemical.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of mismatched candidates to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of mismatched cache keys to write/process per repair batch.",
    )
    parser.add_argument(
        "--repair-mode",
        choices=("delete", "recompute"),
        default="delete",
        help=(
            "delete removes poisoned records only; recompute removes them and then "
            "warms the cache by calling the prediction API in batches."
        ),
    )
    parser.add_argument(
        "--resolver-url",
        default=None,
        help="CIM resolver lookup endpoint used by --repair-mode recompute.",
    )
    parser.add_argument(
        "--predict-url",
        default=None,
        help="Predictor models predict endpoint used by --repair-mode recompute.",
    )
    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=DEFAULT_PREDICTION_BATCH_SIZE,
        help="Number of SMILES per prediction API POST batch in --repair-mode recompute.",
    )
    parser.add_argument(
        "--prediction-gateway-split-threshold",
        type=int,
        default=DEFAULT_PREDICTION_GATEWAY_SPLIT_THRESHOLD,
        help=(
            "For prediction API timeout/502/503/504 failures, split batches only while "
            "the failing batch is larger than this size. Smaller failing batches are "
            "counted as failures without retrying each SMILES individually. Use 1 to "
            "split gateway failures down to singles."
        ),
    )
    parser.add_argument(
        "--resolver-timeout",
        type=float,
        default=None,
        help="Resolver request timeout in seconds.",
    )
    parser.add_argument(
        "--predict-timeout",
        type=float,
        default=None,
        help="Prediction API request timeout in seconds.",
    )
    parser.add_argument(
        "--mongo-cursor-batch-size",
        type=int,
        default=DEFAULT_CURSOR_BATCH_SIZE,
        help="Number of Mongo documents fetched per cursor page.",
    )
    parser.add_argument(
        "--mongo-socket-timeout-ms",
        type=int,
        default=None,
        help=(
            "Override Mongo socketTimeoutMS for this cleanup run. Use 0 to disable. "
            "Defaults to CACHE_CLEANUP_MONGO_SOCKET_TIMEOUT_MS or PyMongo's no-timeout default."
        ),
    )
    parser.add_argument(
        "--mongo-cursor-retries",
        type=int,
        default=3,
        help="Retry a Mongo scan page this many times after a cursor read failure.",
    )
    parser.add_argument(
        "--mongo-cursor-retry-sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between Mongo scan page retries.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Log scan progress after this many Mongo documents. Use 0 to disable.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Log up to this many sample mismatches. Use 0 to disable samples.",
    )
    parser.add_argument(
        "--report-jsonl",
        type=Path,
        help="Optional path for a JSONL report of mismatched records.",
    )
    parser.add_argument(
        "--mismatch-keys-file",
        type=Path,
        default=DEFAULT_MISMATCH_KEYS_FILE,
        help=(
            "Path for the scan output file containing one mismatched cache key per line. "
            "The repair phase reads this file after the full scan completes."
        ),
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Delete mismatched records from Mongo. Default is dry-run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logger verbosity.",
    )
    return parser.parse_args()


def resolve_args_from_env(args: argparse.Namespace) -> None:
    if args.mongo_uri is None:
        args.mongo_uri = os.getenv("MONGO_URI")
    if args.database is None:
        args.database = os.getenv("MONGO_DATABASE", "predictor")
    if args.resolver_url is None:
        args.resolver_url = os.getenv("RESOLVER_LOOKUP_API", DEFAULT_RESOLVER_LOOKUP_API)
    if args.predict_url is None:
        args.predict_url = os.getenv("PREDICTOR_MODELS_PREDICT_API", DEFAULT_PREDICT_API)
    if args.resolver_timeout is None:
        args.resolver_timeout = float(os.getenv("RESOLVER_LOOKUP_TIMEOUT", "30"))
    if args.predict_timeout is None:
        args.predict_timeout = float(os.getenv("PREDICTOR_MODELS_PREDICT_TIMEOUT", "300"))


def resolve_mongo_socket_timeout_ms(args: argparse.Namespace) -> int | None:
    socket_timeout_ms = args.mongo_socket_timeout_ms
    if socket_timeout_ms is None:
        raw_value = os.getenv("CACHE_CLEANUP_MONGO_SOCKET_TIMEOUT_MS")
        if raw_value is None:
            return None
        socket_timeout_ms = int(raw_value)

    if socket_timeout_ms < 0:
        raise ValueError("--mongo-socket-timeout-ms cannot be negative")
    if socket_timeout_ms == 0:
        return None
    return socket_timeout_ms


def build_mongo_client(args: argparse.Namespace) -> MongoClient:
    timeout_ms = _env_int("MONGO_SERVER_SELECTION_TIMEOUT_MS", 5000)
    socket_timeout_ms = resolve_mongo_socket_timeout_ms(args)
    app_name = os.getenv("MONGO_APP_NAME", "predictor_models_cache_identity_cleanup")

    if args.mongo_uri:
        client_kwargs = {
            "appname": app_name,
            "serverSelectionTimeoutMS": timeout_ms,
            "connectTimeoutMS": _env_int("MONGO_CONNECT_TIMEOUT_MS", timeout_ms),
        }
        if socket_timeout_ms is not None:
            client_kwargs["socketTimeoutMS"] = socket_timeout_ms
        return MongoClient(args.mongo_uri, **client_kwargs)

    client_kwargs = {
        "host": os.getenv("MONGO_HOST", "localhost"),
        "port": _env_int("MONGO_PORT", 27017),
        "authSource": os.getenv("MONGO_AUTH_SOURCE", "admin"),
        "appname": app_name,
        "serverSelectionTimeoutMS": timeout_ms,
        "connectTimeoutMS": _env_int("MONGO_CONNECT_TIMEOUT_MS", timeout_ms),
    }
    if socket_timeout_ms is not None:
        client_kwargs["socketTimeoutMS"] = socket_timeout_ms

    username = os.getenv("MONGO_USER", "root")
    password = os.getenv("MONGO_PASSWORD")
    if username:
        client_kwargs["username"] = username
    if password:
        client_kwargs["password"] = password

    return MongoClient(**client_kwargs)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_model_id_from_key(key: str) -> str | None:
    match = KEY_RE.fullmatch(key.strip())
    if match is None:
        return None
    return match.group(2)


def build_scan_query(args: argparse.Namespace) -> dict[str, Any]:
    if args.key:
        return {"key": {"$in": list(dict.fromkeys(args.key))}}

    clauses: list[dict[str, Any]] = [
        {"key": {"$regex": r"^[A-Z]{14}-[A-Z]{10}-[A-Z]-"}},
        {"prediction.chemicalIdentifiers": {"$type": "object"}},
        {"prediction.chemicalIdentifiers.inchiKey": {"$type": "string"}},
    ]
    if args.only_without_standardized:
        clauses.append({"prediction.standardizedChemical": {"$exists": False}})
    return {"$and": clauses}


def mismatch_projection() -> dict[str, int]:
    return {
        "key": 1,
        "prediction.chemicalIdentifiers.inchiKey": 1,
        "prediction.chemicalIdentifiers.smiles": 1,
        "prediction.chemicalIdentifiers.canonicalSmiles": 1,
        "prediction.chemicalIdentifiers.name": 1,
        "prediction.standardizedChemical": 1,
    }


def build_resume_scan_query(query: dict[str, Any], last_seen_key: str | None) -> dict[str, Any]:
    if last_seen_key is None:
        return query

    resume_clause = {"key": {"$gt": last_seen_key}}
    if not query:
        return resume_clause
    return {"$and": [query, resume_clause]}


def mismatch_from_doc(
    doc: dict[str, Any],
    *,
    model_ids: set[str],
    only_without_standardized: bool,
    stats: CleanupStats | None = None,
) -> MismatchCandidate | None:
    key = doc.get("key")
    if not isinstance(key, str):
        if stats is not None:
            stats.skipped_missing_key_inchi += 1
        return None

    key_inchi_key = prediction_cache_key_inchi_key(key)
    model_id = _parse_model_id_from_key(key)
    if key_inchi_key is None or model_id is None:
        if stats is not None:
            stats.skipped_missing_key_inchi += 1
        return None

    if model_ids and model_id not in model_ids:
        return None

    prediction = doc.get("prediction")
    if not isinstance(prediction, dict):
        return None

    has_standardized_chemical = isinstance(prediction.get("standardizedChemical"), dict)
    if only_without_standardized and has_standardized_chemical:
        return None

    chemical = prediction.get("chemicalIdentifiers")
    if not isinstance(chemical, dict):
        return None

    raw_chemical_inchi_key = chemical.get("inchiKey")
    chemical_inchi_key = normalize_inchi_key(raw_chemical_inchi_key)
    if chemical_inchi_key is None:
        if stats is not None:
            stats.skipped_missing_chemical_inchi += 1
        return None

    if inchi_keys_match_connectivity(key_inchi_key, chemical_inchi_key):
        return None

    return MismatchCandidate(
        doc_id=doc["_id"],
        key=key,
        model_id=model_id,
        key_inchi_key=key_inchi_key,
        chemical_inchi_key=chemical_inchi_key,
        chemical_inchi_key_raw=raw_chemical_inchi_key,
        chemical_smiles=chemical.get("smiles") if isinstance(chemical.get("smiles"), str) else None,
        chemical_canonical_smiles=(
            chemical.get("canonicalSmiles") if isinstance(chemical.get("canonicalSmiles"), str) else None
        ),
        chemical_name=chemical.get("name") if isinstance(chemical.get("name"), str) else None,
        has_standardized_chemical=has_standardized_chemical,
    )


def iter_mismatch_batches(
    collection,
    args: argparse.Namespace,
    stats: CleanupStats,
):
    query = build_scan_query(args)
    projection = mismatch_projection()
    model_ids = {str(model_id) for model_id in args.model_id}

    candidate_buffer = []
    last_seen_key = None
    selected_candidates = 0
    limit_reached = False

    while not limit_reached:
        retries_remaining = args.mongo_cursor_retries

        while True:
            cursor = (
                collection.find(
                    build_resume_scan_query(query, last_seen_key),
                    projection=projection,
                )
                .sort("key", 1)
                .hint("key_idx")
                .limit(args.mongo_cursor_batch_size)
                .batch_size(args.mongo_cursor_batch_size)
            )
            page_doc_count = 0
            try:
                for doc in cursor:
                    page_doc_count += 1
                    last_seen_key = doc.get("key")
                    stats.scanned_documents += 1
                    if args.progress_every > 0 and stats.scanned_documents % args.progress_every == 0:
                        LOGGER.info(
                            "Scan progress: scanned_documents=%s candidates=%s last_seen_key=%s",
                            stats.scanned_documents,
                            selected_candidates,
                            last_seen_key,
                        )

                    candidate = mismatch_from_doc(
                        doc,
                        model_ids=model_ids,
                        only_without_standardized=args.only_without_standardized,
                        stats=stats,
                    )
                    if candidate is None:
                        continue

                    candidate_buffer.append(candidate)
                    selected_candidates += 1
                    if args.limit is not None and selected_candidates >= args.limit:
                        limit_reached = True
                        break

                break
            except PyMongoError as exc:
                if retries_remaining <= 0:
                    raise

                retries_remaining -= 1
                LOGGER.warning(
                    "Mongo cursor read failed; retrying scan page from last_seen_key=%s "
                    "retries_remaining=%s error=%s",
                    last_seen_key,
                    retries_remaining,
                    exc,
                )
                if args.mongo_cursor_retry_sleep > 0:
                    time.sleep(args.mongo_cursor_retry_sleep)
            finally:
                cursor.close()

        while len(candidate_buffer) >= args.batch_size:
            yield candidate_buffer[: args.batch_size]
            candidate_buffer = candidate_buffer[args.batch_size :]

        if page_doc_count < args.mongo_cursor_batch_size:
            break

    if candidate_buffer:
        yield candidate_buffer


def build_delete_filter(candidate: MismatchCandidate) -> dict[str, Any]:
    return {
        "_id": candidate.doc_id,
        "key": candidate.key,
        "prediction.chemicalIdentifiers.inchiKey": candidate.chemical_inchi_key_raw,
    }


def write_report_rows(report_path: Path | None, candidates: list[MismatchCandidate]) -> None:
    if report_path is None or not candidates:
        return

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as report_file:
        for candidate in candidates:
            row = asdict(candidate)
            row["doc_id"] = str(candidate.doc_id)
            report_file.write(json.dumps(row, sort_keys=True) + "\n")


def reset_output_file(path: Path | None) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_mismatch_keys(keys_path: Path, candidates: list[MismatchCandidate]) -> None:
    if not candidates:
        return

    keys_path.parent.mkdir(parents=True, exist_ok=True)
    with keys_path.open("a", encoding="utf-8") as keys_file:
        for candidate in candidates:
            keys_file.write(f"{candidate.key}\n")


def count_file_records(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open("r", encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def iter_key_file_batches(path: Path, batch_size: int):
    batch = []

    with path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            key = raw_line.strip()
            if not key:
                continue

            batch.append(key)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def _ordered_unique(values: list[str]) -> list[str]:
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _chunk_sequence(items, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def build_inchi_key_resolver_payload(inchi_keys: list[str]) -> dict[str, Any]:
    return {
        "fuzzy": "Not",
        "ids": list(inchi_keys),
        "idsType": "InChIKey",
        "mol": False,
        "filters": {},
        "format": "UNKNOWN",
    }


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.upper() == "N/A":
        return None
    return text


def _extract_chemical(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, list):
        for item in payload:
            chemical = _extract_chemical(item)
            if chemical:
                return chemical
        return None

    if not isinstance(payload, dict):
        return None

    chemical = payload.get("chemical")
    if isinstance(chemical, dict):
        return chemical

    chemicals = payload.get("chemicals")
    if isinstance(chemicals, list):
        for item in chemicals:
            chemical = _extract_chemical(item)
            if chemical:
                return chemical

    if any(
        field in payload
        for field in ("chemId", "cid", "sid", "casrn", "name", "smiles", "canonicalSmiles", "inchiKey")
    ):
        return payload

    return None


def _extract_resolver_items(payload: Any) -> list[Any] | None:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("chemicals", "results", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
        return [payload]

    return None


def _match_resolver_item_index(raw_item: Any, input_size: int) -> int | None:
    if not isinstance(raw_item, dict):
        return None

    for key in ("id", "recordId", "index", "idx"):
        value = raw_item.get(key)
        if value is None:
            continue
        try:
            index = int(str(value).strip())
        except ValueError:
            continue
        if 0 <= index < input_size:
            return index

    return None


def _match_resolver_item_query(raw_item: Any, inchi_keys: list[str], used_indexes: set[int]) -> int | None:
    if not isinstance(raw_item, dict):
        return None

    query_inchi_key = normalize_inchi_key(raw_item.get("query") or raw_item.get("input"))
    if query_inchi_key is None:
        return None

    for index, inchi_key in enumerate(inchi_keys):
        if index in used_indexes:
            continue
        if inchi_keys_match_connectivity(inchi_key, query_inchi_key):
            return index

    return None


def _match_resolver_item_chemical(
    chemical: dict[str, Any] | None,
    inchi_keys: list[str],
    used_indexes: set[int],
) -> int | None:
    if not isinstance(chemical, dict):
        return None

    chemical_inchi_key = normalize_inchi_key(chemical.get("inchiKey"))
    if chemical_inchi_key is None:
        return None

    for index, inchi_key in enumerate(inchi_keys):
        if index in used_indexes:
            continue
        if inchi_keys_match_connectivity(inchi_key, chemical_inchi_key):
            return index

    return None


def parse_inchi_key_resolver_payload(payload: Any, inchi_keys: list[str]) -> dict[str, dict[str, Any]]:
    items = _extract_resolver_items(payload)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Unexpected resolver response: {payload!r}")

    results = {}
    used_indexes: set[int] = set()
    use_positional_fallback = len(items) == len(inchi_keys)

    for item_position, item in enumerate(items):
        chemical = _extract_chemical(item)
        if not isinstance(chemical, dict):
            continue

        match_index = _match_resolver_item_index(item, len(inchi_keys))
        if match_index is not None and match_index in used_indexes:
            match_index = None
        if match_index is None:
            match_index = _match_resolver_item_query(item, inchi_keys, used_indexes)
        if match_index is None:
            match_index = _match_resolver_item_chemical(chemical, inchi_keys, used_indexes)
        if match_index is None and use_positional_fallback and item_position not in used_indexes:
            match_index = item_position

        if match_index is None or match_index in used_indexes:
            continue

        input_inchi_key = inchi_keys[match_index]
        chemical_inchi_key = normalize_inchi_key(chemical.get("inchiKey"))
        if chemical_inchi_key is not None and not inchi_keys_match_connectivity(input_inchi_key, chemical_inchi_key):
            LOGGER.warning(
                "Resolver returned nonmatching chemical for InChIKey=%s resolved_inchi=%s",
                input_inchi_key,
                chemical_inchi_key,
            )
            continue

        used_indexes.add(match_index)
        results[input_inchi_key] = chemical

    return results


def lookup_inchi_keys_with_fallback(
    session: requests.Session,
    resolver_url: str,
    inchi_keys: list[str],
    timeout: float,
    stats: CleanupStats,
) -> dict[str, dict[str, Any]]:
    if not inchi_keys:
        return {}

    stats.resolver_batches += 1
    try:
        response = session.post(
            resolver_url,
            json=build_inchi_key_resolver_payload(inchi_keys),
            headers={"accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        return parse_inchi_key_resolver_payload(response.json(), inchi_keys)
    except (RequestException, ValueError) as exc:
        if len(inchi_keys) == 1:
            LOGGER.warning(
                "Resolver lookup failed for one InChIKey; skipping inchi_key=%s error=%s",
                inchi_keys[0],
                exc,
            )
            return {}

        midpoint = len(inchi_keys) // 2
        LOGGER.warning(
            "Resolver batch failed; splitting batch_size=%s left=%s right=%s error=%s",
            len(inchi_keys),
            midpoint,
            len(inchi_keys) - midpoint,
            exc,
        )
        results = lookup_inchi_keys_with_fallback(
            session,
            resolver_url,
            inchi_keys[:midpoint],
            timeout,
            stats,
        )
        results.update(
            lookup_inchi_keys_with_fallback(
                session,
                resolver_url,
                inchi_keys[midpoint:],
                timeout,
                stats,
            )
        )
        return results


def _json_model_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _predict_batch_once(
    session: requests.Session,
    predict_url: str,
    model_id: str,
    smiles_values: list[str],
    timeout: float,
) -> int:
    response = session.post(
        predict_url,
        json={"model_id": _json_model_id(model_id), "smiles": smiles_values},
        headers={"accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        return len(smiles_values)

    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(f"Prediction API returned error: {payload.get('error')}")

    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        failures = sum(1 for item in results if isinstance(item, dict) and item.get("error"))
        return len(results) - failures

    return len(smiles_values)


def _http_status_from_exception(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _is_gateway_or_timeout_prediction_failure(exc: BaseException) -> bool:
    if isinstance(exc, Timeout):
        return True

    status_code = _http_status_from_exception(exc)
    return status_code in {502, 503, 504}


def _should_split_prediction_failure(
    exc: BaseException,
    batch_size: int,
    gateway_split_threshold: int,
) -> bool:
    if batch_size <= 1:
        return False

    if _is_gateway_or_timeout_prediction_failure(exc):
        return batch_size > gateway_split_threshold

    return True


def predict_smiles_with_fallback(
    session: requests.Session,
    predict_url: str,
    model_id: str,
    smiles_values: list[str],
    timeout: float,
    gateway_split_threshold: int,
    stats: CleanupStats,
) -> int:
    if not smiles_values:
        return 0

    stats.prediction_batches += 1
    stats.prediction_requests += len(smiles_values)
    try:
        success_count = _predict_batch_once(session, predict_url, model_id, smiles_values, timeout)
        stats.recomputed_predictions += success_count
        stats.prediction_failures += max(0, len(smiles_values) - success_count)
        return success_count
    except (RequestException, ValueError) as exc:
        if not _should_split_prediction_failure(exc, len(smiles_values), gateway_split_threshold):
            stats.prediction_failures += len(smiles_values)
            LOGGER.warning(
                "Prediction recompute failed without further splitting; model_id=%s batch_size=%s "
                "gateway_or_timeout=%s threshold=%s first_smiles=%s error=%s",
                model_id,
                len(smiles_values),
                _is_gateway_or_timeout_prediction_failure(exc),
                gateway_split_threshold,
                smiles_values[0] if smiles_values else None,
                exc,
            )
            return 0

        midpoint = len(smiles_values) // 2
        LOGGER.warning(
            "Prediction batch failed; splitting model_id=%s batch_size=%s left=%s right=%s error=%s",
            model_id,
            len(smiles_values),
            midpoint,
            len(smiles_values) - midpoint,
            exc,
        )
        return predict_smiles_with_fallback(
            session,
            predict_url,
            model_id,
            smiles_values[:midpoint],
            timeout,
            gateway_split_threshold,
            stats,
        ) + predict_smiles_with_fallback(
            session,
            predict_url,
            model_id,
            smiles_values[midpoint:],
            timeout,
            gateway_split_threshold,
            stats,
        )


def resolve_recompute_smiles(
    session: requests.Session,
    candidates: list[MismatchCandidate],
    args: argparse.Namespace,
    stats: CleanupStats,
) -> dict[str, str]:
    unique_inchi_keys = _ordered_unique([candidate.key_inchi_key for candidate in candidates])
    resolved_by_inchi_key = lookup_inchi_keys_with_fallback(
        session,
        args.resolver_url,
        unique_inchi_keys,
        args.resolver_timeout,
        stats,
    )
    stats.resolver_hits += len(resolved_by_inchi_key)

    smiles_by_key = {}
    for candidate in candidates:
        resolved_chemical = resolved_by_inchi_key.get(candidate.key_inchi_key)
        if not resolved_chemical:
            stats.skipped_without_resolver_match += 1
            continue

        smiles = _clean_text(resolved_chemical.get("smiles")) or _clean_text(
            resolved_chemical.get("canonicalSmiles")
        )
        if smiles is None:
            stats.skipped_without_resolver_match += 1
            continue

        smiles_by_key[candidate.key] = smiles

    LOGGER.info(
        "Resolved recompute SMILES: candidates=%s unique_inchi_keys=%s resolved_inchi_keys=%s resolved_cache_keys=%s",
        len(candidates),
        len(unique_inchi_keys),
        len(resolved_by_inchi_key),
        len(smiles_by_key),
    )
    return smiles_by_key


def recompute_deleted_candidates(
    session: requests.Session,
    candidates: list[MismatchCandidate],
    smiles_by_key: dict[str, str],
    args: argparse.Namespace,
    stats: CleanupStats,
) -> None:
    smiles_by_model_id: dict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        smiles = smiles_by_key.get(candidate.key)
        if smiles:
            smiles_by_model_id[candidate.model_id].append(smiles)

    if not smiles_by_model_id:
        LOGGER.info("No prediction recompute requests to send for this deleted batch")
        return

    for model_id, smiles_values in sorted(smiles_by_model_id.items()):
        unique_smiles = _ordered_unique(smiles_values)
        LOGGER.info(
            "Recomputing predictions for model_id=%s unique_smiles=%s prediction_batch_size=%s",
            model_id,
            len(unique_smiles),
            args.prediction_batch_size,
        )
        batch_number = 0
        for smiles_batch in _chunk_sequence(unique_smiles, args.prediction_batch_size):
            batch_number += 1
            failures_before = stats.prediction_failures
            success_count = predict_smiles_with_fallback(
                session,
                args.predict_url,
                model_id,
                smiles_batch,
                args.predict_timeout,
                args.prediction_gateway_split_threshold,
                stats,
            )
            LOGGER.info(
                "Prediction recompute batch complete: model_id=%s batch=%s requested=%s "
                "recomputed=%s failures=%s cumulative_recomputed=%s",
                model_id,
                batch_number,
                len(smiles_batch),
                success_count,
                stats.prediction_failures - failures_before,
                stats.recomputed_predictions,
            )


def log_mismatch_samples(
    candidates: list[MismatchCandidate],
    args: argparse.Namespace,
    logged_samples: int,
) -> int:
    if args.sample_size <= 0 or logged_samples >= args.sample_size:
        return logged_samples

    remaining_sample_slots = args.sample_size - logged_samples
    for candidate in candidates[:remaining_sample_slots]:
        LOGGER.info(
            "Mismatch sample: key=%s key_inchi=%s chemical_inchi=%s chemical_name=%s chemical_smiles=%s",
            candidate.key,
            candidate.key_inchi_key,
            candidate.chemical_inchi_key,
            candidate.chemical_name,
            candidate.chemical_smiles,
        )
        logged_samples += 1

    return logged_samples


def scan_mismatches_to_files(collection, args: argparse.Namespace, stats: CleanupStats) -> int:
    reset_output_file(args.mismatch_keys_file)
    reset_output_file(args.report_jsonl)

    LOGGER.info(
        "Starting scan phase: keys_file=%s report_jsonl=%s progress_every=%s",
        args.mismatch_keys_file,
        args.report_jsonl,
        args.progress_every,
    )

    logged_samples = 0
    for candidate_batch in iter_mismatch_batches(collection, args, stats):
        stats.candidates += len(candidate_batch)
        stats.planned_deletes += len(candidate_batch)
        write_mismatch_keys(args.mismatch_keys_file, candidate_batch)
        write_report_rows(args.report_jsonl, candidate_batch)
        logged_samples = log_mismatch_samples(candidate_batch, args, logged_samples)

    file_records = count_file_records(args.mismatch_keys_file)
    LOGGER.info(
        "Scan phase complete: scanned_documents=%s mismatches=%s keys_file=%s file_records=%s",
        stats.scanned_documents,
        stats.candidates,
        args.mismatch_keys_file,
        file_records,
    )
    return file_records


def fetch_live_mismatch_candidates_for_keys(
    collection,
    keys: list[str],
    args: argparse.Namespace,
) -> list[MismatchCandidate]:
    if not keys:
        return []

    model_ids = {str(model_id) for model_id in args.model_id}
    docs_by_key = {}
    cursor = collection.find(
        {"key": {"$in": list(dict.fromkeys(keys))}},
        projection=mismatch_projection(),
    ).hint("key_idx")
    try:
        for doc in cursor:
            key = doc.get("key")
            if isinstance(key, str):
                docs_by_key[key] = doc
    finally:
        cursor.close()

    candidates = []
    seen_keys = set()
    for key in keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        doc = docs_by_key.get(key)
        if doc is None:
            continue

        candidate = mismatch_from_doc(
            doc,
            model_ids=model_ids,
            only_without_standardized=args.only_without_standardized,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def process_repair_batch(
    collection,
    session: requests.Session,
    keys: list[str],
    batch_number: int,
    total_batches: int,
    args: argparse.Namespace,
    stats: CleanupStats,
) -> None:
    LOGGER.info(
        "Repair batch start: batch=%s/%s keys=%s repair_mode=%s",
        batch_number,
        total_batches,
        len(keys),
        args.repair_mode,
    )

    candidates = fetch_live_mismatch_candidates_for_keys(collection, keys, args)
    if not candidates:
        LOGGER.info(
            "Repair batch complete: batch=%s/%s keys=%s live_mismatches=0 deleted=0 "
            "resolved_cache_keys=0 recomputed=0 prediction_failures=0 skipped_live_mismatch=%s",
            batch_number,
            total_batches,
            len(keys),
            len(keys),
        )
        return

    resolver_batches_before = stats.resolver_batches
    resolver_hits_before = stats.resolver_hits
    prediction_batches_before = stats.prediction_batches
    prediction_requests_before = stats.prediction_requests
    recomputed_before = stats.recomputed_predictions
    prediction_failures_before = stats.prediction_failures
    skipped_resolver_before = stats.skipped_without_resolver_match
    smiles_by_key = {}
    if args.repair_mode == "recompute":
        smiles_by_key = resolve_recompute_smiles(session, candidates, args, stats)

    result = collection.bulk_write(
        [DeleteOne(build_delete_filter(candidate)) for candidate in candidates],
        ordered=False,
    )
    stats.deleted_documents += result.deleted_count

    if args.repair_mode != "recompute":
        LOGGER.info(
            "Repair batch complete: batch=%s/%s keys=%s live_mismatches=%s deleted=%s "
            "resolved_cache_keys=0 recomputed=0 prediction_failures=0 skipped_live_mismatch=%s",
            batch_number,
            total_batches,
            len(keys),
            len(candidates),
            result.deleted_count,
            len(keys) - len(candidates),
        )
        return

    recompute_deleted_candidates(session, candidates, smiles_by_key, args, stats)
    LOGGER.info(
        "Repair batch complete: batch=%s/%s keys=%s live_mismatches=%s deleted=%s "
        "resolved_cache_keys=%s skipped_live_mismatch=%s "
        "resolver_batches=%s resolver_hits=%s prediction_batches=%s prediction_requests=%s "
        "recomputed=%s prediction_failures=%s skipped_without_resolver_match=%s",
        batch_number,
        total_batches,
        len(keys),
        len(candidates),
        result.deleted_count,
        len(smiles_by_key),
        len(keys) - len(candidates),
        stats.resolver_batches - resolver_batches_before,
        stats.resolver_hits - resolver_hits_before,
        stats.prediction_batches - prediction_batches_before,
        stats.prediction_requests - prediction_requests_before,
        stats.recomputed_predictions - recomputed_before,
        stats.prediction_failures - prediction_failures_before,
        stats.skipped_without_resolver_match - skipped_resolver_before,
    )


def repair_from_keys_file(
    collection,
    session: requests.Session,
    args: argparse.Namespace,
    stats: CleanupStats,
    file_records: int,
) -> None:
    if file_records == 0:
        LOGGER.info("Repair phase skipped: mismatch keys file is empty")
        return

    total_batches = (file_records + args.batch_size - 1) // args.batch_size
    LOGGER.info(
        "Starting repair phase from keys file: keys_file=%s file_records=%s batch_size=%s total_batches=%s",
        args.mismatch_keys_file,
        file_records,
        args.batch_size,
        total_batches,
    )
    for batch_number, key_batch in enumerate(
        iter_key_file_batches(args.mismatch_keys_file, args.batch_size),
        start=1,
    ):
        process_repair_batch(
            collection,
            session,
            key_batch,
            batch_number,
            total_batches,
            args,
            stats,
        )


def cleanup(collection, args: argparse.Namespace) -> CleanupStats:
    stats = CleanupStats()
    session = requests.Session()

    file_records = scan_mismatches_to_files(collection, args, stats)

    if not args.write:
        action = "delete and recompute" if args.repair_mode == "recompute" else "delete"
        LOGGER.info(
            "Dry-run complete after scan: would %s %s mismatched cache document(s) from keys_file=%s",
            action,
            file_records,
            args.mismatch_keys_file,
        )
        return stats

    repair_from_keys_file(collection, session, args, stats, file_records)

    return stats


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mongo_cursor_batch_size <= 0:
        raise ValueError("--mongo-cursor-batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.mongo_socket_timeout_ms is not None and args.mongo_socket_timeout_ms < 0:
        raise ValueError("--mongo-socket-timeout-ms cannot be negative")
    if args.mongo_cursor_retries < 0:
        raise ValueError("--mongo-cursor-retries cannot be negative")
    if args.mongo_cursor_retry_sleep < 0:
        raise ValueError("--mongo-cursor-retry-sleep cannot be negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")
    if args.sample_size < 0:
        raise ValueError("--sample-size cannot be negative")
    if args.prediction_batch_size <= 0:
        raise ValueError("--prediction-batch-size must be positive")
    if args.prediction_gateway_split_threshold <= 0:
        raise ValueError("--prediction-gateway-split-threshold must be positive")
    if args.resolver_timeout is not None and args.resolver_timeout <= 0:
        raise ValueError("--resolver-timeout must be positive")
    if args.predict_timeout is not None and args.predict_timeout <= 0:
        raise ValueError("--predict-timeout must be positive")
    if args.report_jsonl is not None and args.report_jsonl.resolve() == args.mismatch_keys_file.resolve():
        raise ValueError("--report-jsonl and --mismatch-keys-file must point to different files")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    client = None
    try:
        validate_args(args)
        if not args.skip_env_file:
            load_env(args.env_file)
        resolve_args_from_env(args)

        client = build_mongo_client(args)
        client.admin.command("ping")
        collection = client[args.database][args.collection]
        LOGGER.info(
            "Starting prediction cache identity cleanup: database=%s collection=%s write=%s "
            "repair_mode=%s keys=%s model_ids=%s only_without_standardized=%s mismatch_keys_file=%s",
            args.database,
            args.collection,
            args.write,
            args.repair_mode,
            len(args.key),
            args.model_id,
            args.only_without_standardized,
            args.mismatch_keys_file,
        )
        stats = cleanup(collection, args)
    except (PyMongoError, ValueError):
        LOGGER.exception("Cleanup failed")
        return 1
    finally:
        if client is not None:
            client.close()

    LOGGER.info(
        "Cleanup complete: scanned_documents=%s candidates=%s planned_deletes=%s "
        "deleted_documents=%s resolver_batches=%s resolver_hits=%s prediction_batches=%s "
        "prediction_requests=%s recomputed_predictions=%s prediction_failures=%s "
        "skipped_missing_key_inchi=%s skipped_missing_chemical_inchi=%s skipped_without_resolver_match=%s",
        stats.scanned_documents,
        stats.candidates,
        stats.planned_deletes,
        stats.deleted_documents,
        stats.resolver_batches,
        stats.resolver_hits,
        stats.prediction_batches,
        stats.prediction_requests,
        stats.recomputed_predictions,
        stats.prediction_failures,
        stats.skipped_missing_key_inchi,
        stats.skipped_missing_chemical_inchi,
        stats.skipped_without_resolver_match,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
