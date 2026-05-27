#!/usr/bin/env python3
"""
Find or delete predictor_models_cache records whose cache-key InChIKey does not
match prediction.chemicalIdentifiers.inchiKey.

These records are unsafe because the cache key belongs to one chemical while the
stored prediction payload belongs to another. Deleting them is usually the right
repair: the next API request recomputes the prediction and overwrites the cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pymongo import DeleteOne, MongoClient
from pymongo.errors import PyMongoError


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


DEFAULT_BATCH_SIZE = 1000
DEFAULT_CURSOR_BATCH_SIZE = 5000
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
    skipped_missing_key_inchi: int = 0
    skipped_missing_chemical_inchi: int = 0


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
        help="Number of delete operations per Mongo bulk_write batch.",
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
        default=100000,
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


def build_resume_scan_query(query: dict[str, Any], last_seen_id: Any | None) -> dict[str, Any]:
    if last_seen_id is None:
        return query

    resume_clause = {"_id": {"$gt": last_seen_id}}
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
    projection = {
        "key": 1,
        "prediction.chemicalIdentifiers.inchiKey": 1,
        "prediction.chemicalIdentifiers.smiles": 1,
        "prediction.chemicalIdentifiers.canonicalSmiles": 1,
        "prediction.chemicalIdentifiers.name": 1,
        "prediction.standardizedChemical": 1,
    }
    model_ids = {str(model_id) for model_id in args.model_id}

    candidate_buffer = []
    last_seen_id = None
    selected_candidates = 0
    limit_reached = False

    while not limit_reached:
        retries_remaining = args.mongo_cursor_retries

        while True:
            cursor = collection.find(
                build_resume_scan_query(query, last_seen_id),
                projection=projection,
            ).sort("_id", 1).limit(args.mongo_cursor_batch_size).batch_size(args.mongo_cursor_batch_size)
            page_doc_count = 0
            try:
                for doc in cursor:
                    page_doc_count += 1
                    last_seen_id = doc["_id"]
                    stats.scanned_documents += 1
                    if args.progress_every > 0 and stats.scanned_documents % args.progress_every == 0:
                        LOGGER.info(
                            "Scan progress: scanned_documents=%s candidates=%s last_seen_id=%s",
                            stats.scanned_documents,
                            selected_candidates,
                            last_seen_id,
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
                    "Mongo cursor read failed; retrying scan page from last_seen_id=%s "
                    "retries_remaining=%s error=%s",
                    last_seen_id,
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

    with report_path.open("a", encoding="utf-8") as report_file:
        for candidate in candidates:
            row = asdict(candidate)
            row["doc_id"] = str(candidate.doc_id)
            report_file.write(json.dumps(row, sort_keys=True) + "\n")


def process_batch(collection, candidates: list[MismatchCandidate], args: argparse.Namespace, stats: CleanupStats) -> None:
    stats.candidates += len(candidates)
    stats.planned_deletes += len(candidates)

    write_report_rows(args.report_jsonl, candidates)

    if args.sample_size > 0 and stats.candidates <= args.sample_size:
        remaining_sample_slots = args.sample_size - (stats.candidates - len(candidates))
        for candidate in candidates[:remaining_sample_slots]:
            LOGGER.info(
                "Mismatch sample: key=%s key_inchi=%s chemical_inchi=%s chemical_name=%s chemical_smiles=%s",
                candidate.key,
                candidate.key_inchi_key,
                candidate.chemical_inchi_key,
                candidate.chemical_name,
                candidate.chemical_smiles,
            )

    if not args.write:
        LOGGER.info("Dry-run: would delete %s mismatched cache document(s) in this batch", len(candidates))
        return

    result = collection.bulk_write(
        [DeleteOne(build_delete_filter(candidate)) for candidate in candidates],
        ordered=False,
    )
    stats.deleted_documents += result.deleted_count
    LOGGER.info("Deleted batch: deleted=%s", result.deleted_count)


def cleanup(collection, args: argparse.Namespace) -> CleanupStats:
    stats = CleanupStats()

    if args.report_jsonl is not None and args.report_jsonl.exists():
        args.report_jsonl.write_text("", encoding="utf-8")

    for candidate_batch in iter_mismatch_batches(collection, args, stats):
        process_batch(collection, candidate_batch, args, stats)

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
            "keys=%s model_ids=%s only_without_standardized=%s",
            args.database,
            args.collection,
            args.write,
            len(args.key),
            args.model_id,
            args.only_without_standardized,
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
        "deleted_documents=%s skipped_missing_key_inchi=%s skipped_missing_chemical_inchi=%s",
        stats.scanned_documents,
        stats.candidates,
        stats.planned_deletes,
        stats.deleted_documents,
        stats.skipped_missing_key_inchi,
        stats.skipped_missing_chemical_inchi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
