#!/usr/bin/env python3
"""
Backfill null prediction.chemicalIdentifiers fields in predictor_models_cache.

The script reads cached prediction documents whose chemicalIdentifiers have null
identifier fields, resolves the cached SMILES values in batches, and writes the
resolved values back into Mongo.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from requests import RequestException


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_RESOLVER_LOOKUP_API = "https://cim-dev.sciencedataexperts.com/api/resolver/lookup"
DEFAULT_BATCH_SIZE = 100
IDENTIFIER_FIELDS = ("cid", "sid", "casrn", "name", "inchi", "inchiKey")
LOGGER = logging.getLogger("chemical_identifier_backfill")


@dataclass(frozen=True)
class CacheCandidate:
    doc_id: Any
    key: str | None
    smiles: str
    missing_fields: tuple[str, ...]


@dataclass
class BackfillStats:
    scanned_documents: int = 0
    candidates: int = 0
    resolver_batches: int = 0
    resolver_hits: int = 0
    planned_updates: int = 0
    matched_updates: int = 0
    modified_updates: int = 0
    skipped_without_smiles: int = 0
    skipped_without_resolver_match: int = 0
    skipped_without_update_fields: int = 0


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
            "Resolve null prediction.chemicalIdentifiers values in the "
            "predictor_models_cache Mongo collection."
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
    parser.add_argument(
        "--database",
        default=None,
        help="Mongo database name.",
    )
    parser.add_argument(
        "--collection",
        default="predictor_models_cache",
        help="Mongo collection name.",
    )
    parser.add_argument(
        "--resolver-url",
        default=None,
        help="CIM resolver lookup endpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of Mongo cache records to process per resolver batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of candidate Mongo records to process.",
    )
    parser.add_argument(
        "--match-mode",
        choices=("all-null", "any-null"),
        default="all-null",
        help="Select records where all identifier fields are null, or where any one is null.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Resolver request timeout in seconds.",
    )
    parser.add_argument(
        "--query-mode",
        choices=("client", "server"),
        default="client",
        help=(
            "client scans Mongo with a lightweight projection and filters in Python; "
            "server uses a Mongo predicate for the null fields."
        ),
    )
    parser.add_argument(
        "--mongo-cursor-batch-size",
        type=int,
        default=1000,
        help="Number of Mongo documents fetched per cursor batch.",
    )
    parser.add_argument(
        "--mongo-socket-timeout-ms",
        type=int,
        default=None,
        help="Override Mongo socketTimeoutMS for this backfill run. Defaults to BACKFILL_MONGO_SOCKET_TIMEOUT_MS or 300000.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Log scan progress after this many Mongo documents. Use 0 to disable.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist updates to Mongo. Default is dry-run.",
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
    if args.timeout is None:
        args.timeout = float(os.getenv("RESOLVER_LOOKUP_TIMEOUT", "30"))


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_mongo_client(args: argparse.Namespace) -> MongoClient:
    timeout_ms = _env_int("MONGO_SERVER_SELECTION_TIMEOUT_MS", 5000)
    socket_timeout_ms = args.mongo_socket_timeout_ms
    if socket_timeout_ms is None:
        socket_timeout_ms = _env_int("BACKFILL_MONGO_SOCKET_TIMEOUT_MS", 300000)
    app_name = os.getenv("MONGO_APP_NAME", "predictor_models_cache_identifier_backfill")

    if args.mongo_uri:
        return MongoClient(
            args.mongo_uri,
            appname=app_name,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=_env_int("MONGO_CONNECT_TIMEOUT_MS", timeout_ms),
            socketTimeoutMS=socket_timeout_ms,
        )

    client_kwargs = {
        "host": os.getenv("MONGO_HOST", "localhost"),
        "port": _env_int("MONGO_PORT", 27017),
        "authSource": os.getenv("MONGO_AUTH_SOURCE", "admin"),
        "appname": app_name,
        "serverSelectionTimeoutMS": timeout_ms,
        "connectTimeoutMS": _env_int("MONGO_CONNECT_TIMEOUT_MS", timeout_ms),
        "socketTimeoutMS": socket_timeout_ms,
    }

    username = os.getenv("MONGO_USER", "root")
    password = os.getenv("MONGO_PASSWORD")
    if username:
        client_kwargs["username"] = username
    if password:
        client_kwargs["password"] = password

    return MongoClient(**client_kwargs)


def _smiles_exists_clause() -> dict[str, Any]:
    return {
        "$or": [
            {
                "prediction.chemicalIdentifiers.smiles": {
                    "$type": "string",
                    "$nin": ["", "N/A"],
                },
            },
            {
                "prediction.chemicalIdentifiers.canonicalSmiles": {
                    "$type": "string",
                    "$nin": ["", "N/A"],
                },
            },
        ],
    }


def build_candidate_query(match_mode: str) -> dict[str, Any]:
    null_field_conditions = [
        {
            f"prediction.chemicalIdentifiers.{field}": {
                "$in": [None, "", "N/A"],
            }
        }
        for field in IDENTIFIER_FIELDS
    ]
    null_match_clause = (
        {"$or": null_field_conditions}
        if match_mode == "any-null"
        else {"$and": null_field_conditions}
    )

    return {
        "$and": [
            {"prediction.chemicalIdentifiers": {"$type": "object"}},
            _smiles_exists_clause(),
            null_match_clause,
        ],
    }


def build_scan_query(query_mode: str, match_mode: str) -> dict[str, Any]:
    if query_mode == "server":
        return build_candidate_query(match_mode)
    return {}


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.upper() == "N/A":
        return None
    return text


def _is_missing_identifier_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or text.upper() == "N/A"
    return False


def _chemical_matches_mode(chemical: dict[str, Any], match_mode: str) -> bool:
    missing_flags = [
        _is_missing_identifier_value(chemical.get(field))
        for field in IDENTIFIER_FIELDS
    ]
    if match_mode == "any-null":
        return any(missing_flags)
    return all(missing_flags)


def _candidate_from_doc(doc: dict[str, Any], match_mode: str = "all-null") -> CacheCandidate | None:
    prediction = doc.get("prediction")
    if not isinstance(prediction, dict):
        return None

    chemical = prediction.get("chemicalIdentifiers")
    if not isinstance(chemical, dict):
        return None

    if not _chemical_matches_mode(chemical, match_mode):
        return None

    smiles = _clean_text(chemical.get("smiles")) or _clean_text(chemical.get("canonicalSmiles"))
    if smiles is None:
        return None

    missing_fields = tuple(
        field
        for field in IDENTIFIER_FIELDS
        if _is_missing_identifier_value(chemical.get(field))
    )
    if not missing_fields:
        return None

    return CacheCandidate(
        doc_id=doc["_id"],
        key=doc.get("key"),
        smiles=smiles,
        missing_fields=missing_fields,
    )


def iter_candidate_batches(
    collection,
    *,
    batch_size: int,
    limit: int | None,
    match_mode: str,
    query_mode: str,
    cursor_batch_size: int,
    progress_every: int,
    stats: BackfillStats,
):
    query = build_scan_query(query_mode, match_mode)
    projection = {
        "key": 1,
        "prediction.chemicalIdentifiers.smiles": 1,
        "prediction.chemicalIdentifiers.canonicalSmiles": 1,
    }
    for field in IDENTIFIER_FIELDS:
        projection[f"prediction.chemicalIdentifiers.{field}"] = 1
    cursor = collection.find(
        query,
        projection=projection,
        no_cursor_timeout=True,
    ).batch_size(cursor_batch_size)
    if query_mode == "server" and limit is not None:
        cursor = cursor.limit(limit)

    batch = []
    try:
        for doc in cursor:
            stats.scanned_documents += 1
            if progress_every > 0 and stats.scanned_documents % progress_every == 0:
                LOGGER.info(
                    "Scan progress: scanned_documents=%s candidates=%s",
                    stats.scanned_documents,
                    stats.candidates + len(batch),
                )

            candidate = _candidate_from_doc(doc, match_mode)
            if candidate is None:
                continue

            batch.append(candidate)
            if limit is not None and query_mode == "client" and stats.candidates + len(batch) >= limit:
                yield batch
                return

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch
    finally:
        cursor.close()


def build_resolver_payload(smiles_values: list[str]) -> dict[str, Any]:
    return {
        "fuzzy": "Not",
        "ids": [str(smiles) for smiles in smiles_values],
        "idsType": "SMILES",
        "mol": False,
        "filters": {},
        "format": "UNKNOWN",
    }


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
        for field in ("chemId", "cid", "sid", "casrn", "name", "smiles", "canonicalSmiles")
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


def _match_item_index(raw_item: Any, input_size: int) -> int | None:
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


def _match_item_smiles(raw_item: Any, smiles_values: list[str], used_indexes: set[int]) -> int | None:
    if not isinstance(raw_item, dict):
        return None

    for key in ("smiles", "canonicalSmiles", "chemId", "query", "input"):
        value = _clean_text(raw_item.get(key))
        if value is None:
            continue
        try:
            index = smiles_values.index(value)
        except ValueError:
            continue
        if index not in used_indexes:
            return index

    return None


def _match_resolver_item(
    raw_item: Any,
    chemical: dict[str, Any] | None,
    smiles_values: list[str],
    used_indexes: set[int],
) -> int | None:
    item_index = _match_item_index(raw_item, len(smiles_values))
    if item_index is not None and item_index not in used_indexes:
        return item_index

    item_smiles_index = _match_item_smiles(raw_item, smiles_values, used_indexes)
    if item_smiles_index is not None:
        return item_smiles_index

    return _match_item_smiles(chemical, smiles_values, used_indexes)


def parse_resolver_payload(payload: Any, smiles_values: list[str]) -> dict[str, dict[str, Any]]:
    items = _extract_resolver_items(payload)
    if not isinstance(items, list) or not items:
        raise ValueError(f"Unexpected resolver response: {payload!r}")

    results = {}
    used_indexes: set[int] = set()
    use_positional_fallback = len(items) == len(smiles_values)

    for item_position, item in enumerate(items):
        chemical = _extract_chemical(item)
        if not isinstance(chemical, dict):
            continue

        match_index = _match_resolver_item(item, chemical, smiles_values, used_indexes)
        if match_index is None and use_positional_fallback and item_position not in used_indexes:
            match_index = item_position

        if match_index is None or match_index in used_indexes:
            continue

        used_indexes.add(match_index)
        results[smiles_values[match_index]] = chemical

    return results


def lookup_resolver_batch(
    session: requests.Session,
    resolver_url: str,
    smiles_values: list[str],
    timeout: float,
) -> dict[str, dict[str, Any]]:
    response = session.post(
        resolver_url,
        json=build_resolver_payload(smiles_values),
        headers={"accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_resolver_payload(response.json(), smiles_values)


def lookup_resolver_batch_with_fallback(
    session: requests.Session,
    resolver_url: str,
    smiles_values: list[str],
    timeout: float,
    stats: BackfillStats,
) -> dict[str, dict[str, Any]]:
    if not smiles_values:
        return {}

    stats.resolver_batches += 1
    try:
        return lookup_resolver_batch(session, resolver_url, smiles_values, timeout)
    except (RequestException, ValueError) as exc:
        if len(smiles_values) == 1:
            LOGGER.warning(
                "Resolver lookup failed for one SMILES; skipping smiles=%s error=%s",
                smiles_values[0],
                exc,
            )
            return {}

        midpoint = len(smiles_values) // 2
        LOGGER.warning(
            "Resolver batch failed; splitting batch_size=%s left=%s right=%s error=%s",
            len(smiles_values),
            midpoint,
            len(smiles_values) - midpoint,
            exc,
        )

        results = lookup_resolver_batch_with_fallback(
            session,
            resolver_url,
            smiles_values[:midpoint],
            timeout,
            stats,
        )
        results.update(
            lookup_resolver_batch_with_fallback(
                session,
                resolver_url,
                smiles_values[midpoint:],
                timeout,
                stats,
            )
        )
        return results


def _resolved_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text or text.upper() == "N/A":
            return None
        return text
    return value


def build_update_fields(
    resolved_chemical: dict[str, Any],
    missing_fields: tuple[str, ...] = IDENTIFIER_FIELDS,
) -> dict[str, Any]:
    update_fields = {}
    for field in missing_fields:
        value = _resolved_value(resolved_chemical.get(field))
        if value is not None:
            update_fields[f"prediction.chemicalIdentifiers.{field}"] = value
    return update_fields


def build_update_filter(doc_id: Any, missing_fields: tuple[str, ...]) -> dict[str, Any]:
    missing_field_filters = []
    for field in missing_fields:
        path = f"prediction.chemicalIdentifiers.{field}"
        missing_field_filters.append({"$or": [{path: None}, {path: ""}, {path: "N/A"}]})

    if not missing_field_filters:
        return {"_id": doc_id}

    return {"_id": doc_id, "$and": missing_field_filters}


def _ordered_unique(values: list[str]) -> list[str]:
    unique_values = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def process_batch(
    collection,
    session: requests.Session,
    candidates: list[CacheCandidate],
    *,
    resolver_url: str,
    timeout: float,
    write: bool,
    stats: BackfillStats,
) -> None:
    docs_by_smiles = defaultdict(list)
    for candidate in candidates:
        docs_by_smiles[candidate.smiles].append(candidate)

    smiles_values = _ordered_unique([candidate.smiles for candidate in candidates])

    resolved_by_smiles = lookup_resolver_batch_with_fallback(
        session,
        resolver_url,
        smiles_values,
        timeout,
        stats,
    )
    stats.resolver_hits += len(resolved_by_smiles)

    operations = []
    for smiles, smile_candidates in docs_by_smiles.items():
        resolved_chemical = resolved_by_smiles.get(smiles)
        if not resolved_chemical:
            stats.skipped_without_resolver_match += len(smile_candidates)
            continue

        for candidate in smile_candidates:
            update_fields = build_update_fields(resolved_chemical, candidate.missing_fields)
            if not update_fields:
                stats.skipped_without_update_fields += 1
                continue

            operations.append(
                UpdateOne(
                    build_update_filter(candidate.doc_id, candidate.missing_fields),
                    {"$set": update_fields},
                )
            )

    stats.planned_updates += len(operations)
    if not operations:
        return

    if not write:
        LOGGER.info("Dry-run: would update %s document(s) in this batch", len(operations))
        return

    result = collection.bulk_write(operations, ordered=False)
    stats.matched_updates += result.matched_count
    stats.modified_updates += result.modified_count
    LOGGER.info(
        "Updated batch: matched=%s modified=%s",
        result.matched_count,
        result.modified_count,
    )


def backfill(collection, args: argparse.Namespace) -> BackfillStats:
    stats = BackfillStats()
    session = requests.Session()

    for candidate_batch in iter_candidate_batches(
        collection,
        batch_size=args.batch_size,
        limit=args.limit,
        match_mode=args.match_mode,
        query_mode=args.query_mode,
        cursor_batch_size=args.mongo_cursor_batch_size,
        progress_every=args.progress_every,
        stats=stats,
    ):
        stats.candidates += len(candidate_batch)
        LOGGER.info(
            "Processing candidate batch: records=%s scanned_documents=%s first_key=%s",
            len(candidate_batch),
            stats.scanned_documents,
            candidate_batch[0].key,
        )
        process_batch(
            collection,
            session,
            candidate_batch,
            resolver_url=args.resolver_url,
            timeout=args.timeout,
            write=args.write,
            stats=stats,
        )

    return stats


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    if args.batch_size <= 0:
        LOGGER.error("--batch-size must be positive")
        return 2
    if args.limit is not None and args.limit <= 0:
        LOGGER.error("--limit must be positive when provided")
        return 2
    if args.mongo_cursor_batch_size <= 0:
        LOGGER.error("--mongo-cursor-batch-size must be positive")
        return 2
    if args.mongo_socket_timeout_ms is not None and args.mongo_socket_timeout_ms <= 0:
        LOGGER.error("--mongo-socket-timeout-ms must be positive when provided")
        return 2
    if args.progress_every < 0:
        LOGGER.error("--progress-every cannot be negative")
        return 2

    if not args.skip_env_file:
        load_env(args.env_file)
    resolve_args_from_env(args)

    client = build_mongo_client(args)
    try:
        client.admin.command("ping")
        collection = client[args.database][args.collection]
        LOGGER.info(
            "Starting chemicalIdentifiers backfill: database=%s collection=%s "
            "batch_size=%s cursor_batch_size=%s mode=%s query_mode=%s write=%s resolver=%s",
            args.database,
            args.collection,
            args.batch_size,
            args.mongo_cursor_batch_size,
            args.match_mode,
            args.query_mode,
            args.write,
            args.resolver_url,
        )
        stats = backfill(collection, args)
    except (PyMongoError, RequestException, ValueError):
        LOGGER.exception("Backfill failed")
        return 1
    finally:
        client.close()

    LOGGER.info(
        "Backfill complete: scanned_documents=%s candidates=%s resolver_batches=%s "
        "resolver_hits=%s planned_updates=%s matched_updates=%s modified_updates=%s skipped_without_smiles=%s "
        "skipped_without_resolver_match=%s skipped_without_update_fields=%s",
        stats.scanned_documents,
        stats.candidates,
        stats.resolver_batches,
        stats.resolver_hits,
        stats.planned_updates,
        stats.matched_updates,
        stats.modified_updates,
        stats.skipped_without_smiles,
        stats.skipped_without_resolver_match,
        stats.skipped_without_update_fields,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
