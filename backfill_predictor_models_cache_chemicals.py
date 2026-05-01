#!/usr/bin/env python3
"""Backfill chemical identifiers in the predictor_models Mongo response cache.

The predictor_models cache is stored in Mongo as:

    {"key": "...", "prediction": {"chemicalIdentifiers": {...}, ...}}

API responses expose ``chemicalIdentifiers`` as ``chemical``. This script
supports both field names so it can also repair legacy or already-formatted
documents.

Examples:
    # Inspect changes without writing.
    python pf_python_model_building/backfill_predictor_models_cache_chemicals.py --limit 10

    # Repair one cached prediction.
    python pf_python_model_building/backfill_predictor_models_cache_chemicals.py \
        --key ZWBAMYVPMDSJGQ-UHFFFAOYSA-N-1065 --apply

    # Repair all matching documents.
    python pf_python_model_building/backfill_predictor_models_cache_chemicals.py --apply
"""

from __future__ import annotations

import argparse
import copy
import json as json_module
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


CHEMICAL_BLOCKS = ("chemicalIdentifiers", "chemical")
STANDARDIZED_BLOCK = "standardizedChemical"
RESOLVER_SKIP_FIELDS = {"id", "imageSrc"}
BLANK_STRINGS = {"", "N/A", "n/a", "null", "None", "none"}
DEFAULT_RESOLVER_LOOKUP_API = "https://cim-dev.sciencedataexperts.com/api/resolver/lookup"
SCRIPT_ENV_DEFAULTS = {
    "MONGO_HOST": "192.168.1.3",
    "MONGO_PORT": "27017",
    "MONGO_USER": "root",
    "MONGO_PASSWORD": "qqq123",
    "MONGO_DATABASE": "predictor",
    "MONGO_CACHE_ENABLED": "true",
}


class SimpleHttpResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text[:500]}")

    def json(self):
        return json_module.loads(self.text)


class UrllibSession:
    def get(self, url, *, params=None, headers=None, timeout=None):
        query = urlencode(params or {})
        request_url = f"{url}?{query}" if query else url
        request = Request(request_url, headers=headers or {}, method="GET")
        return self._open(request, timeout=timeout)

    def post(self, url, *, json=None, headers=None, timeout=None):
        request_headers = dict(headers or {})
        request_headers.setdefault("content-type", "application/json")
        data = json_module.dumps(json or {}).encode("utf-8")
        request = Request(url, data=data, headers=request_headers, method="POST")
        return self._open(request, timeout=timeout)

    @staticmethod
    def _open(request, *, timeout=None):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                return SimpleHttpResponse(response.status, body)
        except Exception as exc:
            response = getattr(exc, "fp", None)
            status = getattr(exc, "code", None)
            if response is not None and status is not None:
                body = response.read().decode("utf-8", errors="replace")
                return SimpleHttpResponse(status, body)
            raise


def get_default_env_file() -> Path:
    script_env_file = Path(__file__).resolve().parent / ".env"
    if script_env_file.is_file():
        return script_env_file
    return Path.cwd() / ".env"


def load_env_file(path: str | os.PathLike[str] | None) -> None:
    if not path:
        return

    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        os.environ[key] = value.strip().strip("\"'")


def _extract_env_file(argv: Sequence[str]) -> str | os.PathLike[str] | None:
    for index, token in enumerate(argv):
        if token == "--env-file" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--env-file="):
            return token.split("=", 1)[1]
    return None


def get_env(*names: str, default=None):
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value not in (None, ""):
            return value
        value = SCRIPT_ENV_DEFAULTS.get(name)
        if value not in (None, ""):
            return value
    return default


def get_int_env(*names: str, default: int, minimum: int | None = 1) -> int:
    raw_value = get_env(*names, default=str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default

    if minimum is not None:
        return max(minimum, value)
    return value


def get_float_env(*names: str, default: float) -> float:
    raw_value = get_env(*names, default=str(default))
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def get_bool_env(*names: str, default: bool) -> bool:
    raw_value = get_env(*names, default=str(default))
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in BLANK_STRINGS
    return False


def normalize_inchi_key(value) -> str | None:
    if not isinstance(value, str):
        return None

    value_text = value.strip()
    if value_text.lower() in {"", "n/a", "null", "none"}:
        return None
    return value_text.upper()


def chemicals_have_same_inchi_key(chemical, standardized_chemical) -> bool:
    if not isinstance(chemical, dict) or not isinstance(standardized_chemical, dict):
        return False

    chemical_inchi_key = normalize_inchi_key(chemical.get("inchiKey"))
    standardized_inchi_key = normalize_inchi_key(standardized_chemical.get("inchiKey"))
    return bool(
        chemical_inchi_key
        and standardized_inchi_key
        and chemical_inchi_key == standardized_inchi_key
    )


def extract_chemical(payload) -> dict | None:
    if isinstance(payload, list):
        for item in payload:
            chemical = extract_chemical(item)
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
            chemical = extract_chemical(item)
            if chemical:
                return chemical

    if any(
        field_name in payload
        for field_name in ("chemId", "cid", "sid", "casrn", "name", "smiles", "canonicalSmiles", "inchi", "inchiKey")
    ):
        return payload

    return None


def normalize_resolver_item(item) -> dict | None:
    chemical = extract_chemical(item)
    if not isinstance(chemical, dict):
        return None

    normalized = {}
    for key, value in chemical.items():
        if key in RESOLVER_SKIP_FIELDS:
            continue
        normalized[key] = value

    additional_props = normalized.get("additionalProps")
    normalized["additionalProps"] = additional_props if isinstance(additional_props, dict) else {}
    return normalized


def build_resolver_payload(smiles_list: Sequence[str]) -> dict:
    return {
        "fuzzy": "Not",
        "ids": [str(smiles) for smiles in smiles_list],
        "idsType": "SMILES",
        "mol": False,
        "filters": {},
        "format": "UNKNOWN",
    }


def build_resolver_get_params(smiles: str) -> dict:
    return {
        "query": smiles,
        "idType": "SMILES",
        "fuzzy": "Not",
        "mol": "false",
    }


def extract_resolver_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("chemicals", "results", "items"):
            items = payload.get(key)
            if isinstance(items, list):
                return items
        return [payload]
    return None


def choose_query_smiles(item, smiles_by_index: Sequence[str], index: int) -> str | None:
    if not isinstance(item, dict):
        return None

    query = item.get("query")
    if isinstance(query, str) and query in smiles_by_index:
        return query

    chemical = extract_chemical(item)
    if isinstance(chemical, dict):
        for field_name in ("smiles", "canonicalSmiles", "chemId"):
            value = chemical.get(field_name)
            if isinstance(value, str) and value in smiles_by_index:
                return value

    if 0 <= index < len(smiles_by_index):
        return smiles_by_index[index]

    return None


def resolve_one_smiles(session, smiles: str, *, resolver_url: str, timeout: float) -> dict | None:
    response = session.get(
        resolver_url,
        params=build_resolver_get_params(smiles),
        headers={"accept": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return normalize_resolver_item(response.json())


def resolve_smiles_batch(
    session,
    smiles_list: Sequence[str],
    *,
    resolver_url: str,
    timeout: float,
    retry_individual: bool,
) -> dict[str, dict]:
    unique_smiles = list(dict.fromkeys(smiles for smiles in smiles_list if isinstance(smiles, str) and smiles.strip()))
    if not unique_smiles:
        return {}

    resolved: dict[str, dict] = {}

    try:
        response = session.post(
            resolver_url,
            json=build_resolver_payload(unique_smiles),
            headers={"accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        items = extract_resolver_items(response.json())
        if not isinstance(items, list):
            raise ValueError("resolver response did not contain an item list")

        for index, item in enumerate(items):
            query_smiles = choose_query_smiles(item, unique_smiles, index)
            if query_smiles is None:
                continue
            chemical = normalize_resolver_item(item)
            if chemical is not None:
                resolved[query_smiles] = chemical
    except Exception as exc:
        logging.warning("Batch resolver request failed for %s SMILES: %s", len(unique_smiles), exc)

    if retry_individual:
        for smiles in unique_smiles:
            if smiles in resolved:
                continue
            try:
                chemical = resolve_one_smiles(
                    session,
                    smiles,
                    resolver_url=resolver_url,
                    timeout=timeout,
                )
            except Exception as exc:
                logging.warning("Resolver lookup failed for smiles=%s: %s", smiles, exc)
                continue
            if chemical is not None:
                resolved[smiles] = chemical

    return resolved


def chemical_smiles_candidates(chemical: dict) -> list[str]:
    candidates = []
    for field_name in ("smiles", "canonicalSmiles", "chemId"):
        value = chemical.get(field_name)
        if isinstance(value, str) and value.strip() and value not in candidates:
            candidates.append(value)
    return candidates


def chemical_needs_resolver(chemical) -> bool:
    return isinstance(chemical, dict) and normalize_inchi_key(chemical.get("inchiKey")) is None


def merge_resolved_chemical(current, resolved) -> tuple[dict, list[str]]:
    if not isinstance(current, dict) or not isinstance(resolved, dict):
        return current, []

    updated = copy.deepcopy(current)
    changed_fields = []

    if "imageSrc" in updated:
        updated.pop("imageSrc", None)
        changed_fields.append("removed:imageSrc")

    for field_name, resolved_value in resolved.items():
        if field_name in RESOLVER_SKIP_FIELDS or is_blank(resolved_value):
            continue

        current_value = updated.get(field_name)
        if current_value != resolved_value:
            updated[field_name] = resolved_value
            changed_fields.append(field_name)

    if not isinstance(updated.get("additionalProps"), dict):
        updated["additionalProps"] = {}
        changed_fields.append("additionalProps")

    return updated, changed_fields


def remove_image_src_only(chemical) -> tuple[dict, list[str]]:
    if not isinstance(chemical, dict) or "imageSrc" not in chemical:
        return chemical, []

    updated = copy.deepcopy(chemical)
    updated.pop("imageSrc", None)
    return updated, ["removed:imageSrc"]


def collect_smiles_for_documents(documents: Sequence[dict]) -> list[str]:
    smiles_list = []
    for document in documents:
        prediction = document.get("prediction")
        if not isinstance(prediction, dict):
            continue

        for block_name in CHEMICAL_BLOCKS:
            chemical = prediction.get(block_name)
            if not chemical_needs_resolver(chemical):
                continue
            for smiles in chemical_smiles_candidates(chemical):
                if smiles not in smiles_list:
                    smiles_list.append(smiles)

    return smiles_list


def build_document_update(document: dict, resolved_by_smiles: dict[str, dict]):
    prediction = document.get("prediction")
    if not isinstance(prediction, dict):
        return None, []

    set_values = {}
    unset_values = {}
    changes = []

    for block_name in CHEMICAL_BLOCKS:
        chemical = prediction.get(block_name)
        if not isinstance(chemical, dict):
            continue

        updated_chemical = None
        changed_fields = []

        if chemical_needs_resolver(chemical):
            resolved_chemical = None
            for smiles in chemical_smiles_candidates(chemical):
                resolved_chemical = resolved_by_smiles.get(smiles)
                if resolved_chemical is not None:
                    break

            if resolved_chemical is not None:
                updated_chemical, changed_fields = merge_resolved_chemical(chemical, resolved_chemical)

        if updated_chemical is None:
            updated_chemical, changed_fields = remove_image_src_only(chemical)

        if changed_fields:
            set_values[f"prediction.{block_name}"] = updated_chemical
            changes.append((block_name, changed_fields))

    compare_chemical = (
        set_values.get("prediction.chemicalIdentifiers")
        or set_values.get("prediction.chemical")
        or prediction.get("chemicalIdentifiers")
        or prediction.get("chemical")
    )
    compare_standardized = prediction.get(STANDARDIZED_BLOCK)
    if chemicals_have_same_inchi_key(compare_chemical, compare_standardized):
        unset_values[f"prediction.{STANDARDIZED_BLOCK}"] = ""
        changes.append((STANDARDIZED_BLOCK, ["removed:sameInchiKey"]))

    if not set_values and not unset_values:
        return None, []

    update = {}
    if set_values:
        update["$set"] = set_values
    if unset_values:
        update["$unset"] = unset_values

    return update, changes


def iter_batches(iterable: Iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_arg_parser() -> argparse.ArgumentParser:
    resolver_api = get_env("RESOLVER_API", "resolver.url")
    default_resolver_lookup_api = (
        urljoin(resolver_api.rstrip("/") + "/", "lookup")
        if resolver_api
        else DEFAULT_RESOLVER_LOOKUP_API
    )
    resolver_lookup_api = get_env(
        "RESOLVER_LOOKUP_API",
        "resolver.lookup.url",
        default=default_resolver_lookup_api,
    )

    parser = argparse.ArgumentParser(
        description="Backfill chemical identifiers in the predictor_models Mongo response cache.",
    )
    parser.add_argument(
        "--env-file",
        default=str(get_default_env_file()),
        help="Path to a .env file loaded before reading Mongo/resolver defaults.",
    )
    parser.add_argument("--apply", action="store_true", help="Write updates to Mongo. Default is dry-run.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to inspect.")
    parser.add_argument("--batch-size", type=int, default=100, help="Mongo/resolver batch size.")
    parser.add_argument("--key", default=None, help="Only inspect one response-cache key.")
    parser.add_argument("--start-after-key", default=None, help="Resume after this cache key.")
    parser.add_argument(
        "--no-single-retry",
        action="store_true",
        help="Do not retry unresolved batch items with individual GET lookups.",
    )

    parser.add_argument(
        "--mongo-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=get_bool_env("MONGO_CACHE_ENABLED", default=True),
        help="Enable Mongo response-cache access.",
    )
    parser.add_argument("--mongo-host", default=get_env("MONGO_HOST", default="localhost"))
    parser.add_argument("--mongo-port", type=int, default=get_int_env("MONGO_PORT", default=27017))
    parser.add_argument("--mongo-user", default=get_env("MONGO_USER", default="root"))
    parser.add_argument("--mongo-password", default=get_env("MONGO_PASSWORD"))
    parser.add_argument("--mongo-auth-source", default=get_env("MONGO_AUTH_SOURCE", default="admin"))
    parser.add_argument("--mongo-database", default=get_env("MONGO_DATABASE", default="predictor"))
    parser.add_argument(
        "--collection",
        default=get_env(
            "PREDICTOR_MODELS_MONGO_COLLECTION",
            default="predictor_models_cache",
        ),
    )
    parser.add_argument(
        "--mongo-server-selection-timeout-ms",
        type=int,
        default=get_int_env("MONGO_SERVER_SELECTION_TIMEOUT_MS", default=2000),
    )
    parser.add_argument(
        "--resolver-url",
        default=resolver_lookup_api,
        help="Resolver /lookup URL.",
    )
    parser.add_argument(
        "--resolver-timeout",
        type=float,
        default=get_float_env("RESOLVER_LOOKUP_TIMEOUT", "resolver.lookup.timeout", default=10.0),
    )
    parser.add_argument(
        "--log-level",
        default=get_env("LOGGING_LEVEL", "logging.level", default="INFO"),
    )
    return parser


def connect_mongo(args):
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise SystemExit("pymongo is required to run this script.") from exc

    client_kwargs = {
        "host": args.mongo_host,
        "port": args.mongo_port,
        "appname": "predictor_models_cache_backfill",
        "serverSelectionTimeoutMS": args.mongo_server_selection_timeout_ms,
    }

    if args.mongo_user:
        client_kwargs["username"] = args.mongo_user
        client_kwargs["password"] = args.mongo_password
        client_kwargs["authSource"] = args.mongo_auth_source
    elif args.mongo_password:
        raise SystemExit("MONGO_PASSWORD/--mongo-password was provided without MONGO_USER/--mongo-user.")

    client = MongoClient(**client_kwargs)
    client.admin.command("ping")
    collection = client[args.mongo_database][args.collection]
    return client, collection


def get_update_one_class():
    try:
        from pymongo import UpdateOne
    except ImportError as exc:
        raise SystemExit("pymongo is required to run this script.") from exc
    return UpdateOne


def create_http_session():
    try:
        import requests
    except ImportError:
        logging.info("requests is not installed; using urllib for resolver HTTP calls")
        return UrllibSession()
    return requests.Session()


def build_mongo_query(args) -> dict:
    query = {"prediction": {"$type": "object"}}
    if args.key is not None:
        query["key"] = args.key
    elif args.start_after_key is not None:
        query["key"] = {"$gt": args.start_after_key}
    return query


def run_backfill(args: argparse.Namespace) -> None:
    if not args.mongo_cache_enabled:
        raise SystemExit("Mongo cache is disabled via MONGO_CACHE_ENABLED/--no-mongo-cache-enabled.")

    _, collection = connect_mongo(args)
    query = build_mongo_query(args)
    projection = {
        "key": 1,
        "prediction.chemicalIdentifiers": 1,
        "prediction.chemical": 1,
        "prediction.standardizedChemical": 1,
    }

    cursor = collection.find(query, projection=projection).sort("key", 1).batch_size(args.batch_size)
    if args.limit is not None:
        cursor = cursor.limit(args.limit)

    total_seen = 0
    total_resolved = 0
    total_changed = 0
    total_written = 0
    total_unresolved = 0
    update_one = get_update_one_class()
    session = create_http_session()

    for documents in iter_batches(cursor, args.batch_size):
        total_seen += len(documents)
        smiles_list = collect_smiles_for_documents(documents)
        resolved_by_smiles = resolve_smiles_batch(
            session,
            smiles_list,
            resolver_url=args.resolver_url,
            timeout=args.resolver_timeout,
            retry_individual=not args.no_single_retry,
        )
        total_resolved += len(resolved_by_smiles)
        total_unresolved += max(0, len(smiles_list) - len(resolved_by_smiles))

        operations = []
        for document in documents:
            update, changes = build_document_update(document, resolved_by_smiles)
            if update is None:
                continue

            total_changed += 1
            key = document.get("key")
            change_text = ", ".join(
                f"{block_name}: {','.join(changed_fields)}"
                for block_name, changed_fields in changes
            )
            log_action = "Updating" if args.apply else "Would update"
            logging.info("%s key=%s fields=%s", log_action, key, change_text)
            operations.append(update_one({"_id": document["_id"]}, update))

        if args.apply and operations:
            result = collection.bulk_write(operations, ordered=False)
            total_written += result.modified_count

    mode = "apply" if args.apply else "dry-run"
    logging.info(
        "Backfill complete mode=%s seen=%s resolved_smiles=%s changed_docs=%s written_docs=%s unresolved_smiles=%s",
        mode,
        total_seen,
        total_resolved,
        total_changed,
        total_written,
        total_unresolved,
    )


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    load_env_file(_extract_env_file(argv) or get_default_env_file())
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.key is not None and args.start_after_key is not None:
        raise SystemExit("--key and --start-after-key cannot be used together")
    if not args.apply:
        logging.info("Running in dry-run mode. Add --apply to write updates.")

    run_backfill(args)


if __name__ == "__main__":
    main()
