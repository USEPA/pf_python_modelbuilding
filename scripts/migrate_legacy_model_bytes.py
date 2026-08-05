#!/usr/bin/env python3
"""
Reload and reserialize legacy pickled models stored in qsar_models.model_bytes.

This should be run inside the pf_python_model_building runtime, where model
dependencies are installed and DB credentials are available.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")


from util.database_utilities import getSession
from util.serialization_compat import deserialize_model, refresh_legacy_serialized_model, serialize_model


CHUNK_SIZE = 26214400
LOGGER = logging.getLogger("legacy_model_migrator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate pickled model bytes to current scikit-learn/xgboost runtime versions.",
    )
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--model-id",
        type=int,
        action="append",
        dest="model_ids",
        help="Model id to migrate. Repeat this flag for multiple models.",
    )
    selection_group.add_argument(
        "--all",
        action="store_true",
        help="Migrate all model ids found in qsar_models.model_bytes.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist the migrated model bytes back to the database. Default is dry-run.",
    )
    parser.add_argument(
        "--updated-by",
        default=os.getenv("USER") or "legacy_model_migrator",
        help="Value written to updated_by when --write is used.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logger verbosity.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def fetch_model_rows(session, model_id: int):
    query = text(
        """
        SELECT id, bytes, created_at, created_by, updated_at, updated_by
        FROM qsar_models.model_bytes
        WHERE fk_model_id = :model_id
        ORDER BY id
        """
    )
    return list(session.execute(query, {"model_id": model_id}))


def fetch_all_model_ids(session) -> list[int]:
    query = text(
        """
        SELECT DISTINCT fk_model_id
        FROM qsar_models.model_bytes
        ORDER BY fk_model_id
        """
    )
    return [row.fk_model_id for row in session.execute(query)]


def combine_model_bytes(rows) -> bytes:
    return b"".join(row.bytes for row in rows)


def chunk_bytes(data: bytes, chunk_size: int = CHUNK_SIZE) -> list[bytes]:
    return [data[idx:idx + chunk_size] for idx in range(0, len(data), chunk_size)]


def deserialize_model_with_warnings(payload: bytes, model_id: int):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = deserialize_model(payload)

    for warning_item in caught:
        LOGGER.warning("model_id=%s load warning: %s", model_id, warning_item.message)

    return model


def replace_model_bytes(session, model_id: int, payload: bytes, updated_by: str, rows) -> None:
    now = datetime.now()
    created_at = rows[0].created_at if rows else now
    created_by = rows[0].created_by if rows and rows[0].created_by else updated_by

    session.execute(
        text("DELETE FROM qsar_models.model_bytes WHERE fk_model_id = :model_id"),
        {"model_id": model_id},
    )

    insert_query = text(
        """
        INSERT INTO qsar_models.model_bytes (
            fk_model_id,
            bytes,
            created_by,
            updated_by,
            created_at,
            updated_at
        ) VALUES (
            :model_id,
            :bytes,
            :created_by,
            :updated_by,
            :created_at,
            :updated_at
        )
        """
    )

    for chunk in chunk_bytes(payload):
        session.execute(
            insert_query,
            {
                "model_id": model_id,
                "bytes": chunk,
                "created_by": created_by,
                "updated_by": updated_by,
                "created_at": created_at,
                "updated_at": now,
            },
        )

    session.commit()


def migrate_model(session, model_id: int, *, write: bool, updated_by: str) -> int:
    rows = fetch_model_rows(session, model_id)
    if not rows:
        LOGGER.error("model_id=%s: no model_bytes rows found", model_id)
        return 1

    old_payload = combine_model_bytes(rows)
    LOGGER.info(
        "model_id=%s: loaded %s chunk(s), %s bytes",
        model_id,
        len(rows),
        len(old_payload),
    )

    model = deserialize_model_with_warnings(old_payload, model_id)
    model, stats = refresh_legacy_serialized_model(model, logger=LOGGER)
    new_payload = serialize_model(model)

    LOGGER.info(
        "model_id=%s: refreshed %s XGBoost object(s), payload changed=%s",
        model_id,
        stats["xgboost_objects"],
        new_payload != old_payload,
    )

    if not write:
        LOGGER.info("model_id=%s: dry-run complete, database unchanged", model_id)
        return 0

    if new_payload == old_payload:
        LOGGER.info("model_id=%s: migrated payload matches existing bytes, skipping write", model_id)
        return 0

    replace_model_bytes(session, model_id, new_payload, updated_by, rows)
    LOGGER.info("model_id=%s: model_bytes updated successfully", model_id)
    return 0


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    session = getSession()
    exit_code = 0

    try:
        model_ids = args.model_ids
        if args.all:
            model_ids = fetch_all_model_ids(session)
            LOGGER.info("Discovered %s model(s) in qsar_models.model_bytes", len(model_ids))

        for model_id in model_ids:
            try:
                result = migrate_model(
                    session,
                    model_id,
                    write=args.write,
                    updated_by=args.updated_by,
                )
                exit_code = max(exit_code, result)
            except Exception:
                session.rollback()
                LOGGER.exception("model_id=%s: migration failed", model_id)
                exit_code = 1
    finally:
        session.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
