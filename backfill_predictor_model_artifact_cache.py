#!/usr/bin/env python3
"""Backfill predictor model artifacts from Postgres into Mongo.

This exports the model data needed by runtime predictions and HTML reports:

* initialized model snapshots, including model metadata, statistics, training/test
  data frames, DSSTox lookup data, and neighbor prediction lookups
* model files used by report links and images: QMRF, Excel summary, scatter plot,
  histogram, and webpage HTML when available

The runtime service reads these artifacts from Mongo first and only falls back to
Postgres when the Mongo artifact is absent or unavailable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_FILE_TYPE_IDS = (1, 2, 3, 4, 5)


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
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def _extract_env_file(argv: Sequence[str]) -> str | None:
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
    return default


def _parse_file_type_ids(values: Sequence[str] | None) -> tuple[int, ...]:
    if not values:
        return DEFAULT_FILE_TYPE_IDS
    type_ids: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            type_ids.append(int(token))
    return tuple(dict.fromkeys(type_ids))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export predictor model artifacts from Postgres into Mongo.",
    )
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument(
        "--model-id",
        type=int,
        action="append",
        help="Model ID to export. Can be repeated. Defaults to all public predictor models.",
    )
    parser.add_argument(
        "--file-type-id",
        action="append",
        help="File type IDs to export, comma-separated or repeated. Defaults to 1,2,3,4,5.",
    )
    parser.add_argument("--skip-models", action="store_true", help="Do not export initialized model snapshots.")
    parser.add_argument("--skip-files", action="store_true", help="Do not export model files.")
    parser.add_argument("--dry-run", action="store_true", help="Read Postgres but do not write Mongo.")
    parser.add_argument("--log-level", default=get_env("LOG_LEVEL", default="INFO"))
    return parser


def resolve_model_ids(explicit_model_ids: Sequence[int] | None) -> list[int]:
    if explicit_model_ids:
        return list(dict.fromkeys(int(model_id) for model_id in explicit_model_ids))

    from sqlalchemy import text
    from model_ws_db_utilities import getSession

    session = getSession()
    try:
        rows = session.execute(
            text(
                """
                SELECT id
                FROM qsar_models.models
                WHERE fk_source_id = 3
                  AND is_public = true
                ORDER BY id
                """
            )
        ).fetchall()
    finally:
        session.close()

    return [int(row[0]) for row in rows]


def backfill_model_snapshot(model_id: int, *, dry_run: bool) -> bool:
    from db import model_cache
    from model_ws_db_utilities import ModelInitializer

    initializer = ModelInitializer()
    model = initializer._init_model_from_postgres(model_id)
    if model is None:
        logging.error("Could not initialize model_id=%s from Postgres", model_id)
        return False

    if dry_run:
        logging.info("Dry run: would write model snapshot model_id=%s", model_id)
        return True

    model_cache.write_model_snapshot(model_id, model)
    logging.info("Wrote model snapshot model_id=%s", model_id)
    return True


def backfill_model_files(model_id: int, file_type_ids: Sequence[int], *, dry_run: bool) -> tuple[int, int]:
    from db import model_cache
    from util.get_model_file import fetch_model_file_from_postgres

    written = 0
    missing = 0
    for file_type_id in file_type_ids:
        try:
            raw_bytes, file_name, mime_type = fetch_model_file_from_postgres(model_id, int(file_type_id))
        except FileNotFoundError:
            missing += 1
            logging.info("No model file found model_id=%s file_type_id=%s", model_id, file_type_id)
            continue
        except Exception:
            missing += 1
            logging.exception("Could not fetch model file model_id=%s file_type_id=%s", model_id, file_type_id)
            continue

        if dry_run:
            logging.info(
                "Dry run: would write model file model_id=%s file_type_id=%s bytes=%s",
                model_id,
                file_type_id,
                len(raw_bytes),
            )
        else:
            model_cache.write_model_file(model_id, int(file_type_id), raw_bytes, file_name, mime_type)
            logging.info(
                "Wrote model file model_id=%s file_type_id=%s bytes=%s",
                model_id,
                file_type_id,
                len(raw_bytes),
            )
        written += 1
    return written, missing


def run_backfill(args: argparse.Namespace) -> None:
    file_type_ids = _parse_file_type_ids(args.file_type_id)
    model_ids = resolve_model_ids(args.model_id)
    if not model_ids:
        raise SystemExit("No model IDs found to export.")

    logging.info(
        "Starting predictor model artifact backfill model_ids=%s file_type_ids=%s skip_models=%s skip_files=%s dry_run=%s",
        model_ids,
        file_type_ids,
        args.skip_models,
        args.skip_files,
        args.dry_run,
    )

    model_success = 0
    file_written = 0
    file_missing = 0
    for model_id in model_ids:
        logging.info("Processing model_id=%s", model_id)
        if not args.skip_models and backfill_model_snapshot(model_id, dry_run=args.dry_run):
            model_success += 1
        if not args.skip_files:
            written, missing = backfill_model_files(model_id, file_type_ids, dry_run=args.dry_run)
            file_written += written
            file_missing += missing

    logging.info(
        "Predictor model artifact backfill complete model_snapshots=%s file_docs=%s missing_files=%s dry_run=%s",
        model_success,
        file_written,
        file_missing,
        args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    load_env_file(_extract_env_file(argv) or DEFAULT_ENV_FILE)
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_backfill(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
