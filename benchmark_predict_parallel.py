import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import random
import threading
import time

import requests


MODEL_IDS = tuple(range(1065, 1071))
DEFAULT_SMILES_FILE = Path("smiles_cache.smi")
FAILED_SMILES_FILE = Path("smiles_failed.tsv")
DEFAULT_ENDPOINT_PATH = "predict"
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_RETRY_JITTER_SECONDS = 0.25


def count_smiles_in_file(path: Path, skip_first: int = 0) -> int:
    count = 0
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            seen += 1
            if seen <= skip_first:
                continue
            count += 1
    return count


def iter_smiles_batches(path: Path, batch_size: int, skip_first: int = 0):
    batch = []
    seen = 0
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            seen += 1
            if seen <= skip_first:
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_job_specs() -> list[dict]:
    return [{"model_id": model_id} for model_id in MODEL_IDS]


def _has_meaningful_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _preview_value(value, max_len: int = 300) -> str:
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)

    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _find_prediction_error(payload, path: str = "response"):
    if isinstance(payload, dict):
        for field_name in ("error", "predictionError"):
            field_value = payload.get(field_name)
            if _has_meaningful_value(field_value):
                return f"{path}.{field_name}={_preview_value(field_value)}"

        for key, value in payload.items():
            nested_reason = _find_prediction_error(value, f"{path}.{key}")
            if nested_reason is not None:
                return nested_reason
        return None

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            nested_reason = _find_prediction_error(item, f"{path}[{index}]")
            if nested_reason is not None:
                return nested_reason

    return None


def _tsv_field(value) -> str:
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def init_failed_smiles_file(path: Path) -> None:
    header = "\t".join(
        (
            "timestamp",
            "model_id",
            "request_idx",
            "attempt",
            "status_code",
            "error_type",
            "error_message",
            "smiles",
        )
    )
    path.write_text(header + "\n", encoding="utf-8")


def append_failed_records(
    output_path: Path,
    records: list[dict],
    file_lock: threading.Lock,
) -> None:
    if not records:
        return

    with file_lock:
        with output_path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(
                    "\t".join(
                        (
                            _tsv_field(record.get("timestamp")),
                            _tsv_field(record.get("model_id")),
                            _tsv_field(record.get("request_idx")),
                            _tsv_field(record.get("attempt")),
                            _tsv_field(record.get("status_code")),
                            _tsv_field(record.get("error_type")),
                            _tsv_field(record.get("error_message")),
                            _tsv_field(record.get("smiles")),
                        )
                    )
                    + "\n"
                )


def _build_failed_record(
    model_id: int,
    request_idx: int,
    attempt: int,
    status_code: int | None,
    error_type: str,
    error_message: str,
    smile: str,
) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_id": model_id,
        "request_idx": request_idx,
        "attempt": attempt,
        "status_code": status_code,
        "error_type": error_type,
        "error_message": error_message,
        "smiles": smile,
    }


def _classify_request_exception(exc: requests.RequestException) -> str:
    if isinstance(exc, requests.Timeout):
        return "timeout"
    if isinstance(exc, requests.ConnectionError):
        return "connection_error"
    if isinstance(exc, requests.HTTPError):
        return "http_error"
    return "request_exception"


def _status_code_from_exception(exc: requests.RequestException) -> int | None:
    response = getattr(exc, "response", None)
    return response.status_code if response is not None else None


def _should_retry_exception(exc: requests.RequestException) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True

    status_code = _status_code_from_exception(exc)
    return status_code in RETRYABLE_STATUS_CODES


def _retry_delay_seconds(base_delay: float, attempt: int) -> float:
    jitter = random.uniform(0.0, DEFAULT_RETRY_JITTER_SECONDS)
    return (base_delay * (2 ** max(0, attempt - 1))) + jitter


def _analyze_prediction_payload(
    payload,
    smiles_batch: list[str],
    model_id: int,
    request_idx: int,
    attempt: int,
    status_code: int,
) -> tuple[int, list[dict]]:
    batch_error_message = None
    if isinstance(payload, dict):
        if _has_meaningful_value(payload.get("error")):
            batch_error_message = f"response.error={_preview_value(payload.get('error'))}"

        if isinstance(payload.get("results"), list):
            payload = payload["results"]
        elif len(smiles_batch) == 1 and any(key in payload for key in ("modelResults", "result", "chemicalIdentifiers", "chemical")):
            payload = [payload]
        else:
            error_message = f"Expected batch results list, got {type(payload).__name__}: {_preview_value(payload)}"
            return 0, [
                _build_failed_record(
                    model_id,
                    request_idx,
                    attempt,
                    status_code,
                    "invalid_response_shape",
                    error_message,
                    smile,
                )
                for smile in smiles_batch
            ]
    elif not isinstance(payload, list):
        error_message = f"Expected JSON list or batch response object, got {type(payload).__name__}: {_preview_value(payload)}"
        return 0, [
            _build_failed_record(
                model_id,
                request_idx,
                attempt,
                status_code,
                "invalid_response_shape",
                error_message,
                smile,
            )
            for smile in smiles_batch
        ]

    processed_count = 0
    failed_records = []
    shared_mismatch_message = None

    if batch_error_message is not None:
        failed_records.extend(
            _build_failed_record(
                model_id,
                request_idx,
                attempt,
                status_code,
                "batch_error",
                batch_error_message,
                smile,
            )
            for smile in smiles_batch
        )
        return 0, failed_records

    if len(payload) != len(smiles_batch):
        shared_mismatch_message = (
            f"Response length mismatch: expected {len(smiles_batch)} items, got {len(payload)}"
        )

    for idx, smile in enumerate(smiles_batch):
        if idx >= len(payload):
            failed_records.append(
                _build_failed_record(
                    model_id,
                    request_idx,
                    attempt,
                    status_code,
                    "response_length_mismatch",
                    shared_mismatch_message or "Missing response item",
                    smile,
                )
            )
            continue

        error_reason = _find_prediction_error(payload[idx], path=f"response[{idx}]")
        if error_reason is not None:
            failed_records.append(
                _build_failed_record(
                    model_id,
                    request_idx,
                    attempt,
                    status_code,
                    "prediction_error",
                    error_reason,
                    smile,
                )
            )
            continue

        if shared_mismatch_message is not None and len(payload) > len(smiles_batch):
            failed_records.append(
                _build_failed_record(
                    model_id,
                    request_idx,
                    attempt,
                    status_code,
                    "response_length_mismatch",
                    shared_mismatch_message,
                    smile,
                )
            )
            continue

        processed_count += 1

    return processed_count, failed_records


def log(message: str, print_lock: threading.Lock, worker_label: str | None = None) -> None:
    prefix = f"[{worker_label}] " if worker_label is not None else ""
    with print_lock:
        print(f"{prefix}{message}", flush=True)


def post_predict_batch(
    session: requests.Session,
    url: str,
    model_id: int,
    smiles_batch: list[str],
    timeout: int,
) -> requests.Response:
    payload = {"smiles": smiles_batch, "model_id": model_id}
    return session.post(url, json=payload, timeout=timeout)


def process_batch_request(
    session: requests.Session,
    url: str,
    model_id: int,
    timeout: int,
    request_idx: int,
    total_batches: int,
    smiles_batch: list[str],
    print_lock: threading.Lock,
    failed_smiles_path: Path,
    failed_smiles_lock: threading.Lock,
    worker_label: str,
    retry_attempts: int,
    retry_backoff: float,
) -> tuple[int, int, float]:
    batch_start = time.perf_counter()
    for attempt in range(1, retry_attempts + 1):
        try:
            response = post_predict_batch(session, url, model_id, smiles_batch, timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            if attempt < retry_attempts and _should_retry_exception(exc):
                delay = _retry_delay_seconds(retry_backoff, attempt)
                log(
                    f"request {request_idx}/{total_batches} failed on attempt "
                    f"{attempt}/{retry_attempts}, retrying in {delay:.2f}s, error: {exc}",
                    print_lock,
                    worker_label=worker_label,
                )
                time.sleep(delay)
                continue

            batch_elapsed = time.perf_counter() - batch_start
            status_code = _status_code_from_exception(exc)
            error_type = _classify_request_exception(exc)
            error_message = str(exc)
            append_failed_records(
                failed_smiles_path,
                [
                    _build_failed_record(
                        model_id,
                        request_idx,
                        attempt,
                        status_code,
                        error_type,
                        error_message,
                        smile,
                    )
                    for smile in smiles_batch
                ],
                failed_smiles_lock,
            )
            log(
                f"request {request_idx}/{total_batches} failed after {attempt} attempt(s), "
                f"batch time: {batch_elapsed:.3f}s, "
                f"saved {len(smiles_batch)} SMILES to {failed_smiles_path}, "
                f"status={status_code}, error: {exc}",
                print_lock,
                worker_label=worker_label,
            )
            return 0, len(smiles_batch), batch_elapsed

        try:
            payload = response.json()
        except ValueError as exc:
            batch_elapsed = time.perf_counter() - batch_start
            error_message = f"Invalid JSON response: {exc}; body={_preview_value(response.text)}"
            append_failed_records(
                failed_smiles_path,
                [
                    _build_failed_record(
                        model_id,
                        request_idx,
                        attempt,
                        response.status_code,
                        "invalid_json",
                        error_message,
                        smile,
                    )
                    for smile in smiles_batch
                ],
                failed_smiles_lock,
            )
            log(
                f"request {request_idx}/{total_batches} returned invalid JSON, "
                f"batch time: {batch_elapsed:.3f}s, saved {len(smiles_batch)} SMILES to "
                f"{failed_smiles_path}",
                print_lock,
                worker_label=worker_label,
            )
            return 0, len(smiles_batch), batch_elapsed

        processed_count, failed_records = _analyze_prediction_payload(
            payload,
            smiles_batch,
            model_id,
            request_idx,
            attempt,
            response.status_code,
        )
        failed_smiles_count = len(smiles_batch) - processed_count
        batch_elapsed = time.perf_counter() - batch_start

        if failed_records:
            append_failed_records(failed_smiles_path, failed_records, failed_smiles_lock)
            first_error = failed_records[0]["error_message"]
            if processed_count == 0:
                log(
                    f"request {request_idx}/{total_batches} failed semantically, "
                    f"batch time: {batch_elapsed:.3f}s, status={response.status_code}, "
                    f"saved {failed_smiles_count} SMILES to {failed_smiles_path}, "
                    f"first_error: {first_error}",
                    print_lock,
                    worker_label=worker_label,
                )
            else:
                log(
                    f"request {request_idx}/{total_batches} completed with item errors, "
                    f"batch time: {batch_elapsed:.3f}s, status={response.status_code}, "
                    f"success_smiles={processed_count}, failed_smiles={failed_smiles_count}, "
                    f"first_error: {first_error}",
                    print_lock,
                    worker_label=worker_label,
                )
            return processed_count, failed_smiles_count, batch_elapsed

        if attempt > 1:
            log(
                f"request {request_idx}/{total_batches} done on retry "
                f"(attempt {attempt}/{retry_attempts}), batch time: {batch_elapsed:.3f}s",
                print_lock,
                worker_label=worker_label,
            )
        else:
            log(
                f"request {request_idx}/{total_batches} done, batch time: {batch_elapsed:.3f}s",
                print_lock,
                worker_label=worker_label,
            )
        return processed_count, 0, batch_elapsed

    batch_elapsed = time.perf_counter() - batch_start
    return 0, len(smiles_batch), batch_elapsed


def run_endpoint_benchmark(
    url: str,
    smiles_file: Path,
    smiles_count: int,
    model_id: int,
    timeout: int,
    batch_size: int,
    skip_first: int,
    print_lock: threading.Lock,
    failed_smiles_path: Path,
    failed_smiles_lock: threading.Lock,
    retry_attempts: int,
    retry_backoff: float,
) -> dict:
    worker_label = f"model_id={model_id}"
    total_batches = (smiles_count + batch_size - 1) // batch_size

    log(f"starting benchmark: {url}", print_lock, worker_label=worker_label)
    log(
        f"batches prepared: {total_batches} "
        f"(batch_size={batch_size}, source=file, skip_first={skip_first}, "
        f"smiles_count={smiles_count})",
        print_lock,
        worker_label=worker_label,
    )

    start = time.perf_counter()
    total_processed = 0
    success_batches = 0
    partial_batches = 0
    failed_batches = 0
    failed_smiles = 0

    with requests.Session() as session:
        for request_idx, smiles_batch in enumerate(
            iter_smiles_batches(
                smiles_file,
                batch_size,
                skip_first=skip_first,
            ),
            start=1,
        ):
            processed_count, failed_smiles_count, _ = process_batch_request(
                session,
                url,
                model_id,
                timeout,
                request_idx,
                total_batches,
                smiles_batch,
                print_lock,
                failed_smiles_path,
                failed_smiles_lock,
                worker_label,
                retry_attempts,
                retry_backoff,
            )
            total_processed += processed_count
            failed_smiles += failed_smiles_count

            if failed_smiles_count == 0:
                success_batches += 1
            elif processed_count > 0:
                partial_batches += 1
            else:
                failed_batches += 1

            log(
                f"progress: success_batches={success_batches}, "
                f"partial_batches={partial_batches}, failed_batches={failed_batches}, "
                f"total smiles processed={total_processed}, failed_smiles={failed_smiles}",
                print_lock,
                worker_label=worker_label,
            )

    elapsed = time.perf_counter() - start
    log(
        f"total: {elapsed:.3f}s "
        f"(success_batches={success_batches}, partial_batches={partial_batches}, "
        f"failed_batches={failed_batches}, failed_smiles={failed_smiles})",
        print_lock,
        worker_label=worker_label,
    )
    return {
        "model_id": model_id,
        "elapsed": elapsed,
        "success_batches": success_batches,
        "partial_batches": partial_batches,
        "failed_batches": failed_batches,
        "total_processed": total_processed,
        "failed_smiles": failed_smiles,
        "smiles_count": smiles_count,
    }


def main():
    script_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {script_start_time}")

    parser = argparse.ArgumentParser(
        description="Benchmark batch prediction endpoint in 6 parallel threads"
    )
    parser.add_argument(
        "--base-url",
        default="https://cim-dev.sciencedataexperts.com/api/predictor_models",
        help="Base API URL prefix (without trailing slash)",
    )
    parser.add_argument(
        "--endpoint-path",
        default=DEFAULT_ENDPOINT_PATH,
        help="Prediction endpoint path to append to --base-url, for example predictDB or predict",
    )
    parser.add_argument(
        "--smiles-file",
        default=str(DEFAULT_SMILES_FILE),
        help="Path to text file with one SMILES per line",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout (seconds) per request"
    )
    parser.add_argument(
        "--batch-size", type=int, default=200, help="SMILES per request"
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=1000000,
        help="Skip first N non-empty SMILES entries from input file",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Total request attempts for retryable failures (includes the first attempt)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Base delay in seconds before retrying a retryable failure",
    )
    parser.add_argument(
        "--failed-smiles-file",
        default=str(FAILED_SMILES_FILE),
        help="TSV file used to store failed SMILES with diagnostics",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.skip_first < 0:
        raise ValueError("--skip-first must be >= 0")
    if args.retry_attempts <= 0:
        raise ValueError("--retry-attempts must be > 0")
    if args.retry_backoff <= 0:
        raise ValueError("--retry-backoff must be > 0")

    smiles_file = Path(args.smiles_file)
    if not smiles_file.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")

    smiles_count = count_smiles_in_file(smiles_file, skip_first=args.skip_first)
    if smiles_count == 0:
        raise ValueError(
            f"No SMILES found in file after --skip-first={args.skip_first}: {smiles_file}"
        )

    base = args.base_url.rstrip("/")
    endpoint_path = args.endpoint_path.strip().lstrip("/")
    if not endpoint_path:
        raise ValueError("--endpoint-path must not be empty")
    predict_url = f"{base}/{endpoint_path}"
    print_lock = threading.Lock()
    failed_smiles_lock = threading.Lock()
    failed_smiles_path = Path(args.failed_smiles_file)
    init_failed_smiles_file(failed_smiles_path)

    print(f"SMILES loaded from file: {smiles_count}")
    print(f"source_smiles_file: {smiles_file}")
    print(f"predict_url: {predict_url}")
    print(f"skip_first: {args.skip_first}")
    print(f"batch_size: {args.batch_size}")
    print(f"retry_attempts: {args.retry_attempts}")
    print(f"retry_backoff: {args.retry_backoff}")
    print(f"model_ids: {', '.join(str(model_id) for model_id in MODEL_IDS)}")
    print(f"failed_smiles_file: {failed_smiles_path}")

    job_specs = build_job_specs()
    print(f"workers: {len(job_specs)}")

    results = []
    overall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=len(job_specs)) as executor:
        future_to_job = {
            executor.submit(
                run_endpoint_benchmark,
                predict_url,
                smiles_file,
                smiles_count,
                job_spec["model_id"],
                args.timeout,
                args.batch_size,
                args.skip_first,
                print_lock,
                failed_smiles_path,
                failed_smiles_lock,
                args.retry_attempts,
                args.retry_backoff,
            ): job_spec
            for job_spec in job_specs
        }

        for future in as_completed(future_to_job):
            job_spec = future_to_job[future]
            worker_label = f"model_id={job_spec['model_id']}"
            try:
                results.append(future.result())
            except Exception as exc:
                log(f"benchmark crashed: {exc}", print_lock, worker_label=worker_label)
                results.append(
                    {
                        "model_id": job_spec["model_id"],
                        "elapsed": None,
                        "success_batches": 0,
                        "partial_batches": 0,
                        "failed_batches": 0,
                        "total_processed": 0,
                        "failed_smiles": 0,
                        "smiles_count": smiles_count,
                        "error": str(exc),
                    }
                )

    overall_elapsed = time.perf_counter() - overall_start

    print("\nSummary")
    for result in sorted(results, key=lambda item: item["model_id"]):
        elapsed = result["elapsed"]
        elapsed_text = f"{elapsed:.3f}s" if elapsed is not None else "failed"
        error_text = f", error={result['error']}" if "error" in result else ""
        print(
            f"model_id={result['model_id']}: total={elapsed_text}, "
            f"smiles_count={result['smiles_count']}, "
            f"success_batches={result['success_batches']}, "
            f"partial_batches={result['partial_batches']}, "
            f"failed_batches={result['failed_batches']}, "
            f"failed_smiles={result['failed_smiles']}, "
            f"total_processed={result['total_processed']}{error_text}"
        )

    print(f"OVERALL: total={overall_elapsed:.3f}s")


if __name__ == "__main__":
    main()
