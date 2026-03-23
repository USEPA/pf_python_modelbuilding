import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading
import time

import requests


MODEL_IDS = tuple(range(1065, 1071))
DEFAULT_SMILES_FILE = Path("smiles_cache.smi")
FAILED_SMILES_FILE = Path("smiles_failed.smi")


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


def append_failed_smiles(
    output_path: Path,
    model_id: int,
    smiles_batch: list[str],
    file_lock: threading.Lock,
) -> None:
    with file_lock:
        with output_path.open("a", encoding="utf-8") as fh:
            for smile in smiles_batch:
                fh.write(f"{smile}-{model_id}\n")


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
) -> tuple[bool, int, float]:
    batch_start = time.perf_counter()
    try:
        response = post_predict_batch(session, url, model_id, smiles_batch, timeout)
        response.raise_for_status()
    except requests.RequestException as first_exc:
        log(
            f"request {request_idx}/{total_batches} failed on first attempt, retrying once, "
            f"error: {first_exc}",
            print_lock,
            worker_label=worker_label,
        )
        try:
            response = post_predict_batch(session, url, model_id, smiles_batch, timeout)
            response.raise_for_status()
        except requests.RequestException as second_exc:
            batch_elapsed = time.perf_counter() - batch_start
            append_failed_smiles(
                failed_smiles_path,
                model_id,
                smiles_batch,
                failed_smiles_lock,
            )
            log(
                f"request {request_idx}/{total_batches} failed after retry, "
                f"batch time: {batch_elapsed:.3f}s, "
                f"saved {len(smiles_batch)} SMILES to {failed_smiles_path}, "
                f"first_error: {first_exc}, second_error: {second_exc}",
                print_lock,
                worker_label=worker_label,
            )
            return False, 0, batch_elapsed

        batch_elapsed = time.perf_counter() - batch_start
        log(
            f"request {request_idx}/{total_batches} done on retry, "
            f"batch time: {batch_elapsed:.3f}s",
            print_lock,
            worker_label=worker_label,
        )
        return True, len(smiles_batch), batch_elapsed

    batch_elapsed = time.perf_counter() - batch_start
    log(
        f"request {request_idx}/{total_batches} done, batch time: {batch_elapsed:.3f}s",
        print_lock,
        worker_label=worker_label,
    )
    return True, len(smiles_batch), batch_elapsed


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
    failed_batches = 0

    with requests.Session() as session:
        for request_idx, smiles_batch in enumerate(
            iter_smiles_batches(
                smiles_file,
                batch_size,
                skip_first=skip_first,
            ),
            start=1,
        ):
            is_success, processed_count, _ = process_batch_request(
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
            )
            if is_success:
                success_batches += 1
                total_processed += processed_count
            else:
                failed_batches += 1

            log(
                f"progress: success_batches={success_batches}, "
                f"failed_batches={failed_batches}, total smiles processed={total_processed}",
                print_lock,
                worker_label=worker_label,
            )

    elapsed = time.perf_counter() - start
    log(
        f"total: {elapsed:.3f}s "
        f"(success_batches={success_batches}, failed_batches={failed_batches})",
        print_lock,
        worker_label=worker_label,
    )
    return {
        "model_id": model_id,
        "elapsed": elapsed,
        "success_batches": success_batches,
        "failed_batches": failed_batches,
        "total_processed": total_processed,
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
        help="Base API URL (without trailing slash)",
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
        "--batch-size", type=int, default=500, help="SMILES per request"
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=250000,
        help="Skip first N non-empty SMILES entries from input file",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.skip_first < 0:
        raise ValueError("--skip-first must be >= 0")

    smiles_file = Path(args.smiles_file)
    if not smiles_file.exists():
        raise FileNotFoundError(f"SMILES file not found: {smiles_file}")

    smiles_count = count_smiles_in_file(smiles_file, skip_first=args.skip_first)
    if smiles_count == 0:
        raise ValueError(
            f"No SMILES found in file after --skip-first={args.skip_first}: {smiles_file}"
        )

    base = args.base_url.rstrip("/")
    predict_url = f"{base}/predict"
    print_lock = threading.Lock()
    failed_smiles_lock = threading.Lock()
    failed_smiles_path = FAILED_SMILES_FILE
    failed_smiles_path.write_text("", encoding="utf-8")

    print(f"SMILES loaded from file: {smiles_count}")
    print(f"source_smiles_file: {smiles_file}")
    print(f"skip_first: {args.skip_first}")
    print(f"batch_size: {args.batch_size}")
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
                        "failed_batches": 0,
                        "total_processed": 0,
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
            f"failed_batches={result['failed_batches']}, "
            f"total_processed={result['total_processed']}{error_text}"
        )

    print(f"OVERALL: total={overall_elapsed:.3f}s")


if __name__ == "__main__":
    main()
