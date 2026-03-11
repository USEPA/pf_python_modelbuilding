import argparse
import time
from pathlib import Path

import requests


def load_smiles_from_file(path: Path) -> list[str]:
    smiles = []
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            smiles.append(line)
    if not smiles:
        raise ValueError(f"No SMILES found in {path}")
    return smiles


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


def post_predict_batch(url: str, model_id: int, smiles_batch: list[str], timeout: int) -> requests.Response:
    payload = {"smiles": smiles_batch, "model_id": model_id}
    response = requests.post(url, json=payload, timeout=timeout)
    return response


def find_failed_smiles(url: str, model_id: int, smiles_batch: list[str], timeout: int) -> list[tuple[str, str]]:
    failed = []
    for smile in smiles_batch:
        try:
            response = post_predict_batch(url, model_id, [smile], timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            failed.append((smile, str(exc)))
    return failed


def append_errors(errors_file: Path, failed: list[tuple[str, str]], run_no: int, request_no: int) -> None:
    if not failed:
        return

    with errors_file.open("a", encoding="utf-8") as fh:
        for smile, err in failed:
            fh.write(f"run={run_no}\trequest={request_no}\tsmiles={smile}\terror={err}\n")


def run_endpoint_benchmark(
    name: str,
    url: str,
    smiles_file: Path,
    smiles_count: int,
    model_id: int,
    timeout: int,
    batch_size: int,
    skip_first: int,
) -> float:
    print(f"\n{name}: {url}")
    errors_file = Path("errors.txt")

    total_batches = (smiles_count + batch_size - 1) // batch_size
    print(
        f"Batches prepared: {total_batches} "
        f"(batch_size={batch_size}, source=file, skip_first={skip_first})"
    )

    start = time.perf_counter()
    total_processed = 0
    success_batches = 0
    failed_batches = 0

    for request_idx, smiles_batch in enumerate(
        iter_smiles_batches(smiles_file, batch_size, skip_first=skip_first),
        start=1,
    ):
        batch_start = time.perf_counter()
        try:
            response = post_predict_batch(url, model_id, smiles_batch, timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            batch_elapsed = time.perf_counter() - batch_start
            failed_batches += 1
            failed_smiles = find_failed_smiles(url, model_id, smiles_batch, timeout)
            if failed_smiles:
                append_errors(errors_file, failed_smiles, 1, request_idx)
                print(
                    f"    identified failed smiles in batch: {len(failed_smiles)} "
                    f"(saved to {errors_file})"
                )
            else:
                append_errors(
                    errors_file,
                    [("<batch-level>", f"batch failed but single-smile checks passed: {exc}")],
                    1,
                    request_idx,
                )
            print(
                f"  request {request_idx}/{total_batches} failed, "
                f"batch time: {batch_elapsed:.3f}s, error: {exc}"
            )
            continue

        batch_elapsed = time.perf_counter() - batch_start
        success_batches += 1
        total_processed += len(smiles_batch)
        print(
            f"  request {request_idx}/{total_batches} done, "
            f"batch time: {batch_elapsed:.3f}s, total smiles processed: {total_processed}"
        )

    elapsed = time.perf_counter() - start
    print(
        f"  total: {elapsed:.3f}s "
        f"(success_batches={success_batches}, failed_batches={failed_batches})"
    )
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark batch prediction endpoint"
    )
    parser.add_argument(
        "--base-url",
        default="http://192.168.1.7:5004/api/predictor_models",
        help="Base API URL (without trailing slash)",
    )
    parser.add_argument(
        "--smiles-file",
        default="smiles_cache.smi",
        help="Path to text file with one SMILES per line",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=1065,
        help="model_id to use in requests",
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout (seconds) per request"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="SMILES per request"
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=230000,
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

    print(f"SMILES loaded from file: {smiles_count}")
    print(f"skip_first: {args.skip_first}")
    print(f"model_id: {args.model_id}")

    elapsed_primary = run_endpoint_benchmark(
        name=f"BENCHMARK ({base})",
        url=predict_url,
        smiles_file=smiles_file,
        smiles_count=smiles_count,
        model_id=args.model_id,
        timeout=args.timeout,
        batch_size=args.batch_size,
        skip_first=args.skip_first,
    )

    print("\nSummary")
    print(f"BENCHMARK: total={elapsed_primary:.3f}s")


if __name__ == "__main__":
    main()
