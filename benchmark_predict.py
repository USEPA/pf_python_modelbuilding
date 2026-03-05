import argparse
import statistics
import time
from pathlib import Path

import requests


def load_smiles(path: Path) -> list[str]:
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


def run_endpoint_benchmark(name: str, url: str, payload: dict, runs: int, timeout: int) -> list[float]:
    print(f"\n{name}: {url}")

    durations = []
    for idx in range(runs):
        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=timeout)
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        durations.append(elapsed)
        print(f"  run {idx + 1}/{runs}: {elapsed:.3f}s")

    return durations


def summarize(name: str, durations: list[float]) -> dict:
    return {
        "name": name,
        "min": min(durations),
        "max": max(durations),
        "avg": statistics.mean(durations),
        "median": statistics.median(durations),
    }


def print_summary(summary: dict):
    print(
        f"{summary['name']}: "
        f"avg={summary['avg']:.3f}s, "
        f"median={summary['median']:.3f}s, "
        f"min={summary['min']:.3f}s, "
        f"max={summary['max']:.3f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare batch prediction endpoints"
    )
    parser.add_argument(
        "--base-url",
        default="http://192.168.1.7:5004/api/predictor_models",
        help="Primary base API URL (without trailing slash)",
    )
    parser.add_argument(
        "--compare-base-url",
        default="http://192.168.1.7:5005/api/predictor_models",
        help="Secondary base API URL for comparison (without trailing slash)",
    )
    parser.add_argument(
        "--smiles-file",
        default="test_smiles1.smi",
        help="Path to text file with one SMILES per line",
    )
    parser.add_argument(
        "--model-id",
        type=int,
        default=1065,
        help="model_id to use in requests",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per endpoint")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout (seconds) per request"
    )
    args = parser.parse_args()

    smiles_path = Path(args.smiles_file)
    smiles = load_smiles(smiles_path)

    payload = {"smiles": smiles, "model_id": args.model_id}

    base = args.base_url.rstrip("/")
    compare_base = args.compare_base_url.rstrip("/")
    predict_url = f"{base}/predict"
    compare_predict_url = f"{compare_base}/predict"

    print(f"SMILES loaded: {len(smiles)}")
    print(f"model_id: {args.model_id}")

    times_primary = run_endpoint_benchmark(
        name=f"PRIMARY ({base})",
        url=predict_url,
        payload=payload,
        runs=args.runs,
        timeout=args.timeout,
    )

    times_compare = run_endpoint_benchmark(
        name=f"COMPARE ({compare_base})",
        url=compare_predict_url,
        payload=payload,
        runs=args.runs,
        timeout=args.timeout,
    )

    print("\nSummary")
    summary_primary = summarize("PRIMARY", times_primary)
    summary_compare = summarize("COMPARE", times_compare)
    print_summary(summary_primary)
    print_summary(summary_compare)

    if summary_compare["avg"] > 0:
        speedup = summary_compare["avg"] / summary_primary["avg"] if summary_primary["avg"] > 0 else float("inf")
        print(f"Speedup (compare/primary by avg): {speedup:.2f}x")


if __name__ == "__main__":
    main()
