import argparse
import statistics
import time
from pathlib import Path
from typing import Optional

import requests

try:
    import pymysql
except ImportError:  # pragma: no cover - runtime dependency check
    pymysql = None


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


def load_smiles_from_mysql(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    table: str,
    column: str,
    limit: Optional[int] = None,
) -> list[str]:
    if pymysql is None:
        raise RuntimeError(
            "pymysql is not installed. Install it with: pip install pymysql"
        )

    query = f"SELECT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL"
    if limit and limit > 0:
        query += " LIMIT %s"

    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )
    try:
        with connection.cursor() as cursor:
            if limit and limit > 0:
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        connection.close()

    smiles = []
    for idx, row in enumerate(rows, start=1):
        if row[0] and str(row[0]).strip():
            smiles.append(str(row[0]).strip())
        if idx % 1000 == 0:
            print(f"MySQL progress: processed {idx} rows")

    if rows and len(rows) % 1000 != 0:
        print(f"MySQL progress: processed {len(rows)} rows")

    if not smiles:
        raise ValueError(
            f"No SMILES found in MySQL source {database}.{table}.{column}"
        )
    return smiles


def chunk_list(values: list[str], chunk_size: int):
    for i in range(0, len(values), chunk_size):
        yield values[i : i + chunk_size]


def post_predict_batch(url: str, model_id: int, smiles_batch: list[str], timeout: int) -> None:
    payload = {"smiles": smiles_batch, "model_id": model_id}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()


def run_endpoint_benchmark(
    name: str,
    url: str,
    smiles: list[str],
    model_id: int,
    runs: int,
    timeout: int,
    batch_size: int,
) -> list[float]:
    print(f"\n{name}: {url}")

    batches = list(chunk_list(smiles, batch_size))
    print(f"Batches prepared: {len(batches)} (batch_size={batch_size}, sequential mode)")

    durations = []
    for idx in range(runs):
        start = time.perf_counter()
        total_processed = 0

        for request_idx, smiles_batch in enumerate(batches, start=1):
            post_predict_batch(url, model_id, smiles_batch, timeout)
            total_processed += len(smiles_batch)
            print(
                f"  run {idx + 1}/{runs}, request {request_idx}/{len(batches)} done, "
                f"total smiles processed: {total_processed}"
            )

        elapsed = time.perf_counter() - start
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
        "--input-source",
        choices=["mysql", "file"],
        default="mysql",
        help="Source of SMILES data",
    )
    parser.add_argument(
        "--smiles-file",
        default="test_smiles1.smi",
        help="Path to text file with one SMILES per line",
    )
    parser.add_argument("--mysql-host", default="192.168.1.3", help="MySQL host")
    parser.add_argument("--mysql-port", type=int, default=3306, help="MySQL port")
    parser.add_argument("--mysql-user", default="root", help="MySQL username")
    parser.add_argument(
        "--mysql-password", default="qqq123", help="MySQL user password"
    )
    parser.add_argument(
        "--mysql-database", default="dsstox_2026_01", help="MySQL database"
    )
    parser.add_argument(
        "--mysql-table", default="compounds", help="MySQL table with SMILES"
    )
    parser.add_argument(
        "--mysql-column", default="smiles", help="MySQL column containing SMILES"
    )
    parser.add_argument(
        "--mysql-limit",
        type=int,
        default=0,
        help="Optional MySQL row limit (0 means no limit)",
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
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="SMILES per request"
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    benchmark_runs = args.runs

    if args.input_source == "mysql":
        if args.runs != 1:
            print("MySQL mode: forcing --runs to 1")
        benchmark_runs = 1
        smiles = load_smiles_from_mysql(
            host=args.mysql_host,
            port=args.mysql_port,
            user=args.mysql_user,
            password=args.mysql_password,
            database=args.mysql_database,
            table=args.mysql_table,
            column=args.mysql_column,
            limit=args.mysql_limit if args.mysql_limit > 0 else None,
        )
    else:
        smiles_path = Path(args.smiles_file)
        smiles = load_smiles_from_file(smiles_path)

    base = args.base_url.rstrip("/")
    compare_base = args.compare_base_url.rstrip("/")
    predict_url = f"{base}/predict"
    compare_predict_url = f"{compare_base}/predict"

    print(f"SMILES loaded: {len(smiles)}")
    print(f"model_id: {args.model_id}")

    times_primary = run_endpoint_benchmark(
        name=f"PRIMARY ({base})",
        url=predict_url,
        smiles=smiles,
        model_id=args.model_id,
        runs=benchmark_runs,
        timeout=args.timeout,
        batch_size=args.batch_size,
    )

    # times_compare = run_endpoint_benchmark(
    #     name=f"COMPARE ({compare_base})",
    #     url=compare_predict_url,
    #     payload=payload,
    #     runs=args.runs,
    #     timeout=args.timeout,
    # )

    print("\nSummary")
    summary_primary = summarize("PRIMARY", times_primary)
    # summary_compare = summarize("COMPARE", times_compare)
    print_summary(summary_primary)
    # print_summary(summary_compare)

    # if summary_compare["avg"] > 0:
    #     speedup = summary_compare["avg"] / summary_primary["avg"] if summary_primary["avg"] > 0 else float("inf")
    #     print(f"Speedup (compare/primary by avg): {speedup:.2f}x")


if __name__ == "__main__":
    main()
