'''
Created on May 12, 2026

@author: TMARTI02
'''

import json
from typing import Optional, Dict, Any, List
import requests
import sys

from API_Utilities import QsarSmilesAPI


def standardizeStructure(smiles: str, use_full_standardize: bool, workflow: str, server_host: str, omit_salts: bool):
    """
    Uses code in QsarSmilesAPI
    """
    chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(server_host=server_host, smiles=smiles, full=use_full_standardize,
                                                       workflow=workflow)
    if code == 400:
        return chemicals, code
    
    # print(len(chemicals))
            
    if len(chemicals) == 0:
        # logging.debug('Standardization failed')
        return f"{smiles} failed standardization" if smiles else 'No Structure', 400

    if len(chemicals) > 1 and omit_salts:
        # print('qsar smiles indicates mixture')
        return f"{smiles}: model can't run mixtures", 400
    
    # print(json.dumps(chemicals, indent=4))

    return chemicals[0]["canonicalSmiles"], 200


def standardize(
    smiles: str,
    use_full_standardize: bool,
    workflow: str,
    server_host: str,
) -> Optional[str]:
    """
    Calls the QSAR-ready standardizer and returns the standardized SMILES string.
    Returns None if the request is unsuccessful or no results are returned.
    """
    standardize_response = call_qsar_ready_standardize_post(
        smiles, use_full_standardize, workflow, server_host
    )

    if standardize_response.status_code == 200:
        json_response = get_response_body(standardize_response, use_full_standardize)
        
        if use_full_standardize:
            return handle_full_output_single_chemical(json_response)
        else:
            return handle_simple_output_single_chemical(json_response)

    return None


def get_response_body(response: requests.Response, full: bool) -> str:
    """
    Parses the response JSON and re-serializes it for pretty printing.
    The 'full' parameter is kept for parity with the Java signature.
    """
    try:
        obj = response.json()
    except ValueError:
        # Fallback if response isn't valid JSON
        return response.text
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def handle_simple_output_single_chemical(json_str: str) -> Optional[str]:
    """
    Expects a JSON array like:
    [
      { "canonicalSmiles": "..." },
      { "canonicalSmiles": "..." },
      ...
    ]
    """
    try:
        results: List[Dict[str, Any]] = json.loads(json_str)
    except ValueError:
        return None

    if not isinstance(results, list):
        return None

    if len(results) == 0:
        return None
    elif len(results) == 1:
        result = results[0]
        return result.get("canonicalSmiles")
    else:
        smiles_parts: List[str] = []
        for result in results:
            cs = result.get("canonicalSmiles")
            if isinstance(cs, str) and cs:
                smiles_parts.append(cs)
        return ".".join(smiles_parts) if smiles_parts else None


def handle_full_output_single_chemical(json_str: str) -> Optional[str]:
    """
    Expects a JSON object with a 'records' array. Records with status == 'SKIPPED'
    are ignored when there are multiple.
    """
    try:
        jo: Dict[str, Any] = json.loads(json_str)
    except ValueError:
        return None

    records = jo.get("records")
    if not isinstance(records, list):
        return None

    if len(records) == 0:
        return None
    elif len(records) == 1:
        record = records[0]
        try:
            return get_smiles_from_record(record)
        except KeyError:
            return None
    else:
        smiles_accum: List[str] = []
        for record in records:
            if isinstance(record, dict) and record.get("status") == "SKIPPED":
                continue
            try:
                smiles_i = get_smiles_from_record(record)
            except KeyError:
                continue
            if isinstance(smiles_i, str) and smiles_i:
                smiles_accum.append(smiles_i)
        return ".".join(smiles_accum) if smiles_accum else None


def get_smiles_from_record(record: Dict[str, Any]) -> str:
    """
    Extracts a SMILES string from a single 'record' in the full output.
    Adjust this logic to match your server's schema.

    Tries several common locations for 'canonicalSmiles' or 'smiles'.
    Raises KeyError if no SMILES can be found.
    """
    # Try top-level common fields
    for key in ("canonicalSmiles", "smiles"):
        val = record.get(key)
        if isinstance(val, str) and val:
            return val

    # Try nested locations frequently used in structured outputs
    candidate_paths = [
        ("result", "canonicalSmiles"),
        ("output", "canonicalSmiles"),
        ("standardized", "canonicalSmiles"),
        ("result", "smiles"),
        ("output", "smiles"),
        ("standardized", "smiles"),
    ]
    for path in candidate_paths:
        d: Any = record
        for k in path:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                d = None
                break
        if isinstance(d, str) and d:
            return d

    raise KeyError("Could not find SMILES in record")


def call_qsar_ready_standardize_post(
    smiles: str, full: bool, workflow: str, server_host: str
) -> requests.Response:
    """
    Calls the QSAR-ready standardizer endpoint using JSON payloads.
    """
    url = server_host.rstrip("/") + "/api/stdizer/chemicals"
    payload = {
        "full": full,
        "options": {"workflow": workflow},
        "chemicals": [{"smiles": smiles}],
    }
    headers = {"Content-Type": "application/json"}
    return requests.post(url, json=payload, headers=headers)


def runExample():
    # smiles = 'CI'
    smiles = 'CCC.CCCC'
    use_full_standardize = False
    server_host = 'https://cim-dev.sciencedataexperts.com/'
    workflow = 'qsar-ready_04242025_0'
    # workflow = 'qsar-ready_06182025'
    omit_salts = True

    result = standardize(smiles, use_full_standardize, workflow, server_host)
    if result is None:
        print("Standardization failed or returned no result.", file=sys.stderr)
    print(result)
    
    smiles, code = standardizeStructure(smiles, use_full_standardize, workflow, server_host, omit_salts)
    print(smiles, code)

# def addQsarReadySmilesToExcelFile():
#
#
#     from dotenv import load_dotenv
#     load_dotenv('../personal.env')
#
#     import os
#     import pandas as pd
#     PROJECT_ROOT = os.getenv("PROJECT_ROOT")
#     dataset_name = 'exp_prop_RBIODEG_301F v1 modeling'
#
#     from pathlib import Path
#
#     input_excel_path = Path(PROJECT_ROOT) / "data" / "models" / dataset_name / "DSSTox_FRAGRANCEBB_20260413.xlsx"
#     df = pd.read_excel(input_excel_path)
#
#     print(df.shape)


def addQsarReadySmilesToExcelFile():

    from dotenv import load_dotenv
    load_dotenv("../personal.env")

    import os
    import csv
    from pathlib import Path
    import pandas as pd

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    SERVER_HOST = "https://cim-dev.sciencedataexperts.com/"
    WORKFLOW = "qsar-ready_04242025_0"
    USE_FULL = False
    OMIT_SALTS = True

    dataset_name = "exp_prop_RBIODEG_301F v1 modeling"
    input_excel_path = (
        Path(PROJECT_ROOT)
        / "data" / "models" / dataset_name
        / "DSSTox_FRAGRANCEBB_20260413.xlsx"
    )

    df = pd.read_excel(input_excel_path)
    print(df.shape)

    # Choose the SMILES column
    input_col = "Structure_SMILES" if "Structure_SMILES" in df.columns else "Structure_Smiles"
    if input_col not in df.columns:
        raise KeyError('Neither "Structure_SMILES" nor "Structure_Smiles" found in DataFrame.')

    # Ensure result column exists
    if "Structure_qsar_ready" not in df.columns:
        df["Structure_qsar_ready"] = pd.NA

    # CSV progress file (for resume)
    progress_path = input_excel_path.with_name(f"{input_excel_path.stem}_qsar_ready_progress.csv")

    # Load existing progress to resume
    if progress_path.exists():
        prog_df = pd.read_csv(
            progress_path,
            na_values=["", "NA", "NaN", "null", "None"],
            dtype={"row_index": "int64"}
        )
        if not prog_df.empty:
            for r in prog_df.itertuples(index=False):
                idx = int(r.row_index)
                val = getattr(r, "Structure_qsar_ready")
                if idx in df.index and pd.isna(df.at[idx, "Structure_qsar_ready"]):
                    df.at[idx, "Structure_qsar_ready"] = val

    # Open progress CSV for appending
    fieldnames = ["row_index", input_col, "Structure_qsar_ready", "status_code"]
    write_header = not progress_path.exists()
    f = open(progress_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    try:
        for row in df.itertuples(index=True):
            row_index = int(row.Index)
            smiles = getattr(row, input_col)

            # Skip if already done
            if not pd.isna(df.at[row_index, "Structure_qsar_ready"]):
                continue

            # Skip missing SMILES
            if pd.isna(smiles) or (isinstance(smiles, str) and not smiles.strip()):
                df.at[row_index, "Structure_qsar_ready"] = pd.NA
                writer.writerow({
                    "row_index": row_index,
                    input_col: None,
                    "Structure_qsar_ready": None,
                    "status_code": None,
                })
                f.flush()
                continue

            # Standardize
            try:
                qsar_smiles, code = standardizeStructure(
                    str(smiles),
                    use_full_standardize=USE_FULL,
                    workflow=WORKFLOW,
                    server_host=SERVER_HOST,
                    omit_salts=OMIT_SALTS,
                )
            except Exception as e:
                print(f"Error standardizing row {row_index}: {e}")
                qsar_smiles, code = None, None

            df.at[row_index, "Structure_qsar_ready"] = qsar_smiles if qsar_smiles else pd.NA

            writer.writerow({
                "row_index": row_index,
                input_col: smiles,
                "Structure_qsar_ready": qsar_smiles,
                "status_code": code,
            })
            f.flush()

            print(f"{row_index}: {smiles} -> {qsar_smiles} (code={code})")
    finally:
        f.close()

    # Write final Excel file alongside input
    output_excel_path = input_excel_path.with_name(
        f"{input_excel_path.stem}_qsar_ready{input_excel_path.suffix}"
    )
    df.to_excel(output_excel_path, index=False)
    print(f"Wrote standardized data to {output_excel_path}")
    print(f"Progress saved incrementally to {progress_path}")


def main() -> int:
    
    # runExample()
    addQsarReadySmilesToExcelFile()


if __name__ == "__main__":
    main()
