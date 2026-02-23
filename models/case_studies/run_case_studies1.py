'''
Created on Feb 3, 2026

@author: TMARTI02
'''

from models.case_studies.run_model_building_db import run_dataset, Results, ParametersGeneticAlgorithm,\
    ParametersImportance
from util import predict_constants as pc

from model_ws_db_utilities import getEngine, getSession
from ModelToExcel import ModelToExcel
import logging
import json
from utils import print_first_row
from _tkinter import create
from models.runGA import num_generations, num_optimizers
import pandas as pd
import os
import requests
from pathlib import Path
import csv
from typing import Optional, List, Dict, Tuple
import math

def run_episuite_csv(url, df_test, output_path):
    
    # Build a set of already-processed SMILES by scanning the existing output file
    processed_smiles = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    s = rec.get("smiles")
                    if isinstance(s, str):
                        processed_smiles.add(s)
                except Exception:
                    # Skip malformed lines
                    continue
        print(f"Found {len(processed_smiles)} SMILES already in {output_path}. Resuming...")
    
    # Reuse the HTTP session for efficiency
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "smiles-client/1.0"
    })
    
    # Append new results to the file (create if it doesn't exist)
    with open(output_path, "a", encoding="utf-8") as f:
        for idx, smiles in df_test["smiles"].dropna().astype(str).items():
            smiles = smiles.strip()
            if not smiles:
                continue
    
            # Skip SMILES already recorded in the output file
            if smiles in processed_smiles:
                # Optionally log the skip
                # print(f"Skipping already processed SMILES at index {idx}: {smiles}")
                continue
    
            record = {
                "index": int(idx),
                "smiles": smiles
            }
    
            try:
                resp = session.get(url, params={"smiles": smiles}, timeout=30)
                resp.raise_for_status()
    
                # Parse JSON if possible; otherwise store raw text
                try:
                    payload = resp.json()
                except ValueError:
                    payload = {"raw": resp.text}
    
                print (idx, smiles, resp.status_code)
                
                record.update({
                    "status_code": resp.status_code,
                    "response": payload
                })
            except requests.RequestException as e:
                record.update({
                    "error": str(e)
                })
    
            # Write one JSON record per line and mark as processed
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_smiles.add(smiles)

    print(f"Done. Responses appended to: {output_path}")



def go_through_test_results(method, csv_output_path_test, df_episuite, split, limits, property_name):

    df_test = pd.read_csv(csv_output_path_test) #for example the predictions file for one our models
    
    mask = df_test['canon_qsar_smiles'].isin(df_episuite['canon_qsar_smiles'])
    df_filtered = df_test[mask]
    
    required_cols = ["canon_qsar_smiles", "exp", "pred"]
    # Select and clean the needed columns
    df_subset = df_filtered[required_cols].copy()
        
    mae = (df_subset['exp'] - df_subset['pred']).abs().mean()
    # print(mae)
            
    # Create the list of dictionaries
    mps = df_subset.to_dict(orient="records")
    # print(json.dumps(mps, indent=4))
    
    path = Path(csv_output_path_test)
    filePathOutScatter = os.path.join(path.parent, split +" set predictions (omit if have episuite exp).png")
    title = method + " "+split+" set results for "+property_name
    from models import make_test_plots as mtp
    mtp.generateScatterPlot3(filePathOutScatter, title, 'Log10(L/kg)', mps, split+' set', limits)
    number = df_subset.shape[0]
    
    print(f"{method}\t{split}\t{number}\t{mae:.2f}")
    return df_subset, number, mae, filePathOutScatter

def go_through_episuite_results(csv_output_path_episuite, csv_output_path_test, split, property):
    
       # Load the EpiSuite compiled CSV and rename columns
    df_epi = pd.read_csv(csv_output_path_episuite)
    
    # Ensure the expected columns exist and rename them
    rename_map = {}
    if "exp" in df_epi.columns:
        rename_map["exp"] = "exp_episuite"
    if "pred" in df_epi.columns:
        rename_map["pred"] = "pred_episuite"
    
   
    df_epi = df_epi.rename(columns=rename_map)

    # Keep only relevant columns and drop duplicate SMILES (keep last occurrence)
    keep_cols = ["smiles"] + [c for c in ["exp_episuite", "pred_episuite"] if c in df_epi.columns]
    df_epi = df_epi[keep_cols].drop_duplicates(subset="smiles", keep="last")
    
    # Load the base CSV
    df_base = pd.read_csv(csv_output_path_test) #for example the predictions file for one our models
    
    # Merge: augment df_base with EpiSuite columns using a left join on "smiles"
    merged_df = df_base.merge(df_epi, on="smiles", how="left")
    
    # merged_df = merged_df[merged_df["exp_episuite"].isna()] # only keep ones without exp from episuite (to make external preds)
        
    if merged_df["exp_episuite"].notna().any():
        merged_df = merged_df[merged_df["exp_episuite"].isna()]
    
    # Optional: inspect result
    # print(f"Merged shape: {merged_df.shape}")
    # print_first_row(merged_df)
    
    # Ensure required columns exist
    required_cols = ["canon_qsar_smiles", "exp", "pred_episuite"]
    # Select and clean the needed columns
    df_subset = merged_df[required_cols].copy()
    # Convert NaN to None for JSON-friendly output
    df_subset = df_subset.where(pd.notna(df_subset), None)
    df_subset = df_subset.rename(columns={"pred_episuite": "pred"})

    mae = (df_subset['exp'] - df_subset['pred']).abs().mean()
    
    # Create the list of dictionaries
    mps = df_subset.to_dict(orient="records")
    # print(json.dumps(mps, indent=4))
    
    path = Path(csv_output_path_episuite)
   
    filePathOutScatter = os.path.join(path.parent, split +" set predictions episuite (omit if have exp).png")
    title = f"Episuite {split} set results for {property}"
    from models import make_test_plots as mtp
    limits = mtp.generateScatterPlot3(filePathOutScatter, title, 'Log10(L/kg)', mps, split+' set')
    
    number = df_subset.shape[0]
    print(f"episuite\t{split}\t{number}\t{mae:.2f}")
    
    return df_subset, number, mae, filePathOutScatter, limits



# pip install dominate
from dominate import document
from dominate.tags import meta, style, h1, section, table, tr, td, div
from dominate.tags import img as img_tag  # alias

def run_episuite(max_cols_per_row: int = 3, img_min_height_px: int = 400, row_gap_px: int = 24):
    url = "https://episuite.app/EpiWebSuite/api/submit"

    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    # dataset_name = "KOC v1 modeling"
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3 modeling'
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    property = "96 hr Fish LC50"


    base_dir = Path(PROJECT_ROOT) / "data" / "models" / dataset_name
    episuite_dir = base_dir / "episuite"
    episuite_dir.mkdir(parents=True, exist_ok=True)

    def pred_csv_path(embedding: str, split: str) -> Path:
        return base_dir / embedding / f"{split} set predictions.csv"

    def episuite_json_path(split: str) -> Path:
        return episuite_dir / f"{split} set predictions episuite.json"

    def episuite_csv_path(split: str) -> Path:
        return episuite_json_path(split).with_suffix(".csv")

    base_embedding_for_df_test = "gcm_WebTEST-default_fs=False"
    rf_custom_embedding = "rf_WebTEST-default_fs=False_custom"

    # splits = ["test", "external"]
    splits = ["test"]
    
    if "KOC" in dataset_name:
        splits.append("external")
    
    other_methods = ["xgb", "reg", "knn"]

    summaries = {split: [] for split in splits}

    for split in splits:
        file_path_csv = pred_csv_path(base_embedding_for_df_test, split)
        df_test = pd.read_csv(file_path_csv)

        output_path_json = episuite_json_path(split)
        csv_output_path_episuite = episuite_csv_path(split)
        
        # Optionally run these if needed:
        run_episuite_csv(url, df_test, str(output_path_json))
        

        #todo add method get fish tox        
        
        if "KOC" in dataset_name:        
            compile_koc_csv(str(output_path_json), str(csv_output_path_episuite))
        elif "Fish" in dataset_name:
            compile_ecosar_csv(str(output_path_json), str(csv_output_path_episuite))

        else:
            print("handle compilation of data for "+dataset_name)
            return

        # continue


        # EpiSuite
        df_episuite, number, mae, filePathOutScatter, limits = go_through_episuite_results(
            str(csv_output_path_episuite), str(file_path_csv), split, property
        )
        summaries[split].append({"method": "EpiSuite", "n": number, "mae": mae, "img": filePathOutScatter})

        # RF custom
        df_subset, number, mae, filePathOutScatter = go_through_test_results(
            "rf", str(pred_csv_path(rf_custom_embedding, split)), df_episuite, split, limits, property
        )
        summaries[split].append({"method": "RF", "n": number, "mae": mae, "img": filePathOutScatter})

        # GCM baseline
        df_subset, number, mae, filePathOutScatter = go_through_test_results(
            "gcm", str(pred_csv_path(base_embedding_for_df_test, split)), df_episuite, split, limits, property
        )
        summaries[split].append({"method": "GCM", "n": number, "mae": mae, "img": filePathOutScatter})

        # Other methods
        for method in other_methods:
            embedding = f"{method}_WebTEST-default_fs=True"
            df_subset, number, mae, filePathOutScatter = go_through_test_results(
                method, str(pred_csv_path(embedding, split)), df_episuite, split, limits, property
            )
            summaries[split].append({"method": method.upper(), "n": number, "mae": mae, "img": filePathOutScatter})

    def rel_for_html(target_path: str, base: Path) -> str:
        if not target_path:
            return ""
        try:
            rel = os.path.relpath(target_path, start=base)
        except Exception:
            rel = target_path
        return Path(rel).as_posix()

    def build_split_html(property, split: str, rows: list, columns_per_row: int, img_h_px: int, v_gap_px: int):
        columns_per_row = min(3, max(1, columns_per_row))
        img_h_px = max(400, img_h_px)

        out_html = episuite_dir / f"{split}_scatter_plots.html"
        doc = document(title=f"{property} Scatter Plots - {split.title()} Split")
        with doc.head:
            meta(charset="utf-8")
            style(f"""
                body {{
                    font-family: Arial, Helvetica, sans-serif;
                    margin: 10px;
                }}
                h1 {{
                    margin: 0 0 6px 0;
                    font-size: 14px;
                    font-weight: 600;
                }}
                .table-wrap {{
                    text-align: center;
                }}
                .plots-table {{
                    display: inline-table;
                    width: auto;
                    table-layout: auto;
                    border-collapse: separate;
                    border-spacing: 6px {v_gap_px}px; /* horizontal 6px, vertical gap between rows */
                    margin: 4px auto;
                }}
                .plots-table td {{
                    padding: 0;
                    border: 0;
                    vertical-align: top;
                }}
                .plot-img {{
                    height: {img_h_px}px;
                    width: auto;
                    max-width: 100%;
                    display: block;
                    margin: 0;
                }}
                .mae-caption {{
                    font-size: 1.15rem;
                    font-weight: 800;
                    color: #222;
                    margin-top: 6px;
                    text-align: center;
                }}
            """)

        with doc:
            h1(f"{property} Scatter Plots - {split.title()} split")

            with section(cls="table-wrap"):
                with table(cls="plots-table"):
                    for i in range(0, len(rows), columns_per_row):
                        with tr():
                            for r in rows[i:i + columns_per_row]:
                                method_name = r.get("method", "")
                                img_path = r.get("img")
                                mae_val = r.get("mae", None)
                                caption_text = f"{method_name} — MAE: {mae_val:.2f}" if mae_val is not None else f"{method_name}"

                                img_src = rel_for_html(img_path, episuite_dir) if img_path else ""
                                if img_src and not Path(os.path.join(episuite_dir, img_src)).exists():
                                    img_src = Path(img_path).as_posix()

                                with td():
                                    if img_src:
                                        img_tag(src=img_src, cls="plot-img", alt="")
                                    div(caption_text, cls="mae-caption")

        out_html.write_text(doc.render(), encoding="utf-8")
        print(f"Wrote compact HTML for {split} split: {out_html}")

    for split in splits:
        build_split_html(
            property,
            split,
            summaries[split],
            columns_per_row=max_cols_per_row,
            img_h_px=img_min_height_px,
            v_gap_px=row_gap_px
        )



def compile_koc_csv(
    input_path: str = "api_responses.ndjson",
    output_csv: str = "compiled_koc.csv"
) -> Tuple[int, List[Dict[str, Optional[float]]]]:
    """
    Read a newline-delimited JSON file (one record per line) where each record contains:
      {
        "smiles": <str>,
        "response": {
            "logKoc": {
                "estimatedValue": { "model": { "logKoc": <number-like> } },
                "selectedValue": { "valueType": <str>, "value": <number-like> }
            }
        },
        ...
      }

    Extract:
      - smiles (string)
      - pred = response["logKoc"]["estimatedValue"]["model"]["logKoc"]
      - exp  = response["logKoc"]["selectedValue"]["value"]
              only when response["logKoc"]["selectedValue"]["valueType"] != "ESTIMATED" (case-insensitive)

    Writes a CSV with columns: smiles, pred, exp

    Returns:
      - number of rows written
      - list of row dicts (for optional further use)
    """

    def to_float(x) -> Optional[float]:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    rows: List[Dict[str, Optional[float]]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue

            smiles = rec.get("smiles")
            response = rec.get("response")

            # Only proceed if we have a dict-like JSON response
            if not isinstance(response, dict):
                # Some lines may have {"raw": "..."}; skip those
                continue

            logkoc = response.get("logKoc")
            if not isinstance(logkoc, dict):
                # Skip if the expected section is missing
                continue

            # Extract predicted logKoc
            pred = None
            try:
                pred = logkoc.get("estimatedValue", {}).get("model", {}).get("logKoc")
            except Exception:
                pred = None

            pred = to_float(pred)

            # Extract experimental value only if valueType != "ESTIMATED"
            exp: Optional[float] = None
            selected = logkoc.get("selectedValue", {})
            selected = selected if isinstance(selected, dict) else {}
            value_type = selected.get("valueType")

            if isinstance(value_type, str) and value_type.strip().upper() != "ESTIMATED":
                exp = to_float(selected.get("value"))

            # Build the row (include if smiles and pred are available)
            if smiles is not None and pred is not None:
                rows.append({
                    "smiles": str(smiles),
                    "pred": pred,
                    "exp": exp
                })

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles", "pred", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows



def compile_ecosar_csv(
    input_path: str = "api_responses.ndjson",
    output_csv: str = "compiled_koc.csv"
) -> Tuple[int, List[Dict[str, Optional[float]]]]:
    """
    Read a newline-delimited JSON file (one record per line) 
    
    Extract:
      - smiles (string)
      - pred 
      - exp  

    Writes a CSV with columns: smiles, pred, exp

    Returns:
      - number of rows written
      - list of row dicts (for optional further use)
    """

    def to_float(x) -> Optional[float]:
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    rows: List[Dict[str, Optional[float]]] = []
    
    # print("eeer")

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                print("error loading json")
                continue

            smiles = rec.get("smiles")
            response = rec.get("response")
            
            # print(smiles)

            # Only proceed if we have a dict-like JSON response
            if not isinstance(response, dict):
                # Some lines may have {"raw": "..."}; skip those
                print("dont have response dict")
                continue

            ecosar = response.get("ecosar")
            if not isinstance(ecosar, dict):
                # Skip if the expected section is missing
                print("dont have ecosar dict")
                continue

            modelResults = ecosar.get("modelResults")


            if not isinstance(modelResults, list):
                # Skip if the expected section is missing
                print("dont have modelResults list")
                continue

            predCount = 0
            pred = None
            
            insideAD=True
            for mr in modelResults:
                if mr["organism"] !="Fish":
                    continue;
                if mr["endpoint"] != "LC50":
                    continue;
                if mr["duration"] != "96-hr":
                    continue;
                predNew = mr["concentration"] # mg/L value
                predNew /= 1000.0; # g/L
                predNew /= response.get("chemicalProperties").get("molecularWeight"); # mol/L
                predNew = -math.log10(predNew); # -logM
                predCount+=1
                if pred is None or predNew > pred:
                    pred = predNew;
                    qsarClass=mr["qsarClass"]
                
            if response["ecosar"]["parameters"]["logKow"]["value"] > mr["maxLogKow"]:
                insideAD=False;    

            exp = None

            start_token = "Available Measured Data from ECOSAR Training Set"
            stop_token = "ECOSAR v2.20 Class-specific Estimations"
            
            started = False
            filtered_lines = []
            
            for raw in ecosar["output"].splitlines():
                line = raw.strip()
            
                # Wait until we hit the start token
                if not started:
                    if start_token in line:
                        started = True
                        # print(smiles, "\n",ecosar["output"])
                    continue
            
                # Once started, stop if we hit stop token or no-data notice
                if "No Data Available" in line:
                    print(smiles, "No Data Available")
                    break
                
                if stop_token in line:
                    break
            
                # Skip separators and "(SW)" lines
                if "---------" in line:
                    continue
                
                if "(SW)" in line:
                    continue
            
                # print("\t", smiles, line)
            
                # Keep only lines that mention Fish, 96h, and LC50
                # if not ("Fish" in line and "96h" in line and "LC50" in line):
                #     continue
                
                # print(smiles, line)
            
                filtered_lines.append(line)
            
            # print(smiles, exp, pred, insideAD)

            if smiles is not None and pred is not None:
                rows.append({
                    "smiles": str(smiles),
                    "pred": pred,
                    "exp": exp
                })

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles", "pred", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows

# Example usage:
# count, data = compile_koc_csv("api_responses.ndjson", "compiled_koc.csv")
# print(f"Wrote {count} rows to compiled_koc.csv")

def run_example():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False)  # OK


def run_Koc():
    create_unique_excel = False
    write_to_db=True
    # write_to_db=True
    dataset_name = "KOC v1 modeling"

    # ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK

    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK

    # embedding = ["ALOGP2","nBnz","MATS6v","nDB","Lop","MATS1p"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, embedding=embedding, write_to_db=write_to_db)

    #
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db)

    # grid = {
    #     # Tree-specific parameters
    #     'estimator__max_depth': [3, 5, 7, 9],'estimator__min_child_weight': [1, 3, 5],'estimator__gamma': [0, 0.1, 0.2, 0.4],
    #     'estimator__subsample': [0.7, 0.8, 0.9, 1.0],'estimator__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    #     'estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],'estimator__n_estimators': [100, 500, 1000] 
    # }
    # params = ParametersImportance(qsar_method='xgb', hyperparameter_grid=grid, descriptor_set_name= "WebTEST-default",
    #                                     ad_measure=ad_measure_model, dataset_name=dataset_name, 
    #                                     run_rfe=True, run_sfs=True, feature_selection=True)
    # run_dataset(dataset_name=dataset_name, qsar_method='xgb', feature_selection=params.feature_selection, params=params, write_to_db=write_to_db)

    
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK
    #
    # grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']}  # matches AD in terms of using 3
    # params = ParametersGeneticAlgorithm(qsar_method='knn', hyperparameter_grid=grid, descriptor_set_name= "WebTEST-default",
    #                                     ad_measure=ad_measure_model, dataset_name=dataset_name, 
    #                                     num_generations = 100, num_optimizers=100,
    #                                     run_rfe=True, run_sfs=False, feature_selection=True)
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=params.feature_selection, params=params, write_to_db=write_to_db)
    #
    # embedding = ['piPC06', 'ALOGP', 'ALOGP2']
    # # embedding = ["piPC06","XLOGP","XLOGP2"]
    # results_dict = run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, embedding=embedding, write_to_db=write_to_db)
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK
    #
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', folder_embedding="rf_WebTEST-default_fs=True", ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK
    # run_dataset(dataset_name=dataset_name, qsar_method='reg', feature_selection=True, ad_measure_model=ad_measure_model,write_to_db=write_to_db)  # OK

    # Does adding my LogKow model prediction as a descriptor help:
    # run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=True, add_LOGP_Martin=True,write_to_db=write_to_db)  # Martin LOGP will show up in final descriptors, but error isnt lower!
    # run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, add_LOGP_Martin=True,write_to_db=write_to_db)  # OK

    # *****************************************************************************************************************************

    # ---- Models to upload: ----
    # Models to upload:
    # for method in ['rf','xgb', 'reg','knn']:
    #     run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
    #                 ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    # run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    embedding = ["ALOGP2","nBnz","MATS6v","ATS1p","nDB","Lop","MATS1p"]
    results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
                               embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)
    

    r = Results()
    r.summarize_model_stats(dataset_name)
    

def run_Koc_knn_ga():
    
    descriptor_set_name = "WebTEST-default"
    dataset_name = "KOC v1 modeling"

    grid = {'estimator__n_neighbors': [3], 'estimator__weights': ['distance']}  # matches AD in terms of using 3
    params = ParametersGeneticAlgorithm(qsar_method='knn', hyperparameter_grid=grid,
                                        descriptor_set_name=descriptor_set_name, dataset_name=dataset_name,
                                        run_rfe=True)
    params.num_optimizers = 100
    params.num_generations = 100
    
    # max_features_array = [3, 5, 10, 15, 20]
    
    max_features_array = [20]
    
    stats_dict = {}
    
    for max_features in max_features_array:
        params.max_features = max_features
        results_dict = run_dataset(dataset_name=dataset_name, qsar_method='knn', feature_selection=True, params=params)
        MAE_Test = results_dict['test_stats']['MAE_Test']
        MAE_Training_CV = results_dict['cv_stats']['MAE_Test']
        
        logging.info(f"max_features: {max_features}, MAE_Test: {MAE_Test:.2f}, MAE_Training_CV: {MAE_Training_CV:.2f}")
    
        stats = {"max_features": max_features, "MAE_Test":MAE_Test, "MAE_Training_CV":MAE_Training_CV}
        stats_dict[max_features] = stats
    
    print(json.dumps(stats_dict, indent=4))


def run_fish_tox():
    
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    write_to_db=True
    create_unique_excel = False

    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False, ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    for method in ['rf','xgb', 'reg','knn']:
        run_dataset(dataset_name=dataset_name, qsar_method=method, feature_selection=True, 
                    ad_measure_model=ad_measure_model,write_to_db=write_to_db, create_unique_excel=create_unique_excel)  # OK
    
    ## embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'xv1', 'Mp', 'AMW'] 
    ## embedding = ['ALOGP', 'XLOGP2', 'MW', 'BEHm3', 'Mp', 'AMW']
    embedding = ['ALOGP', 'ALOGP2', 'MW', 'Mp', 'AMW']
    
    results_dict = run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, 
                               embedding=embedding, write_to_db=write_to_db, create_unique_excel=create_unique_excel)

    
    
    Results.summarize_model_stats(dataset_name)


def test_model_summary():
    engine = getEngine()
    session = getSession()
    model_id = 1065
    excel_path = "summary.xlsx"
    test = ModelToExcel(engine, session, model_id, excel_path)
    test.create_excel()


def test_model_summary_local():
    dataset_name = "KOC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='rf', feature_selection=False, create_detailed_excel=True)  # OK

    

# split_num
# fk_data_point_id

if __name__ == '__main__':
    # run_example()
    # run_Koc_knn_ga()
    
    run_episuite()
    
    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()

    # run_Koc()
    # run_fish_tox()
    # test_create_model()
    # test_model_summary()
    # test_model_summary_local()


