'''
Created on Mar 5, 2026

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

import pandas as pd
import os
import requests
from pathlib import Path
import csv
from typing import Optional, List, Dict, Tuple, Any

import math
    # Build HTML with dominate
from dominate import document
from dominate.tags import meta, style, h2, table, thead, tbody, tr, th, td, div, img, h1, section, span

from util.indigo_utils import IndigoUtils


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




# Indigo image helper (same as your single-method function)


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


def build_prediction_table_css(
    cell_padding: int,
    width: int,
    height: int,
    smiles_col_px: int,
    table_max_width: int | None = None,
) -> str:
    container_max_w_css = f"max-width:{table_max_width}px;" if table_max_width is not None else ""
    return f"""
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      .content-wrap {{
        {container_max_w_css}
        margin: 0;
      }}
      table {{
        border-collapse: collapse;
        width: auto;
        margin: 0;
        table-layout: auto;
      }}
      th, td {{
        border: 1px solid #ddd;
        padding: {cell_padding}px;
        vertical-align: bottom;
      }}
      img {{
          height: 50%;
          width: 50%;
      }}
      th {{ background: #f0f0f0; }}
      tr:nth-child(even) {{ background: #fafafa; }}
      td.num {{ text-align: right; }}
      th.smiles-col, td.smiles-col {{ max-width: {smiles_col_px}px; width: {smiles_col_px}px; }}
      .smiles-cell {{ width: {width}px; display: flex; flex-direction: column; align-items: center; }}
      .smiles-text {{ margin-top: 6px; font-family: monospace; font-size: 12px; word-break: break-all; }}
      .placeholder {{
        width: {width}px;
        height: {height}px;
        display:flex; align-items:center; justify-content:center;
        background:#f8f8f8; color:#888; border:1px solid #ddd;
      }}
      /* Nested mini table for split/exp/methods (columns) */
      .mini-table {{
        margin-top: 8px;
        border-collapse: collapse;
        width: 100%;
        font-size: 18px;
      }}
      .mini-table th, .mini-table td {{
        border: 1px solid #e5e5e5;
        padding: 4px 6px;
      }}
      .mini-table th {{
        background: #f9f9f9;
        text-align: center;
        color: #333;
        font-weight: 600;
        white-space: nowrap;
      }}
      .mini-table td.num {{
        text-align: center;
        font-variant-numeric: tabular-nums;
      }}
    """


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


def build_split_html(episuite_dir, property, split: str, rows: list, columns_per_row: int, img_h_px: int, v_gap_px: int):
    columns_per_row = min(3, max(1, columns_per_row))
    # img_h_px = max(400, img_h_px)

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
        h2(f"{property} Scatter Plots - {split.title()} split")

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
                                    img(src=img_src, cls="plot-img", alt="")
                                div(caption_text, cls="mae-caption")

    out_html.write_text(doc.render(), encoding="utf-8")
    print(f"Wrote compact HTML for {split} split: {out_html}")


def rel_for_html(target_path: str, base: Path) -> str:
    if not target_path:
        return ""
    try:
        rel = os.path.relpath(target_path, start=base)
    except Exception:
        rel = target_path
    return Path(rel).as_posix()





def build_all_methods_dom_html(
    property_name:str = None, 
    big_df: pd.DataFrame=None,
    out_html_path=None,
    smiles_col="canon_qsar_smiles",
    exp_col="exp",
    pred_cols=None,           # auto-detect if None (columns starting with 'pred_')
    width=240,
    height=120,
    max_rows=None,
    include_abs_err=False,    # not shown in mini-table by default
    cell_padding=8,
    table_max_width=None,
    sort_within_split=False,  # optional: sort within split
    columns_per_row=3,        # number of “cards” per grid row
    split=None, 
):
    """
    Build an HTML where each grid cell contains:
      - structure image
      - SMILES text
      - a nested mini-table with columns: split, exp, EpiSuite, gcm, rf, xgb, reg, knn (when present).
    Rows are ordered by decreasing EpiSuite absolute error.
    The main table has no header; items are laid out in a grid with `columns_per_row` cells per row.
    """
    df = big_df.copy()

    # Detect prediction columns
    if pred_cols is None:
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No prediction columns found. Expected columns named 'pred_<method>'.")

    # Normalize numerics
    df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
    for c in pred_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify EpiSuite column (case-insensitive)
    episuite_col = next((c for c in pred_cols if "episuite" in c.lower()), None)
    if episuite_col is None:
        raise KeyError("EpiSuite prediction column not found among pred_* columns. Expected 'pred_EpiSuite' or similar.")

    # Compute error and sort
    df["episuite_abs_err"] = (df[episuite_col] - df[exp_col]).abs()
    if sort_within_split and "split" in df.columns:
        df = df.sort_values(by=["split", "episuite_abs_err"], ascending=[True, False], na_position="last", kind="stable")
    else:
        df = df.sort_values(by="episuite_abs_err", ascending=False, na_position="last", kind="stable")

    if max_rows is not None:
        df = df.head(max_rows)

    # Map labels -> column names (e.g., 'gcm' -> 'pred_gcm')
    label_to_col = {}
    for c in pred_cols:
        label_raw = c.replace("pred_", "")
        label_to_col[label_raw.lower()] = c

    # Desired mini-table column order (include only if present)
    desired_order = ["split", "exp", "EpiSuite", "gcm", "rf", "xgb", "reg", "knn"]
    mini_columns = []
    for label in desired_order:
        if label == "split":
            if "split" in df.columns:
                mini_columns.append(("split", "split"))
        elif label == "exp":
            if exp_col in df.columns:
                mini_columns.append(("exp", exp_col))
        elif label == "EpiSuite":
            if episuite_col in df.columns:
                mini_columns.append(("epi", episuite_col))
        else:
            col = label_to_col.get(label.lower())
            if col in df.columns:
                mini_columns.append((label, col))

    # Renderer and CSS
    iu = IndigoUtils()
    smiles_col_px = width + 2 * cell_padding
    css = build_prediction_table_css(
        cell_padding=cell_padding,
        width=width,
        height=height,
        smiles_col_px=smiles_col_px,
        table_max_width=table_max_width,
    )

    def fmt_num(x):
        return "" if pd.isna(x) else f"{x:.2f}"

    if property_name is not None:        
        title = property_name +" All Methods Prediction Report (EpiSuite error sorted)"
    else:
        title = "All Methods Prediction Report (EpiSuite error sorted)"
        
    doc = document(title=title)
    with doc.head:
        meta(charset="utf-8")
        style(css)

    with doc:
        with div(_class="content-wrap"):
            h2(title)

            # Main grid table WITHOUT header
            grid_tbl = table()
            grid_body = grid_tbl.add(tbody())

            # Build rows of `columns_per_row` cells
            cells_in_row = 0
            current_tr = None

            for _, row in df.iterrows():

                current_split = row.get("split")
                
                if split is not None and split != current_split:
                    continue
                
                if current_tr is None or cells_in_row == columns_per_row:
                    current_tr = grid_body.add(tr())
                    cells_in_row = 0

                smi = "" if pd.isna(row.get(smiles_col)) else str(row.get(smiles_col))
                b64 = iu.smiles_png_b64_indigo(smi, width=width*2, height=height*2, trim=True)

                cell_td = current_tr.add(td(_class="smiles-col"))
                with cell_td.add(div(_class="smiles-cell")):
                    # Image / placeholder
                    if b64:
                        img(src=f"data:image/png;base64,{b64}", alt=smi)
                    else:
                        div("N/A", _class="placeholder")
                    # SMILES text
                    div(smi, _class="smiles-text")

                    # Nested mini-table: columns (header + one row)
                    mini = table(_class="mini-table")
                    # Header row
                    with mini.add(thead()).add(tr()):
                        for label, _src in mini_columns:
                            th(label)
                    # Data row
                    mini_body = mini.add(tbody())
                    with mini_body.add(tr()):
                        for label, src in mini_columns:
                            if label == "split":
                                val = str(row.get("split", ""))
                                td(val)
                            elif label == "exp":
                                td(fmt_num(row.get(exp_col)), _class="num")
                            else:
                                td(fmt_num(row.get(src)), _class="num")

                    # Optional: EpiSuite |err| as an extra column in the mini-table
                    if include_abs_err:
                        # Append an extra column (header + value) at the end
                        # Rebuild the header row with an extra th would complicate markup;
                        # simplest is to add a second mini-table row below
                        with mini_body.add(tr()):
                            td("EpiSuite |err|")
                            td(fmt_num(row.get("episuite_abs_err")), _class="num")

                cells_in_row += 1

            # Optionally, you could pad the last row with empty cells so every row has equal columns:
            # while cells_in_row and cells_in_row < columns_per_row:
            #     current_tr.add(td())
            #     cells_in_row += 1

    html_str = doc.render()

    if out_html_path:
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    return html_str


def build_all_methods_dom_html_old(
    big_df: pd.DataFrame,
    out_html_path=None,
    smiles_col="canon_qsar_smiles",
    exp_col="exp",
    pred_cols=None,           # auto-detect if None (columns starting with 'pred_')
    width=240,
    height=180,
    max_rows=None,
    include_abs_err=False,    # hidden by default
    cell_padding=8,
    table_max_width=None,
):
    """
    Build one HTML table with exp and predictions from all methods in big_df.
    Rows are ordered by EpiSuite absolute error (descending) if a 'pred_EpiSuite' (or case-insensitive match) column is present.
    Error columns are hidden unless include_abs_err=True.
    """
    df = big_df.copy()

    # Choose prediction columns
    if pred_cols is None:
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not pred_cols:
        raise ValueError("No prediction columns found. Expected columns named 'pred_<method>'.")

    # Normalize numerics
    df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
    for c in pred_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Identify EpiSuite prediction column (case-insensitive)
    episuite_col = next((c for c in pred_cols if "episuite" in c.lower()), None)

    # Order by EpiSuite absolute error if available
    if episuite_col is not None:
        df["_sort_key"] = (df[episuite_col] - df[exp_col]).abs()
        df = df.sort_values("_sort_key", ascending=False, na_position="last")
        pred_cols = [episuite_col] + [c for c in pred_cols if c != episuite_col]
    else:
        df["_sort_key"] = 0  # keep input order

    if max_rows is not None:
        df = df.head(max_rows)

    def fmt_num(x):
        return "" if pd.isna(x) else f"{x:.2f}"

    # Labels for headers
    method_labels = {c: c.replace("pred_", "") for c in pred_cols}

    iu = IndigoUtils()

    # CSS
    smiles_col_px = width + 2 * cell_padding
    css = build_prediction_table_css(
        cell_padding=cell_padding,
        width=width,
        height=height,
        smiles_col_px=smiles_col_px,
        table_max_width=table_max_width,
    )

    title = "All Methods Prediction Report (ordered by EpiSuite error)"
    doc = document(title=title)
    with doc.head:
        meta(charset="utf-8")
        style(css)

    with doc:
        with div(_class="content-wrap"):
            h2("Predicted vs. Experimental (ordered by EpiSuite |pred - exp|)")
            tbl = table()
            with tbl.add(thead()).add(tr()):
                th("Structure", _class="smiles-col")
                th("exp")
                for c in pred_cols:
                    th(method_labels[c])
                    if include_abs_err:
                        th(f"|err| {method_labels[c]}")

            body = tbl.add(tbody())

            for _, row in df.iterrows():
                smi = "" if pd.isna(row.get(smiles_col)) else str(row.get(smiles_col))
                exp_val = row.get(exp_col)


                b64 = iu.smiles_png_b64_indigo(smi, width=width, height=height, trim=True)

                # print (smi, b64)

                with body.add(tr()):
                    smiles_td = td(_class="smiles-col")
                    with smiles_td.add(div(_class="smiles-cell")):
                        if b64:
                            img(src=f"data:image/png;base64,{b64}", width=str(width), height=str(height), alt=smi)
                        else:
                            div("N/A", _class="placeholder")
                        div(smi, _class="smiles-text")

                    td(fmt_num(exp_val), _class="num")

                    for c in pred_cols:
                        pred_val = row.get(c)
                        td(fmt_num(pred_val), _class="num")
                        if include_abs_err:
                            err_val = abs((pred_val or float('nan')) - (exp_val or float('nan'))) if pd.notna(pred_val) and pd.notna(exp_val) else float('nan')
                            td(fmt_num(err_val), _class="num")

    html_str = doc.render()

    if out_html_path:
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    return html_str


def build_episuite_dom_html(
    method=None,
    df_episuite=None,
    out_html_path=None,
    smiles_col="smiles",
    pred_col="pred",
    exp_col="exp",
    width=240,
    height=180,
    max_rows=None,
    cell_padding=8,
    table_max_width=None
):
    """
    Build a single-method HTML table: adds abs_err = |pred - exp|, sorts by abs_err desc.
    """
    title = method.upper() + " Error Report"

    df = df_episuite.copy()
    df[pred_col] = pd.to_numeric(df[pred_col], errors="coerce")
    df[exp_col] = pd.to_numeric(df[exp_col], errors="coerce")
    df["abs_err"] = (df[pred_col] - df[exp_col]).abs()
    df_sorted = df.sort_values("abs_err", ascending=False, na_position="last")
    if max_rows is not None:
        df_sorted = df_sorted.head(max_rows)

    iu = IndigoUtils()

    smiles_col_px = width + 2 * cell_padding
    css = build_prediction_table_css(
        cell_padding=cell_padding,
        width=width,
        height=height,
        smiles_col_px=smiles_col_px,
        table_max_width=table_max_width,
    )

    doc = document(title=title)
    with doc.head:
        meta(charset="utf-8")
        style(css)

    with doc:
        with div(_class="content-wrap"):
            h2(method + " Predicted vs. Experimental Value (sorted by |pred - exp| desc)")

            tbl = table()
            with tbl.add(thead()).add(tr()):
                th("Structure", _class="smiles-col")
                th("exp")
                th("pred")
                th("|pred - exp|")

            body = tbl.add(tbody())

            def fmt_num(x):
                return "" if pd.isna(x) else f"{x:.2f}"

            for _, row in df_sorted.iterrows():
                smi = "" if pd.isna(row.get(smiles_col)) else str(row.get(smiles_col))
                pred_val = row.get(pred_col)
                exp_val = row.get(exp_col)
                abs_err_val = row.get("abs_err")

                b64 = iu.smiles_png_b64_indigo(smi, width=width, height=height, trim=True)

                with body.add(tr()):
                    smiles_td = td(_class="smiles-col")
                    with smiles_td.add(div(_class="smiles-cell")):
                        if b64:
                            img(src=f"data:image/png;base64,{b64}", width=str(width), height=str(height), alt=smi)
                        else:
                            div("N/A", _class="placeholder")
                        div(smi, _class="smiles-text")

                    td(fmt_num(exp_val), _class="num")
                    td(fmt_num(pred_val), _class="num")
                    td(fmt_num(abs_err_val), _class="num")

    html_str = doc.render()

    if out_html_path:
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write(html_str)

    return html_str


def compile_koc_csv(
    input_path: str = "api_responses.ndjson",
    output_csv: str = "compiled_koc.csv"
) -> Tuple[int, List[Dict[str, Optional[float]]]]:
    """
    From NDJSON records, extract smiles, pred (logKoc model estimate), and exp (selectedValue if not ESTIMATED).
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
                continue

            smiles = rec.get("smiles")
            response = rec.get("response")
            if not isinstance(response, dict):
                continue

            logkoc = response.get("logKoc")
            if not isinstance(logkoc, dict):
                continue

            pred = to_float(logkoc.get("estimatedValue", {}).get("model", {}).get("logKoc"))

            exp: Optional[float] = None
            selected = logkoc.get("selectedValue", {})
            selected = selected if isinstance(selected, dict) else {}
            value_type = selected.get("valueType")
            if isinstance(value_type, str) and value_type.strip().upper() != "ESTIMATED":
                exp = to_float(selected.get("value"))

            if smiles is not None and pred is not None:
                rows.append({"smiles": str(smiles), "pred": pred, "exp": exp})

    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles", "pred", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows


def compile_ecosar_csv(
    input_path: str = "api_responses.ndjson",
    output_csv: str = "compiled_ecosar.csv"
) -> Tuple[int, List[Dict[str, Optional[float]]]]:
    """
    From NDJSON records, extract smiles, pred (ECOSAR derived), and exp (if found).
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
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            smiles = rec.get("smiles")
            response = rec.get("response")
            if not isinstance(response, dict):
                continue

            ecosar = response.get("ecosar")
            if not isinstance(ecosar, dict):
                continue

            modelResults = ecosar.get("modelResults")
            if not isinstance(modelResults, list):
                continue

            pred = None
            for mr in modelResults:
                if mr.get("organism") != "Fish":
                    continue
                if mr.get("endpoint") != "LC50":
                    continue
                if mr.get("duration") != "96-hr":
                    continue
                # mg/L -> g/L -> mol/L -> -logM
                predNew = mr.get("concentration")
                if predNew is None:
                    continue
                try:
                    mw = response.get("chemicalProperties", {}).get("molecularWeight")
                    predNew = float(predNew) / 1000.0 / float(mw)
                    predNew = -math.log10(predNew)
                except Exception:
                    continue
                if pred is None or predNew > pred:
                    pred = predNew

            # Experimental extraction placeholder (none found in this example)
            exp = None

            if smiles is not None and pred is not None:
                rows.append({"smiles": str(smiles), "pred": pred, "exp": exp})

    with open(output_csv, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["smiles", "pred", "exp"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), rows




def run_episuite(
    max_cols_per_row: int = 3,
    img_min_height_px: int = 400,
    row_gap_px: int = 24,
    force_rerun: bool = False,    # set True to force EpiSuite submit/compile
):
    def pred_csv_path(embedding: str, split: str) -> Path:
        return base_dir / embedding / f"{split} set predictions.csv"

    def episuite_json_path(split: str) -> Path:
        return episuite_dir / f"{split} set predictions episuite.json"

    def episuite_csv_path(split: str) -> Path:
        return episuite_json_path(split).with_suffix(".csv")

    url = "https://episuite.app/EpiWebSuite/api/submit"
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")

    dataset_name = "KOC v1 modeling"
    property_name = "Koc"

    # Examples:
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3 modeling'
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    # property = "96 hr Fish LC50"

    base_dir = Path(PROJECT_ROOT) / "data" / "models" / dataset_name
    episuite_dir = base_dir / "episuite"
    episuite_dir.mkdir(parents=True, exist_ok=True)

    base_embedding_for_df_test = "gcm_WebTEST-default_fs=False"
    other_methods = ["rf", "xgb", "reg", "knn"]

    # Build splits list without duplicates
    splits = ["test"]
    if "KOC" in dataset_name:
        splits.append("external")
    # de-dup while preserving order
    splits = list(dict.fromkeys(splits))

    summaries = {split: [] for split in splits}
    compiled_all = []

    for split in splits:
        file_path_csv = pred_csv_path(base_embedding_for_df_test, split)
        df_test = pd.read_csv(file_path_csv)

        output_path_json = episuite_json_path(split)
        csv_output_path_episuite = episuite_csv_path(split)

        # Submit/compile only if needed (or forced)
        if force_rerun or not output_path_json.exists() or not csv_output_path_episuite.exists():
            run_episuite_csv(url, df_test, str(output_path_json))
            if "KOC" in dataset_name:
                compile_koc_csv(str(output_path_json), str(csv_output_path_episuite))
            elif "Fish" in dataset_name:
                compile_ecosar_csv(str(output_path_json), str(csv_output_path_episuite))
            else:
                print("handle compilation of data for " + dataset_name)
                return
        else:
            print(f"Skipping EpiSuite submit/compile for '{split}' (existing outputs found).")

        # EpiSuite
        df_episuite, number, mae, filePathOutScatter, limits = go_through_episuite_results(
            str(csv_output_path_episuite), str(file_path_csv), split, property_name
        )
        summaries[split].append({"method": "EpiSuite", "n": number, "mae": mae, "img": filePathOutScatter})

        # GCM baseline
        method = "gcm"
        df_gcm, number, mae, filePathOutScatter = go_through_test_results(
            method, str(pred_csv_path(base_embedding_for_df_test, split)), df_episuite, split, limits, property_name
        )
        summaries[split].append({"method": method, "n": number, "mae": mae, "img": filePathOutScatter})

        # Other methods
        method_frames = {"gcm": df_gcm}
        for method in other_methods:
            embedding = f"{method}_WebTEST-default_fs=True"
            df_subset, number, mae, filePathOutScatter = go_through_test_results(
                method, str(pred_csv_path(embedding, split)), df_episuite, split, limits, property_name
            )
            summaries[split].append({"method": method.upper(), "n": number, "mae": mae, "img": filePathOutScatter})
            method_frames[method] = df_subset

        # ==== MERGE per split ====
        if "exp" in df_episuite.columns:
            exp_col = "exp"
        elif property in df_episuite.columns:
            exp_col = property
        else:
            raise KeyError("Experimental column not found in df_episuite (expected 'exp' or property).")

        base = (
            df_episuite[["canon_qsar_smiles", exp_col]]
            .rename(columns={exp_col: "exp"})
            .drop_duplicates(subset=["canon_qsar_smiles"])
        )
        base["split"] = split

        merged = base.copy()
        if "pred" in df_episuite.columns:
            merged = merged.merge(
                df_episuite[["canon_qsar_smiles", "pred"]].rename(columns={"pred": "pred_EpiSuite"}),
                on="canon_qsar_smiles",
                how="left",
            )

        for m, df_m in method_frames.items():
            if "pred" not in df_m.columns:
                continue
            df_m_small = (
                df_m[["canon_qsar_smiles", "pred"]]
                .drop_duplicates(subset=["canon_qsar_smiles"])
                .rename(columns={"pred": f"pred_{m}"})
            )
            merged = merged.merge(df_m_small, on="canon_qsar_smiles", how="left")

        compiled_all.append(merged)

    # ===== After all splits: build once =====
    big_df = pd.concat(compiled_all, ignore_index=True)

    for split in splits:
        
        out_html_path = episuite_dir / f"{split}_all_methods_predictions.html"
        build_all_methods_dom_html(
            property_name=property_name,
            big_df=big_df,
            out_html_path=out_html_path,
            smiles_col="canon_qsar_smiles",
            exp_col="exp",
            width=img_min_height_px*4/3,
            height=img_min_height_px,
            table_max_width=1200,
            columns_per_row=2,        # 3 cells per grid row
            include_abs_err=False,    # keep the mini-table minimal
            split = split
        )
                    

    # Compact scatter-plot HTML per split
    for split in splits:
        build_split_html(
            episuite_dir,
            property,
            split,
            summaries[split],
            columns_per_row=max_cols_per_row,
            img_h_px=img_min_height_px,
            v_gap_px=row_gap_px,
        )

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    run_episuite()
