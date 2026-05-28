'''
Created on Mar 2, 2026

@author: TMARTI02
'''

import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from models.db_utilities.dataset_utilities_db import get_instances, get_training_ids, getDatapointsLookup
from utils import print_first_row
from datetime import datetime
from sklearn.model_selection import KFold, StratifiedKFold
from sqlalchemy import text, bindparam
from pathlib import Path
import os
from typing import Dict, Optional, Tuple
from util.database_utilities import getSession

    
    
# @deprecated("use make_representative_and_inner_cv_splits")   
def find_representative_cv_splits(datasetName, descriptorSetName, user, n_splits=5, 
                              random_state=0, write_to_db=False, shuffle=True):
    
    print('enter find_representative_cv_splits')

    session = getSession()
    # data = getDatapoints(session, datasetName, descriptorSetName) # all the datapoints no splitting
    
    import util.predict_constants as pc
    
    splittingName = pc.SPLITTING_RND_REPRESENTATIVE

    df = get_training_ids(session, datasetName, descriptorSetName,splittingName=splittingName)
    
    # print(df.shape)
    # print_first_row(df)
    
    required = {'fk_data_point_id', 'qsar_property_value'}
    if not required.issubset(df.columns):
        raise ValueError("df must contain 'fk_data_point_id' and 'qsar_property_value' columns")

    # Ensure integer fk_data_point_id values
    ids_num = pd.to_numeric(df['fk_data_point_id'], errors='coerce')
    if ids_num.isna().any():
        raise ValueError("Found null/invalid 'fk_data_point_id' values")

    if not np.all(np.isclose(ids_num.values, np.round(ids_num.values))):
        raise ValueError("Non-integer 'fk_data_point_id' values detected (found fractional parts)")

    ids_int = ids_num.astype(np.int64).to_numpy()

    # Target
    y = df['qsar_property_value'].to_numpy()
    n = len(df)

    # Choose CV strategy
    is_binary = pd.Series(y).isin([0, 1]).all()
    if is_binary:
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_iter = splitter.split(np.zeros(n), y)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_iter = splitter.split(np.zeros(n), None)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(np.zeros(n), None)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        # Training rows
        for idx in train_idx:
            rows.append({'fk_data_point_id': int(ids_int[idx]), 'fold': fold_idx, 'split_num': 0})
        # Prediction (test) rows
        for idx in test_idx:
            rows.append({'fk_data_point_id': int(ids_int[idx]), 'fold': fold_idx, 'split_num': 1})

    assign_df = pd.DataFrame(rows, columns=['fk_data_point_id', 'fold', 'split_num'])
    assign_df['fk_data_point_id'] = assign_df['fk_data_point_id'].astype('int64')
    assign_df['split_num'] = assign_df['split_num'].astype('int8')
    
    created_at = datetime.now()
    
    merged = (
        assign_df
        .assign(
            created_at=created_at,
            updated_at=created_at,
            created_by=user,
            updated_by=user,
        )
        .assign(fk_splitting_id=lambda d: (d['fold'] + 2).astype('int64')) # add 2 to get the right foreign key from the fold #
        .drop(columns=['fold'])
    )

    print_first_row(merged)
    print(merged.shape)
    
    records = merged.replace({np.nan: None}).to_dict(orient="records")

    # for record in records:
    #     print(record["fk_data_point_id"], record["fk_splitting_id"])
    # print(records)
    
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None) # make a new session so will commit correctly
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Count splittings loaded for RND_REPRESENTATIVE_CV# splittings: {count}")        


    


    
    
    
# @deprecated("use make_representative_and_inner_cv_splits")
def find_representative_split(datasetName, descriptorSetName, remove_log_p_descriptors, user, n_threads, n_splits=5, 
                              random_state=0, write_to_db=False):
    
    print('enter find_representative_split')

    session = getSession()
    data = get_instances(session, datasetName, descriptorSetName) # all the datapoints no splitting
    # print_first_row(data)

    print(f"Shape of input df:{data.shape}")


    # Columns: col0 = ID, col1 = label/target, remaining = features
    y = data[data.columns[1]].to_numpy()
    X = data.drop(columns=[data.columns[0], data.columns[1]])
    ids = data[data.columns[0]].to_numpy()

    # Optionally remove logP descriptors from the feature set
    if remove_log_p_descriptors:
        X = X.loc[:, ~X.columns.str.contains(r'log[_ ]?p', case=False, regex=True)]

    # Binary classification if labels are strictly in {0, 1}; otherwise regression
    is_binary = pd.Series(y).isin([0, 1]).all()
    print(f"is_binary:{is_binary}")

    # Choose CV strategy
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if is_binary else
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    # Choose a baseline estimator (replace with your own if needed)
    if is_binary:
        base_model = RandomForestClassifier(
            n_estimators=100, min_impurity_decrease=1e-5,
            random_state=random_state, n_jobs=n_threads, class_weight='balanced'
        )
    else:
        base_model = RandomForestRegressor(
            n_estimators=100, min_impurity_decrease=1e-5,
            random_state=random_state, n_jobs=n_threads
        )

    # Build folds using sklearn and run CV
    folds = list(splitter.split(X, y if is_binary else None))
    scores = []
    y_pred = np.empty_like(y, dtype=np.float64)  # holds out-of-fold predictions

    print(f"Fold\tRMSE\tMAE")
    for i, (train_idx, test_idx) in enumerate(folds):
        model = clone(base_model)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[test_idx])
        y_pred[test_idx] = preds

        if is_binary:
            scores.append(balanced_accuracy_score(y[test_idx], preds))
        else:
            rmse = root_mean_squared_error(y[test_idx], preds)
            mae = mean_absolute_error(y[test_idx], preds)
            scores.append(rmse)
            print(f"{i+1}\t{rmse:.3f}\t{mae:.3f}")

    # Compute pooled score across all out-of-fold predictions
    if is_binary:
        pooled_score = balanced_accuracy_score(y, y_pred)
    else:
        pooled_score = root_mean_squared_error(y, y_pred)

    # Find the fold whose score is closest to pooled score
    scores_arr = np.asarray(scores)
    representative_split = int(np.argmin(np.abs(scores_arr - pooled_score)))
    prediction_idx = folds[representative_split][1]  # test indices for representative split

    metric_name = "balanced accuracy" if is_binary else "RMSE"
    print(f"\npooled {metric_name} over all {n_splits} folds = {pooled_score}")
    print("The split with the most representative score uses fold " +
          str(representative_split + 1) + " for prediction set")

    # Build results aligned to original row order
    # t_p: 0 for training, 1 for prediction
    t_p = np.zeros(len(y), dtype=np.int8)
    t_p[prediction_idx] = 1

    results = pd.DataFrame({
        'qsar_smiles': ids,
        'exp': y,
        'pred': y_pred,
        'split_num': t_p
    })
    
    df_lookup = getDatapointsLookup(session, datasetName)
    # print_first_row(df_lookup)
    # print(df_lookup.shape)

    merged = (
        results[["qsar_smiles", "split_num"]]
        .merge( df_lookup[["qsar_smiles", "fk_data_point_id"]], on="qsar_smiles", how="inner")
        [["fk_data_point_id", "split_num"]]
    )

    # print(merged.to_dict(orient="records"))
    
    created_at = datetime.now()
    
    # Inject uniform constants; updated_at uses the same value as created_at
    merged = merged.assign(
        created_at=created_at,
        updated_at=created_at,
        created_by=user,
        updated_by=user,
        fk_splitting_id=1
    )

    print_first_row(merged)
    print(merged.shape)
    
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None) # make a new session so will commit correctly
        records = merged.replace({np.nan: None}).to_dict(orient="records")
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Count splittings loaded for RND_REPRESENTATIVE splitting: {count}")        
        
    
    
    
def preview_and_delete_splittings_by_dataset(
    dataset_name,
    split_ids=None,      # e.g., [1] or [2,3,4,5,6]; if None, target all split IDs for the dataset
    head_n=50,           # how many rows to print in preview
    dry_run=True,        # True = preview only; False = perform DELETE
):
    """
    Preview and optionally delete rows in qsar_datasets.data_points_in_splittings for a given dataset,
    using a JOIN on datasets -> data_points -> data_points_in_splittings.

    Arguments:
      - dataset_name: exact name in qsar_datasets.datasets.name
      - split_ids: optional list of fk_splitting_id values to restrict the operation
      - head_n: number of rows to show in preview
      - dry_run: if True, only preview; if False, perform the deletion (irreversible)

    Returns dict with counts and, if deleted, the reported rowcount.
    """
    session = getSession()

    # Build optional filter for split_ids
    filter_split = ""
    params = {"dataset_name": dataset_name}
    if split_ids is not None and len(split_ids) > 0:
        filter_split = " AND dpis.fk_splitting_id IN :split_ids"
        params["split_ids"] = split_ids
        split_bind = {"split_ids": bindparam("split_ids", expanding=True)}
    else:
        split_bind = {}

    # 1) Count
    count_sql = text(f"""
        SELECT COUNT(dpis.id)
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp
          ON d.id = dp.fk_dataset_id
        JOIN qsar_datasets.data_points_in_splittings dpis
          ON dpis.fk_data_point_id = dp.id
        WHERE d.name = :dataset_name
        {filter_split}
    """).bindparams(**split_bind)

    total = session.execute(count_sql, params).scalar() or 0
    print(f"Total matching rows for dataset '{dataset_name}'"
          f"{' and split_ids=' + str(split_ids) if split_ids else ''}: {total}")

    # 2) Preview rows
    if head_n and head_n > 0:
        preview_sql = text(f"""
            SELECT dpis.id,
                   dpis.fk_data_point_id,
                   dpis.fk_splitting_id,
                   dpis.split_num,
                   dpis.created_at,
                   dpis.created_by
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp
              ON d.id = dp.fk_dataset_id
            JOIN qsar_datasets.data_points_in_splittings dpis
              ON dpis.fk_data_point_id = dp.id
            WHERE d.name = :dataset_name
            {filter_split}
            ORDER BY dpis.fk_splitting_id, dpis.split_num, dpis.fk_data_point_id
            LIMIT :head_n
        """).bindparams(**split_bind)

        for row in session.execute(preview_sql, {**params, "head_n": head_n}):
            print(
                f"id={row.id}, "
                f"fk_data_point_id={row.fk_data_point_id}, "
                f"fk_splitting_id={row.fk_splitting_id}, "
                f"split_num={row.split_num}, "
                f"created_at={row.created_at}, "
                f"created_by={row.created_by}"
            )

    if dry_run:
        print("Dry run: no deletions performed.")
        return {
            "dataset_name": dataset_name,
            "split_ids": split_ids,
            "total_matching": int(total),
            "deleted": 0
        }

    # 3) DELETE with JOIN (PostgreSQL USING clause)
    delete_sql = text(f"""
        DELETE FROM qsar_datasets.data_points_in_splittings dpis
        USING qsar_datasets.data_points dp,
              qsar_datasets.datasets d
        WHERE dpis.fk_data_point_id = dp.id
          AND dp.fk_dataset_id = d.id
          AND d.name = :dataset_name
          {filter_split}
    """).bindparams(**split_bind)

    res = session.execute(delete_sql, params)
    session.commit()

    # rowcount is typically supported on Postgres; fallback to recount if needed
    deleted = res.rowcount if (res.rowcount is not None and res.rowcount >= 0) else None
    if deleted is None:
        # Fallback recount after delete
        remaining = session.execute(count_sql, params).scalar() or 0
        deleted = int(total) - int(remaining)

    print(f"Deleted {deleted} rows for dataset '{dataset_name}'.")
    return {
        "dataset_name": dataset_name,
        "split_ids": split_ids,
        "total_matching": int(total),
        "deleted": int(deleted)
    }
    

# ---- Helpers ----
def build_splitter(y, n_splits, shuffle, random_state):
    # Binary classification if strictly {0,1}; else regression
    is_binary = pd.Series(y).isin([0, 1]).all()
    if is_binary:
        classes, counts = np.unique(y, return_counts=True)
        # Fallback to KFold if not enough samples per class for StratifiedKFold
        if counts.min() >= n_splits:
            return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), True
        else:
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), True
    else:
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), False


def rows_for_assignment(idx_array: np.ndarray, split_num: int, fk_splitting_id: int):
    return pd.DataFrame(
        {
            "fk_data_point_id": idx_array.astype("int64"),
            "split_num": np.full(len(idx_array), split_num, dtype="int8"),
            "fk_splitting_id": np.full(len(idx_array), fk_splitting_id, dtype="int64"),
        }
    )



def fetch_smiles_for_dataset(session, dataset_name, descriptor_set_name='WebTEST-default'):
    
    """
    Only includes rows present in the given descriptor set.
    """
    sql = text("""
        SELECT dp.canon_qsar_smiles
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
        JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
        JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
        WHERE d.name = :datasetName
          AND ds.name = :descriptorSetName
        ORDER BY dp.id
    """)
    rows = session.execute(
        sql,
        {
            "datasetName": dataset_name,
            "descriptorSetName": descriptor_set_name,
        },
    ).fetchall()

    if not rows:
        raise ValueError(
            f"No rows found for dataset='{dataset_name}', descriptorSet='{descriptor_set_name}'"
        )

    smiles_list = [r[0] for r in rows]
    return pd.DataFrame({"canon_qsar_smiles": smiles_list})



def print_class_balance_for_folds(y, folds, title="Outer"):
    """
    Print per-fold class balance for a list of (train_idx, test_idx) folds.

    Parameters
    ----------
    y : array-like
        The target labels (classification).
    folds : list[tuple[np.ndarray, np.ndarray]]
        A list of (train_idx, test_idx) index arrays.
    title : str
        A label to print (e.g., "Outer" or "Inner").
    """
    def counts(arr):
        u, c = np.unique(arr, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))
    
    def fmt_line(counts_dict, total):
        parts = []
        for cls in sorted(counts_dict, key=lambda x: str(x)):
            cnt = counts_dict[cls]
            perc = 100.0 * cnt / total if total else 0.0
            parts.append(f"{cls}:{cnt} ({perc:.1f}%)")
        return " | ".join(parts)

    y = np.asarray(y)
    overall = counts(y)
    print(f"{title} - overall: n={len(y)} -> {fmt_line(overall, len(y))}")
    for i, (tr_idx, te_idx) in enumerate(folds):
        tr_counts = counts(y[tr_idx])
        te_counts = counts(y[te_idx])
        print(f"{title} fold {i}:")
        print(f"  Train n={len(tr_idx)}: {fmt_line(tr_counts, len(tr_idx))}")
        print(f"  Test  n={len(te_idx)}: {fmt_line(te_counts, len(te_idx))}")


def make_representative_and_inner_cv_splits(
    datasetName,
    datasetName2,
    descriptorSetName,
    remove_log_p_descriptors,
    user,
    n_outer_splits=5,
    n_inner_splits=5,
    random_state=0,
    n_threads=None,
    write_to_db=False,
    shuffle=True,
):
    """
    Computes:
      1) A representative outer split (1 fold used as prediction/holdout) chosen by closeness to pooled OOF performance.
      2) Inner CV folds on the remaining training portion.

    Writes both assignments in one go if write_to_db=True.

    Produces rows:
      fk_data_point_id, split_num (0=train, 1=prediction), fk_splitting_id
      - fk_splitting_id == 1 for the representative outer split
      - fk_splitting_id == 2 + inner_fold_idx for the inner CV folds

    Additionally:
      - If datasetName2 is provided, clone the same split assignments (by qsar_smiles) to datasetName2,
        writing a separate splittings.xlsx for datasetName2 (and optional DB writes).
    """
    
    # ---- Helpers (new) ----
    def make_assignment_blueprint(ids_local, rep_train_idx_local, rep_test_idx_local, inner_folds_abs_local):
        """
        Build a "blueprint" DataFrame of assignments keyed by qsar_smiles.
        Columns: ['qsar_smiles', 'split_num', 'fk_splitting_id'].
        """
        smiles = np.asarray(ids_local)

        # Outer representative split (fk_splitting_id=1)
        outer_df = pd.concat(
            [
                pd.DataFrame({
                    "qsar_smiles": smiles[rep_train_idx_local],
                    "split_num": 0,
                    "fk_splitting_id": 1
                }),
                pd.DataFrame({
                    "qsar_smiles": smiles[rep_test_idx_local],
                    "split_num": 1,
                    "fk_splitting_id": 1
                }),
            ],
            ignore_index=True
        )

        # Inner folds (fk_splitting_id=2+idx)
        inner_rows = []
        if inner_folds_abs_local is not None and len(inner_folds_abs_local) > 0:
            for inner_fold_idx, (inner_tr_abs, inner_te_abs) in enumerate(inner_folds_abs_local):
                inner_rows.append(pd.DataFrame({
                    "qsar_smiles": smiles[inner_tr_abs],
                    "split_num": 0,
                    "fk_splitting_id": 2 + inner_fold_idx
                }))
                inner_rows.append(pd.DataFrame({
                    "qsar_smiles": smiles[inner_te_abs],
                    "split_num": 1,
                    "fk_splitting_id": 2 + inner_fold_idx
                }))
        inner_df = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame(columns=["qsar_smiles", "split_num", "fk_splitting_id"])

        return pd.concat([outer_df, inner_df], ignore_index=True)

    def map_blueprint_to_dataset_rows(session_local, dataset_name_local, blueprint_df, user_local, created_at_local):
        """
        Map the blueprint assignments (by qsar_smiles) to fk_data_point_id for the given dataset.
        Returns a DataFrame with columns:
            ['fk_data_point_id', 'split_num', 'fk_splitting_id', 'created_at', 'updated_at', 'created_by', 'updated_by'].
        """
        df_lookup_local = getDatapointsLookup(session_local, dataset_name_local)  # must contain 'qsar_smiles', 'fk_data_point_id'
        rows_local = (
            blueprint_df.merge(df_lookup_local[["qsar_smiles", "fk_data_point_id"]], on="qsar_smiles", how="inner")
                        .drop(columns=["qsar_smiles"])
                        .drop_duplicates(["fk_data_point_id", "fk_splitting_id", "split_num"])
                        .reset_index(drop=True)
        )
        # Add audit fields
        rows_local = rows_local.assign(
            created_at=created_at_local,
            updated_at=created_at_local,
            created_by=user_local,
            updated_by=user_local
        )
        # Ensure types
        rows_local["fk_data_point_id"] = rows_local["fk_data_point_id"].astype("int64")
        rows_local["split_num"] = rows_local["split_num"].astype("int64")
        rows_local["fk_splitting_id"] = rows_local["fk_splitting_id"].astype("int64")
        return rows_local

    def compute_df_targets_for_dataset(session_local, dataset_name_local, descriptor_set_name_local, remove_log_p_local):
        """
        Obtain y targets keyed by fk_data_point_id for a given dataset, to enable class-balance tabs.
        Returns a DataFrame with ['fk_data_point_id', 'y'].
        """
        df_instances_local = get_instances(session_local, dataset_name_local, descriptor_set_name_local)
        X_l, y_l, ids_l = get_X_y_ids(df_instances_local, remove_log_p_local)
        df_lookup_local = getDatapointsLookup(session_local, dataset_name_local)
        map_df_local = pd.DataFrame({"qsar_smiles": ids_l})
        map_df_local = map_df_local.merge(df_lookup_local[["qsar_smiles", "fk_data_point_id"]], on="qsar_smiles", how="inner")
        df_targets_local = pd.DataFrame(
            {
                "fk_data_point_id": pd.Series(map_df_local["fk_data_point_id"].astype("int64").to_numpy(), dtype="int64"),
                "y": y_l[: len(map_df_local)],  # align to merged rows
            }
        )
        return df_targets_local

    def write_assignments_excel(base_dir_root, dataset_name_local, rows_local, is_binary_local, df_targets_local):
        """
        Write the assignments (and class-balance tabs if classification) to models/<dataset>/splittings.xlsx.
        """
        excel_path_local = Path(base_dir_root) / "data" / "models" / dataset_name_local / "splittings.xlsx"
        excel_path_local.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(excel_path_local, engine="openpyxl") as writer:
            rows_local.to_excel(writer, sheet_name="assignments", index=False)

            if is_binary_local and not rows_local.empty and df_targets_local is not None and not df_targets_local.empty:
                merged_local = rows_local[["fk_data_point_id", "split_num", "fk_splitting_id"]] \
                    .merge(df_targets_local, on="fk_data_point_id", how="left")
                for sid in sorted(merged_local["fk_splitting_id"].unique()):
                    df_sid = merged_local.loc[merged_local["fk_splitting_id"] == sid].copy()
                    if df_sid["y"].isna().all():
                        continue
                    counts = (
                        df_sid.groupby(["split_num, y".split(", ")[0], "y"])
                              .size()
                              .rename("count")
                              .reset_index()
                    )
                    counts["percent"] = counts["count"] / counts.groupby("split_num")["count"].transform("sum")
                    counts = counts.sort_values(by=["split_num", "y"]).reset_index(drop=True)
                    sheet_name = f"class_balance_fk{sid}"
                    counts.to_excel(writer, sheet_name=sheet_name, index=False)

        print(excel_path_local)
        return excel_path_local

    # ---- Helpers (existing/original) ----
    def build_model(is_binary, n_threads, random_state):
        if is_binary:
            return RandomForestClassifier(
                n_estimators=100,
                min_impurity_decrease=1e-5,
                random_state=random_state,
                n_jobs=n_threads,
                class_weight='balanced'
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                min_impurity_decrease=1e-5,
                random_state=random_state,
                n_jobs=n_threads
            )

    def get_X_y_ids(df_overall_set, remove_log_p):
        y_local = df_overall_set[df_overall_set.columns[1]].to_numpy()
        X_local = df_overall_set.drop(columns=[df_overall_set.columns[0], df_overall_set.columns[1]])
        ids_local = df_overall_set[df_overall_set.columns[0]].to_numpy()
        if remove_log_p:
            X_local = X_local.loc[:, ~X_local.columns.str.contains(r'log[_ ]?p', case=False, regex=True)]
        return X_local, y_local, ids_local

    def get_set_aside_smiles_once(session, datasetName2_local):
        if datasetName2_local is None:
            return None
        df_set_aside = fetch_smiles_for_dataset(session, datasetName2_local)
        return set(df_set_aside['canon_qsar_smiles'].dropna().unique())

    def build_outer_folds_simple(X_local, y_local, n_splits, shuffle_local, random_state_local, is_binary_local):
        outer_splitter_local, _ = build_splitter(y_local, n_splits, shuffle_local, random_state_local)
        return list(outer_splitter_local.split(X_local, y_local if is_binary_local else None))

    def build_outer_folds_with_aside(df_overall, y_local, n_splits, shuffle_local, random_state_local, is_binary_local, set_aside_smiles_local):
        # Partition by whether ID is in set_aside_smiles
        mask = df_overall['ID'].isin(set_aside_smiles_local)
        df_in_set = df_overall.loc[mask].copy()
        df_not_in_set = df_overall.loc[~mask].copy()

        y_in_set = df_in_set[df_in_set.columns[1]].to_numpy()
        y_not_in_set = df_not_in_set[df_not_in_set.columns[1]].to_numpy()

        outer_splitter_in_set, _ = build_splitter(y_in_set, n_splits, shuffle_local, random_state_local)
        outer_splitter_not_in_set, _ = build_splitter(y_not_in_set, n_splits, shuffle_local, random_state_local)

        folds_in_set = list(outer_splitter_in_set.split(
            np.zeros(len(y_in_set)),
            y_in_set if is_binary_local else None
        ))
        folds_not_in_set = list(outer_splitter_not_in_set.split(
            np.zeros(len(y_not_in_set)),
            y_not_in_set if is_binary_local else None
        ))

        pos_in_set = df_overall.index.get_indexer(df_in_set.index)
        pos_not_in_set = df_overall.index.get_indexer(df_not_in_set.index)

        merged_folds = []
        for (tr_in, te_in), (tr_out, te_out) in zip(folds_in_set, folds_not_in_set):
            tr_global = np.concatenate([pos_in_set[tr_in], pos_not_in_set[tr_out]])
            te_global = np.concatenate([pos_in_set[te_in], pos_not_in_set[te_out]])
            merged_folds.append((tr_global, te_global))
        return merged_folds

    def build_inner_folds_simple(y_local, rep_train_idx_local, n_splits, shuffle_local, random_state_local, is_binary_local):
        inner_splitter_local, _ = build_splitter(y_local[rep_train_idx_local], n_splits, shuffle_local, random_state_local)
        inner_folds_abs_local = []
        for inner_tr_rel, inner_te_rel in inner_splitter_local.split(
            np.zeros(len(rep_train_idx_local)),
            y_local[rep_train_idx_local] if is_binary_local else None
        ):
            inner_tr_abs = rep_train_idx_local[inner_tr_rel]
            inner_te_abs = rep_train_idx_local[inner_te_rel]
            inner_folds_abs_local.append((inner_tr_abs, inner_te_abs))
        return inner_folds_abs_local

    def build_inner_folds_with_aside(y_local, rep_train_idx_local, n_splits, shuffle_local, random_state_local, is_binary_local, df_overall, set_aside_smiles_local):
        # Attempt "fancy" inner folds; fall back to simple if too few samples in a subset
        mask_full = df_overall['ID'].isin(set_aside_smiles_local)
        mask_on_train = mask_full.iloc[rep_train_idx_local].to_numpy()

        idx_in_set = rep_train_idx_local[mask_on_train]
        idx_not_in_set = rep_train_idx_local[~mask_on_train]

        can_do_fancy = (
            len(idx_in_set) >= n_splits and
            len(idx_not_in_set) >= n_splits
        )

        if not can_do_fancy:
            return build_inner_folds_simple(y_local, rep_train_idx_local, n_splits, shuffle_local, random_state_local, is_binary_local)

        inner_splitter_in_set, _ = build_splitter(y_local[idx_in_set], n_splits, shuffle_local, random_state_local)
        inner_splitter_not_in_set, _ = build_splitter(y_local[idx_not_in_set], n_splits, shuffle_local, random_state_local)

        folds_in_set = list(
            inner_splitter_in_set.split(
                np.zeros(len(idx_in_set)),
                y_local[idx_in_set] if is_binary_local else None
            )
        )
        folds_not_in_set = list(
            inner_splitter_not_in_set.split(
                np.zeros(len(idx_not_in_set)),
                y_local[idx_not_in_set] if is_binary_local else None
            )
        )

        inner_folds_abs_local = []
        for (tr_in_rel, te_in_rel), (tr_out_rel, te_out_rel) in zip(folds_in_set, folds_not_in_set):
            tr_abs = np.concatenate([idx_in_set[tr_in_rel], idx_not_in_set[tr_out_rel]])
            te_abs = np.concatenate([idx_in_set[te_in_rel], idx_not_in_set[te_out_rel]])
            inner_folds_abs_local.append((tr_abs, te_abs))

        return inner_folds_abs_local

    # ---- Load df_overall_set once ----
    session = getSession()

    df_overall_set = get_instances(session, datasetName, descriptorSetName)
    print(f"Input shape = {df_overall_set.shape}")

    # Build X, y, ids once
    X, y, ids = get_X_y_ids(df_overall_set, remove_log_p_descriptors)

    # Decide task type once (classification vs regression)
    _tmp_splitter, is_binary = build_splitter(y, n_outer_splits, shuffle, random_state)

    # Precompute set_aside_smiles (only once if datasetName2 is provided)
    set_aside_smiles = get_set_aside_smiles_once(session, datasetName2)

    # ---- Build outer folds (refactored) ----
    if datasetName2 is None:
        folds = build_outer_folds_simple(X, y, n_outer_splits, shuffle, random_state, is_binary)
    else:
        folds = build_outer_folds_with_aside(
            df_overall_set, y, n_outer_splits, shuffle, random_state, is_binary, set_aside_smiles
        )
        print('done creating fancy folds')

    base_model = build_model(is_binary, n_threads, random_state)

    # ---- Report outer class balance (classification only) ----
    if is_binary and len(folds) > 0:
        print_class_balance_for_folds(
            y,
            folds,
            title="Outer class balance"
        )

    # ---- Outer CV to pick representative fold ----
    y_oof = np.empty_like(y, dtype=float)
    fold_scores = []

    if is_binary:
        print("fold\tBA")
    else:
        print("fold\tRMSE\tMAE")

    for i, (tr_idx, te_idx) in enumerate(folds):
        m = clone(base_model)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        preds = m.predict(X.iloc[te_idx])
        y_oof[te_idx] = preds

        if is_binary:
            BA = balanced_accuracy_score(y[te_idx], preds)
            print(f"{i}\t{BA:.3f}")
            fold_scores.append(BA)
        else:
            RMSE = root_mean_squared_error(y[te_idx], preds)
            MAE = mean_absolute_error(y[te_idx], preds)
            fold_scores.append(RMSE)
            print(f"{i}\t{RMSE:.3f}\t{MAE:.3f}")

    # Pooled score across all OOF predictions
    pooled = balanced_accuracy_score(y, y_oof) if is_binary else root_mean_squared_error(y, y_oof)

    # Representative fold: closest to pooled
    rep_fold = int(np.argmin(np.abs(np.asarray(fold_scores) - pooled)))
    rep_test_idx = folds[rep_fold][1]
    rep_train_idx = folds[rep_fold][0]

    if is_binary:
        print(f"Best fold:{rep_fold} with BA = {fold_scores[rep_fold]:.3f}, pooled value = {pooled:.3f}")
    else:
        print(f"Best fold:{rep_fold} with RMSE = {fold_scores[rep_fold]:.3f}, pooled value = {pooled:.3f}")

    # ---- Inner CV on training portion only ----
    if n_inner_splits and n_inner_splits > 1 and len(rep_train_idx) >= n_inner_splits:
        # Build inner folds using the same choice as for outer (simple vs fancy), but reuse set_aside_smiles
        if datasetName2 is None:
            inner_folds_abs = build_inner_folds_simple(
                y, rep_train_idx, n_inner_splits, shuffle, random_state, is_binary
            )
        else:
            inner_folds_abs = build_inner_folds_with_aside(
                y, rep_train_idx, n_inner_splits, shuffle, random_state, is_binary,
                df_overall_set, set_aside_smiles
            )

        # Print inner class balance (classification only)
        if is_binary and len(inner_folds_abs) > 0:
            print_class_balance_for_folds(
                y,
                inner_folds_abs,
                title="Inner class balance (within outer-train)"
            )
    else:
        inner_folds_abs = []

    # ---- Build split "blueprint" by qsar_smiles and map to datasets ----
    created_at = datetime.now()

    blueprint = make_assignment_blueprint(
        ids_local=ids,
        rep_train_idx_local=rep_train_idx,
        rep_test_idx_local=rep_test_idx,
        inner_folds_abs_local=inner_folds_abs
    )

    # Map blueprint to datasetName
    rows_dataset1 = map_blueprint_to_dataset_rows(
        session_local=session,
        dataset_name_local=datasetName,
        blueprint_df=blueprint,
        user_local=user,
        created_at_local=created_at
    )

    # Compute df_targets for datasetName (for class-balance tabs)
    df_targets_dataset1 = pd.DataFrame(
        {
            "fk_data_point_id": rows_dataset1["fk_data_point_id"].drop_duplicates().to_numpy()
        }
    ).merge(
        compute_df_targets_for_dataset(
            session_local=session,
            dataset_name_local=datasetName,
            descriptor_set_name_local=descriptorSetName,
            remove_log_p_local=remove_log_p_descriptors
        ),
        on="fk_data_point_id",
        how="left"
    ).drop_duplicates(subset=["fk_data_point_id"])

    # Write Excel for datasetName
    base_dir_root = Path(os.getenv("PROJECT_ROOT")) / ""
    excel_path_dataset1 = write_assignments_excel(
        base_dir_root=base_dir_root,
        dataset_name_local=datasetName,
        rows_local=rows_dataset1,
        is_binary_local=is_binary,
        df_targets_local=df_targets_dataset1
    )

    # Optionally clone to datasetName2
    if datasetName2:
        rows_dataset2 = map_blueprint_to_dataset_rows(
            session_local=session,
            dataset_name_local=datasetName2,
            blueprint_df=blueprint,
            user_local=user,
            created_at_local=created_at
        )

        if not rows_dataset2.empty:
            # Compute df_targets for datasetName2 (for class-balance tabs)
            df_targets_dataset2 = pd.DataFrame(
                {
                    "fk_data_point_id": rows_dataset2["fk_data_point_id"].drop_duplicates().to_numpy()
                }
            ).merge(
                compute_df_targets_for_dataset(
                    session_local=session,
                    dataset_name_local=datasetName2,
                    descriptor_set_name_local=descriptorSetName,
                    remove_log_p_local=remove_log_p_descriptors
                ),
                on="fk_data_point_id",
                how="left"
            ).drop_duplicates(subset=["fk_data_point_id"])

            excel_path_dataset2 = write_assignments_excel(
                base_dir_root=base_dir_root,
                dataset_name_local=datasetName2,
                rows_local=rows_dataset2,
                is_binary_local=is_binary,
                df_targets_local=df_targets_dataset2
            )
        else:
            print(f"No overlapping qsar_smiles between {datasetName} splits and {datasetName2}; nothing to write for datasetName2.")
            excel_path_dataset2 = None
    else:
        rows_dataset2 = pd.DataFrame(columns=["fk_data_point_id", "split_num", "fk_splitting_id", "created_at", "updated_at", "created_by", "updated_by"])
        excel_path_dataset2 = None

    # ---- Optional DB write(s) ----
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None)  # independent session for commit

        records1 = rows_dataset1.replace({np.nan: None}).to_dict(orient="records")
        count1 = dbl.create_many_chunked(table="data_points_in_splittings", records=records1)
        print(f"Wrote {count1} rows for {datasetName} across representative split and inner CV folds")

        if datasetName2 and not rows_dataset2.empty:
            records2 = rows_dataset2.replace({np.nan: None}).to_dict(orient="records")
            count2 = dbl.create_many_chunked(table="data_points_in_splittings", records=records2)
            print(f"Wrote {count2} rows for {datasetName2} (cloned from {datasetName})")

    # ---- Return info (maintain original shape; add cloned paths for convenience) ----
    # Reconstruct the per-dataset1 rows to match original keys
    outer_rows_dataset1 = rows_dataset1[rows_dataset1["fk_splitting_id"] == 1][["fk_data_point_id", "split_num", "fk_splitting_id"]].copy()
    inner_rows_dataset1 = rows_dataset1[rows_dataset1["fk_splitting_id"] != 1][["fk_data_point_id", "split_num", "fk_splitting_id"]].copy()

    return {
        "outer_assignments": outer_rows_dataset1.reset_index(drop=True),
        "inner_assignments": inner_rows_dataset1.reset_index(drop=True),
        "representative_fold_index": rep_fold,
        "pooled_score": float(pooled),
        "per_fold_scores": list(map(float, fold_scores)),
        "is_binary": bool(is_binary),
        "excel_path_dataset1": str(excel_path_dataset1),
        "excel_path_dataset2": str(excel_path_dataset2) if excel_path_dataset2 else None,
    }

def create_splittings(datasetName,  datasetName2=None, descriptorSetName = 'WebTEST-default'):
    """
    Creates entries in data_points_in_splittings table in database
    """
    
    write_to_db = True
    delete_old = False
    
    user="tmarti02"
    remove_log_p_descriptors = False
    n_outer_splits = 5
    n_inner_splits = 5
    random_state = 0
    n_threads = 8
    shuffle = True

    
    if delete_old:
        preview_and_delete_splittings_by_dataset(
            dataset_name=datasetName,
            split_ids=None,
            head_n=10,
            dry_run=False
        )
        
        if datasetName2 is not None:
            preview_and_delete_splittings_by_dataset(
                dataset_name=datasetName2,
                split_ids=None,
                head_n=10,
                dry_run=False
            )
            
    # old way:
    # find_representative_split(datasetName, descriptorSetName, remove_log_p_descriptors, user, n_threads=4, n_splits=5, random_state=42,
    #                           write_to_db=write_to_db)
    #
    # find_representative_cv_splits(datasetName, descriptorSetName, user, n_splits=5, random_state=42,
    #                         write_to_db=write_to_db)

    # all in one:
    make_representative_and_inner_cv_splits(datasetName, datasetName2, descriptorSetName, remove_log_p_descriptors, user, 
                                            n_outer_splits, n_inner_splits, random_state, n_threads, write_to_db, shuffle)




def create_inner_splittings(datasetName, descriptorSetName = 'WebTEST-default'):
    """
    Creates entries in data_points_in_splittings table in database
    """
    
    write_to_db = True
    # write_to_db = False
    delete_old = False
    
    user="tmarti02"
    n_inner_splits = 5
    random_state = 0
    n_threads = 8
    shuffle = True

    outerSplittingName='RND_REPRESENTATIVE'
    
    if delete_old:
        preview_and_delete_splittings_by_dataset(
            dataset_name=datasetName,
            split_ids=None,
            head_n=10,
            dry_run=False
        )
        
    make_inner_cv_splits(datasetName, descriptorSetName, outerSplittingName, user, n_inner_splits, random_state, n_threads, write_to_db, shuffle)
    




def make_inner_cv_splits(
    datasetName: str,
    descriptorSetName: str,
    outerSplittingName: str = "RND_REPRESENTATIVE",
    user: str = "system",
    n_inner_splits: int = 5,
    random_state: int = 0,
    n_threads: Optional[int] = None,  # kept for API compatibility; not used here
    write_to_db: bool = False,
    shuffle: bool = True,
) -> Dict:
    """
    Create ONLY inner CV splits, assuming the OUTER splitting has already been created
    and stored in the database using the standard naming:

      - Outer: 'RND_REPRESENTATIVE' (commonly id=1)
      - Inner folds: 'RND_REPRESENTATIVE_CV1' .. 'RND_REPRESENTATIVE_CV5' (commonly ids 2..6)

    What this does:
      - Pulls the TRAINING portion (split_num=0) of the specified outer split from the DB
        (restricted to points present in the given descriptor set).
      - Builds inner CV folds within that training subset using build_splitter(...)
        (StratifiedKFold for binary {0,1} if feasible; otherwise KFold).
      - Uses the standard inner split names 'RND_REPRESENTATIVE_CV{i}' to resolve fk_splitting_id values.
      - If write_to_db=True, writes assignments to qsar_datasets.data_points_in_splittings.
      - Writes an Excel spreadsheet with:
          - Sheet "assignments": inner fold assignments (fk_data_point_id, split_num, fk_splitting_id, metadata)
          - Sheet "class_balance": fraction of positive compounds (y==1) in train/test for:
              - the specified outer split
              - each inner fold

    Returns
    -------
    dict with keys:
      - inner_assignments: DataFrame with columns
          fk_data_point_id, split_num (0=train, 1=prediction), fk_splitting_id,
          created_at, updated_at, created_by, updated_by
      - is_binary: bool
      - fold_to_splitting_id: List[int] (fk_splitting_id used for each inner fold index)
      - n_training_points: int
      - excel_path: str (path to the written spreadsheet)
    """

    # ---- Helpers (specific to this function) ----
    def fetch_outer_train_test_targets(
        session, dataset_name: str, descriptor_set_name: str, outer_splitting_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Pull both TRAINING (split_num=0) and TEST/PREDICTION (split_num=1) portions of the given outer splitting.
        Returns:
          fk_ids_train, y_train, fk_ids_test, y_test
        Only includes rows present in the given descriptor set.
        """
        sql = text("""
            SELECT dp.id AS fk_data_point_id, dp.qsar_property_value, dpis.split_num
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
            JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
            JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
            JOIN qsar_datasets.splittings s ON s.id = dpis.fk_splitting_id
            WHERE d.name = :datasetName
              AND ds.name = :descriptorSetName
              AND s.name = :outerSplittingName
            ORDER BY dp.id
        """)
        rows = session.execute(
            sql,
            {
                "datasetName": dataset_name,
                "descriptorSetName": descriptor_set_name,
                "outerSplittingName": outer_splitting_name,
            },
        ).fetchall()

        if not rows:
            raise ValueError(
                f"No rows found for dataset='{dataset_name}', descriptorSet='{descriptor_set_name}', "
                f"outerSplitting='{outer_splitting_name}'."
            )

        train_ids, train_y, test_ids, test_y = [], [], [], []
        for fk_id, y_val, split_num in rows:
            if split_num == 0:
                train_ids.append(int(fk_id))
                train_y.append(y_val)
            elif split_num == 1:
                test_ids.append(int(fk_id))
                test_y.append(y_val)
        return (
            np.asarray(train_ids, dtype="int64"),
            np.asarray(train_y, dtype=float),
            np.asarray(test_ids, dtype="int64"),
            np.asarray(test_y, dtype=float),
        )

    def resolve_standard_inner_ids(session, k: int) -> Tuple[np.ndarray, list]:
        """
        Resolve splitting IDs for standard inner folds:
          names: 'RND_REPRESENTATIVE_CV1' .. 'RND_REPRESENTATIVE_CV{k}'
        Returns:
          (np.ndarray[int64] of IDs, list[str] of names)
        """
        if k > 5:
            raise ValueError("Standard scheme supports up to 5 inner folds: RND_REPRESENTATIVE_CV1..CV5")
        ids = []
        names = []
        for i in range(1, k + 1):
            nm = f"RND_REPRESENTATIVE_CV{i}"
            res = session.execute(
                text("SELECT id FROM qsar_datasets.splittings WHERE name = :name"),
                {"name": nm},
            ).fetchone()
            if not res:
                raise ValueError(
                    f"Required splitting name '{nm}' was not found in qsar_datasets.splittings. "
                    "Please create it or ensure the standard mapping (ids 2..6) exists."
                )
            ids.append(int(res[0]))
            names.append(nm)
        return np.asarray(ids, dtype="int64"), names

    def frac_positive(y: np.ndarray) -> float:
        """
        Fraction of y==1 among non-NaN entries. Returns np.nan if denominator is 0.
        """
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            return np.nan
        return float((y[mask] == 1).sum() / mask.sum())

    # ---- Load outer subsets for class-balance reporting ----
    session = getSession()
    fk_ids_outer_tr, y_outer_tr, fk_ids_outer_te, y_outer_te = fetch_outer_train_test_targets(
        session, datasetName, descriptorSetName, outerSplittingName
    )

    # ---- Prepare the outer training subset for inner splitting ----
    fk_ids_all, y_all = fk_ids_outer_tr.copy(), y_outer_tr.copy()

    # Optional: drop NaN targets if any (cannot stratify/split on NaN)
    if np.isnan(y_all).any():
        mask = ~np.isnan(y_all)
        dropped = int((~mask).sum())
        if dropped > 0:
            print(f"Dropping {dropped} rows with NaN targets from outer training subset before inner CV.")
        fk_ids_all = fk_ids_all[mask]
        y_all = y_all[mask]

    n_train = len(fk_ids_all)
    if n_inner_splits is None or n_inner_splits < 2:
        # Nothing to do; still write spreadsheet with outer class balance
        is_binary_outer = pd.Series(np.concatenate([y_outer_tr, y_outer_te])).dropna().isin([0, 1]).all()
        class_balance_rows = []

        # Outer balance
        class_balance_rows.append({
            "level": "outer",
            "name": outerSplittingName,
            "fold_index": None,
            "subset": "train",
            "n": int((~np.isnan(y_outer_tr)).sum()),
            "n_pos": int((y_outer_tr[~np.isnan(y_outer_tr)] == 1).sum()) if is_binary_outer else np.nan,
            "frac_pos": frac_positive(y_outer_tr) if is_binary_outer else np.nan,
        })
        class_balance_rows.append({
            "level": "outer",
            "name": outerSplittingName,
            "fold_index": None,
            "subset": "test",
            "n": int((~np.isnan(y_outer_te)).sum()),
            "n_pos": int((y_outer_te[~np.isnan(y_outer_te)] == 1).sum()) if is_binary_outer else np.nan,
            "frac_pos": frac_positive(y_outer_te) if is_binary_outer else np.nan,
        })
        class_balance_df = pd.DataFrame(class_balance_rows)

        # Empty assignments
        inner_rows = pd.DataFrame(
            columns=["fk_data_point_id", "split_num", "fk_splitting_id", "created_at", "updated_at", "created_by", "updated_by"]
        )

        # Write spreadsheet
        excel_path = _write_inner_splits_excel(datasetName, inner_rows, class_balance_df)
        print(excel_path)

        return {
            "inner_assignments": inner_rows.copy(),
            "is_binary": False,
            "fold_to_splitting_id": [],
            "n_training_points": int(n_train),
            "excel_path": str(excel_path),
        }

    if n_train < n_inner_splits:
        raise ValueError(
            f"Not enough training samples ({n_train}) to create {n_inner_splits} inner folds."
        )
    if n_inner_splits > 5:
        raise ValueError(
            f"n_inner_splits={n_inner_splits} exceeds standard available folds (RND_REPRESENTATIVE_CV1..CV5)."
        )

    print(f"Outer training subset size = {n_train}")

    # ---- Build inner splitter (helper is expected to be imported/defined elsewhere) ----
    inner_splitter, is_binary = build_splitter(y_all, n_inner_splits, shuffle, random_state)

    # Iterate splits: pass y when is_binary, else None (KFold ignores y if provided)
    split_iter = inner_splitter.split(np.zeros(n_train), y_all if is_binary else None)

    # Resolve standard inner splitting IDs and names (RND_REPRESENTATIVE_CV1..CVK)
    fold_to_splitting_id, inner_split_names = resolve_standard_inner_ids(session, n_inner_splits)

    # ---- Build assignment rows for each inner fold and class balance table ----
    inner_rows_parts = []
    class_balance_rows = []

    # Outer class balance (always reported)
    is_binary_outer = pd.Series(np.concatenate([y_outer_tr, y_outer_te])).dropna().isin([0, 1]).all()
    class_balance_rows.append({
        "level": "outer",
        "name": outerSplittingName,
        "fold_index": None,
        "subset": "train",
        "n": int((~np.isnan(y_outer_tr)).sum()),
        "n_pos": int((y_outer_tr[~np.isnan(y_outer_tr)] == 1).sum()) if is_binary_outer else np.nan,
        "frac_pos": frac_positive(y_outer_tr) if is_binary_outer else np.nan,
    })
    class_balance_rows.append({
        "level": "outer",
        "name": outerSplittingName,
        "fold_index": None,
        "subset": "test",
        "n": int((~np.isnan(y_outer_te)).sum()),
        "n_pos": int((y_outer_te[~np.isnan(y_outer_te)] == 1).sum()) if is_binary_outer else np.nan,
        "frac_pos": frac_positive(y_outer_te) if is_binary_outer else np.nan,
    })

    for inner_fold_idx, (inner_tr_rel, inner_te_rel) in enumerate(split_iter):
        fk_tr = fk_ids_all[inner_tr_rel]
        fk_te = fk_ids_all[inner_te_rel]
        y_tr = y_all[inner_tr_rel]
        y_te = y_all[inner_te_rel]
        fk_splitting_id = int(fold_to_splitting_id[inner_fold_idx])

        inner_rows_parts.append(rows_for_assignment(fk_tr, split_num=0, fk_splitting_id=fk_splitting_id))
        inner_rows_parts.append(rows_for_assignment(fk_te, split_num=1, fk_splitting_id=fk_splitting_id))

        # Class balance for this inner fold
        class_balance_rows.append({
            "level": "inner",
            "name": inner_split_names[inner_fold_idx],
            "fold_index": inner_fold_idx,
            "subset": "train",
            "n": int((~np.isnan(y_tr)).sum()),
            "n_pos": int((y_tr[~np.isnan(y_tr)] == 1).sum()) if is_binary else np.nan,
            "frac_pos": frac_positive(y_tr) if is_binary else np.nan,
        })
        class_balance_rows.append({
            "level": "inner",
            "name": inner_split_names[inner_fold_idx],
            "fold_index": inner_fold_idx,
            "subset": "test",
            "n": int((~np.isnan(y_te)).sum()),
            "n_pos": int((y_te[~np.isnan(y_te)] == 1).sum()) if is_binary else np.nan,
            "frac_pos": frac_positive(y_te) if is_binary else np.nan,
        })

    inner_rows = pd.concat(inner_rows_parts, ignore_index=True)

    # ---- Stamp metadata ----
    created_at = datetime.now()
    inner_rows = inner_rows.assign(
        created_at=created_at,
        updated_at=created_at,
        created_by=user,
        updated_by=user,
    )

    # Class balance DataFrame
    class_balance_df = pd.DataFrame(class_balance_rows)

    # ---- Optional: write to DB ----
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None)  # independent session for commit
        records = inner_rows.replace({np.nan: None}).to_dict(orient="records")
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Wrote {count} rows for {n_inner_splits} inner CV folds into qsar_datasets.data_points_in_splittings.")

    # ---- Write spreadsheet (assignments + class balance) ----
    excel_path = _write_inner_splits_excel(datasetName, inner_rows, class_balance_df)
    print(excel_path)

    # ---- Return info ----
    return {
        "inner_assignments": inner_rows.copy(),
        "is_binary": bool(is_binary),
        "fold_to_splitting_id": [int(x) for x in fold_to_splitting_id],
        "n_training_points": int(n_train),
        "excel_path": str(excel_path),
    }


def delete_splitting_rows_for_dataset(session, dataset_name: str) -> int:
    """
    Delete rows in qsar_datasets.data_points_in_splittings linked to the given dataset name.
    Returns the number of rows deleted.
    """
    sql = text("""
        DELETE FROM qsar_datasets.data_points_in_splittings AS dpis
        WHERE dpis.fk_data_point_id IN (
            SELECT dp.id
            FROM qsar_datasets.data_points AS dp
            JOIN qsar_datasets.datasets AS d
              ON dp.fk_dataset_id = d.id
            WHERE d.name = :dataset_name
        )
    """)
    try:
        result = session.execute(sql, {"dataset_name": dataset_name})
        session.commit()
        return result.rowcount
    except Exception:
        session.rollback()
        raise


def run_deletes():
    session = getSession()
    datasets = [
        "exp_prop_PERCENT_BIODEGRADATION_301F v1 modeling",
        "exp_prop_RBIODEG_301F v1 modeling",
        "exp_prop_PERCENT_BIODEGRADATION_RIFM_CHEMREG",
        "exp_prop_RBIODEG_RIFM_CHEMREG",
    ]
    total = 0
    try:
        for ds in datasets:
            deleted = delete_splitting_rows_for_dataset(session, ds)
            print(f"{ds}: deleted {deleted} rows")
            total += deleted
        print(f"Total deleted across datasets: {total}")
    finally:
        session.close()


def _write_inner_splits_excel(datasetName: str, inner_rows: pd.DataFrame, class_balance_df: pd.DataFrame) -> Path:
    """
    Helper to write the Excel file with two sheets:
      - 'assignments': inner_rows
      - 'class_balance': class_balance_df
    Writes to: $PROJECT_ROOT/data/models/{datasetName}/inner_splittings.xlsx
    """
    project_root = os.getenv("PROJECT_ROOT") or "."
    excel_path = Path(project_root) / "data" / "models" / datasetName / "inner_splittings.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_path) as writer:
        inner_rows.to_excel(writer, sheet_name="assignments", index=False)
        class_balance_df.to_excel(writer, sheet_name="class_balance", index=False)

    return excel_path



if __name__ == '__main__':
    
    # test getting the detailed property data:
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    
    # dataset_name = 'KOC v1 modeling'
    # dataset_name = 'exp_prop_RBIODEG_RIFM_BY_DTXSID'    
    # dataset_name = 'exp_prop_RBIODEG_NITE_OPPT v1.0'
    # dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    # create_splittings(dataset_name)
    
    # create_inner_splittings(dataset_name)
                      
    # run_deletes()
                          
    dataset_name = 'exp_prop_RBIODEG_301F v1 modeling'
    dataset_name2 = 'exp_prop_RBIODEG_RIFM_CHEMREG'
    create_splittings(dataset_name, datasetName2=dataset_name2)
