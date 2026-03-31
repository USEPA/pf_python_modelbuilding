'''
Created on Mar 2, 2026

Fetch QSAR dataset rows by splitting a large query into smaller queries, then
assemble results including:
- all parameters as exp_details_all_parameters
- dynamic fields exp_details_{parameter_name}_{attr} (e.g., value_point_estimate)

Requirements:
- SQLAlchemy >= 1.4
- A PostgreSQL DB URL in DB_URL (e.g., postgresql+psycopg2://user:pass@host/db)

@author: TMARTI02
'''

from typing import Dict, List, Tuple, Optional, Any, Literal
from sqlalchemy.sql.elements import TextClause
import os
import re
import pandas as pd
import webbrowser
from sqlalchemy import text, bindparam

DuplicateStrategy = Literal["first", "list", "id_suffix"]

class ExpDataGetter:
    
    def _params_query(self, 
        prop_value_ids: List[int]
    ) -> Tuple[TextClause, Dict[str, Any]]:
        """
        Fetch all parameter values for a set of property value IDs.
        Returns (TextClause, params dict).
        """
        # Use IN :pv_ids with expanding bindparam for bulk fetch
        sql = text("""
            SELECT
                pvparam.fk_property_value_id AS prop_value_id,
                pvparam.fk_parameter_id      AS parameter_id,
                param.name               AS parameter_name,
                pvparam.value_point_estimate,
                pvparam.value_min,
                pvparam.value_max,
                pvparam.value_text,
                pvparam.value_qualifier,
                pvparam.fk_unit_id           AS unit_id,
                uparam.abbreviation      AS value_units
            FROM exp_prop.parameter_values AS pvparam
            JOIN exp_prop.parameters AS param
              ON param.id = pvparam.fk_parameter_id
            LEFT JOIN exp_prop.units AS uparam
              ON uparam.id = pvparam.fk_unit_id
            WHERE pvparam.fk_property_value_id IN :pv_ids
        """).bindparams(bindparam("pv_ids", expanding=True))
        return sql, {"pv_ids": prop_value_ids}



    def _base_query(self, 
        dataset_name: Optional[str]=None,
        snapshot_id: Optional[int]=None,
        keep_only: bool=True
    ) -> Tuple[TextClause, Dict[str, Any]]:
        """
        Build the base query. Filters are optional; if omitted, they won’t be applied.
        Returns (TextClause, params dict).
        """
        where_clauses = ["1=1"]
        params: Dict[str, Any] = {}

        if dataset_name is not None:
            where_clauses.append("d.name = :dataset_name")
            params["dataset_name"] = dataset_name

        if snapshot_id is not None:
            where_clauses.append("dr.fk_dsstox_snapshot_id = :snapshot_id")
            params["snapshot_id"] = snapshot_id

        # Qualify 'keep' to avoid ambiguity; adjust alias if needed
        if keep_only:
            where_clauses.append("pv.keep = TRUE")

        where_sql = " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                pv.id AS prop_value_id,
                dp.canon_qsar_smiles,
                sc.source_dtxsid,
                sc.source_casrn,
                sc.source_chemical_name,
                sc.source_smiles,
                dr.dtxsid AS mapped_dtxsid,
                dr.dtxcid AS mapped_dtxcid,
                dr.casrn AS mapped_casrn,
                dr.smiles AS mapped_smiles,
                dr.preferred_name AS mapped_chemical_name,
                dr.mol_weight as mapped_mol_weight,
                p.name_ccd AS prop_name,
                'experimental' AS prop_type,
                pc.name AS prop_category,
                d."name" AS dataset,
                dpc.property_value AS prop_value,
                u.abbreviation_ccd AS prop_unit,
                dp.qsar_property_value,
                dp.qsar_exp_prop_property_values_id, 
                u2.abbreviation_ccd AS qsar_property_unit,
                pv.value_original AS prop_value_original,
                pv.value_text AS prop_value_text,
                pv.page_url AS direct_url,
                pv.document_name AS brief_citation,
                pv.notes AS notes,
                pv.qc_flag AS qc_flag,
                pv.keep_reason AS flag_reason,
                pv.value_max AS value_max,
                pv.value_min AS value_min,
                sc.source_dtxrid AS source_dtxrid
            FROM qsar_datasets.data_points AS dp
            JOIN qsar_datasets.data_point_contributors AS dpc
                ON dpc.fk_data_point_id = dp.id
            JOIN exp_prop.property_values AS pv
                ON dpc.exp_prop_property_values_id = pv.id
            JOIN exp_prop.source_chemicals AS sc
                ON sc.id = pv.fk_source_chemical_id
            JOIN qsar_datasets.datasets AS d
                ON dp.fk_dataset_id = d.id
            JOIN qsar_datasets.properties AS p
                ON d.fk_property_id = p.id
            JOIN qsar_datasets.units AS u
                ON u.id = d.fk_unit_id_contributor
            JOIN qsar_datasets.units AS u2
                ON u2.id = d.fk_unit_id
            LEFT JOIN qsar_datasets.properties_in_categories AS pic
                ON p.id = pic.fk_property_id
            LEFT JOIN qsar_datasets.property_categories AS pc
                ON pic.fk_property_category_id = pc.id
            LEFT JOIN qsar_models.dsstox_records AS dr
                ON dr.dtxcid = dpc.dtxcid
            WHERE {where_sql}
        """)
        return sql, params
        
    
    def _sources_query(self, prop_value_ids: List[int]) -> Tuple[TextClause, Dict[str, Any]]:
        """
        Fetch literature/public source metadata for a set of property value IDs.
        Returns (TextClause, params dict).
        """
        sql = text("""
            SELECT
                pv.id AS prop_value_id,
                ps."name"        AS public_source_name,
                ps.description   AS public_source_description,
                ps.url           AS public_source_url,
                ls."name"        AS literature_source_name,
                ls.citation      AS literature_source_citation,
                ls.doi           AS literature_source_doi,
                ps2."name"       AS public_source_original_name,
                ps2.description  AS public_source_original_description,
                ps2.url          AS public_source_original_url
            FROM exp_prop.property_values AS pv
            LEFT JOIN exp_prop.literature_sources AS ls
                ON pv.fk_literature_source_id = ls.id
            LEFT JOIN exp_prop.public_sources AS ps
                ON pv.fk_public_source_id = ps.id
            LEFT JOIN exp_prop.public_sources AS ps2
                ON pv.fk_public_source_original_id = ps2.id
            WHERE pv.id IN :pv_ids
        """).bindparams(bindparam("pv_ids", expanding=True))
        return sql, {"pv_ids": prop_value_ids}
    
    
    def _sanitize_param_name(self, name: str) -> str:
        """
        Convert a parameter name to a safe, lowercase, underscore-separated key.
        - lowercases
        - replaces any non-alphanumeric with underscores
        - collapses multiple underscores
        - trims leading/trailing underscores
        """
        s = (name or "").strip().lower()
        s = re.sub(r'[^a-z0-9]+', '_', s)
        s = re.sub(r'_+', '_', s).strip('_')
        return s
    
    
    def _safe_param_key(self, name: str, parameter_id: Optional[int]) -> str:
        """
        Return a safe key derived from parameter name; fallback to param_{id} if name sanitizes empty.
        """
        key = self._sanitize_param_name(name) if name else ""
        if not key:
            key = f"param_{parameter_id}" if parameter_id is not None else "param"
        return key
    
    
    def _add_parameter_dynamic_fields(self, 
        row: Dict[str, Any],
        params: List[Dict[str, Any]],
        *,
        prefix: str="exp_details",
        duplicate_strategy: DuplicateStrategy="list",        
    ) -> Dict[str, Any]:
        """
        Add dynamic fields exp_details_{parameter_name}_{attr} to 'row'.
    
        Guarantees that parameter_id and unit_id fields are NOT added to the dynamic fields.
        Disambiguates duplicate parameter names using occurrence index when duplicate_strategy="id_suffix".
        """
        # Defensive: never allow id fields to slip in, even if passed
        
        include_fields: Tuple[str, ...]=(
            "value_point_estimate",
            "value_text",
            "unit_abbreviation",
            "value_qualifier",
            "value_min",
            "value_max",
            "value_units",
            # "parameter_id",
            # "unit_id",
        )
        
        disallowed = {"parameter_id", "unit_id"}
    
        # Count occurrences by sanitized parameter name
        name_counts: Dict[str, int] = {}
        for p in params:
            base_key = self._sanitize_param_name(p.get("parameter_name", "")) or "param"
            name_counts[base_key] = name_counts.get(base_key, 0) + 1
    
        # Track index per name for id_suffix disambiguation
        name_seen_index: Dict[str, int] = {}
    
        for p in params:
            base_key = self._sanitize_param_name(p.get("parameter_name", "")) or "param"
    
            if name_counts.get(base_key, 1) > 1 and duplicate_strategy == "id_suffix":
                idx = name_seen_index.get(base_key, 0) + 1
                name_seen_index[base_key] = idx
                base_key = f"{base_key}_{idx}"
                # For "first" and "list", we keep the base_key unchanged
    
            for field in include_fields:
                value = p.get(field)
                dyn_key = f"{prefix}_{base_key}_{field}"
    
                if duplicate_strategy == "first":
                    if dyn_key not in row:
                        row[dyn_key] = value
                elif duplicate_strategy == "list":
                    if dyn_key not in row:
                        row[dyn_key] = value
                    else:
                        if not isinstance(row[dyn_key], list):
                            row[dyn_key] = [row[dyn_key]]
                        row[dyn_key].append(value)
                elif duplicate_strategy == "id_suffix":
                    row[dyn_key] = value
    
        return row
    
    
    
    def _get_params_as_dict(self, params):
        # Convenience dict keyed by parameter_name (collect duplicates into list)
        params_by_name: Dict[str, Any] = {}
        for p in params:
            name = p.get("parameter_name")
            if name in params_by_name:
                existing = params_by_name[name]
                if isinstance(existing, list):
                    existing.append(p)
                else:
                    params_by_name[name] = [existing, p]
            else:
                params_by_name[name] = p
    
    
    @staticmethod
    def _drop_all_null_columns(
        df: pd.DataFrame,
        *,
        consider_blank_strings: bool = True,
        consider_empty_containers: bool = True,
        consider_containers_with_only_nulls: bool = True,
    ) -> pd.DataFrame:
        """
        Drop columns that have no meaningful values across all rows.
    
        A value is considered "null" if:
          - It is NaN/None (pandas.isna is True), OR
          - consider_blank_strings=True and it's a blank/whitespace-only string, OR
          - consider_empty_containers=True and it's an empty list/tuple/set/dict, OR
          - consider_containers_with_only_nulls=True and a container whose elements/values
            are all effectively null by the same rules (one-level deep).
        """
        def _isna(v: Any) -> bool:
            return pd.isna(v)
    
        def _is_blank_string(v: Any) -> bool:
            return isinstance(v, str) and v.strip() == ""
    
        def _is_container(v: Any) -> bool:
            return isinstance(v, (list, tuple, set, dict))
    
        def _container_is_empty(v: Any) -> bool:
            if isinstance(v, dict):
                return len(v) == 0
            return isinstance(v, (list, tuple, set)) and len(v) == 0
    
        def _container_all_null(v: Any) -> bool:
            if isinstance(v, dict):
                if not v:
                    return True
                return all(_is_effectively_null(val) for val in v.values())
            elif isinstance(v, (list, tuple, set)):
                if not v:
                    return True
                return all(_is_effectively_null(val) for val in v)
            return False
    
        def _is_effectively_null(v: Any) -> bool:
            if _isna(v):
                return True
            if consider_blank_strings and _is_blank_string(v):
                return True
            if consider_empty_containers and _is_container(v):
                if _container_is_empty(v):
                    return True
                if consider_containers_with_only_nulls and _container_all_null(v):
                    return True
            return False
    
        cols_to_keep = []
        for col in df.columns:
            series = df[col]
            if series.map(lambda v: not _is_effectively_null(v)).any():
                cols_to_keep.append(col)
    
        return df.loc[:, cols_to_keep]
    


    def get_mapped_property_values(
        self, 
        session,
        dataset_name: Optional[str] = None,
        snapshot_id: Optional[int] = None,
        keep_only: bool = True,
        *,
        add_dynamic_fields: bool = True,
        dynamic_prefix: str = "exp_details",
        duplicate_strategy: str = "id_suffix",  # DuplicateStrategy alias for simplicity
        params_as_list: bool = False,        
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Run small queries and assemble results into a DataFrame, including
        source metadata and optional dynamic fields per parameter. Returns flat
        dataframe with parameters additional columns and the unique list of parameters.
    
        Returns:
            df: pandas DataFrame (sorted by canon_qsar_smiles)
            unique_params: sorted list of unique parameter names in the data
        """
        # Base rows
        base_text, base_params = self._base_query(
            dataset_name=dataset_name,
            snapshot_id=snapshot_id,
            keep_only=keep_only,
        )
        base_rows = session.execute(base_text, base_params).mappings().all()
        if not base_rows:
            return pd.DataFrame(), []
    
        pv_ids = [r["prop_value_id"] for r in base_rows]
    
        # Parameters
        params_text, params_params = self._params_query(pv_ids)
        param_rows = session.execute(params_text, params_params).mappings().all()
    
        # Build parameter info per prop value, omitting parameter_id and unit_id
        params_by_pv: Dict[int, List[Dict[str, Any]]] = {}
        unique_param_names: set[str] = set()
    
        for pr in param_rows:
            pv_id = pr["prop_value_id"]
            pname = pr.get("parameter_name")
            if pname:
                unique_param_names.add(pname)
    
            params_by_pv.setdefault(pv_id, []).append({
                "parameter_name": pname,
                "value_point_estimate": pr.get("value_point_estimate"),
                "value_min": pr.get("value_min"),   # ensure your _params_query returns these
                "value_max": pr.get("value_max"),
                "value_text": pr.get("value_text"),
                "value_qualifier": pr.get("value_qualifier"),
                "value_units": pr.get("value_units"),
            })
    
        # Sources
        sources_text, sources_params = self._sources_query(pv_ids)
        source_rows = session.execute(sources_text, sources_params).mappings().all()
        sources_by_pv: Dict[int, Dict[str, Any]] = {}
        for sr in source_rows:
            sources_by_pv[sr["prop_value_id"]] = {
                "public_source_name": sr.get("public_source_name"),
                "public_source_description": sr.get("public_source_description"),
                "public_source_url": sr.get("public_source_url"),
                "literature_source_name": sr.get("literature_source_name"),
                "literature_source_citation": sr.get("literature_source_citation"),
                "literature_source_doi": sr.get("literature_source_doi"),
                "public_source_original_name": sr.get("public_source_original_name"),
                "public_source_original_description": sr.get("public_source_original_description"),
                "public_source_original_url": sr.get("public_source_original_url"),
            }
    
        # Assemble rows and add dynamic parameter fields OR params list
        results: List[Dict[str, Any]] = []
        for row in base_rows:
            pv_id = row["prop_value_id"]
            params = params_by_pv.get(pv_id, [])
            sources = sources_by_pv.get(pv_id, {})
    
            combined = {**row, **sources}
    
            if params_as_list:
                combined["params"] = params
                
                # print(json.dumps(params,indent=4))
                
            elif add_dynamic_fields and params:
                # assumes you have self._add_parameter_dynamic_fields
                self._add_parameter_dynamic_fields(
                    combined,
                    params,
                    prefix=dynamic_prefix,
                    duplicate_strategy=duplicate_strategy,
                )
    
            results.append(combined)
    
        # Convert to DataFrame and clean types
        df = pd.DataFrame.from_records(results)
        df = df.convert_dtypes()       # uses pandas nullable dtypes; missing -> pd.NA
        # df = df.replace({pd.NA: None}) # easier JSON export

        df = self._drop_all_null_columns(df)

    
        # Sort for consistency
        sort_cols = [c for c in ["canon_qsar_smiles", "prop_value"] if c in df.columns]
        if sort_cols:
            df.sort_values(by=sort_cols, inplace=True)    

        # Build unique parameter list (sorted)
        unique_params = sorted(unique_param_names)
    
        return df, unique_params  


if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
      
    # print(os.getenv('DEV_QSAR_DATABASE'))
    from util.database_utilities import getSession  
    session = getSession()
    dataset_name = 'KOC v1 modeling'
    property_name = "LogKoc"
    snapshot_id = 4
    duplicate_strategy="id_suffix"
    params_as_list = True
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    path_segments = [PROJECT_ROOT, "data", "models", dataset_name, "raw_data.xlsx"]
    excel_path = os.path.join(*path_segments)
    
    path_segments = [PROJECT_ROOT, "data", "models", dataset_name, "raw_data.json"]
    json_path = os.path.join(*path_segments)

    edg = ExpDataGetter()
    
    # df_pv2, param_names = edg2.get_mapped_property_values(session, dataset_name, snapshot_id, params_as_list=False)
    # for idx, row in df_pv2.iterrows():
    #     row_dict = row.to_dict()
    #     if row.source_casrn == '40487-42-1' :  
    #         if row.prop_value is not None and abs(row.prop_value-6500)<1:
    #             safe_row = to_json_safe(row_dict, omit_nulls=True)
    #             json_str = json.dumps(safe_row, indent=4)
    #             print(json_str)


    df_pv2, param_names = edg.get_mapped_property_values(session, dataset_name, snapshot_id, params_as_list=True)
    df_pv2.to_json(json_path, orient='records', indent=4)
    
    # for idx, row in df_pv2.iterrows():
    #     row_dict = row.to_dict()
    #     if row_dict["params"] is not None:
    #         print(json.dumps(row_dict,indent=4))

    
    from report_creator_dict import ReportCreator
    rc = ReportCreator()
    es = rc.RawExpDataSection()
    html = es.create_exp_records_webpage(df_pv2, param_names, title_text="Experimental Property Records for " + property_name)
    
    path_segments = [PROJECT_ROOT, "data", "models", dataset_name, "raw_data.html"]
    temp_file_path = os.path.join(*path_segments)

    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(html)
        webbrowser.open('file://' + os.path.realpath(temp_file_path))


