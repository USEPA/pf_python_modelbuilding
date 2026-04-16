import pickle
import re

import pandas as pd
from sqlalchemy import URL, text, create_engine
from sqlalchemy.orm import sessionmaker
import math
import numpy as np
from typing import Optional, Dict, Any, Iterable, Tuple, Union
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name
import os
import traceback
import json
from sklearn2pmml.pipeline import PMMLPipeline as PMMLPipeline
from sklearn.model_selection import KFold
from model_ws_db_utilities import ModelInitializer
from models.ModelBuilder import Model
from models.db_utilities.raw_exp_data_db import ExpDataGetter
from models.db_utilities.dataset_utilities_db import getMappedDatapoints
import applicability_domain.applicability_domain_utilities as adu
from dataclasses import dataclass, field
from pathlib import Path
import warnings

import logging
import coloredlogs

logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

logging_level = logging.INFO

level_styles = {
    "debug": {"color": "cyan"},
    "info": {"color": "yellow"},
    "warning": {"color": "red", "bold": True},
    "error": {"color": "white", "background": "red"}
}

coloredlogs.install(level=logging_level, milliseconds=True, level_styles=level_styles,
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

pd.options.mode.chained_assignment = None  # default='warn'

# ============================================================
# STORAGE CLASS
# ============================================================

@dataclass
class ModelDataObjects:
    """Container for model data, dataframes, and configurations for Excel generation.
    
    Stores model objects, experimental dataframes, Excel sheet dataframes, query arguments,
    and the DataQuerier instance. Can be initialized with a model_id to auto-populate data
    from the database, or with pre-constructed model and dataframe objects for custom workflows.
    """

    # Model object and metadata (passed to DataQuerier)
    model_id: Optional[int] = None
    model: Optional[Any] = None

    # Pre-constructed dataframes related to the Model object (passed to DataQuerier)
    df_pv: Optional[pd.DataFrame] = None
    df_gmd: Optional[pd.DataFrame] = None
    df_gmd_external: Optional[pd.DataFrame] = None
    experimental_parameters: Optional[pd.DataFrame] = None

    # Pre-constructed dataframes for Excel sheets
    cover_sheet_df: Optional[pd.DataFrame] = None
    statistics_df: Optional[pd.DataFrame] = None
    records_df: Optional[pd.DataFrame] = None
    records_field_descriptions_df: Optional[pd.DataFrame] = None
    model_descriptors_df: Optional[pd.DataFrame] = None
    model_descriptor_values_df: Optional[pd.DataFrame] = None
    training_cv_predictions_df: Optional[pd.DataFrame] = None
    test_set_predictions_df: Optional[pd.DataFrame] = None
    external_predictions_df: Optional[pd.DataFrame] = None

    # Optional arguments for queries (passed to DataQuerier)
    query_args: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Optional arguments for transformations (passed to DataTransformer)
    transformation_args: Optional[Dict[str, Any]] = field(default_factory=dict)

    # DataQuerier object to handle queries (best left as None for automatic initialization, but can be pre-constructed)
    data_querier: Optional[Any] = None

    def __post_init__(self):
        logging.info("Initializing ModelDataObjects...")
        if self.model_id is not None:
            logging.info(f"model_id provided, initializing using DataQuerier for model_id={self.model_id}...")
            if self.model is not None:
                logging.warning(f"Both model_id and model provided, priority given to model_id, database will be queried for model_id={self.model_id}...")
            
            data_querier = DataQuerier(model_id=self.model_id, query_args=self.query_args)
            self.data_querier = data_querier

            self.model = data_querier.model

            self.df_pv = data_querier.df_pv
            self.df_gmd = data_querier.df_gmd
            self.df_gmd_external = data_querier.df_gmd_external
            self.experimental_parameters = data_querier.experimental_parameters

            self.cover_sheet_df = data_querier.query_cover_sheet_df()
            self.statistics_df = data_querier.query_statistics_df()
            self.records_df = data_querier.query_records_df()
            self.records_field_descriptions_df = DataTransformer.get_records_field_descriptions_df(self.experimental_parameters)
            self.model_descriptors_df = data_querier.query_model_descriptors_df()
            self.model_descriptor_values_df = data_querier.query_model_descriptor_values_df()
            self.training_cv_predictions_df = data_querier.query_training_cv_predictions_df()
            self.test_set_predictions_df = data_querier.query_test_set_predictions_df()
            self.external_predictions_df = data_querier.query_external_predictions_df()
            self.superheaders = DataTransformer.get_superheaders(self.records_df, self.experimental_parameters["Field"].tolist())

        elif self.model is not None:
            logging.info("Initializing with provided model and dataframes...")
            if self.df_pv is None:
                logging.warning("df_pv is None, will be queried from the database using the provided model")
            if self.df_gmd is None:
                logging.warning("df_gmd is None, will be queried from the database using the provided model")
            if getattr(self.model, "external_dataset_name", False) and self.df_gmd_external is None:
                logging.warning("df_gmd_external is None, will be queried from the database using the provided model")
            
            self.model_id = getattr(self.model, "modelId", None)

            data_querier = DataQuerier(model=self.model, df_pv=self.df_pv, df_gmd=self.df_gmd, df_gmd_external=self.df_gmd_external, experimental_parameters=self.experimental_parameters, query_args=self.query_args)
            self.data_querier = data_querier

            self.df_pv = data_querier.df_pv
            self.df_gmd = data_querier.df_gmd
            self.df_gmd_external = data_querier.df_gmd_external
            self.experimental_parameters = data_querier.experimental_parameters

            self.cover_sheet_df = data_querier.query_cover_sheet_df()
            self.statistics_df = data_querier.query_statistics_df()
            self.records_df = data_querier.query_records_df()
            self.records_field_descriptions_df = DataTransformer.get_records_field_descriptions_df(self.experimental_parameters)
            self.model_descriptors_df = data_querier.query_model_descriptors_df()
            self.model_descriptor_values_df = data_querier.query_model_descriptor_values_df()
            self.training_cv_predictions_df = data_querier.query_training_cv_predictions_df()
            self.test_set_predictions_df = data_querier.query_test_set_predictions_df()
            self.external_predictions_df = data_querier.query_external_predictions_df()
            self.superheaders = DataTransformer.get_superheaders(self.records_df)            

        else:
            logging.info("Not initialized with model_id or model, manual initialization required")


# ============================================================
# EXCEL FORMATTING HELPER CLASS
# ============================================================

class ExcelFormatter:
    """Handles Excel formatting, column management, filtering, and layout operations.
    
    Provides static methods for setting column widths, adding filters, managing headers,
    handling accidental formulas, adding subtotals, and creating hyperlinks between sheets.
    """
    
    @staticmethod
    def set_column_width(
        writer: Any,
        sheet_name: str,
        df: pd.DataFrame,
        col_width_pad: int=6,
        min_col_width: int=8,
        how: str="header",
        first_col_format: Optional[Any] = None
        ) -> list[int]:
        """Set column widths in Excel sheet based on content.
        
        Args:
            writer: pandas ExcelWriter object.
            sheet_name: Name of the worksheet to format.
            df: DataFrame corresponding to the sheet (used to get column names and widths).
            col_width_pad: Extra padding to add to calculated width. Defaults to 6.
            min_col_width: Minimum width for any column. Defaults to 8.
            how: Strategy for width calculation - 'header' uses header text width, 'full' considers all cell values. Defaults to 'header'.
            first_col_format: Optional cell format to apply to the first column only.
        
        Returns:
            list[int] of column widths.
        """
        worksheet = writer.sheets[sheet_name]
        
        col_widths = []
        for col_idx in range(len(df.columns)):
            if how == "header":
                header_entry = df.columns[col_idx]
                header_width = len(header_entry)
                col_width = max(header_width, min_col_width) + col_width_pad
                col_widths.append(col_width)
                worksheet.set_column(col_idx, col_idx, col_width, first_col_format if col_idx == 0 else None)
            elif how == "full":
                header_entry = df.columns[col_idx]
                header_width = len(str(header_entry))
                col_width = max(header_width, min_col_width)
                max_width = col_width
                for row_idx in range(df.shape[0]):
                    cell_value = df.iloc[row_idx, col_idx]
                    if cell_value is not None:
                        cell_value = str(cell_value)
                        max_width = max(max_width, len(cell_value))
                col_width = max_width + col_width_pad
                col_widths.append(col_width)
                worksheet.set_column(col_idx, col_idx, col_width, first_col_format if col_idx == 0 else None)
        return col_widths
    
    @staticmethod
    def set_sig_figs(
        df: pd.DataFrame,
        columns: list[str],
        sig_figs: int = 3,
        in_place: bool = True
        ) -> Optional[pd.DataFrame]:
        """Set numerical columns to have the desired number of significant figures.
        
        Args:
            df: DataFrame corresponding to the sheet (used to get column names and widths).
            columns: A list of columns to apply significant figures to. If None, all numerical columns are formatted.
            sig_figs: Number of significant figures for numerical columns. Defaults to 3.
            in_place: Whether to perform the transformation in place on the provided DataFrame. Defaults to True.
        """
        numerical_types = [np.float64, np.float32, float]

        logging.debug(f"Columns:\n\t{df.columns.tolist()}\nTypes:\n\t{df.dtypes.to_dict()}")

        if not in_place:
            df = df.copy()
        
        if columns is None:
            for col in df.columns.tolist():
                if df[col].dtype in numerical_types:
                    df[col] = df[col].apply(lambda x: f"{x:.{sig_figs}g}").astype(float)
                    logging.debug(f"Formatted column '{col}' with {sig_figs} significant figures:\n\t{df[col].iloc[:5].tolist()}")
        else:
            for col in columns:
                if df[col].dtype in numerical_types:
                    df[col] = df[col].apply(lambda x: f"{x:.{sig_figs}g}").astype(float)
                    logging.debug(f"Formatted column '{col}' with {sig_figs} significant figures:\n\t{df[col].iloc[:5].tolist()}")
        
        if not in_place:
            return df
    
    @staticmethod
    def get_header_row(has_subtotals: bool = False, has_superheaders: bool = False) -> int:
        """Get the row number where column headers are located.
        
        Args:
            has_subtotals: Whether the sheet has subtotal rows. Defaults to False.
            has_superheaders: Whether the sheet has superheader rows above headers. Defaults to False.
        
        Returns:
            int: Zero-based row index where headers are located.
        """
        if has_superheaders:
            return 3 if has_subtotals else 1
        return 2 if has_subtotals else 0
    
    @staticmethod
    def get_data_start_row(has_subtotals: bool = False, has_superheaders: bool = False) -> int:
        """Get the row number where data starts (first row after headers).
        
        Args:
            has_subtotals: Whether the sheet has subtotal rows. Defaults to False.
            has_superheaders: Whether the sheet has superheader rows above headers. Defaults to False.
        
        Returns:
            int: Zero-based row index where the first data row is located.
        """
        if has_superheaders:
            return 4 if has_subtotals else 1
        return 3 if has_subtotals else 1

    @staticmethod
    def add_subtotals(writer: Any, sheet_name: str, df: pd.DataFrame) -> None:
        """Add SUBTOTAL formulas above table headers to count visible cells per column.
        
        Args:
            writer: pandas ExcelWriter object.
            sheet_name: Name of the worksheet to add subtotals to.
            df: DataFrame whose columns will have subtotal formulas added.
        """
        ws = writer.sheets[sheet_name]
        nrows, ncols = df.shape
        
        for col_idx in range(ncols):
            start_cell = xl_rowcol_to_cell(3, col_idx)
            end_cell = xl_rowcol_to_cell(nrows + 2, col_idx)
            range_str = f"{start_cell}:{end_cell}"
            formula = f"=SUBTOTAL(3,{range_str})"
            ws.write_formula(0, col_idx, formula)
        
        logging.debug(f"Added SUBTOTAL formulas to {sheet_name} with {ncols} columns")

    @staticmethod
    def add_filter(writer: Any, sheet_name: str, df: pd.DataFrame, has_subtotals: bool = False, has_superheaders: bool = False) -> None:
        """Add autofilter dropdown to header row for easy data filtering.
        
        Args:
            writer: pandas ExcelWriter object.
            sheet_name: Name of the worksheet to add filter to.
            df: DataFrame corresponding to the sheet (used to determine data range).
            has_subtotals: Whether the sheet has subtotal rows. Defaults to False.
            has_superheaders: Whether the sheet has superheader rows. Defaults to False.
        """
        ws = writer.sheets[sheet_name]
        nrows, ncols = df.shape
        header_row = ExcelFormatter.get_header_row(has_subtotals=has_subtotals, has_superheaders=has_superheaders)
        data_end_row = nrows + (1 if has_subtotals else 0) + (1 if has_superheaders else 0)
        ws.autofilter(header_row, 0, data_end_row, ncols - 1)

    @staticmethod
    def clean_col_titles(s: str) -> str:
        """Convert snake_case database column names to readable title case.
        
        Args:
            s: Snake_case column name from database (e.g., 'exp_details_ph_str').
        
        Returns:
            str: Cleaned, title-cased column name with special handling for chemical terms (e.g., 'pH', 'SMILES', 'CASRN').
        """
        val = re.sub(r"exp_details_", "", s)
        val = re.sub(r"_str", "", val)
        val = " ".join(val.split("_")).title()
        val = re.sub(r"(\W*)Ph(\W*)", r"\1pH\2", val)
        val = re.sub(r"\sId\s?$", " ID", val)
        val = re.sub(r"\sCid\s?$", " CID", val)
        val = re.sub(r"Qc", "QC", val)
        val = re.sub(r"Url", "URL", val)
        val = re.sub(r"Doi", "DOI", val)
        val = re.sub(r"Dtxsid", "DTXSID", val)
        val = re.sub(r"Dtxrid", "DTXRID", val)
        val = re.sub(r"Dtxcid", "DTXCID", val)
        val = re.sub(r"Casrn", "CASRN", val)
        val = re.sub(r"(\W*)Cas(\W*)", r"\1CAS\2", val)
        val = re.sub(r"Qsar", "QSAR", val)
        val = re.sub(r"Smiles", "SMILES", val)
        val = re.sub(r"Pubchem", "PubChem", val)
        val = re.sub(r"Ad(\W*)", "AD\1", val)
        return val

    @staticmethod
    def handle_accidental_formulas(df: pd.DataFrame, how: str="formula") -> pd.DataFrame:
        """Escape special characters in cells that Excel would interpret as formula starts.
        
        Prevents cells starting with '=', '+', '-', or '@' from being interpreted as formulas.
        
        Args:
            df: DataFrame to process.
            how: Escaping method - 'formula' wraps in ="..." formula, 'quote' prefixes with single quote. Defaults to 'formula'.
        
        Returns:
            pd.DataFrame: DataFrame with escaped formula characters in all cells and column names.
        """
        if how == "formula":
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f'="{x}"' if isinstance(x, str) and x and x[0] in ('=', '+', '-', '@') else x
                )
                if col[0] in ("=", "+", "-", "@"):
                    df = df.rename(columns={col: f'="{col}"'})
        elif how == "quote":
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"'{x}" if isinstance(x, str) and x and x[0] in ('=', '+', '-', '@') else x
                )
                if col[0] in ("=", "+", "-", "@"):
                    df = df.rename(columns={col: f"'{col}"})
        
        return df
    
    @staticmethod
    def add_hyperlinks_to_sheet(
        writer: pd.ExcelWriter,
        source_sheet: str,
        target_sheet: str,
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        link_column: str = "Exp Prop ID",
        has_subtotals: bool=True,
        source_has_superheaders: bool=False,
        target_has_superheaders: bool=False
        ) -> None:
        """
        Add hyperlinks from source sheet to target sheet based on a matching column.
        Uses xlsxwriter to add hyperlinks. Avoids needing to close and reopen the file,
        and Windows specific dependencies.
        This approach preserves chart formatting by avoiding openpyxl's file manipulation.
        
        Args:
            writer: Excel writer instance
            source_sheet: Name of the source sheet (e.g., "Training CV Predictions")
            target_sheet: Name of the target sheet (e.g., "Records")
            df_source: Source dataframe (used to verify column exists and get values)
            df_target: Target dataframe (used to create the mapping)
            link_column: Column name to match on (must exist in both dataframes)
        """
        logging.info(f"Processing hyperlinks for {source_sheet}")
        
        
        if df_source is None or df_source.empty:
            logging.error(f"Source dataframe {source_sheet} is empty or None. Skipping hyperlinks.")
            return
        
        if link_column not in df_source.columns:
            logging.error(f"Column '{link_column}' not found in source dataframe {source_sheet}.")
            return
        
        if link_column not in df_target.columns:
            logging.error(f"Column '{link_column}' not found in target dataframe {target_sheet}.")
            return
        
        # Get the worksheets
        wb = writer.book
        ws_source = writer.sheets[source_sheet]
                
        # Find the column index in the target sheet for the link_column (1-based for Excel)
        target_link_col_index = df_target.columns.get_loc(link_column) + 1
        target_link_col_letter = xl_rowcol_to_cell(0, target_link_col_index - 1)[0]  # Get the column letter using xlsxwriter utility

        # Create a mapping of link_column values to row numbers in target sheet
        # df_target is 0-indexed, but Excel rows are 1-indexed (with row 1 being header)
        target_row_map = {}
        for idx, val in enumerate(df_target[link_column]):
            excel_row = idx + ExcelFormatter.get_data_start_row(has_subtotals=has_subtotals, has_superheaders=target_has_superheaders) + 1  # Accounts for header and optional subtotal row
            key = str(val).strip() if pd.notna(val) else None
            target_row_map[key] = excel_row
        
        logging.debug(f"Created target mapping with {len(target_row_map)} entries for {target_sheet}")
        logging.debug(f"Target link column: '{link_column}' at Excel column {target_link_col_letter}")
        
        # Find the source column letter for the link column (where we'll add hyperlinks)
        source_link_col_index = df_source.columns.get_loc(link_column) + 1
        source_link_col_letter = xl_rowcol_to_cell(0, source_link_col_index - 1)[0]  # Get the column letter using xlsxwriter utility
        
        # Add hyperlinks to source sheet
        hyperlinks_added = 0
        unmatched_count = 0

        try:
            for src_idx, src_val in enumerate(df_source[link_column]):
                src_val_str = str(src_val).strip() if pd.notna(src_val) else None
                excel_row = src_idx + ExcelFormatter.get_data_start_row(has_subtotals=has_subtotals, has_superheaders=source_has_superheaders) + 1  # Accounts for header and optional subtotal row and superheader row in Records
                
                if src_val_str and src_val_str in target_row_map:
                    target_row = target_row_map[src_val_str]
                    
                    # Create the cell reference for the hyperlink
                    # Format for SubAddress: SheetName!CellAddress (no #)
                    address = f"internal:'{target_sheet}'!{target_link_col_letter}{target_row}"
                    
                    try:
                        # Get the cell range in the source sheet using Excel's Range method
                        cell_range = f"{source_link_col_letter}{excel_row}"
                        
                        # Add the hyperlink using Excel COM API
                        screen_tip = f"Link to {target_sheet} Row {target_row}"

                        # Catch UserWarning from xlsxwriter when hyperlink limit is exceeded
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            ws_source.write_url(cell_range, url=address, string=src_val_str, tip=screen_tip)
                            
                            # Check if a UserWarning about hyperlinks was raised
                            if w and any(issubclass(warning.category, UserWarning) for warning in w):
                                logging.warning(f"Hyperlink limit exceeded in {source_sheet}. Maximum hyperlinks reached. Skipping remaining {len(df_source) - src_idx} rows.")
                                # Count remaining unmatched cells and exit loop
                                unmatched_count += sum(1 for idx in range(src_idx, len(df_source)) if str(df_source[link_column].iloc[idx]).strip() in target_row_map)
                                break

                        hyperlinks_added += 1
                    except Exception as e:
                        logging.warning(f"Failed to write hyperlink for {source_sheet} row {excel_row}: {e}")
                else:
                    unmatched_count += 1
        
        except Exception as e:
            logging.error(f"Error adding hyperlinks: {e}")
        
        logging.info(f"Added {hyperlinks_added} hyperlinks from {source_sheet} to {target_sheet} ({unmatched_count} unmatched)")


# ============================================================
# CHART BUILDER HELPER CLASS
# ============================================================

class ChartBuilder:
    """Handles chart creation and axis calculations for prediction plots.
    
    Provides static methods for creating scatter plots with equal axes, calculating optimal
    axis bounds with padding, and computing nice integer tick spacing for chart axes.
    """
    
    @staticmethod
    def nice_integer_major_unit(span: int, target_ticks: int=5) -> int:
        """Calculate a nice integer major unit for chart axis tick spacing.
        
        Args:
            span: The range of values on the axis.
            target_ticks: Desired number of tick marks. Defaults to 5.
        
        Returns:
            int: A nice round number for axis tick spacing (e.g., 1, 2, 5, 10, 50, etc.).
        """
        raw = max(1, int(math.ceil(span / max(1, target_ticks))))
        exp = int(math.floor(math.log10(raw))) if raw > 0 else 0
        base = raw / (10 ** exp)
        if base <= 1:
            step_base = 1
        elif base <= 2:
            step_base = 2
        elif base <= 5:
            step_base = 5
        else:
            step_base = 10
        return int(step_base * (10 ** exp))

    @staticmethod
    def compute_equal_axis_bounds(
        x_values: Iterable[float],
        y_values: Iterable[float],
        pad_ratio: float=0.02,
        integer_ticks: bool=True,
        log_plot: bool=True,
        target_ticks: int=5,
    ) -> Tuple[float, float, Optional[float]]:
        """Calculate equal bounds for both axes of a scatter plot with padding.
        
        Ensures both axes span the same range (min to max) for unbiased visual comparison.
        
        Args:
            x_values: X-axis values (numeric).
            y_values: Y-axis values (numeric).
            pad_ratio: Fraction of data range to use for padding. Defaults to 0.02 (2%).
            integer_ticks: Whether to round bounds to integers. Defaults to True.
            log_plot: Whether plot uses logarithmic scale. Defaults to True.
            target_ticks: Desired number of tick marks on axis. Defaults to 5.
        
        Returns:
            Tuple[float, float, Optional[float]]: (min_bound, max_bound, major_unit) where major_unit is None for log plots.
        """
        xs = [float(v) for v in x_values if v is not None and v == v]
        ys = [float(v) for v in y_values if v is not None and v == v]
        if not xs or not ys:
            raise ValueError("No numeric values provided for axis scaling.")
    
        mn = min(min(xs), min(ys))
        mx = max(max(xs), max(ys))
    
        if mn == mx:
            mn -= 0.5
            mx += 0.5
        else:
            span = mx - mn
            mn -= span * pad_ratio
            mx += span * pad_ratio
    
        if integer_ticks:
            int_min = math.floor(mn)
            int_max = math.ceil(mx)
            if int_min == int_max:
                int_min -= 1
                int_max += 1
            if log_plot:
                major_unit = 1
            else:
                major_unit = ChartBuilder.nice_integer_major_unit(int_max - int_min, target_ticks=target_ticks)
            
            return float(int_min), float(int_max), float(major_unit)
    
        return mn, mx, None
    
    @staticmethod
    def get_min_max(unit_name: str, exps: Iterable[float], preds: Iterable[float]) -> Tuple[float, float]:
        min_value = min(min(exps), min(preds))
        max_value = max(max(exps), max(preds))
        # Check if "log" is in unit_name
        if "log" in unit_name.lower():
            min_int = int(np.floor(min_value))
            max_int = int(np.ceil(max_value))
            # Determine if padding is needed
            if (min_value - min_int) < 0.5:
                min_value = min_int - 1
            else:
                min_value = min_int
            if (max_int - max_value) < 0.5:
                max_value = max_int + 1
            else:
                max_value = max_int
            
            # ax.set_xticks(range(min_value, max_value + 1))
            # ax.set_yticks(range(min_value, max_value + 1))
            
        elif unit_name == "°C":
            min_value = (np.floor(min_value / 50) * 50) - 50
            max_value = (np.ceil(max_value / 50) * 50) + 50
        
        else:
            padding = (max_value - min_value) * 0.05
            min_value -= padding
            max_value += padding
            
        return (min_value, max_value)
    
    @staticmethod
    def get_major_unit(unit_name: str, min_limit: int, max_limit: int) -> int:
        """Get the major tick unit for the axis.
        
        Args:
            unit_name: 
            min_limit: 
            max_limit: 

        Returns:
            int: The major tick unit for the axis.
        """
        if "log" in unit_name.lower():
            return 1
        else:
            return (max_limit - min_limit) // 5

    @staticmethod
    def add_plot(
        writer: Any,
        workbook: Any,
        sheet_name: str,
        sheet_name_plot: str,
        df: pd.DataFrame,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        chart_size_px: int=520,
        pad_ratio: float=0.02,
        integer_ticks: bool=True,
        log_plot: bool=True,
        yx_offset_rows: int=3,
        property_name: Optional[str] = None,
        property_units: Optional[str] = None,
        has_subtotals: bool = False,
        has_superheaders: bool = False) -> None:
        """Add a scatter plot comparing predicted vs observed/experimental values.
        
        Creates a square scatter plot with equal axes, data points, and y=x reference line.
        
        Args:
            writer: pandas ExcelWriter object.
            workbook: xlsxwriter workbook object.
            sheet_name: Name of worksheet containing the data.
            sheet_name_plot: Name of worksheet where chart will be inserted.
            df: DataFrame with data columns.
            x_col: Column name for x-axis ('Exp', 'Observed', etc.). Defaults to 'Exp'.
            y_col: Column name for y-axis ('Pred', 'Predicted', etc.). Defaults to 'Pred'.
            chart_size_px: Square chart size in pixels. Defaults to 520.
            pad_ratio: Axis padding as fraction of data range. Defaults to 0.02.
            integer_ticks: Use integer-based tick spacing. Defaults to True.
            log_plot: Use log scale (affects tick calculation). Defaults to True.
            yx_offset_rows: Empty rows between data and y=x reference line points. Defaults to 3.
            property_name: Property name for chart title (e.g., 'log Koc'). Defaults to None.
            property_units: Property units for axis labels (e.g., 'mg/L'). Defaults to None.
            has_subtotals: Whether data rows include subtotal formulas. Defaults to False.
            has_superheaders: Whether sheet has superheader rows. Defaults to False.
        """
        worksheet = writer.sheets[sheet_name]
        worksheet_plot = writer.sheets[sheet_name_plot]
        worksheet.hide_gridlines(2)
        nrows = len(df)
        
        data_start_row = ExcelFormatter.get_data_start_row(has_subtotals=has_subtotals, has_superheaders=has_superheaders)
        
        if x_col is None:
            x_col = "Exp"
            x_col_name = "Experimental"
        elif x_col == "Exp":
            x_col_name = "Experimental"
        else:
            x_col_name = x_col
        if y_col is None:
            y_col = "Pred"
            y_col_name = "Predicted"
        elif y_col == "Pred":
            y_col_name = "Predicted"
        else:
            y_col_name = y_col
        try:
            x_col_idx = df.columns.get_loc(x_col)
            y_col_idx = df.columns.get_loc(y_col)
        except KeyError as e:
            logging.error(f"Dataframe columns:\n\t{list(df.columns)}")
            raise ValueError(f"Column not found: {e}")

        # mn, mx, major_unit = ChartBuilder.compute_equal_axis_bounds(
        #     df[x_col], df[y_col], pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=log_plot, target_ticks=5)
        
        mn, mx = ChartBuilder.get_min_max(unit_name=property_units, exps=df[x_col], preds=df[y_col])
        major_unit = ChartBuilder.get_major_unit(unit_name=property_units, min_limit=mn, max_limit=mx)
        
        chart = workbook.add_chart({"type":"scatter", "subtype":"straight_with_markers"})
        title = f"{sheet_name} for {property_name}" if property_name is not None else f"{y_col.capitalize()} vs {x_col.capitalize()}"
        title_len = len(title)
        title_font_size = 18 - 4*(title_len // 30)
        series_name = f"{y_col.capitalize()} vs {x_col.capitalize()}"
        chart.set_title({
            "name": title,
            "overlay": False,
            "name_font": {
                "size": title_font_size
            }
            })
        chart.set_style(10)
        
        data_end_row = data_start_row + nrows - 1
        chart.add_series({
                "name":series_name,
                "categories":[sheet_name, data_start_row, x_col_idx, data_end_row, x_col_idx],
                "values":[sheet_name, data_start_row, y_col_idx, data_end_row, y_col_idx],
                "marker":{
                    "type":"circle",
                    "size":6,
                    "border":{
                        "color":"#000000",
                        "width": 0.25
                        }
                    },
                "line":{
                    "none":True
                    }
                })
        
        yx_row_start = data_end_row + 1 + yx_offset_rows
        worksheet.write_number(yx_row_start, x_col_idx, mn)
        worksheet.write_number(yx_row_start, y_col_idx, mn)
        worksheet.write_number(yx_row_start + 1, x_col_idx, mx)
        worksheet.write_number(yx_row_start + 1, y_col_idx, mx)
        
        chart.add_series({
                "name":"y = x",
                "categories":[sheet_name, yx_row_start, x_col_idx, yx_row_start + 1, x_col_idx],
                "values":[sheet_name, yx_row_start, y_col_idx, yx_row_start + 1, y_col_idx],
                "marker":{"type":"none"},
                "line":{"color":"#1f4e79", "width":2.25}})
        
        x_axis_label = x_col if property_units is None else f"{x_col_name} ({property_units})"
        x_axis_opts = {
            "name":x_axis_label,
            "min":mn, "max":mx,
            "num_format":"0",
            "crossing": "min",
            "major_gridlines":{"visible":False},
            "minor_gridlines":{"visible":False}}
        y_axis_label = y_col if property_units is None else f"{y_col_name} ({property_units})"
        y_axis_opts = {
            "name":y_axis_label,
            "min":mn, "max":mx,
            "num_format":"0",
            "crossing": "min",
            "major_gridlines":{"visible":False},
            "minor_gridlines":{"visible":False}}
        if major_unit is not None:
            x_axis_opts["major_unit"] = major_unit
            y_axis_opts["major_unit"] = major_unit
        
        chart.set_x_axis(x_axis_opts)
        chart.set_y_axis(y_axis_opts)
                
        chart.set_size({"width":chart_size_px, "height":chart_size_px})
        chart.set_plotarea({
                "fill":{"none":True},
                "border":{"color":"#666666", "width":1.0},
                "layout":{
                    "x":0.12,
                    "y":0.10,
                    "width":0.84,
                    "height":0.80}})
        chart.set_legend({
                "overlay":True,
                "layout":{"x":0.7, "y":0.75, "width":0.25, "height":0.1}})
        
        if sheet_name == sheet_name_plot:
            chart_start_col = len(df.columns)
            chart_position = xl_rowcol_to_cell(data_start_row, chart_start_col)
            worksheet_plot.insert_chart(chart_position, chart, {"x_offset":20, "y_offset":10})
        else:
            worksheet_plot.insert_chart(0, 0, chart, {'x_scale': 1.0, 'y_scale': 1.0})


# ============================================================
# DATA QUERIER HELPER CLASS
# ============================================================

class DataQuerier:
    """Handles database connections and data retrieval for models and experimental data.
    
    Manages SQLAlchemy engine/session creation, model queries, property value retrieval,
    descriptor value retrieval, and all sheet dataframe generation. Can be initialized with
    pre-loaded data for custom workflows or configured to query the database on-demand.
    """
    
    def __init__(
            self,
            model_id: Optional[int] = None,
            engine: Optional[Any] = None,
            session: Optional[Any] = None,
            query_args: Optional[dict[str, dict[str, Any]]] = None,
            model: Optional[Any] = None,
            df_pv: Optional[pd.DataFrame] = None,
            df_gmd: Optional[pd.DataFrame] = None,
            df_gmd_external: Optional[pd.DataFrame] = None,
            experimental_parameters: Optional[pd.DataFrame] = None
        ):
        """Initialize DataQuerier for database access and data retrieval.

        Either a model_id or a model object should be passed at initialization.
        
        Args:
            model_id: Database ID of the model. Defaults to None.
            engine: SQLAlchemy engine for database connection. If None, creates new engine from environment variables.
            session: SQLAlchemy session for queries. If None, creates new session from engine.
            query_args: Dictionary of query argument sets for different query methods. Defaults to empty dict.
            model: Pre-loaded Model object (if None, will be queried from database).
            df_pv: Pre-loaded property values dataframe (if None, will be queried from database).
            df_gmd: Pre-loaded descriptor values dataframe (if None, will be queried from database).
            df_gmd_external: Pre-loaded external dataset descriptor values (if None, will be queried from database).
        """
        self.model_id = model_id

        self.engine = engine or DataQuerier.getEngine()
        self.session = session or DataQuerier.getSession(self.engine)

        self.model = model
        self.dataset_name = None
        self.dataset_name_external = None

        self.df_pv = df_pv
        self.df_gmd = df_gmd
        self.df_gmd_external = df_gmd_external

        self.experimental_parameters = experimental_parameters

        self.query_args = query_args if query_args is not None else {}

        self.post_init()

    def post_init(self):
        logging.info("Initializing DataQuerier...")
        self.model = self.query_model()
        if self.model is not None:
            self.dataset_name = getattr(self.model, "datasetName", None)
            self.dataset_name_external = getattr(self.model, "external_dataset_name", None)
        
        self.query_df_pv(**self.query_args.get("query_df_pv", {}))
        self.query_df_gmd(external=False)
        self.query_df_gmd(external=True)
        self.query_experimental_parameters()

    @staticmethod
    def getEngine(connect_args: Optional[dict] = {}) -> Any:
        """Create SQLAlchemy engine for PostgreSQL connection.
        
        Uses environment variables for database credentials:
            DEV_QSAR_USER, DEV_QSAR_PASS, DEV_QSAR_HOST, DEV_QSAR_PORT, DEV_QSAR_DATABASE
        
        Args:
            connect_args: Additional arguments to pass to URL.create(). Defaults to empty dict.
        
        Returns:
            Any: SQLAlchemy engine configured for PostgreSQL connection.
        """
        connect_url = URL.create(
            drivername='postgresql+psycopg2',
            username=os.getenv('DEV_QSAR_USER'),
            password=os.getenv('DEV_QSAR_PASS'),
            host=os.getenv('DEV_QSAR_HOST', 'localhost'),
            port=os.getenv('DEV_QSAR_PORT', 5432),
            database=os.getenv('DEV_QSAR_DATABASE'),
            **connect_args
        )
        engine = create_engine(connect_url, echo=False)
        return engine

    @staticmethod
    def getSession(engine: Any=None) -> Any:
        """Create SQLAlchemy session for database queries.
        
        Args:
            engine: SQLAlchemy engine instance. If None, creates new engine.
        
        Returns:
            Any: SQLAlchemy session object for executing queries.
        """
        if engine is None:
            engine = DataQuerier.getEngine()
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    
    def query_model(self) -> Optional[Model]:
        """Query the database for the model object.
        
        Returns:
            Optional[Model]: Model object with all properties and data, or None if query fails.
        """
        if self.model is not None:
            return self.model

        try:            
            logging.info(f"Loading model {self.model_id} from database")
            initializer: ModelInitializer = ModelInitializer()
            model: Model = initializer.initModel(self.model_id)
            logging.info(f"Done loading model {self.model_id} from database")
            
            if model is None:
                logging.error(f"Failed to load model {self.model_id}")
                return None

            self.model = model
            return model
        
        except Exception as e:
            logging.error(f"Error retrieving model {self.model_id}: {e}")
            traceback.print_exc()
            return None

    def query_df_pv(self, snapshot_id: int = 4, duplicate_strategy: str = "id_suffix") -> Optional[pd.DataFrame]:
        """Query property values dataframe from the database using ExpDataGetter.
        
        Args:
            snapshot_id: Database snapshot ID to use for query. Defaults to 4.
            duplicate_strategy: How to handle duplicate property values - 'id_suffix' or 'keep_first'. Defaults to 'id_suffix'.
        
        Returns:
            Optional[pd.DataFrame]: Property values dataframe with experimental data, or None if query fails.
        """
        if self.df_pv is not None:
            return self.df_pv
        elif self.dataset_name is not None:
            dataset_name = self.dataset_name
        else:
            return None
        
        try:
            logging.info("Loading df_pv from database")
            edg = ExpDataGetter()
            df_pv, _ = edg.get_mapped_property_values(self.session, dataset_name, snapshot_id, duplicate_strategy=duplicate_strategy)
            logging.info(f"Done loading df_pv from database")
            
            if df_pv is None:
                logging.error(f"Failed to query df_pv")
                return None

            self.df_pv = df_pv
            return df_pv
        
        except Exception as e:
            logging.error(f"Error retrieving df_pv: {e}")
            traceback.print_exc()
            return None

    def query_df_gmd(self, external: bool = False) -> Optional[pd.DataFrame]:
        """Query descriptor values (molecular properties) dataframe from the database.
        
        Args:
            external: If False, retrieves training dataset descriptors; if True, retrieves external dataset descriptors. Defaults to False.
        
        Returns:
            Optional[pd.DataFrame]: Descriptor values dataframe (GMD), or None if query fails.
        """
        if external and self.df_gmd_external is not None:
            # If external df_gmd already initialized, return it immediately
            return self.df_gmd_external
        elif external and self.dataset_name_external is not None:
            # If external df_gmd not already initialized and model has an external dataset, prepare to query it
            dataset_name = self.dataset_name_external
        elif external and self.dataset_name_external is None:
            # If external df_gmd not already initialized and model doesn't have an external dataset, return nothing
            logging.warning("Provided model has no external dataset provided, skipping initialization of external GMD\n\t(If this is not desired, set dataset_name_external on the DataQuerier object before running query_df_gmd with external = True)")
            return None
        elif not external and self.df_gmd is not None:
            # If df_gmd already initialized, return it immediately
            return self.df_gmd
        elif not external and self.dataset_name is not None:
            # If df_gmd not already initialized, prepare to query it from the dataset name
            dataset_name = self.dataset_name
        else:
            # If none of the above cases occurred, there is some kind of error
            logging.error("Must set dataset_name on DataQuerier object prior to querying GMD")
            return None
        
        try:
            logging.info("Loading df_gmd from database")
            df_gmd = getMappedDatapoints(self.session, dataset_name)
            logging.info(f"Done loading df_gmd from database")
            
            if df_gmd is None:
                logging.error(f"Failed to query df_gmd")
                return None

            if external:
                self.df_gmd_external = df_gmd
            else:
                self.df_gmd = df_gmd
            return df_gmd
        
        except Exception as e:
            logging.error(f"Error retrieving df_gmd: {e}")
            traceback.print_exc()
            return None
        
    def query_experimental_parameters(self) -> Optional[pd.DataFrame]:
        """Query database for experimental parameters.
        
        Returns:
            pd.DataFrame: DataFrame with experimental parameters and their descriptions.
        """
        if self.experimental_parameters is not None:
            return self.experimental_parameters

        logging.info("Querying experimental parameters from database")

        sql = text("""
            select p.name as "Field", p.description as "Description"
            from exp_prop.parameters p;
        """)

        param_rows = self.session.execute(sql).mappings().all()
        if not param_rows:
            logging.warning("No experimental parameters found in the database.")
            return None

        experimental_parameters = pd.DataFrame(param_rows)
        experimental_parameters["Field"] = experimental_parameters["Field"].apply(lambda x: ExcelFormatter.clean_col_titles(x))
        experimental_parameters = experimental_parameters[["Field", "Description"]]

        self.experimental_parameters = experimental_parameters

        return self.experimental_parameters

    def query_cover_sheet_df(self) -> pd.DataFrame:
        """Query database for model summary information to populate the cover sheet.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model metadata (name, property, dataset, etc.).
        """
        logging.info(f"Building Cover Sheet from Model {self.model_id}")
        model = self.query_model()
        model_id = getattr(model, "modelId", None)
        model_id = int(model_id) if model_id is not None else None

        summary_dict = {
            "Model ID": [model_id],
            "Model Name": [model.modelName],
            "Property Name": [model.propertyName],
            "Property Description": [model.propertyDescription],
            "Property Units": [model.unitsModel],
            "Dataset Name": [model.datasetName],
            "Dataset Description": [model.datasetDescription],
            "nTraining": [model.num_training],
            "nTest": [model.num_prediction],
            "Method Name": [model.modelMethod],
            "Method Description": [model.modelMethodDescription],
            "Applicability Domain": [model.applicabilityDomainName]
        }
        if getattr(model, "external_dataset_name", False):
            summary_dict["External Dataset Name"] = [model.external_dataset_name]
            summary_dict["External Dataset Description"] = [model.external_dataset_description]
            summary_dict["nExternal"] = [model.num_external]
        summary = pd.DataFrame(summary_dict)

        # self.cover_sheet_df = summary

        logging.info(f"Finished building Cover Sheet from Model {self.model_id}")

        return summary

    def query_statistics_df(self) -> pd.DataFrame:
        """
        Query database for model performance statistics.
        
        Retrieves training, cross-validation, test, and applicability domain statistics from the database.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model performance metrics rounded to 2 decimal places.
        """
        logging.info(f"Building Statistics from Model {self.model_id}")

        model = self.query_model()

        statistics_dict = {
            "nTraining": [model.num_training],
            "nTest": [model.num_prediction],
            "RSQ_Training": [model.modelStatistics.get("PearsonRSQ_Training", None)],
            "RMSE_Training": [model.modelStatistics.get("RMSE_Training", None)],
            "MAE_Training": [model.modelStatistics.get("MAE_Training", None)],
            "RSQ_CV_Training": [model.modelStatistics.get("PearsonRSQ_CV_Training", None)],
            "RMSE_CV_Training": [model.modelStatistics.get("RMSE_CV_Training", None)],
            "MAE_CV_Training": [model.modelStatistics.get("MAE_CV_Training", None)],
            "RSQ_Test": [model.modelStatistics.get("PearsonRSQ_Test", None)],
            "RMSE_Test": [model.modelStatistics.get("RMSE_Test", None)],
            "MAE_Test": [model.modelStatistics.get("MAE_Test", None)],
            "Q2_Test": [model.modelStatistics.get("Q2_Test", None)],
            "MAE_Test_Inside_AD": [model.modelStatistics.get("MAE_Test_inside_AD", None)],
            "MAE_Test_Outside_AD": [model.modelStatistics.get("MAE_Test_outside_AD", None)],
            "Coverage_Test": [model.modelStatistics.get("Coverage_Test", None)]
        }
        statistics = pd.DataFrame(statistics_dict).apply(lambda x: round(x, 2))

        # self.statistics_df = statistics
        
        logging.info(f"Finished building Statistics from Model {self.model_id}")

        return statistics
    
    def query_records_df(self) -> pd.DataFrame:
        """
        Query database for detailed experimental records.
                
        Returns:
            pd.DataFrame: Records dataframe from database (currently returns empty result).
        """
        logging.info(f"Building Records from Model {self.model_id}")
        
        df_pv = self.query_df_pv()
        records_df = DataTransformer.get_records_df(df_pv)
        logging.info(f"Finished building Records from Model {self.model_id}")

        # self.records_df = records_df

        # DataTransformer.get_superheaders(records_df)

        return records_df
    
    def query_model_descriptors_df(self) -> pd.DataFrame:
        """
        Query database for model descriptors and their definitions.
                
        Returns:
            pd.DataFrame: Model descriptors from database (currently returns empty result).
        """
        logging.info(f"Building Model Descriptors from Model {self.model_id}")
        
        model = self.query_model()

        results_dict = {}
        results_dict["model_details"] = {}
        results_dict["model_details"]["embedding"] = ["Intercept", *model.embedding]

        method_name = getattr(model, "qsar_method", False) or getattr(model, "regressor_name", False) or ""
        # method_name = getattr(model, "regressor_name", "") if method_name == "" else method_name
        if any(method in method_name for method in ["reg", "las", "gcm"]):
            coefficients_df = DataTransformer.get_model_coefficients(model)
            results_dict["model_details"]["model_coefficients"] = coefficients_df
        else:
            logging.warning(f"Model type {method_name} does not support coefficient retrieval, coefficient values will not be added to Model Descriptors sheet")
        
        model_descriptors = DataTransformer.get_model_descriptors_df(results_dict)

        # self.model_descriptors_df = model_descriptors

        logging.info(f"Finished building Model Descriptors from Model {self.model_id}")

        return model_descriptors
    
    def query_model_descriptor_values_df(self) -> pd.DataFrame:
        """
        Query database for predictions and descriptor values.
                
        Returns:
            pd.DataFrame: Predictions and descriptor values from database (currently returns empty result).
        """
        logging.info(f"Building Model Descriptor Values from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd(external=False)

        df_preds_training_cv = model.df_preds_training_cv.rename(columns={"canon_qsar_smiles": "id"})

        training = pd.merge(model.df_training, df_gmd, left_on="ID", right_on="canon_qsar_smiles", how="left")
        training = pd.merge(training, df_preds_training_cv, left_on="ID", right_on="id", how="left")

        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_col = np.zeros(len(model.df_training), dtype=int)
        for fold_index, (train_index, val_index) in enumerate(kfold_splitter.split(model.df_training)):
            fold_col[val_index] = fold_index + 1

        training["Fold"] = fold_col
        training["Set"] = training.Fold.apply(lambda x: f"Training CV, Fold {x}")

        df_preds_test = model.df_preds_test.rename(columns={"canon_qsar_smiles": "id"})

        test = pd.merge(model.df_prediction, df_gmd, left_on="ID", right_on="canon_qsar_smiles", how="left")
        test = pd.merge(test, df_preds_test, left_on="ID", right_on="id", how="left")

        test["Set"] = "Test"

        temp = pd.concat([test, training], ignore_index=True)

        headers = model.embedding
        header_columns = {}
        for header in headers:
            header_columns[header] = temp[header]

        model_descriptor_values_dict = {
            "Exp Prop ID": temp["qsar_exp_prop_property_values_id_first"],
            "DTXCID": temp["dtxcid"],
            "CASRN": temp["casrn"],
            "Preferred Name": temp["preferred_name"],
            "Canonical QSAR Ready Smiles": temp["canon_qsar_smiles"],
            f"Observed ({model.unitsModel})": temp["exp"],
            f"Predicted ({model.unitsModel})": temp["pred"],
            "Set": temp["Set"],
            **header_columns
        }
        model_descriptor_values_df = pd.DataFrame(model_descriptor_values_dict)
        model_descriptor_values_df = ExcelFormatter.handle_accidental_formulas(model_descriptor_values_df)

        # self.model_descriptor_values_df = model_descriptor_values_df

        logging.info(f"Finished building Model Descriptor Values from Model {self.model_id}")

        return model_descriptor_values_df
    
    def query_training_cv_predictions_df(self) -> pd.DataFrame:
        """
        Query database for training set cross-validation predictions.
                
        Returns:
            pd.DataFrame: Training predictions from database (currently returns empty result).
        """
        logging.info(f"Building Training CV Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd(external=False)

        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_col = np.zeros(len(model.df_preds_training_cv), dtype=int)
        for fold_index, (train_index, val_index) in enumerate(kfold_splitter.split(model.df_preds_training_cv)):
            fold_col[val_index] = fold_index + 1

        df_preds_training_cv = model.df_preds_training_cv.rename(columns={"canon_qsar_smiles": "id"})

        temp = pd.merge(df_preds_training_cv, df_gmd, left_on="id", right_on="canon_qsar_smiles", how="left")

        training_cv_predictions_dict = {
            "Exp Prop ID": temp["qsar_exp_prop_property_values_id_first"],
            "Canon QSAR SMILES": temp["canon_qsar_smiles"],
            "Exp": temp["exp"],
            "Pred": temp["pred"],
            "CV Fold": fold_col,
            "DTXCID": temp["dtxcid"],
            "DTXSID": temp["dtxsid"],
            "CASRN": temp["casrn"],
            "Preferred Name": temp["preferred_name"],
            "SMILES": temp["smiles"],
            "Mol Weight": temp["mol_weight"]
        }
        training_cv_predictions_df = pd.DataFrame(training_cv_predictions_dict)

        # self.training_cv_predictions_df = training_cv_predictions_df

        logging.info(f"Finished building Training CV Predictions from Model {self.model_id}")

        return training_cv_predictions_df

    def query_test_set_predictions_df(self) -> pd.DataFrame:
        """
        Query database for test set predictions.
                
        Returns:
            pd.DataFrame: Test predictions from database (currently returns empty result).
        """
        logging.info(f"Building Test Set Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd(external=False)

        df_preds_test = model.df_preds_test.rename(columns={"canon_qsar_smiles": "id"})

        temp = pd.merge(df_preds_test, df_gmd, left_on="id", right_on="canon_qsar_smiles", how="left")

        ads = model.applicabilityDomainName.split(" and ")
        ad_test_columns = {}
        for ad in ads:
            df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                    train_df=model.df_training.copy(), test_df=model.df_prediction.copy(),
                    remove_log_p=model.remove_log_p_descriptors,
                    embedding=model.embedding, applicability_domain=ad,
                    filterColumnsInBothSets=False,
                    returnTrainingAD=False)
            ad_test_columns[f"AD {ad}"] = df_ad_output["AD"]

        test_predictions_dict = {
            "Exp Prop ID": temp["qsar_exp_prop_property_values_id_first"],
            "Canon QSAR SMILES": temp["canon_qsar_smiles"],
            "Exp": temp["exp"],
            "Pred": temp["pred"],
            **ad_test_columns,
            "DTXCID": temp["dtxcid"],
            "DTXSID": temp["dtxsid"],
            "CASRN": temp["casrn"],
            "Preferred Name": temp["preferred_name"],
            "SMILES": temp["smiles"],
            "Mol Weight": temp["mol_weight"]
        }
        test_set_predictions_df = pd.DataFrame(test_predictions_dict)

        # self.test_set_predictions_df = test_set_predictions_df

        logging.info(f"Finished building Test Set Predictions from Model {self.model_id}")

        return test_set_predictions_df
    
    def query_external_predictions_df(self) -> Optional[pd.DataFrame]:
        """
        Query database for external validation set predictions.
                
        Returns:
            pd.DataFrame: External predictions from database (currently returns empty result).
        """
        if self.dataset_name_external is None:
            logging.info(f"External dataset name is None for provided Model (model_id = {self.model_id}), skipping External Predictions sheet")
            return None
        
        logging.info(f"Building External Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd_external = self.query_df_gmd(external=True)

        df_preds_external = model.df_preds_external.rename(columns={"canon_qsar_smiles": "id"})

        temp = pd.merge(df_preds_external, df_gmd_external, left_on="id", right_on="canon_qsar_smiles", how="left")

        ads = model.applicabilityDomainName.split(" and ")
        ad_test_columns = {}
        for ad in ads:
            df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                    train_df=model.df_training.copy(), test_df=model.df_external.copy(),
                    remove_log_p=model.remove_log_p_descriptors,
                    embedding=model.embedding, applicability_domain=ad,
                    filterColumnsInBothSets=False,
                    returnTrainingAD=False)
            ad_test_columns[f"AD {ad}"] = df_ad_output["AD"]

        external_predictions_dict = {
            "Exp Prop ID": temp["qsar_exp_prop_property_values_id_first"],
            "Canon QSAR SMILES": temp["canon_qsar_smiles"],
            "Exp": temp["exp"],
            "Pred": temp["pred"],
            **ad_test_columns,
            "DTXCID": temp["dtxcid"],
            "DTXSID": temp["dtxsid"],
            "CASRN": temp["casrn"],
            "Preferred Name": temp["preferred_name"],
            "SMILES": temp["smiles"],
            "Mol Weight": temp["mol_weight"]
        }
        external_predictions_df = pd.DataFrame(external_predictions_dict)

        # self.external_predictions_df = external_predictions_df

        logging.info(f"Finished building External Predictions from Model {self.model_id}")

        return external_predictions_df


# ============================================================
# DATA TRANSFORMER HELPER CLASS
# ============================================================

class DataTransformer:
    """Handles data transformation, formatting, and dataframe generation for Excel sheets.
    
    Provides static methods for formatting numeric values, transforming experimental metadata,
    generating dataframes for all Excel sheets, and extracting/formatting model coefficients.
    """
    
    @staticmethod
    def set_significant_digits(value: float, significant_digits: int) -> str:
        """Format a numeric value with a specified number of significant figures.
        
        Args:
            value: Numeric value to format.
            significant_digits: Number of significant figures to display. Must be >= 0.
        
        Returns:
            str: Formatted number as string with specified significant figures.
        
        Raises:
            ValueError: If significant_digits < 0.
        """
        if significant_digits < 0:
            raise ValueError("significant_digits must be >= 0")
        
        return f"{value:.{significant_digits}g}"

    @staticmethod
    def get_formatted_value(dvalue: float, nsig: int=3, exp_nsig: int=2) -> str:
        """Format a numeric value with automatic exponential notation when appropriate.
        
        Uses exponential notation for very small (< 0.01) or very large (> 1e3) numbers.
        
        Args:
            dvalue: Numeric value to format, or None.
            nsig: Significant figures for normal notation. Defaults to 3.
            exp_nsig: Significant figures for exponential notation. Defaults to 2.
        
        Returns:
            str: Formatted number, or None if input is None or formatting fails.
        """
        if dvalue is None:
            return None
        try:
            if dvalue != 0 and (abs(dvalue) < 0.01 or abs(dvalue) > 1e3):
                return f"{dvalue:.{exp_nsig}e}"
            
            return DataTransformer.set_significant_digits(dvalue, nsig)
        except:
            return None

    @staticmethod
    def parameter_cols_to_str(parameter_dict: dict, sig_figs: int=3) -> str:
        """Format experimental parameter values into a readable string.
        
        Converts point estimates, ranges, or text values into a single display string.
        
        Args:
            parameter_dict: Dictionary with keys 'value_point_estimate', 'value_min', 'value_max', 'value_text', 'value_units'.
            sig_figs: Significant figures for numeric formatting. Defaults to 3.
        
        Returns:
            str: Formatted parameter string (e.g., '25.5 °C' or '20 < value < 30 mg/L'), or None if no values.
        """
        point_estimate: str = DataTransformer.get_formatted_value(parameter_dict.get('value_point_estimate'), sig_figs)
        str_val_min: str = DataTransformer.get_formatted_value(parameter_dict.get('value_min'), sig_figs)
        str_val_max: str = DataTransformer.get_formatted_value(parameter_dict.get('value_max'), sig_figs)
        str_text: str = parameter_dict.get("value_text")
        value_units: str = parameter_dict.get("value_units")
        value_units = value_units if value_units is not None and value_units != "Text" else ""
        if not str_val_min is None and not str_val_max is None:
            return f"{str_val_min} {value_units} < value < {str_val_max} {value_units}"
        elif not str_val_min is None:
            return f" > {str_val_min} {value_units}"
        elif not str_val_max is None:
            return f" < {str_val_max} {value_units}"
        elif not point_estimate is None:
            return f"{point_estimate} {value_units}"
        elif not str_text is None:
            return str_text
        else:
            return None

    @staticmethod
    def create_parameter_dict(row: pd.Series, param_name: str) -> dict:
        """Extract experimental parameter values from a dataframe row into a dictionary.
        
        Args:
            row: DataFrame row (Series) with exp_details_* columns.
            param_name: Parameter name (e.g., 'ph', 'temperature') to extract values for.
        
        Returns:
            dict: Dictionary with keys 'value_point_estimate', 'value_min', 'value_max', 'value_text', 'value_units'.
        """
        parameter_dict = {}

        point_est_col = f"exp_details_{param_name}_value_point_estimate"
        min_col = f"exp_details_{param_name}_value_min"
        max_col = f"exp_details_{param_name}_value_max"
        text_col = f"exp_details_{param_name}_value_text"
        units_col = f"exp_details_{param_name}_value_units"
        
        if point_est_col in row.index:
            parameter_dict['value_point_estimate'] = None if pd.isna(row[point_est_col]) else row[point_est_col]
        if min_col in row.index:
            parameter_dict['value_min'] = None if pd.isna(row[min_col]) else row[min_col]
        if max_col in row.index:
            parameter_dict['value_max'] = None if pd.isna(row[max_col]) else row[max_col]
        if text_col in row.index:
            parameter_dict['value_text'] = None if pd.isna(row[text_col]) else row[text_col]
        if units_col in row.index:
            parameter_dict["value_units"] = None if pd.isna(row[units_col]) else row[units_col]
        else:
            parameter_dict["value_units"] = ""
                        
        return parameter_dict

    @staticmethod
    def convert_exp_details_to_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Convert experimental detail columns into formatted summary strings.
        
        Args:
            df: DataFrame with exp_details_* columns (from database query).
        
        Returns:
            pd.DataFrame: DataFrame with new exp_details_*_str columns containing formatted strings.
        """
        df_result = df.copy()
        
        parameter_cols = [col for col in df.columns if col.startswith("exp_details_")]
        
        unique_parameters = set()
        for col in parameter_cols:
            match_obj = re.match(r"exp_details_(.*?)_(value.*)", col)
            if match_obj:
                unique_parameters.add(match_obj.group(1))
        
        for parameter in sorted(unique_parameters):
            col_name = f"exp_details_{parameter}_str"
            df_result[col_name] = df.apply(
                lambda row: DataTransformer.parameter_cols_to_str(DataTransformer.create_parameter_dict(row, parameter)),
                axis=1
            )
        
        return df_result

    @staticmethod
    def get_cover_sheet_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """Generate cover sheet dataframe from model results dictionary.
        
        Args:
            results_dict: Dictionary with 'model_details' key containing model metadata.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model summary information.
        """
        cover_sheet_df = {
            "Model ID": [results_dict["model_details"].get("modelId", None)],
            "Model Name": [results_dict["model_details"].get("modelName", None)],
            "Property Name": [results_dict["model_details"].get("propertyName", None)],
            "Property Description": [results_dict["model_details"].get("propertyDescription", None)],
            "Property Units": [results_dict["model_details"].get("unitsModel", None)],
            "Dataset Name": [results_dict["model_details"].get("datasetName", None)],
            "Dataset Description": [results_dict["model_details"].get("datasetDescription", None)],
            "nTraining": [results_dict["model_details"].get("numTraining", None)],
            "nTest": [results_dict["model_details"].get("numPrediction", None)],
            "Method Name": [results_dict["model_details"].get("qsar_method", None)],
            "Method Description": [results_dict["model_details"].get("qsar_method_description", None)],
            "Applicability Domain": [results_dict["model_details"].get("applicabilityDomainName", None)]
        }
        if results_dict["model_details"].get("externalDatasetName", False):
            cover_sheet_df["External Dataset Name"] = [results_dict["model_details"].get("externalDatasetName", None)]
            cover_sheet_df["External Dataset Description"] = [results_dict["model_details"].get("externalDatasetDescription", None)]
            cover_sheet_df["nExternal"] = [results_dict["model_details"].get("numExternal", None)]
        cover_sheet_df = pd.DataFrame.from_dict(cover_sheet_df)
        return cover_sheet_df

    @staticmethod
    def get_statistics_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """Generate statistics dataframe from model results dictionary.
        
        Args:
            results_dict: Dictionary with 'model_statistics' key containing performance metrics.
        
        Returns:
            pd.DataFrame: Single-row dataframe with training, CV, test, and AD statistics.
        """
        statistics_df = {
            "nTraining": [results_dict["model_details"].get("numTraining", None)],
            "nTest": [results_dict["model_details"].get("numPrediction", None)],
            "RSQ_Training": [results_dict["model_statistics"].get("training_stats", {}).get("PearsonRSQ_Training", None)],
            "RMSE_Training": [results_dict["model_statistics"].get("training_stats", {}).get("RMSE_Training", None)],
            "MAE_Training": [results_dict["model_statistics"].get("training_stats", {}).get("MAE_Training", None)],
            "RSQ_CV_Training": [results_dict["model_statistics"].get("cv_stats", {}).get("PearsonRSQ_CV_Training", None)],
            "RMSE_CV_Training": [results_dict["model_statistics"].get("cv_stats", {}).get("RMSE_CV_Training", None)],
            "MAE_CV_Training": [results_dict["model_statistics"].get("cv_stats", {}).get("MAE_CV_Training", None)],
            "RSQ_Test": [results_dict["model_statistics"].get("test_stats", {}).get("PearsonRSQ_Test", None)],
            "RMSE_Test": [results_dict["model_statistics"].get("test_stats", {}).get("RMSE_Test", None)],
            "MAE_Test": [results_dict["model_statistics"].get("test_stats", {}).get("MAE_Test", None)],
            "Q2_Test": [results_dict["model_statistics"].get("test_stats", {}).get("Q2_Test", None)],
            "MAE_Test_Inside_AD": [results_dict["model_statistics"].get("test_stats_AD", {}).get("MAE_Test_inside_AD", None)],
            "MAE_Test_Outside_AD": [results_dict["model_statistics"].get("test_stats_AD", {}).get("MAE_Test_outside_AD", None)],
            "Coverage_Test": [results_dict["model_statistics"].get("test_stats_AD", {}).get("Coverage_Test", None)],
        }
        statistics_df = pd.DataFrame.from_dict(statistics_df)
        return statistics_df

    @staticmethod
    def get_records_df(df_pv: pd.DataFrame) -> pd.DataFrame:
        """Generate experimental records dataframe from property values dataframe.
        
        Args:
            df_pv: Property values dataframe from ExpDataGetter.
        
        Returns:
            pd.DataFrame: Experimental records with identifiers, sources, properties, and measurement details.
        """
        df_pv = DataTransformer.convert_exp_details_to_strings(df_pv)
        str_columns = [col for col in df_pv.columns if col.startswith("exp_details_") and col.endswith("_str")]
        exp_details_columns = {ExcelFormatter.clean_col_titles(col): df_pv[col] for col in str_columns}
        records_df = {
            "Exp Prop ID": df_pv.get("prop_value_id", None),
            "Canon QSAR SMILES": df_pv.get("canon_qsar_smiles", None),
            "Page URL": df_pv.get("direct_url", None),
            "Public Source Name": df_pv.get("public_source_name", None),
            "Public Source URL": df_pv.get("public_source_url", None),
            "Literature Source Citation": df_pv.get("literature_source_citation", None),
            "Literature Source DOI": df_pv.get("literature_source_doi", None),
            "Source DTXRID": df_pv.get("source_dtxrid", None),
            "Source DTXSID": df_pv.get("source_dtxsid", None),
            "Source CASRN": df_pv.get("source_casrn", None),
            "Source Chemical Name": df_pv.get("source_chemical_name", None),
            "Source SMILES": df_pv.get("source_smiles", None),
            "Mapped DTXCID": df_pv.get("mapped_dtxcid", None),
            "Mapped DTXSID": df_pv.get("mapped_dtxsid", None),
            "Mapped CAS": df_pv.get("mapped_casrn", None),
            "Mapped Chemical Name": df_pv.get("mapped_chemical_name", None),
            "Mapped SMILES": df_pv.get("mapped_smiles", None),
            "Mapped Molweight": df_pv.get("mapped_mol_weight", None),
            "Value Original": df_pv.get("prop_value_original", None),
            "Value Max": df_pv.get("value_max", None),
            "Value Min": df_pv.get("value_min", None),
            "Value Point Estimate": df_pv.get("prop_value", None),
            "Value Units": df_pv.get("prop_unit", None),
            "QSAR Property Value": df_pv.get("qsar_property_value", None),
            "QSAR Property Units": df_pv.get("qsar_property_unit", None),
            **exp_details_columns,
            "Notes": df_pv.get("notes", None),
            "QC Flag": df_pv.get("qc_flag", None),
            "Flag Reason": df_pv.get("flag_reason", None),
        }
        records_df = pd.DataFrame.from_dict(records_df)
        return records_df

    @staticmethod
    def get_superheaders(records_df: pd.DataFrame, experimental_details_columns: Optional[list] = None) -> dict:
        """Generate logical groupings (superheaders) for records columns.
        
        Args:
            records_df: Records dataframe to categorize.
            experimental_details_columns: List of experimental details column names. 
        
        Returns:
            dict: Dictionary mapping superheader names (e.g., 'Identifiers', 'Source Metadata') to lists of column names.
        """
        if experimental_details_columns is None:
            experimental_details_columns = ["Measurement Method", "Media", "Percentage Organic Carbon", "Percentage Organic Matter", "pH", "Soil Type", "Temperature", "Testing Conditions", "Notes", "Pressure", "Reliability"]

        superheaders = {
            "Identifiers": ["Exp Prop ID", "Canon QSAR SMILES"],
            "Literature Source Metadata": ["Page URL", "Public Source Name", "Public Source URL", "Literature Source Citation", "Literature Source DOI"],
            "Source Chemical Metadata": ["Source DTXRID", "Source DTXSID", "Source CASRN", "Source Chemical Name", "Source SMILES"],
            "Mapped DSSTox Record Metadata": ["Mapped DTXCID", "Mapped DTXSID", "Mapped CAS", "Mapped Chemical Name", "Mapped SMILES", "Mapped Molweight"],
            "Property Value Data": ["Value Original", "Value Max", "Value Min", "Value Point Estimate", "Value Units", "QSAR Property Value", "QSAR Property Units"],
            "Experimental Details": experimental_details_columns,
            "Quality Control": ["QC Flag", "Flag Reason"]
        }

        superheaders["Other"] = [col for col in records_df.columns if col not in [val for sublist in superheaders.values() for val in sublist]]

        empty_superheaders = []
        for superheader in superheaders.keys():
            if superheaders[superheader]:
                superheaders[superheader] = [col for col in records_df.columns if col in superheaders[superheader]]
                if not superheaders[superheader]:
                    empty_superheaders.append(superheader)
            else:
                empty_superheaders.append(superheader)
        for superheader in empty_superheaders:
            superheaders.pop(superheader)

        return superheaders

    @staticmethod
    def get_records_field_descriptions_df(experimental_parameters: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Load field descriptions from resource file.

        Args:
            experimental_parameters: Optional dataframe with experimental parameter descriptions.
        
        Returns:
            pd.DataFrame: Records field names and descriptions from 'resources/records_field_descriptions.txt'.
        """
        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        path_segments = [PROJECT_ROOT, "resources", "records_field_descriptions.txt"]
        records_field_descriptions_df = pd.read_csv(os.path.join(*path_segments), sep="\t")
        records_field_descriptions_df.columns = records_field_descriptions_df.columns.str.strip()
        records_field_descriptions_df = records_field_descriptions_df.map(lambda x: x.strip() if isinstance(x, str) else x)

        if experimental_parameters is not None:
            # Merge with experimental parameters if provided
            records_field_descriptions_df = pd.concat([records_field_descriptions_df, experimental_parameters], ignore_index=True)
        else:
            # Merge with local experimental parameters
            path_segments_params = [PROJECT_ROOT, "resources", "experimental_parameters_descriptions.txt"]
            experimental_parameters_df = pd.read_csv(os.path.join(*path_segments_params), sep="\t")
            experimental_parameters_df.columns = experimental_parameters_df.columns.str.strip()
            experimental_parameters_df = experimental_parameters_df.map(lambda x: x.strip() if isinstance(x, str) else x)

            records_field_descriptions_df = pd.concat([records_field_descriptions_df, experimental_parameters_df], ignore_index=True)
        logging.debug(f"records_field_descriptions_df:\n\tFields: {records_field_descriptions_df.Field.tolist()}")
        return records_field_descriptions_df

    @staticmethod
    def get_model_descriptors_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """Generate model descriptors dataframe with definitions and classification.
        
        Args:
            results_dict: Dictionary with 'model_details' containing descriptor embedding and optional coefficients.
        
        Returns:
            pd.DataFrame: Descriptors with definitions, classification, and optionally coefficients.
        """
        model_descriptors_df = pd.DataFrame(results_dict["model_details"]["embedding"], columns=["Symbol"])

        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        path_segments = [PROJECT_ROOT, "resources", "variable definitions-ed.txt"]
        
        variable_definitions_df = pd.read_csv(os.path.join(*path_segments), sep="\t")

        model_descriptors_df['Symbol'] = model_descriptors_df['Symbol'].astype(str)
        variable_definitions_df['Symbol'] = variable_definitions_df['Symbol'].astype(str)

        temp = model_descriptors_df.merge(variable_definitions_df, on="Symbol", how="left")
        temp = temp.rename(columns={"Symbol": "Descriptor", "Category": "Class"})

        try:
            has_coefficients = bool(results_dict.get("model_details", {}).get("model_coefficients", False))
        except Exception:
            has_coefficients = False

        if has_coefficients:
            coefficients = pd.DataFrame(results_dict["model_details"]["model_coefficients"])
            coefficients = coefficients.rename(columns={"name": "Descriptor", "coefficient": "Coefficient", "std_error": "Standard Error"})
            coefficients["Descriptor"] = coefficients["Descriptor"].astype(str)

            result = temp.merge(coefficients, on="Descriptor", how="right")
            result = ExcelFormatter.handle_accidental_formulas(result, how="formula")
        else:
            result = ExcelFormatter.handle_accidental_formulas(temp, how="formula")

        return result

    @staticmethod
    def get_model_descriptor_values_df(results_dict: Dict[str, Any], df_pred_cv: pd.DataFrame, df_pred_test: pd.DataFrame, df_training_model: pd.DataFrame, df_test_model: pd.DataFrame) -> pd.DataFrame:
        """Generate dataframe combining predictions with all descriptor values.
        
        Args:
            results_dict: Dictionary with model details (units, etc.).
            df_pred_cv: Cross-validation predictions for training set.
            df_pred_test: Predictions for test set.
            df_training_model: Training set descriptor values.
            df_test_model: Test set descriptor values.
        
        Returns:
            pd.DataFrame: All compounds with observed/predicted values and descriptor values.
        """
        units = results_dict["model_details"].get("unitsModel", "Units")
        columns = ["exp_prop_id", "dtxcid", "casrn", "preferred_name", "canon_qsar_smiles", "exp", "pred"]

        test = df_pred_test.loc[:, columns]
        test["Set"] = "Test"

        training = df_pred_cv.loc[:, columns]
        training["Set"] = df_pred_cv.cv_fold.apply(lambda x: f"Training CV, Fold {x + 1}")  # TODO: Test on local dataframe to ensure this is correct

        full = pd.concat([test, training], ignore_index=True)
        full["canon_qsar_smiles"] = full["canon_qsar_smiles"].astype(str)
        full = full.rename(columns={
            "dtxcid": "DTXCID",
            "casrn": "CASRN",
            "preferred_name": "Preferred Name",
            "canon_qsar_smiles": "Canonical QSAR Ready Smiles",
            "exp": f"Observed ({units})",
            "pred": f"Predicted ({units})"
        })

        test_descriptors = df_test_model.drop(columns=["Property"])
        training_descriptors = df_training_model.drop(columns=["Property"])

        full_descriptors = pd.concat([test_descriptors, training_descriptors], ignore_index=True)
        full_descriptors["ID"] = full_descriptors["ID"].astype(str)
        full_descriptors = full_descriptors.rename(columns={"ID": "Canonical QSAR Ready Smiles"})

        final = full.merge(full_descriptors, on="Canonical QSAR Ready Smiles", how="left")
        final = final.rename(columns={
            "exp_prop_id": "Exp Prop ID"
        })

        final = ExcelFormatter.handle_accidental_formulas(final)

        return final

    @staticmethod
    def get_training_cv_predictions_df(df_training_cv: pd.DataFrame) -> pd.DataFrame:
        """Format training set cross-validation predictions for Excel sheet.
        
        Args:
            df_training_cv: Cross-validation predictions dataframe.
        
        Returns:
            pd.DataFrame: Formatted predictions ready for Excel output.
        """
        exp_prop_id = df_training_cv.pop("exp_prop_id")
        df_training_cv.insert(0, "Exp Prop ID", exp_prop_id)
        df_training_cv.rename(columns={col: ExcelFormatter.clean_col_titles(col) for col in df_training_cv.columns}, inplace=True)
        return df_training_cv

    @staticmethod
    def get_test_set_predictions_df(df_test: pd.DataFrame, actual_ads: Optional[list]=None) -> pd.DataFrame:
        """Format test set predictions for Excel sheet.
        
        Args:
            df_test: Test set predictions dataframe.
            actual_ads: List of applicability domain names to keep columns for. Others are dropped. Defaults to None (keep all).
        
        Returns:
            pd.DataFrame: Formatted predictions ready for Excel output.
        """
        exp_prop_id = df_test.pop("exp_prop_id")
        df_test.insert(0, "Exp Prop ID", exp_prop_id)

        if actual_ads is not None:
            applicability_domain_cols = [col for col in df_test.columns if "AD" in col]
            for col in applicability_domain_cols:
                temp_name = col.replace("AD_", "")
                temp_name = temp_name.replace("_", " ")
                if temp_name not in actual_ads:
                    df_test.pop(col)
        
        df_test.rename(columns={col: ExcelFormatter.clean_col_titles(col) for col in df_test.columns}, inplace=True)
        return df_test

    @staticmethod
    def get_external_predictions_df(df_ext: pd.DataFrame, ad_columns: Optional[list|str] = None, df_training: Optional[pd.DataFrame] = None, remove_log_p_descriptors: Optional[bool] = True, embedding: Optional[list[str]] = None) -> pd.DataFrame:
        """Format external/validation set predictions for Excel sheet.
        
        Args:
            df_ext: External dataset predictions dataframe.
            ad_columns: List of applicability domain column names to include.
            df_training: Training dataset dataframe for handling applicability domains.
            remove_log_p_descriptors: Whether to remove logP descriptors. Defaults to True.
            embedding: List of embedding column names.
        
        Returns:
            pd.DataFrame: Formatted predictions ready for Excel output.
        """
        exp_prop_id = df_ext.pop("exp_prop_id")
        df_ext.insert(0, "Exp Prop ID", exp_prop_id)

        if ad_columns and df_training:
            if isinstance(ad_columns, str):
                ads = ad_columns.split(" and ")
            else:
                ads = ad_columns
            ad_test_columns = {}
            for ad in ads:
                df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                        train_df=df_training.copy(), test_df=df_ext.copy(),
                        remove_log_p=remove_log_p_descriptors,
                        embedding=embedding, applicability_domain=ad,
                        filterColumnsInBothSets=False,
                        returnTrainingAD=False)
                ad_test_columns[f"AD {ad}"] = df_ad_output["AD"]

        df_ext.rename(columns={col: ExcelFormatter.clean_col_titles(col) for col in df_ext.columns}, inplace=True)
        return df_ext
    
    @staticmethod
    def get_model_coefficients(model: Model) -> Optional[pd.DataFrame]:
        """Extract model coefficients and standard errors from model object.
        
        Args:
            model: Model object with regression coefficients (for reg, las, gcm methods only).
        
        Returns:
            Optional[pd.DataFrame]: Dataframe with descriptor names and coefficients, or None if model type doesn't support coefficients or extraction fails.
        """
        try:            
            if model is None:
                logging.error(f"Failed to load model")
                return None
            method_name = getattr(model, "qsar_method", False) or getattr(model, "regressor_name", False) or ""
            if not any(method in method_name for method in ["reg", "las", "gcm"]):
                logging.warning(f"Model has QSAR method that does not support coefficient retrieval: {method_name}")
                return None
                        
            df_training = model.df_training
            y = df_training[df_training.columns[1]]
            X = df_training[model.embedding]

            coefficients_json = model.getOriginalRegressionCoefficients2(X, y)
            coefficients_dict = json.loads(coefficients_json)
            
            return coefficients_dict
            
        except Exception as e:
            logging.error(f"Error retrieving coefficients: {e}")
            traceback.print_exc()
            return None


# ============================================================
# EXCEL CREATOR CLASS
# ============================================================

class ModelToExcel:
    """Orchestrates creation of Excel workbooks with QSAR model summary reports.
    
    Manages the generation of all Excel sheets (cover sheet, statistics, records, predictions, etc.),
    formatting, charts, and hyperlinks. Uses ModelDataObjects for data management and delegaes
    formatting to ExcelFormatter, charting to ChartBuilder, and data queries to DataQuerier.
    """
    
    def __init__(
            self,
            model_data_objects: ModelDataObjects,
            excel_path: str = "summary.xlsx",
            # Plotting/formatting options
            exclude_blank_columns: bool = True,
            include_value_original: bool = False,
            include_qc_columns: bool = False,
            display_dropped_columns: bool = False,
            add_subtotals: bool = True,
            create_records_superheaders: bool = True,
            log_plot: bool = True,
            colors: list[str] = ["#CCFFCC", "#CCCCFF", "#FFCCCC", "#FFFFCC", "#CCFFFF"]
        ) -> None:
        """Initialize ModelToExcel instance for generating QSAR model summary reports.
        
        Orchestrates Excel workbook generation using data from ModelDataObjects and formatting
        helpers. All sheet dataframes are stored from ModelDataObjects during initialization.
        
        Args:
            model_data_objects: ModelDataObjects instance containing model, dataframes, and query configuration.
            excel_path: Output file path for generated Excel workbook. Defaults to 'summary.xlsx'.
            exclude_blank_columns: Remove completely empty columns from records sheet. Defaults to True.
            include_value_original: Include original (non-standardized) measured values in records. Defaults to False.
            include_qc_columns: Include QC flag and reason columns in records. Defaults to False.
            display_dropped_columns: Show which columns were dropped in field descriptions sheet. Defaults to False.
            add_subtotals: Add row count formulas above headers in data sheets. Defaults to True.
            create_records_superheaders: Group record columns under superheaders by category. Defaults to True.
            log_plot: Use logarithmic scale for prediction plot axes. Defaults to True.
            colors: Color palette for superheader cells (list of hex color strings). Defaults to EPA color palette.
        """
        self.excel_path = Path(excel_path)

        self.exclude_blank_columns = exclude_blank_columns
        self.include_value_original = include_value_original
        self.include_qc_columns = include_qc_columns
        self.display_dropped_columns = display_dropped_columns
        self.add_subtotals = add_subtotals
        self.create_records_superheaders = create_records_superheaders
        self.log_plot = log_plot
        self.colors = colors
        
        # Dataframes for sheets, generated first and stored in model_data_objects
        self.model = getattr(model_data_objects, "model", None)
        self.cover_sheet_df = getattr(model_data_objects, "cover_sheet_df", None)
        self.statistics_df = getattr(model_data_objects, "statistics_df", None)
        self.records_df = getattr(model_data_objects, "records_df", None)
        self.records_field_descriptions_df = getattr(model_data_objects, "records_field_descriptions_df", None)
        self.model_descriptors_df = getattr(model_data_objects, "model_descriptors_df", None)
        self.model_descriptor_values_df = getattr(model_data_objects, "model_descriptor_values_df", None)
        self.training_cv_predictions_df = getattr(model_data_objects, "training_cv_predictions_df", None)
        self.test_set_predictions_df = getattr(model_data_objects, "test_set_predictions_df", None)
        self.external_predictions_df = getattr(model_data_objects, "external_predictions_df", None)

    def cover_sheet(self, writer: Any, cover_sheet: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """Create the cover sheet in the Excel workbook with model summary information.
        
        Transposes the cover sheet dataframe (properties as rows, values as column), applies formatting
        with bold labels and left alignment, and freezes the first column for readability.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            cover_sheet: Cover sheet dataframe with model metadata. If None, returns None.
        
        Returns:
            pd.DataFrame: The transposed cover sheet dataframe that was written, or None if input was None.
        """
        if cover_sheet is None:
            return None
        
        cover_sheet = cover_sheet.transpose().round(2)
        cover_sheet.reset_index(inplace=True, names=[""])

        workbook = writer.book
        worksheet = workbook.add_worksheet("Summary")
        
        # Create formats
        bold_format = workbook.add_format({"bold": True, "align": "left", "valign": "vcenter"})
        left_align_format = workbook.add_format({"align": "left", "valign": "vcenter"})
        
        # Manually write the cover sheet data
        for row_num, row_data in enumerate(cover_sheet.values):
            for col_num, value in enumerate(row_data):
                # First column (labels) uses bold format, rest use left align
                cell_format = bold_format if col_num == 0 else left_align_format
                worksheet.write(row_num, col_num, value, cell_format)
        
        worksheet.freeze_panes(0, 1)

        ExcelFormatter.set_column_width(writer, "Summary", cover_sheet, how="full", first_col_format=bold_format, col_width_pad=1, min_col_width=5)

        return cover_sheet

    def statistics(self, writer: Any, statistics: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """Create the statistics sheet with model performance metrics.
        
        Organizes statistics into color-coded sections (Training, Cross-Validation, Test, Applicability Domain),
        uses rich text formatting for superscripts (R², subscripts), and includes an equation reference image.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            statistics: Statistics dataframe with R², RMSE, MAE for all dataset splits. If None, returns None.
        
        Returns:
            pd.DataFrame: The statistics dataframe that was written, or None if input was None.
        """
        if statistics is None:
            return None

        workbook = writer.book
        worksheet = workbook.add_worksheet("Statistics")

        format_center = workbook.add_format({
            "align": "center"
        })
        format_super = workbook.add_format({
            "font_script": 1
        })
        format_sub = workbook.add_format({
            "font_script": 2
        })
        format_number = workbook.add_format({
            "align": "center",
            "num_format": "0.00"
        })
        merge_format_training = workbook.add_format({
            "bold": True,
            "align": "center",
            "fg_color": "#d3d3d3"
        })
        merge_format_cv = workbook.add_format({
            "bold": True,
            "align": "center",
            "fg_color": "#ccffcc"
        })
        merge_format_test = workbook.add_format({
            "bold": True,
            "align": "center",
            "fg_color": "#ccccff"
        })
        merge_format_ad = workbook.add_format({
            "bold": True,
            "align": "center",
            "fg_color": "#ffffcc"
        })

        # Make section headers
        worksheet.merge_range("A1:C1", f"Training Set ({statistics.at[0, 'nTraining']})", merge_format_training)
        worksheet.merge_range("D1:F1", f"5-Fold CV ({statistics.at[0, 'nTraining']})", merge_format_cv)
        worksheet.merge_range("A5:C5", f"Test Set ({statistics.at[0, 'nTest']})", merge_format_test)
        worksheet.merge_range("D5:F5", f"Test Set Applicability Domain Statistics", merge_format_ad)

        # Make section sub-headers (column titles)
        worksheet.write_rich_string("A2", "R", format_super, "2", format_center)
        worksheet.write_rich_string("D2", "R", format_super, "2", format_center)
        worksheet.write_rich_string("A6", "R", format_super, "2", format_center)

        worksheet.write_string("B2", "RMSE", format_center)
        worksheet.write_string("E2", "RMSE", format_center)
        worksheet.write_string("B6", "RMSE", format_center)

        worksheet.write_string("C2", "MAE", format_center)
        worksheet.write_string("F2", "MAE", format_center)
        worksheet.write_string("C6", "MAE", format_center)

        worksheet.write_rich_string("D6", "MAE", format_sub, "Test", " Inside AD", format_center)
        worksheet.write_rich_string("E6", "MAE", format_sub, "Test", " Outside AD", format_center)
        worksheet.write_string("F6", "Fraction Inside AD", format_center)

        # Write statistics
        worksheet.write_number("A3", statistics.at[0, "RSQ_Training"], format_number)
        worksheet.write_number("B3", statistics.at[0, "RMSE_Training"], format_number)
        worksheet.write_number("C3", statistics.at[0, "MAE_Training"], format_number)

        worksheet.write_number("D3", statistics.at[0, "RSQ_CV_Training"], format_number)
        worksheet.write_number("E3", statistics.at[0, "RMSE_CV_Training"], format_number)
        worksheet.write_number("F3", statistics.at[0, "MAE_CV_Training"], format_number)

        worksheet.write_number("A7", statistics.at[0, "RSQ_Test"], format_number)
        worksheet.write_number("B7", statistics.at[0, "RMSE_Test"], format_number)
        worksheet.write_number("C7", statistics.at[0, "MAE_Test"], format_number)

        worksheet.write_number("D7", statistics.at[0, "MAE_Test_Inside_AD"], format_number)
        worksheet.write_number("E7", statistics.at[0, "MAE_Test_Outside_AD"], format_number)
        worksheet.write_number("F7", statistics.at[0, "Coverage_Test"], format_number)

        ExcelFormatter.set_column_width(writer, "Statistics", statistics, how="full")

        img_path = os.path.join(os.getenv("PROJECT_ROOT"), "resources", "continuous_equations.png")
        try:
            worksheet.insert_image("A8", img_path, {"x_scale": 0.7, "y_scale": 0.7, "x_offset": 10, "y_offset": 2})
        except Exception as e:
            logging.error(f"Error inserting image in Statistics sheet: {e}")

        return statistics

    def records(self, writer: Any, records: Optional[pd.DataFrame]=None, add_subtotals: bool=True, exclude_blank_columns: bool=True, include_qc_columns: bool=False, include_value_original: bool=False, superheaders: dict=None) -> pd.DataFrame:
        """
        Create the records sheet in the Excel workbook with detailed experimental data.
        
        Includes autofilter for easy data filtering, frozen header row, and appropriately sized columns.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            records (pd.DataFrame, optional): Records dataframe. If None, queries from database or uses instance dataframe.
        
        Returns:
            pd.DataFrame: The records dataframe that was written.
        """
        if records is None:
            return None
        
        if exclude_blank_columns:
            records = records.dropna(axis=1, how="all")
        if not include_qc_columns:
            records = records.drop(columns=["QC Flag", "Flag Reason"], errors="ignore")
        if not include_value_original:
            records = records.drop(columns=["Value Original"], errors="ignore")
        
        workbook = writer.book
        worksheet = workbook.add_worksheet("Records")

        if self.create_records_superheaders:
            superheaders = DataTransformer.get_superheaders(records) if superheaders is None else superheaders

            if superheaders:
                colors = self.colors
                merge_formats = [workbook.add_format({"bold": True, "align": "center", "fg_color": color}) for color in colors]

                other_cols = superheaders.get("Other", [])
                records = records[[col for col in records.columns if col not in other_cols] + other_cols]

                i = 0
                for superheader in superheaders.keys():
                        col_idxs = [records.columns.get_loc(col) for col in superheaders[superheader]]
                        if not col_idxs:
                            continue

                        start_col = xl_col_to_name(min(col_idxs))  # Get the column letter using xlsxwriter utility
                        end_col = xl_col_to_name(max(col_idxs))  # Get the column letter using xlsxwriter utility
                        superheader_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals, has_superheaders=self.create_records_superheaders)

                        logging.debug(f"Processing superheader: {superheader}\n\t{superheaders[superheader]}\n\tMerging Range: {start_col}{superheader_row}:{end_col}{superheader_row}")

                        worksheet.merge_range(f"{start_col}{superheader_row}:{end_col}{superheader_row}", superheader, merge_formats[i%len(merge_formats)])
                        i += 1
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals, has_superheaders=self.create_records_superheaders)
        records.to_excel(writer, sheet_name="Records", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Records"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "Records", records)
            worksheet.freeze_panes(3 + [0, 1][self.create_records_superheaders], 0)
        else:
            worksheet.freeze_panes(1 + [0, 1][self.create_records_superheaders], 0)

        ExcelFormatter.set_column_width(writer, "Records", records, col_width_pad=6, how="header")
        ExcelFormatter.add_filter(writer, "Records", records, has_subtotals=add_subtotals, has_superheaders=self.create_records_superheaders)

        return records
    
    def records_field_descriptions(self, writer: Any, records_field_descriptions: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """Create the records field descriptions sheet with documentation for all fields.
        
        Shows field names and descriptions for all columns in the records sheet. Can optionally group
        fields under superheader categories and show which columns were excluded. Includes frozen header
        and autofilter for searching.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            records_field_descriptions: Field documentation dataframe. If None, returns None.
        
        Returns:
            pd.DataFrame: The formatted field descriptions dataframe that was written, or None if input was None.
        """
        if records_field_descriptions is None:
            return None
        
        if self.exclude_blank_columns:
            records = self.data_querier.query_records_df() if self.records_df is None else self.records_df
            temp = records.dropna(axis=1, how="all")
            dropped_columns = set(records.columns) - set(temp.columns)
            records_field_descriptions = records_field_descriptions[records_field_descriptions["Field"].isin(temp.columns)]

            if not self.include_qc_columns:
                dropped_columns.update({"QC Flag", "Flag Reason"})
            if not self.include_value_original:
                dropped_columns.update({"Value Original"})

            records_field_descriptions = records_field_descriptions[~records_field_descriptions["Field"].isin(dropped_columns)]
            records_field_descriptions["Field"] = pd.Categorical(records_field_descriptions["Field"], categories=temp.columns, ordered=True)
            records_field_descriptions = records_field_descriptions.sort_values("Field")
            if self.display_dropped_columns and dropped_columns:
                records_field_descriptions = pd.concat([records_field_descriptions, pd.DataFrame({"Field": ["Dropped Columns"], "Description": [", ".join(list(dropped_columns))]})])
        
        workbook = writer.book
        worksheet = workbook.add_worksheet("Records Field Descriptions")

        startcol=0
        if self.create_records_superheaders and self.records_df is not None:
            superheaders = DataTransformer.get_superheaders(self.records_df)
            colors = self.colors
            merge_formats = [workbook.add_format({"bold": True, "align": "center", "valign": "vcenter", "fg_color": color}) for color in colors]
            bold_format = workbook.add_format({"bold": True, "align": "center"})

            worksheet.write_string("A1", "Field Group", bold_format)

            records_field_descriptions["Field Group"] = records_field_descriptions["Field"].map(lambda x: next((k for k, v in superheaders.items() if x in v), "Other"))
            records_field_descriptions.reset_index(drop=True, inplace=True)

            i = 0
            for superheader in superheaders.keys():
                    row_idxs = records_field_descriptions[records_field_descriptions["Field"].isin(superheaders[superheader])].index.tolist()
                    if not row_idxs:
                        continue

                    start_row = min(row_idxs) + 2
                    end_row = max(row_idxs) + 2

                    logging.debug(f"Processing superheader: {superheader}\n\t{superheaders[superheader]}\n\tMerging Range: A{start_row}:A{end_row}")

                    worksheet.merge_range(f"A{start_row}:A{end_row}", superheader, merge_formats[i%len(merge_formats)])
                    i += 1

            startcol=1
        
        records_field_descriptions[["Field", "Description"]].to_excel(writer, sheet_name="Records Field Descriptions", index=False, startcol=startcol)
        records_field_descriptions = records_field_descriptions[["Field Group", "Field", "Description"]]

        workbook = writer.book
        worksheet = writer.sheets["Records Field Descriptions"]
        worksheet.freeze_panes(1, 0)

        ExcelFormatter.set_column_width(writer, "Records Field Descriptions", records_field_descriptions, how="full", col_width_pad=1, min_col_width=5)
        ExcelFormatter.add_filter(writer, "Records Field Descriptions", records_field_descriptions)

        return records_field_descriptions
    
    def model_descriptors(self, writer: Any, model_descriptors: Optional[pd.DataFrame]=None, add_subtotals: bool=True) -> pd.DataFrame:
        """
        Create the model descriptors sheet in the Excel workbook.
        
        Lists all descriptors used in the model with their definitions and classifications. Includes autofilter and properly sized columns.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            model_descriptors (pd.DataFrame, optional): Model descriptors dataframe. If None, queries from database or uses instance dataframe.
        
        Returns:
            pd.DataFrame: The model descriptors dataframe that was written.
        """
        if model_descriptors is None:
            return None
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals)
        if "Coefficient" in model_descriptors.columns and "Standard Error" in model_descriptors.columns:
            ExcelFormatter.set_sig_figs(model_descriptors, sig_figs=3, columns=["Coefficient", "Standard Error"])
        model_descriptors.to_excel(writer, sheet_name="Model Descriptors", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptors"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "Model Descriptors", model_descriptors)
            worksheet.freeze_panes(3, 0)
        else:
            worksheet.freeze_panes(1, 0)

        col_widths = ExcelFormatter.set_column_width(writer, "Model Descriptors", model_descriptors, how="full")
        ExcelFormatter.add_filter(writer, "Model Descriptors", model_descriptors, has_subtotals=add_subtotals)

        return model_descriptors
    
    def model_descriptor_values(self, writer: Any, model_descriptor_values: Optional[pd.DataFrame]=None, add_subtotals: bool=True) -> pd.DataFrame:
        """Create the model descriptor values sheet with predictions and descriptor values.
        
        Shows all training and test compounds with their observed/predicted values, fold assignments,
        and all descriptor values used in the model. Includes subtotals, autofilter, and column widths.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            model_descriptor_values: Predictions and descriptor values for all data points. If None, returns None.
            add_subtotals: Add row count formulas above headers. Defaults to True.
        
        Returns:
            pd.DataFrame: The formatted descriptor values dataframe that was written, or None if input was None.
        """
        if model_descriptor_values is None:
            return None
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals)
        # format_cols = [col for col in model_descriptor_values.columns if col.startswith("Observed") or col.startswith("Predicted")]
        ExcelFormatter.set_sig_figs(model_descriptor_values, sig_figs=3, columns=None)
        model_descriptor_values.to_excel(writer, sheet_name="Model Descriptor Values", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptor Values"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "Model Descriptor Values", model_descriptor_values)
            worksheet.freeze_panes(3, 0)
        else:
            worksheet.freeze_panes(1, 0)

        col_widths = ExcelFormatter.set_column_width(writer, "Model Descriptor Values", model_descriptor_values, min_col_width=7, col_width_pad=5, how="header")
        ExcelFormatter.add_filter(writer, "Model Descriptor Values", model_descriptor_values, has_subtotals=add_subtotals)

        return model_descriptor_values
    
    def training_cv_predictions(self, writer: Any, training_cv_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=6, min_col_width: int=8, property_name: Optional[str]=None, property_units: Optional[str]=None) -> pd.DataFrame:
        """Create the training CV predictions sheet with observed vs predicted scatter plot.
        
        Displays cross-validation predictions for training set compounds with integrated scatter plot,
        compound metadata, fold assignments, and autofilter. Plot includes y=x reference line and equal axes.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            training_cv_predictions: Training cross-validation predictions dataframe. If None, returns None.
            add_subtotals: Add row count formulas. Defaults to True.
            x_col: Column for x-axis (observed values). Defaults to 'Exp'.
            y_col: Column for y-axis (predicted values). Defaults to 'Pred'.
            chart_size_px: Chart size in pixels. Defaults to 520.
            pad_ratio: Axis padding ratio. Defaults to 0.02.
            integer_ticks: Use integer tick spacing. Defaults to True.
            yx_offset_rows: Empty rows before y=x reference points. Defaults to 3.
            col_width_pad: Column width padding. Defaults to 6.
            min_col_width: Minimum column width. Defaults to 8.
            property_name: Property name for chart title. Defaults to None.
            property_units: Property units for axis labels. Defaults to None.
        
        Returns:
            pd.DataFrame: The formatted training predictions dataframe that was written, or None if input was None.
        """
        if training_cv_predictions is None:
            return None
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals)
        ExcelFormatter.set_sig_figs(training_cv_predictions, sig_figs=3, columns=["Exp", "Pred"])
        training_cv_predictions.to_excel(writer, sheet_name="Training CV Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Training CV Predictions"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "Training CV Predictions", training_cv_predictions)
            worksheet.freeze_panes(3, 0)
        else:
            worksheet.freeze_panes(1, 0)

        col_widths = ExcelFormatter.set_column_width(writer, "Training CV Predictions", training_cv_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header")
        ExcelFormatter.add_filter(writer, "Training CV Predictions", training_cv_predictions, has_subtotals=add_subtotals)
        
        ChartBuilder.add_plot(writer, workbook, "Training CV Predictions", "Training CV Predictions", training_cv_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)

        return training_cv_predictions
    
    def test_set_predictions(self, writer: Any, test_set_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=6, min_col_width: int=8, property_name: Optional[str]=None, property_units: Optional[str]=None) -> pd.DataFrame:
        """Create the test set predictions sheet with observed vs predicted scatter plot and AD information.
        
        Displays test set predictions with compound identifiers, observed/predicted values, applicability domain
        membership, compound metadata, and integrated scatter plot with y=x reference line and equal axes.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            test_set_predictions: Test set predictions dataframe with AD columns. If None, returns None.
            add_subtotals: Add row count formulas. Defaults to True.
            x_col: Column for x-axis (observed values). Defaults to 'Exp'.
            y_col: Column for y-axis (predicted values). Defaults to 'Pred'.
            chart_size_px: Chart size in pixels. Defaults to 520.
            pad_ratio: Axis padding ratio. Defaults to 0.02.
            integer_ticks: Use integer tick spacing. Defaults to True.
            yx_offset_rows: Empty rows before y=x reference points. Defaults to 3.
            col_width_pad: Column width padding. Defaults to 6.
            min_col_width: Minimum column width. Defaults to 8.
            property_name: Property name for chart title. Defaults to None.
            property_units: Property units for axis labels. Defaults to None.
        
        Returns:
            pd.DataFrame: The formatted test predictions dataframe that was written, or None if input was None.
        """
        if test_set_predictions is None:
            return None
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals)
        ExcelFormatter.set_sig_figs(test_set_predictions, sig_figs=3, columns=["Exp", "Pred"])
        test_set_predictions.to_excel(writer, sheet_name="Test Set Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Test Set Predictions"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "Test Set Predictions", test_set_predictions)
            worksheet.freeze_panes(3, 0)
        else:
            worksheet.freeze_panes(1, 0)

        col_widths = ExcelFormatter.set_column_width(writer, "Test Set Predictions", test_set_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header")
        ExcelFormatter.add_filter(writer, "Test Set Predictions", test_set_predictions, has_subtotals=add_subtotals)

        ChartBuilder.add_plot(writer, workbook, "Test Set Predictions", "Test Set Predictions", test_set_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)

        return test_set_predictions

    def external_predictions(self, writer: Any, external_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=6, min_col_width: int=8, property_name: Optional[str]=None, property_units: Optional[str]=None) -> Optional[pd.DataFrame]:
        """Create the external validation set predictions sheet with scatter plot.
        
        Displays external/validation dataset predictions with observed/predicted values, compound metadata, and
        integrated scatter plot with y=x reference line and equal axes. Only created if external data is available.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            external_predictions: External dataset predictions dataframe. If None, returns None.
            add_subtotals: Add row count formulas. Defaults to True.
            x_col: Column for x-axis (observed values). Defaults to 'Exp'.
            y_col: Column for y-axis (predicted values). Defaults to 'Pred'.
            chart_size_px: Chart size in pixels. Defaults to 520.
            pad_ratio: Axis padding ratio. Defaults to 0.02.
            integer_ticks: Use integer tick spacing. Defaults to True.
            yx_offset_rows: Empty rows before y=x reference points. Defaults to 3.
            col_width_pad: Column width padding. Defaults to 6.
            min_col_width: Minimum column width. Defaults to 8.
            property_name: Property name for chart title. Defaults to None.
            property_units: Property units for axis labels. Defaults to None.
        
        Returns:
            pd.DataFrame: The formatted external predictions dataframe that was written, or None if input was None.
        """
        if external_predictions is None:
            return None
        
        start_row = ExcelFormatter.get_header_row(has_subtotals=add_subtotals)
        ExcelFormatter.set_sig_figs(external_predictions, sig_figs=3, columns=["Exp", "Pred"])
        external_predictions.to_excel(writer, sheet_name="External Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["External Predictions"]
        if add_subtotals:
            ExcelFormatter.add_subtotals(writer, "External Predictions", external_predictions)
            worksheet.freeze_panes(3, 0)
        else:
            worksheet.freeze_panes(1, 0)

        col_widths = ExcelFormatter.set_column_width(writer, "External Predictions", external_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header")
        ExcelFormatter.add_filter(writer, "External Predictions", external_predictions, has_subtotals=add_subtotals)
        
        ChartBuilder.add_plot(writer, workbook, "External Predictions", "External Predictions", external_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)
        
        return external_predictions

    def create_excel(
            self,
            x_col: str=None,
            y_col: str=None,
            chart_size_px: int=520,
            pad_ratio: float=0.02,
            integer_ticks: bool=True,
            yx_offset_rows: int=3,
            col_width_pad: int=6,
            min_col_width: int=8
        ) -> None:
        """Generate complete Excel workbook with all model summary sheets and charts.
        
        Orchestrates creation of the final Excel file by calling sheet methods in order, extracting
        property metadata, and adding hyperlinks between related sheets. Creates sheets for model
        summary, statistics, experimental records, descriptors, predictions, and charts.
        
        Args:
            x_col: Column name for x-axis (observed values) in prediction plots. Defaults to 'Exp'.
            y_col: Column name for y-axis (predicted values) in prediction plots. Defaults to 'Pred'.
            chart_size_px: Square chart size in pixels for all prediction plots. Defaults to 520.
            pad_ratio: Axis padding as fraction of data range. Defaults to 0.02 (2%).
            integer_ticks: Use integer-based tick spacing on chart axes. Defaults to True.
            yx_offset_rows: Empty rows between data and y=x reference line points. Defaults to 3.
            col_width_pad: Extra padding to add to calculated column widths. Defaults to 4.
            min_col_width: Minimum width for any column. Defaults to 5 characters.
        """
        logging.info("Creating detailed Excel...")
        self.excel_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.excel_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            logging.info("Creating Cover Sheet...")
            self.cover_sheet(writer, self.cover_sheet_df)

            logging.info("Creating Statistics...")
            self.statistics(writer, self.statistics_df)

            logging.info("Creating Records...")
            df = self.records(writer, self.records_df, add_subtotals=self.add_subtotals, exclude_blank_columns=self.exclude_blank_columns, include_qc_columns=self.include_qc_columns, include_value_original=self.include_value_original)

            logging.info("Creating Records Field Descriptions...")
            df = self.records_field_descriptions(writer, self.records_field_descriptions_df)

            logging.info("Creating Model Descriptors...")
            df = self.model_descriptors(writer, self.model_descriptors_df, add_subtotals=self.add_subtotals)

            logging.info("Creating Model Descriptor Values...")
            df = self.model_descriptor_values(writer, self.model_descriptor_values_df, add_subtotals=self.add_subtotals)

            try:
                property_name = self.model.propertyName if self.model is not None else self.cover_sheet_df["Property Name"].iloc[0] if not self.cover_sheet_df["Property Name"].empty else None
                if "koc" in property_name.lower():
                    property_name = "log Koc"
            except Exception as e:
                property_name = None
            try:
                property_units = self.model.unitsModel if self.model is not None else self.cover_sheet_df["Property Units"].iloc[0] if not self.cover_sheet_df["Property Units"].empty else None
            except Exception as e:
                property_units = None

            logging.info("Creating Training CV Predictions...")
            df = self.training_cv_predictions(writer, self.training_cv_predictions_df, add_subtotals=self.add_subtotals, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, yx_offset_rows=yx_offset_rows, col_width_pad=col_width_pad, min_col_width=min_col_width, property_name=property_name, property_units=property_units)

            logging.info("Creating Test Set Predictions...")
            df = self.test_set_predictions(writer, self.test_set_predictions_df, add_subtotals=self.add_subtotals, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, yx_offset_rows=yx_offset_rows, col_width_pad=col_width_pad, min_col_width=min_col_width, property_name=property_name, property_units=property_units)

            if self.external_predictions_df is not None:
                logging.info("Creating External Predictions...")
                df = self.external_predictions(writer, self.external_predictions_df, add_subtotals=self.add_subtotals, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, yx_offset_rows=yx_offset_rows, col_width_pad=col_width_pad, min_col_width=min_col_width, property_name=property_name, property_units=property_units)

            # logging.info("Done creating detailed Excel!")
            # logging.info("Done with initial passthrough of all sheets!")

            # Add Hyperlinks
            logging.info("Adding hyperlinks...")
            try:
                ExcelFormatter.add_hyperlinks_to_sheet(writer, "Records", "Training CV Predictions", self.records_df, self.training_cv_predictions_df, has_subtotals=self.add_subtotals, source_has_superheaders=self.create_records_superheaders)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ExcelFormatter.add_hyperlinks_to_sheet(writer, "Records", "Test Set Predictions", self.records_df, self.test_set_predictions_df, has_subtotals=self.add_subtotals, source_has_superheaders=self.create_records_superheaders)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ExcelFormatter.add_hyperlinks_to_sheet(writer, "Training CV Predictions", "Records", self.training_cv_predictions_df, self.records_df, has_subtotals=self.add_subtotals, target_has_superheaders=self.create_records_superheaders)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ExcelFormatter.add_hyperlinks_to_sheet(writer, "Test Set Predictions", "Records", self.test_set_predictions_df, self.records_df, has_subtotals=self.add_subtotals, target_has_superheaders=self.create_records_superheaders)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            
            logging.info("Done creating detailed Excel!")


# ============================================================
# TEST FUNCTIONS AND MISCELLANEOUS STUFF
# ============================================================

def custom_encoder(obj: Any) -> Any:
    """Custom JSON encoder for model objects and dataframes.
    
    Args:
        obj: Object to encode (PMMLPipeline or DataFrame).
    
    Returns:
        dict: Dictionary representation of the object, or original object if not handled.
    """
    if isinstance(obj, PMMLPipeline):
        return obj.__dict__
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')


def query_example() -> None:
    """Example: Generate Excel report for model queried from the database.
    
    Creates ModelDataObjects from model_id, which automatically queries all necessary
    data from the database, then generates the Excel workbook.
    """
    model_id = 1746
    file_path = os.path.join(PROJECT_ROOT, "data", "excel_summaries", f"model_{model_id}_summary.xlsx")

    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()


def local_example() -> None:
    """Example: Generate Excel report for a locally constructed model.
    
    Loads pre-computed model and data objects from a pickle file (local_model_data.pkl),
    then generates the Excel workbook using local data instead of querying database.
    """
    try:
        with open("local_model_data.pkl", "rb") as f:
            stuff = pickle.load(f)
        model = stuff.get("model")
        df_pv = stuff.get("df_pv")
        df_gmd = stuff.get("df_gmd")
        df_gmd_external = stuff.get("df_gmd_external")

        file_path = os.path.join(PROJECT_ROOT, "data", "excel_summaries", "test", "test_summary_refactor_local.xlsx")
        
        mdo = ModelDataObjects(model=model, df_pv=df_pv, df_gmd=df_gmd, df_gmd_external=df_gmd_external)
        mte = ModelToExcel(mdo, file_path)
        mte.create_excel()
    except Exception as e:
        logging.error(f"Error in local_example: {e}")


def test_model_details_pv() -> None:
    """Testing/debugging: Query model and property values from database.
    
    Demonstrates direct use of DataQuerier to retrieve model object and property values,
    then saves them to JSON and pickle files for inspection and testing.
    """
    engine = DataQuerier.getEngine()
    session = DataQuerier.getSession(engine)
    model_id = 1746
    test = DataQuerier(engine=engine, session=session, model_id=model_id)
    model = test.model
    print(model.__dict__)

    with open("test_model_details.json", "w") as f:
        json.dump(model.__dict__, f, indent=4, default=custom_encoder)
    with open("test_model.pkl", "wb") as f:
        f.write(pickle.dumps(model))

    with open("test_experimental_parameters.pkl", "wb") as f:
        f.write(pickle.dumps(test.experimental_parameters))
    
    df_pv = test.query_df_pv()
    with open("test_df_pv.pkl", "wb") as f:
        pickle.dump(df_pv, f)


def test_model_details_gmd() -> None:
    """Testing/debugging: Query descriptor values from database.
    
    Demonstrates retrieval of molecular descriptor values (GMD dataframes) for both
    training and external datasets, then saves them to pickle files for inspection.
    """
    engine = DataQuerier.getEngine()
    session = DataQuerier.getSession(engine)
    dataset_name = "KOC v1 modeling"
    df_gmd = getMappedDatapoints(session, dataset_name)

    with open("test_df_gmd.pkl", "wb") as f:
        pickle.dump(df_gmd, f)
    
    dataset_name_external = "KOC v1 external"
    df_gmd_external = getMappedDatapoints(session, dataset_name_external)

    with open("test_df_gmd_external.pkl", "wb") as f:
        pickle.dump(df_gmd_external, f)


def test_query_old_models() -> None:
    """Testing/debugging: Generate Excel report for multiple older models queried from the database.
    
    Creates ModelDataObjects from the model_id's provided, which automatically queries all necessary
    data from the database, then generates the Excel workbook.
    """
    model_ids = list(range(1065, 1071))
    # model_ids = [1070]
    for model_id in model_ids:
        file_path = os.path.join(PROJECT_ROOT, "data", "excel_summaries", f"{model_id}_summary.xlsx")
        mdo = ModelDataObjects(model_id=model_id)
        mte = ModelToExcel(mdo, file_path)
        mte.create_excel()


if __name__ == "__main__":
    # query_example()
    # local_example()
    # test_model_details_pv()
    # test_model_details_gmd()
    test_query_old_models()
