import pickle

import pandas as pd
from sqlalchemy import URL, text, create_engine
from sqlalchemy.orm import sessionmaker
import math
import numpy as np
from typing import Optional, Dict, Any, Iterable, Tuple, Union
from xlsxwriter.utility import xl_rowcol_to_cell
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

import logging
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, force=True)


class ModelToExcel:
    def __init__(
            self,
            excel_path: str = "summary.xlsx",
            cover_sheet_df: Optional[pd.DataFrame] = None,
            statistics_df: Optional[pd.DataFrame] = None,
            training_set_df: Optional[pd.DataFrame] = None,
            test_set_df: Optional[pd.DataFrame] = None,
            records_df: Optional[pd.DataFrame] = None,
            records_field_descriptions_df: Optional[pd.DataFrame] = None,
            training_cv_predictions_df: Optional[pd.DataFrame] = None,
            test_set_predictions_df: Optional[pd.DataFrame] = None,
            external_predictions_df: Optional[pd.DataFrame] = None,
            model_descriptors_df: Optional[pd.DataFrame] = None,
            model_descriptor_values_df: Optional[pd.DataFrame] = None,
            engine = None,
            session = None,
            model_id: int = 1746,
            log_plot: bool = True,
            add_subtotals: bool = True,
            dataset_name: Optional[str] = None,
            snapshot_id: Optional[int] = None,
            duplicate_strategy: Optional[str] = None,
            model: Optional[Any] = None,
            df_pv: Optional[pd.DataFrame] = None,
            df_gmd: Optional[pd.DataFrame] = None,
            exclude_blank_columns: bool = True,
            include_qc_columns: bool = False,
            include_value_original: bool = False,
            display_dropped_columns: bool = False,
            dataset_name_external: Optional[str] = None,
            df_gmd_external: Optional[pd.DataFrame] = None
        ) -> None:
        """
        Initialize ModelToExcel instance for generating QSAR model summary reports.
        
        Args:
            excel_path: Output path for the generated Excel file. Defaults to "summary.xlsx".
            cover_sheet_df: DataFrame containing high-level model information.
            statistics_df: DataFrame with model performance statistics.
            training_set_df: DataFrame with training set data.
            test_set_df: DataFrame with test set data.
            records_df: DataFrame with detailed experimental records.
            records_field_descriptions_df: DataFrame describing fields in the records sheet.
            training_cv_predictions_df: DataFrame with cross-validation predictions on training set.
            test_set_predictions_df: DataFrame with predictions on test set.
            external_predictions_df: DataFrame with predictions on external/validation set.
            model_descriptors_df: DataFrame with model descriptor definitions.
            model_descriptor_values_df: DataFrame with descriptor values for each prediction.
            engine: SQLAlchemy engine for database queries. If None, creates a new engine.
            session: SQLAlchemy session for database queries. If None, creates a new session.
            model_id: Database ID of the model. Defaults to 1065.
            log_plot: Whether to use log scale for prediction plots. Defaults to True.
            add_subtotals: Whether to add subtotals in the relevant sheets. Defaults to False.
        """
        # TODO: Determine correct default for excel_path given model_id
        #       Maybe write a new method that can be ran post init to
        #       change the excel_path based on a query of the database?
        if engine is None:
            engine = ModelToExcel.getEngine()
        if session is None:
            session = ModelToExcel.getSession(engine)
        
        self.excel_path = excel_path
        self.cover_sheet_df = cover_sheet_df
        self.statistics_df = statistics_df
        self.training_set_df = training_set_df
        self.test_set_df = test_set_df
        self.records_df = records_df
        self.records_field_descriptions_df = records_field_descriptions_df
        self.training_cv_predictions_df = training_cv_predictions_df
        self.test_set_predictions_df = test_set_predictions_df
        self.external_predictions_df = external_predictions_df
        self.model_descriptors_df = model_descriptors_df
        self.model_descriptor_values_df = model_descriptor_values_df
        self.engine = engine
        self.session = session
        self.model_id = model_id
        self.log_plot = log_plot
        self.add_subtotals = add_subtotals
        self.dataset_name = dataset_name
        self.snapshot_id = snapshot_id
        self.duplicate_strategy = duplicate_strategy
        self.model = model
        self.df_pv = df_pv
        self.df_gmd = df_gmd
        self.exclude_blank_columns = exclude_blank_columns
        self.include_qc_columns = include_qc_columns
        self.include_value_original = include_value_original
        self.display_dropped_columns = display_dropped_columns
        self.dataset_name_external = dataset_name_external
        self.df_gmd_external = df_gmd_external
    

    @staticmethod
    def get_cover_sheet_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract model information from results dictionary to create cover sheet dataframe.
        
        Args:
            results_dict (dict): Dictionary containing model details and metadata.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model overview information including property, dataset, and method details.
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
            cover_sheet_df["nExternal"] = [results_dict["model_details"].get("numExternal", None)]
        cover_sheet_df = pd.DataFrame.from_dict(cover_sheet_df)
        return cover_sheet_df
    

    def query_cover_sheet_df(self) -> pd.DataFrame:
        """
        Query database for model summary information to populate cover sheet.
        
        Retrieves model details including property name, dataset information, method details, and training/test set sizes.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model overview from database.
        """
        logging.info(f"Building Cover Sheet from Model {self.model_id}")

        model = self.query_model()

        summary_dict = {
            "Model ID": [int(model.modelId)],
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
        if hasattr(model, "external_dataset_name") and model.external_dataset_name:
            summary_dict["External Dataset Name"] = [model.external_dataset_name]
            summary_dict["nExternal"] = [model.num_external]
        summary = pd.DataFrame(summary_dict)

        self.cover_sheet_df = summary
        self.dataset_name = model.datasetName # Set dataset_name for use in other queries

        logging.info(f"Finished building Cover Sheet from Model {self.model_id}")

        return summary
    

    def cover_sheet(self, writer: Any, cover_sheet: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """
        Create the cover sheet in the Excel workbook with model summary information.
        
        Formats the cover sheet with bold labels and centered alignment. Freezes the first column for better readability.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            cover_sheet (pd.DataFrame, optional): Cover sheet dataframe. If None, queries from database or uses instance dataframe.
        
        Returns:
            pd.DataFrame: The cover sheet dataframe that was written.
        """
        if cover_sheet is None:
            cover_sheet = self.query_cover_sheet_df() if self.cover_sheet_df is None else self.cover_sheet_df
        
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

        ModelToExcel.set_column_width(writer, "Summary", cover_sheet, how="full", first_col_format=bold_format)

        return cover_sheet


    @staticmethod
    def get_statistics_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract model performance statistics from results dictionary.
        
        Args:
            results_dict (dict): Dictionary containing model statistics and details.
        
        Returns:
            pd.DataFrame: Single-row dataframe with model performance metrics including R², RMSE, MAE for training, CV, test, and applicability domain.
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

        self.statistics_df = statistics
        
        logging.info(f"Finished building Statistics from Model {self.model_id}")

        return statistics
    

    def statistics(self, writer: Any, statistics: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """
        Create the statistics sheet in the Excel workbook with formatted model performance metrics.
        
        Organizes statistics into sections (Training, Cross-Validation, Test, Applicability Domain) with color-coded headers. Includes equation reference image.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            statistics (pd.DataFrame, optional): Statistics dataframe. If None, queries from database or uses instance dataframe.
        
        Returns:
            pd.DataFrame: The statistics dataframe that was written.
        """
        if statistics is None:
            statistics = self.query_statistics_df() if self.statistics_df is None else self.statistics_df

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

        ModelToExcel.set_column_width(writer, "Statistics", statistics, how="full")

        img_path = os.path.join(os.getenv("PROJECT_ROOT"), "resources", "equations.png")        
        worksheet.insert_image("A8", img_path, {"x_scale": 0.7, "y_scale": 0.7, "x_offset": 10, "y_offset": 2})

        return statistics


    @staticmethod
    def get_records_df(df_pv: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and structure experimental record information from property values dataframe.
        
        Transforms raw property value data into a detailed records format with chemical mapping, source information, and experimental conditions.
        
        Args:
            df_pv (pd.DataFrame): Property values dataframe containing raw experimental and chemical data.
        
        Returns:
            pd.DataFrame: Records dataframe with standardized columns for chemical, source, and experimental information.
        """
        records_df = {
            "exp_prop_id": df_pv.get("prop_value_id", None),
            "canon_qsar_smiles": df_pv.get("canon_qsar_smiles", None),
            "page_url": df_pv.get("direct_url", None),
            "public_source_name": df_pv.get("public_source_name", None),
            "public_source_url": df_pv.get("public_source_url", None),
            # "public_source_original_name": None,  # Doesn't seem to be used/useful
            # "public_source_original_url": None,  # Doesn't seem to be used/useful
            "literature_source_citation": df_pv.get("literature_source_citation", None),
            "literature_source_doi": df_pv.get("literature_source_doi", None),
            "source_dtxrid": df_pv.get("source_dtxrid", None),
            "source_dtxsid": df_pv.get("source_dtxsid", None),
            "source_casrn": df_pv.get("source_casrn", None),
            "source_chemical_name": df_pv.get("source_chemical_name", None),
            "source_smiles": df_pv.get("source_smiles", None),
            "mapped_dtxcid": df_pv.get("mapped_dtxcid", None),
            "mapped_dtxsid": df_pv.get("mapped_dtxsid", None),
            "mapped_cas": df_pv.get("mapped_casrn", None),
            "mapped_chemical_name": df_pv.get("mapped_chemical_name", None),
            "mapped_smiles": df_pv.get("mapped_smiles", None),
            "mapped_molweight": df_pv.get("mapped_mol_weight", None),
            "value_original": df_pv.get("prop_value_original", None),
            "value_max": df_pv.get("value_max", None),  # Edited
            "value_min": df_pv.get("value_min", None),  # Edited
            "value_point_estimate": df_pv.get("prop_value", None),
            "value_units": df_pv.get("prop_unit", None),
            "qsar_property_value": df_pv.get("qsar_property_value", None),
            "qsar_property_units": df_pv.get("qsar_property_unit", None),
            "temperature_c": df_pv.get("exp_details_temperature_value_point_estimate", None),
            # "pressure_mmHg": None,  # Not currently stored in exp_prop.property_values
            "pH": df_pv.get("exp_details_ph_value_point_estimate", None),
            "notes": df_pv.get("notes", None),  # Added
            "qc_flag": df_pv.get("qc_flag", None),  # Added
            "flag_reason": df_pv.get("flag_reason", None)  # Added
        }
        records_df = pd.DataFrame.from_dict(records_df)
        return records_df
    

    def query_records_df(self) -> pd.DataFrame:
        """
        Query database for detailed experimental records.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: Records dataframe from database (currently returns empty result).
        """
        # TODO: Determine if this is the correct way to build the Records df, or if we should use the Model object
        logging.info(f"Building Records from Model {self.model_id}")
        if self.dataset_name is None:
            try:
                model = self.query_model()
                self.dataset_name = model.datasetName
            except Exception as e:
                logging.warning("dataset_name not set and failed to query from database. Cannot query records without dataset_name. Please set dataset_name or ensure it can be queried from the database with the provided model_id.")
                raise e
        
        df_pv = self.query_df_pv()
        records_df = ModelToExcel.get_records_df(df_pv)
        logging.info(f"Finished building Records from Model {self.model_id}")
        self.records_df = records_df
        return records_df
    

    def records(self, writer: Any, records: Optional[pd.DataFrame]=None, add_subtotals: bool=True, exclude_blank_columns: bool=True, include_qc_columns: bool=False, include_value_original: bool=False) -> pd.DataFrame:
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
            records = self.query_records_df() if self.records_df is None else self.records_df
        
        if exclude_blank_columns:
            records = records.dropna(axis=1, how="all")
        if not include_qc_columns:
            records = records.drop(columns=["qc_flag", "flag_reason"], errors="ignore")
        if not include_value_original:
            records = records.drop(columns=["value_original"], errors="ignore")
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        records.to_excel(writer, sheet_name="Records", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Records"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "Records", records)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Records", records, col_width_pad=5, how="header", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "Records", records, has_subtotals=add_subtotals)

        return records


    @staticmethod
    def get_records_field_descriptions_df() -> pd.DataFrame:
        """
        Create dataframe with detailed descriptions of all fields in the records sheet.
        
        Provides documentation for each column explaining the source and meaning of the data.
        
        Returns:
            pd.DataFrame: Two-column dataframe mapping field names to their descriptions.
        """
        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        path_segments = [PROJECT_ROOT, "resources", "records_field_descriptions.txt"]
        
        # Load in variable definitions
        records_field_descriptions_df = pd.read_csv(os.path.join(*path_segments), sep="\t")
        return records_field_descriptions_df
    

    def records_field_descriptions(self, writer: Any, records_field_descriptions: Optional[pd.DataFrame]=None) -> pd.DataFrame:
        """
        Create the records field descriptions sheet in the Excel workbook.
        
        Provides documentation for all fields in the records sheet with frozen header and autofilter.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            records_field_descriptions (pd.DataFrame, optional): Field descriptions dataframe. If None, uses static method or instance dataframe.
        
        Returns:
            pd.DataFrame: The field descriptions dataframe that was written.
        """
        if records_field_descriptions is None:
            records_field_descriptions = ModelToExcel.get_records_field_descriptions_df() if self.records_field_descriptions_df is None else self.records_field_descriptions_df
        
        if self.exclude_blank_columns:
            records = self.query_records_df() if self.records_df is None else self.records_df
            temp = records.dropna(axis=1, how="all")
            dropped_columns = set(records.columns) - set(temp.columns)
            records_field_descriptions = records_field_descriptions[records_field_descriptions["Field"].isin(temp.columns)]

            if not self.include_qc_columns:
                dropped_columns.update({"qc_flag", "flag_reason"})
            if not self.include_value_original:
                dropped_columns.update({"value_original"})

            records_field_descriptions = records_field_descriptions[~records_field_descriptions["Field"].isin(dropped_columns)]
            if self.display_dropped_columns and dropped_columns:
                records_field_descriptions = pd.concat([records_field_descriptions, pd.DataFrame({"Field": ["Dropped Columns"], "Description": [", ".join(list(dropped_columns))]})])

        records_field_descriptions.to_excel(writer, sheet_name="Records Field Descriptions", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Records Field Descriptions"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Records Field Descriptions", records_field_descriptions, how="full")
        ModelToExcel.add_filter(writer, "Records Field Descriptions", records_field_descriptions)

        return records_field_descriptions


    @staticmethod
    def get_model_descriptors_df(results_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract model descriptors from results dictionary and merge with variable definitions.
        
        Loads model descriptors from embedding data, matches them with variable definitions from file, and handles special formula characters.
        
        Args:
            results_dict (dict): Dictionary containing model details and embedding (descriptors) information.
        
        Returns:
            pd.DataFrame: Descriptors with their definitions and classifications, with formula characters escaped.
        """
        # Load in model descriptors
        model_descriptors_df = pd.DataFrame(results_dict["model_details"]["embedding"], columns=["Symbol"])

        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        path_segments = [PROJECT_ROOT, "resources", "variable definitions-ed.txt"]
        
        # Load in variable definitions
        variable_definitions_df = pd.read_csv(os.path.join(*path_segments), sep="\t")

        # Convert both Symbol columns to strings explicitly
        model_descriptors_df['Symbol'] = model_descriptors_df['Symbol'].astype(str)
        variable_definitions_df['Symbol'] = variable_definitions_df['Symbol'].astype(str)

        # Merge the model's descriptors with their respective definitions and rename columns
        temp = model_descriptors_df.merge(variable_definitions_df, on="Symbol", how="left")
        temp = temp.rename(columns={"Symbol": "Descriptor", "Category": "Class"})

        if "model_details" in results_dict and "model_coefficients" in results_dict["model_details"]:
            coefficients = pd.DataFrame(results_dict["model_details"]["model_coefficients"])
            coefficients = coefficients.rename(columns={"name": "Descriptor", "coefficient": "Coefficient", "std_error": "Standard Error"})
            coefficients["Descriptor"] = coefficients["Descriptor"].astype(str)

            result = temp.merge(coefficients, on="Descriptor", how="right")

            result = ModelToExcel.handle_accidental_formulas(result, how="formula")
        else:
            result = ModelToExcel.handle_accidental_formulas(temp, how="formula")

        return result
    

    def query_model_descriptors_df(self) -> pd.DataFrame:
        """
        Query database for model descriptors and their definitions.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: Model descriptors from database (currently returns empty result).
        """
        logging.info(f"Building Model Descriptors from Model {self.model_id}")
        
        model = self.query_model()

        results_dict = {}
        results_dict["model_details"] = {}
        results_dict["model_details"]["embedding"] = ["Intercept", *model.embedding]

        if any(method in model.qsar_method for method in ["reg", "las", "gcm"]):
            coefficients_df = self.query_model_coefficients()
            results_dict["model_details"]["model_coefficients"] = coefficients_df
        else:
            logging.warning(f"Model type {model.qsar_method} does not support coefficient retrieval.")
        
        model_descriptors = ModelToExcel.get_model_descriptors_df(results_dict)

        self.model_descriptors_df = model_descriptors

        logging.info(f"Finished building Model Descriptors from Model {self.model_id}")

        return model_descriptors
    

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
            model_descriptors = self.query_model_descriptors_df() if self.model_descriptors_df is None else self.model_descriptors_df
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        model_descriptors.to_excel(writer, sheet_name="Model Descriptors", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptors"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "Model Descriptors", model_descriptors)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Model Descriptors", model_descriptors, how="full", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "Model Descriptors", model_descriptors, has_subtotals=add_subtotals)

        return model_descriptors


    @staticmethod
    def get_model_descriptor_values_df(results_dict: Dict[str, Any], df_pred_cv: pd.DataFrame, df_pred_test: pd.DataFrame, df_training_model: pd.DataFrame, df_test_model: pd.DataFrame) -> pd.DataFrame:
        """
        Create dataframe combining predictions with descriptor values for all data points.
        
        Merges experimental and predicted values for training and test sets with their corresponding descriptor values, renaming columns for clarity.
        
        Args:
            results_dict (dict): Dictionary containing model details including units.
            df_pred_cv (pd.DataFrame): Training set predictions with cross-validation fold information.
            df_pred_test (pd.DataFrame): Test set predictions.
            df_training_model (pd.DataFrame): Training set descriptor values.
            df_test_model (pd.DataFrame): Test set descriptor values.
        
        Returns:
            pd.DataFrame: Combined dataframe with predictions and descriptor values for all points, with formula characters escaped.
        """
        # Get the units of the model (for the Observed and Predicted columns) and the columns to pull
        units = results_dict["model_details"].get("unitsModel", "Units")
        columns = ["exp_prop_id", "dtxcid", "casrn", "preferred_name", "canon_qsar_smiles", "exp", "pred"]

        # Get the experimental and predicted values for the test set
        test = df_pred_test.loc[:, columns]
        test["Set"] = "Test"

        # Get the experimental and predicted values for the training set
        training = df_pred_cv.loc[:, columns]
        training["Set"] = df_pred_cv.cv_fold.apply(lambda x: f"Training, Fold {x}")

        # Concatenate the test and training sets, and clean the columns
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

        # Get the descriptor values for the test set
        test_descriptors = df_test_model.drop(columns=["Property"])

        # Get the descriptor values for the training set
        training_descriptors = df_training_model.drop(columns=["Property"])

        # Concatenate the test and training set predictors, and clean the columns
        full_descriptors = pd.concat([test_descriptors, training_descriptors], ignore_index=True)
        full_descriptors["ID"] = full_descriptors["ID"].astype(str)
        full_descriptors = full_descriptors.rename(columns={"ID": "Canonical QSAR Ready Smiles"})

        # Merge the full set of experimental and predicted values with the model descriptor values
        final = full.merge(full_descriptors, on="Canonical QSAR Ready Smiles", how="left")

        final = ModelToExcel.handle_accidental_formulas(final)

        return final


    def query_model_descriptor_values_df(self) -> pd.DataFrame:
        """
        Query database for predictions and descriptor values.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: Predictions and descriptor values from database (currently returns empty result).
        """
        logging.info(f"Building Model Descriptor Values from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd()

        training = pd.merge(model.df_training, df_gmd, left_on="ID", right_on="canon_qsar_smiles", how="left")
        training = pd.merge(training, model.df_preds_training_cv, left_on="ID", right_on="id", how="left")

        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_col = np.zeros(len(model.df_training), dtype=int)
        for fold_index, (train_index, val_index) in enumerate(kfold_splitter.split(model.df_training)):
            fold_col[val_index] = fold_index

        training["Fold"] = fold_col
        training["Set"] = training.Fold.apply(lambda x: f"Training, Fold {x}")

        test = pd.merge(model.df_prediction, df_gmd, left_on="ID", right_on="canon_qsar_smiles", how="left")
        test = pd.merge(test, model.df_preds_test, left_on="ID", right_on="id", how="left")

        test["Set"] = "Test"

        temp = pd.concat([test, training], ignore_index=True)

        headers = model.embedding
        header_columns = {}
        for header in headers:
            header_columns[header] = temp[header]

        model_descriptor_values_dict = {
            "exp_prop_id": temp["qsar_exp_prop_property_values_id_first"],
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
        model_descriptor_values_df = ModelToExcel.handle_accidental_formulas(model_descriptor_values_df)

        self.model_descriptor_values_df = model_descriptor_values_df

        logging.info(f"Finished building Model Descriptor Values from Model {self.model_id}")

        return model_descriptor_values_df
    

    def model_descriptor_values(self, writer: Any, model_descriptor_values: Optional[pd.DataFrame]=None, add_subtotals: bool=True) -> pd.DataFrame:
        """
        Create the model descriptor values sheet in the Excel workbook.
        
        Shows predictions and descriptor values for all compounds in training and test sets. Includes autofilter and appropriately sized columns.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            model_descriptor_values (pd.DataFrame, optional): Descriptor values dataframe. If None, queries from database or uses instance dataframe.
        
        Returns:
            pd.DataFrame: The descriptor values dataframe that was written.
        """
        if model_descriptor_values is None:
            model_descriptor_values = self.query_model_descriptor_values_df() if self.model_descriptor_values_df is None else self.model_descriptor_values_df
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        model_descriptor_values.to_excel(writer, sheet_name="Model Descriptor Values", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptor Values"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "Model Descriptor Values", model_descriptor_values)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Model Descriptor Values", model_descriptor_values, min_col_width=7, col_width_pad=5, how="header", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "Model Descriptor Values", model_descriptor_values, has_subtotals=add_subtotals)

        return model_descriptor_values
    

    @staticmethod
    def get_training_cv_predictions_df(df_training_cv: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare training set cross-validation predictions for Excel output.
        
        Moves exp_prop_id to the first column for proper hyperlink anchoring.
        
        Args:
            df_training_cv (pd.DataFrame): Training cross-validation predictions dataframe.
        
        Returns:
            pd.DataFrame: Formatted predictions dataframe with exp_prop_id in first column.
        """
        exp_prop_id = df_training_cv.pop("exp_prop_id")
        df_training_cv.insert(0, "exp_prop_id", exp_prop_id)
        return df_training_cv
    

    def query_training_cv_predictions_df(self) -> pd.DataFrame:
        """
        Query database for training set cross-validation predictions.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: Training predictions from database (currently returns empty result).
        """
        logging.info(f"Building Training CV Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd()

        kfold_splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_col = np.zeros(len(model.df_preds_training_cv), dtype=int)
        for fold_index, (train_index, val_index) in enumerate(kfold_splitter.split(model.df_preds_training_cv)):
            fold_col[val_index] = fold_index

        temp = pd.merge(model.df_preds_training_cv, df_gmd, left_on="id", right_on="canon_qsar_smiles", how="left")

        training_cv_predictions_dict = {
            "exp_prop_id": temp["qsar_exp_prop_property_values_id_first"],
            "canon_qsar_smiles": temp["canon_qsar_smiles"],
            "exp": temp["exp"],
            "pred": temp["pred"],
            "cv_fold": fold_col,
            "dtxcid": temp["dtxcid"],
            "dtxsid": temp["dtxsid"],
            "casrn": temp["casrn"],
            "preferred_name": temp["preferred_name"],
            "smiles": temp["smiles"],
            "mol_weight": temp["mol_weight"]
        }
        training_cv_predictions_df = pd.DataFrame(training_cv_predictions_dict)

        self.training_cv_predictions_df = training_cv_predictions_df

        logging.info(f"Finished building Training CV Predictions from Model {self.model_id}")

        return training_cv_predictions_df
    

    def training_cv_predictions(self, writer: Any, training_cv_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=5, min_col_width: int=7, property_name: Optional[str]=None, property_units: Optional[str]=None) -> pd.DataFrame:
        """
        Create the training CV predictions sheet in the Excel workbook with scatter plot.
        
        Displays cross-validation predictions with observed vs predicted plot, autofilter, and appropriate column widths.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            training_cv_predictions (pd.DataFrame, optional): Training predictions dataframe. If None, queries from database or uses instance dataframe.
            x_col (str): Column name to use for x-axis in prediction plots. If None, defaults to 'exp'.
            y_col (str): Column name to use for y-axis in prediction plots. If None, defaults to 'pred'.
            chart_size_px (int): Square chart size in pixels. Defaults to 520.
            pad_ratio (float): Axis padding as fraction of data span. Defaults to 0.02.
            integer_ticks (bool): Whether to use integer-based tick spacing. Defaults to True.
            yx_offset_rows (int): Empty rows between data and y=x helper points. Defaults to 3.
            col_width_pad (int): Extra padding for column widths. Defaults to 5.
            min_col_width (int): Minimum column width in characters. Defaults to 7.
            property_name (str, optional): Name of the property being modeled. Used for chart labeling. If None, uses y_col name.
            property_units (str, optional): Units of the property (e.g., 'mg/L', 'log scale'). Appended to y-axis label if provided.
        
        Returns:
            pd.DataFrame: The training predictions dataframe that was written.
        """
        if training_cv_predictions is None:
            training_cv_predictions = self.query_training_cv_predictions_df() if self.training_cv_predictions_df is None else self.training_cv_predictions_df
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        training_cv_predictions.to_excel(writer, sheet_name="Training CV Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Training CV Predictions"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "Training CV Predictions", training_cv_predictions)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Training CV Predictions", training_cv_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "Training CV Predictions", training_cv_predictions, has_subtotals=add_subtotals)
        
        ModelToExcel.add_plot(writer, workbook, "Training CV Predictions", "Training CV Predictions", training_cv_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)

        return training_cv_predictions
    

    @staticmethod
    def get_test_set_predictions_df(df_test: pd.DataFrame, actual_ads: Optional[list]=None) -> pd.DataFrame:
        """
        Prepare test set predictions for Excel output.
        
        Moves exp_prop_id to the first column for proper hyperlink anchoring.
        
        Args:
            df_test (pd.DataFrame): Test set predictions dataframe.
        
        Returns:
            pd.DataFrame: Formatted predictions dataframe with exp_prop_id in first column.
        """
        exp_prop_id = df_test.pop("exp_prop_id")
        df_test.insert(0, "exp_prop_id", exp_prop_id)

        if actual_ads is not None:
            applicability_domain_cols = [col for col in df_test.columns if "AD" in col]
            for col in applicability_domain_cols:
                temp_name = col.replace("AD_", "")
                temp_name = temp_name.replace("_", " ")
                if temp_name not in actual_ads:
                    df_test.pop(col)
        return df_test
    

    def query_test_set_predictions_df(self) -> pd.DataFrame:
        """
        Query database for test set predictions.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: Test predictions from database (currently returns empty result).
        """
        logging.info(f"Building Test Set Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd = self.query_df_gmd()

        temp = pd.merge(model.df_preds_test, df_gmd, left_on="id", right_on="canon_qsar_smiles", how="left")

        ads = model.applicabilityDomainName.split(" and ")
        ad_test_columns = {}
        for ad in ads:
            df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                    train_df=model.df_training.copy(), test_df=model.df_prediction.copy(),
                    remove_log_p=model.remove_log_p_descriptors,
                    embedding=model.embedding, applicability_domain=ad,
                    filterColumnsInBothSets=False,
                    returnTrainingAD=False)
            ad_test_columns["AD_" + ad.replace(" ", "_")] = df_ad_output["AD"]

        test_predictions_dict = {
            "exp_prop_id": temp["qsar_exp_prop_property_values_id_first"],
            "canon_qsar_smiles": temp["canon_qsar_smiles"],
            "exp": temp["exp"],
            "pred": temp["pred"],
            **ad_test_columns,
            "dtxcid": temp["dtxcid"],
            "dtxsid": temp["dtxsid"],
            "casrn": temp["casrn"],
            "preferred_name": temp["preferred_name"],
            "smiles": temp["smiles"],
            "mol_weight": temp["mol_weight"]
        }
        test_set_predictions_df = pd.DataFrame(test_predictions_dict)

        self.test_set_predictions_df = test_set_predictions_df

        logging.info(f"Finished building Test Set Predictions from Model {self.model_id}")

        return test_set_predictions_df
    

    def test_set_predictions(self, writer: Any, test_set_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=5, min_col_width: int=7, property_name: Optional[str]=None, property_units: Optional[str]=None) -> pd.DataFrame:
        """
        Create the test set predictions sheet in the Excel workbook with scatter plot.
        
        Displays test set predictions with observed vs predicted plot, autofilter, and appropriate column widths.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            test_set_predictions (pd.DataFrame, optional): Test predictions dataframe. If None, queries from database or uses instance dataframe.
            x_col (str): Column name to use for x-axis in prediction plots. If None, defaults to 'exp'.
            y_col (str): Column name to use for y-axis in prediction plots. If None, defaults to 'pred'.
            chart_size_px (int): Square chart size in pixels. Defaults to 520.
            pad_ratio (float): Axis padding as fraction of data span. Defaults to 0.02.
            integer_ticks (bool): Whether to use integer-based tick spacing. Defaults to True.
            yx_offset_rows (int): Empty rows between data and y=x helper points. Defaults to 3.
            col_width_pad (int): Extra padding for column widths. Defaults to 5.
            min_col_width (int): Minimum column width in characters. Defaults to 7.
            property_name (str, optional): Name of the property being modeled. Used for chart labeling. If None, uses y_col name.
            property_units (str, optional): Units of the property (e.g., 'mg/L', 'log scale'). Appended to y-axis label if provided.
        
        Returns:
            pd.DataFrame: The test predictions dataframe that was written.
        """
        if test_set_predictions is None:
            test_set_predictions = self.query_test_set_predictions_df() if self.test_set_predictions_df is None else self.test_set_predictions_df
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        test_set_predictions.to_excel(writer, sheet_name="Test Set Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["Test Set Predictions"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "Test Set Predictions", test_set_predictions)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Test Set Predictions", test_set_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "Test Set Predictions", test_set_predictions, has_subtotals=add_subtotals)
        
        ModelToExcel.add_plot(writer, workbook, "Test Set Predictions", "Test Set Predictions", test_set_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)

        return test_set_predictions
    

    @staticmethod
    def get_external_predictions_df(df_ext: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare external/validation set predictions for Excel output.
        
        Moves exp_prop_id to the first column for proper hyperlink anchoring.
        
        Args:
            df_ext (pd.DataFrame): External predictions dataframe.
        
        Returns:
            pd.DataFrame: Formatted predictions dataframe with exp_prop_id in first column.
        """
        exp_prop_id = df_ext.pop("exp_prop_id")
        df_ext.insert(0, "exp_prop_id", exp_prop_id)
        return df_ext
    

    def query_external_predictions_df(self) -> pd.DataFrame:
        """
        Query database for external validation set predictions.
        
        Note: This method is a placeholder and requires SQL implementation.
        
        Returns:
            pd.DataFrame: External predictions from database (currently returns empty result).
        """
        logging.info(f"Building External Predictions from Model {self.model_id}")

        model = self.query_model()
        df_gmd_external = self.query_df_gmd_external()

        temp = pd.merge(model.df_preds_external, df_gmd_external, left_on="id", right_on="canon_qsar_smiles", how="left")

        external_predictions_dict = {
            "exp_prop_id": temp["qsar_exp_prop_property_values_id_first"],
            "canon_qsar_smiles": temp["canon_qsar_smiles"],
            "exp": temp["exp"],
            "pred": temp["pred"],
            "dtxcid": temp["dtxcid"],
            "dtxsid": temp["dtxsid"],
            "casrn": temp["casrn"],
            "preferred_name": temp["preferred_name"],
            "smiles": temp["smiles"],
            "mol_weight": temp["mol_weight"]
        }
        external_predictions_df = pd.DataFrame(external_predictions_dict)

        self.external_predictions_df = external_predictions_df

        logging.info(f"Finished building External Predictions from Model {self.model_id}")

        return external_predictions_df
    

    def external_predictions(self, writer: Any, external_predictions: Optional[pd.DataFrame]=None, add_subtotals: bool=True, x_col: str=None, y_col: str=None, chart_size_px: int=520, pad_ratio: float=0.02, integer_ticks: bool=True, yx_offset_rows: int=3, col_width_pad: int=5, min_col_width: int=7, property_name: Optional[str]=None, property_units: Optional[str]=None) -> Optional[pd.DataFrame]:
        """
        Create the external predictions sheet in the Excel workbook with scatter plot.
        
        Displays external/validation set predictions with observed vs predicted plot (if data available), autofilter, and appropriate column widths.
        
        Args:
            writer: pandas ExcelWriter object for writing to the workbook.
            external_predictions (pd.DataFrame, optional): External predictions dataframe. If None, queries from database or uses instance dataframe.
            x_col (str): Column name to use for x-axis in prediction plots. If None, defaults to 'exp'.
            y_col (str): Column name to use for y-axis in prediction plots. If None, defaults to 'pred'.
            chart_size_px (int): Square chart size in pixels. Defaults to 520.
            pad_ratio (float): Axis padding as fraction of data span. Defaults to 0.02.
            integer_ticks (bool): Whether to use integer-based tick spacing. Defaults to True.
            yx_offset_rows (int): Empty rows between data and y=x helper points. Defaults to 3.
            col_width_pad (int): Extra padding for column widths. Defaults to 5.
            min_col_width (int): Minimum column width in characters. Defaults to 7.
            property_name (str, optional): Name of the property being modeled. Used for chart labeling. If None, uses y_col name.
            property_units (str, optional): Units of the property (e.g., 'mg/L', 'log scale'). Appended to y-axis label if provided.
        
        Returns:
            pd.DataFrame: The external predictions dataframe that was written, or None if no data available.
        """
        if external_predictions is None:
            external_predictions = self.query_external_predictions_df() if self.external_predictions_df is None else self.external_predictions_df
            if external_predictions is None:
                return external_predictions
        
        start_row = ModelToExcel.get_header_row(has_subtotals=add_subtotals)
        external_predictions.to_excel(writer, sheet_name="External Predictions", index=False, startrow=start_row)

        workbook = writer.book
        worksheet = writer.sheets["External Predictions"]
        if add_subtotals:
            ModelToExcel.add_subtotals(writer, "External Predictions", external_predictions)
            worksheet.freeze_panes(2, 0)
        else:
            worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "External Predictions", external_predictions, min_col_width=min_col_width, col_width_pad=col_width_pad, how="header", has_subtotals=add_subtotals)
        ModelToExcel.add_filter(writer, "External Predictions", external_predictions, has_subtotals=add_subtotals)
        
        ModelToExcel.add_plot(writer, workbook, "External Predictions", "External Predictions", external_predictions, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=self.log_plot, yx_offset_rows=yx_offset_rows, property_name=property_name, property_units=property_units, has_subtotals=add_subtotals)
        
        return external_predictions


    @staticmethod
    def initialize_from_model(model: Model, args, **kwargs) -> 'ModelToExcel':
        """
        Initialize a ModelToExcel instance from a Model object.
        
        Extracts necessary dataframes and parameters from the Model object to populate the ModelToExcel instance for Excel export.
        
        Args:
            model (Model): The model object containing all relevant data and parameters.
        """
        mte = ModelToExcel(model_id=model.modelId, model=model, args=args, **kwargs)

        mte.cover_sheet_df = ModelToExcel.query_cover_sheet_df(model.results_dict)
        mte.statistics_df = ModelToExcel.query_statistics_df(model.results_dict)
        mte.records_df = ModelToExcel.query_records_df(model.results_dict)

        mte.model_descriptors_df = ModelToExcel.query_model_descriptors_df(model.results_dict)
        mte.model_descriptor_values_df = ModelToExcel.query_model_descriptor_values_df(model.results_dict, model.df_pred_cv, model.df_pred_test, model.df_training_model, model.df_test_model)
        mte.training_cv_predictions_df = ModelToExcel.query_training_cv_predictions_df(model.df_pred_cv)
        mte.test_set_predictions_df = ModelToExcel.query_test_set_predictions_df(model.df_pred_test)
        mte.external_predictions_df = ModelToExcel.query_external_predictions_df(model.df_ext) if model.df_ext is not None else None

        mte.log_plot = "log" in model.unitsModel.lower()

        return mte
    
    
    @staticmethod
    def set_column_width(
        writer: Any,
        sheet_name: str,
        df: pd.DataFrame,
        col_width_pad: int=5,
        min_col_width: int=7,
        how: str="header",
        first_col_format: Optional[Any] = None,
        has_subtotals: bool = False) -> None:
        """
        Set column widths in Excel sheet based on content or header width.
        
        Args:
            writer: pandas ExcelWriter object for accessing the worksheet.
            sheet_name (str): Name of the sheet to format.
            df (pd.DataFrame): Dataframe whose columns determine width.
            col_width_pad (int): Padding to add to calculated width. Defaults to 4.
            min_col_width (int): Minimum column width. Defaults to 5.
            how (str): Method for calculating width - 'header' uses header length, 'full' scans all data. Defaults to 'header'.
            first_col_format: Optional cell format to apply to the first column.
            has_subtotals (bool): Whether subtotals are present above headers. Defaults to False.
        """
        worksheet = writer.sheets[sheet_name]
        data_start_row = ModelToExcel.get_data_start_row(has_subtotals)
        
        for col_idx in range(len(df.columns)):
            if how == "header":
                header_entry = df.columns[col_idx]
                header_width = len(header_entry)
                col_width = max(header_width, min_col_width) + col_width_pad
                worksheet.set_column(col_idx, col_idx, col_width, first_col_format if col_idx == 0 else None)
            elif how == "full":
                # Find the maximum width needed for data in this column
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
                worksheet.set_column(col_idx, col_idx, col_width, first_col_format if col_idx == 0 else None)


    @staticmethod
    def get_header_row(has_subtotals: bool = False) -> int:
        """
        Get the row number where headers are located.
        
        Args:
            has_subtotals (bool): Whether subtotals are present above headers. Defaults to False.
        
        Returns:
            int: The 0-based row index of the header row.
        """
        return 2 if has_subtotals else 0
    

    @staticmethod
    def get_data_start_row(has_subtotals: bool = False) -> int:
        """
        Get the row number where data starts (first row after headers).
        
        Args:
            has_subtotals (bool): Whether subtotals are present above headers. Defaults to False.
        
        Returns:
            int: The 0-based row index of the first data row.
        """
        return 3 if has_subtotals else 1


    @staticmethod
    def add_subtotals(writer: Any, sheet_name: str, df: pd.DataFrame) -> None:
        """
        Add SUBTOTAL formulas above table headers to count visible cells per column.
        
        Inserts a row at the top of the sheet with SUBTOTAL(3, column_range) formulas,
        which count non-empty cells in each column that are currently visible (respecting filters).
        This shifts the data down one row and requires using has_subtotals=True in other methods.
        
        Args:
            writer: pandas ExcelWriter object for accessing the worksheet.
            sheet_name (str): Name of the sheet where subtotals will be added.
            df (pd.DataFrame): Dataframe providing column information.
        """
        ws = writer.sheets[sheet_name]
        nrows, ncols = df.shape
        
        # Add SUBTOTAL formulas to each column
        # SUBTOTAL(3, range) uses COUNTA function to count non-empty cells
        # The data now starts at row 4 (0-indexed row 3) after insertion
        for col_idx in range(ncols):
            start_cell = xl_rowcol_to_cell(3, col_idx)  # Ensure we are using the correct row index for the formula range
            end_cell = xl_rowcol_to_cell(nrows + 2, col_idx)  # Data ends at row nrows + 3 (accounting for header and subtotal row)
            # Range: from row 4 to row (nrows + 4) since we inserted 1 row
            # (original row 3 is now row 4 due to the insertion)
            range_str = f"{start_cell}:{end_cell}"
            formula = f"=SUBTOTAL(3,{range_str})"
            ws.write_formula(0, col_idx, formula)
        
        logging.debug(f"Added SUBTOTAL formulas to {sheet_name} with {ncols} columns")


    @staticmethod
    def add_filter(writer: Any, sheet_name: str, df: pd.DataFrame, has_subtotals: bool = False) -> None:
        """
        Add autofilter to a sheet for easy data filtering.
        
        Enables filter dropdown buttons on all column headers. Automatically adjusts for subtotal rows.
        
        Args:
            writer: pandas ExcelWriter object for accessing the worksheet.
            sheet_name (str): Name of the sheet to add filter to.
            df (pd.DataFrame): Dataframe providing shape information for filter range.
            has_subtotals (bool): Whether subtotals are present above headers. Defaults to False.
        """
        # Get the worksheet object
        ws = writer.sheets[sheet_name]

        # Determine the range (0-indexed for xlsxwriter)
        nrows, ncols = df.shape
        
        # Get the header row based on whether subtotals exist
        header_row = ModelToExcel.get_header_row(has_subtotals)

        # Add an auto-filter to the full range (header row position; data ends at row nrows + offset for subtotals)
        data_end_row = nrows + (1 if has_subtotals else 0)
        ws.autofilter(header_row, 0, data_end_row, ncols - 1)
    

    @staticmethod
    def handle_accidental_formulas(df: pd.DataFrame, how: str="formula") -> pd.DataFrame:
        """
        Escape special characters in cells that Excel would interpret as formula starts.
        
        Prevents cells starting with =, +, -, or @ from being treated as formulas by Excel.
        
        Args:
            df (pd.DataFrame): Dataframe to process.
            how (str): Escaping method - 'formula' wraps in formula quotes, 'quote' prepends single quote. Defaults to 'formula'.
        
        Returns:
            pd.DataFrame: Processed dataframe with escaped special characters.
        """
        # Excel treats cells starting with =, +, -, @ as formulas
        # Use an explicit formula to allow values to display properly as is, but alter the actual stored value
        if how == "formula":
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f'="{x}"' if isinstance(x, str) and x and x[0] in ('=', '+', '-', '@') else x
                )
                if col[0] in ("=", "+", "-", "@"):
                    df = df.rename(columns={col: f'="{col}"'})
        # Prepend a single quote to treat them as literal strings, making a more minor alteration to the stored value
        elif how == "quote":
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: f"'{x}" if isinstance(x, str) and x and x[0] in ('=', '+', '-', '@') else x
                )
                if col[0] in ("=", "+", "-", "@"):
                    df = df.rename(columns={col: f"'{col}"})
        
        return df


    @staticmethod
    def nice_integer_major_unit(span: int, target_ticks: int=5) -> int:
        """
        Calculate a nice integer major unit for chart axis tick spacing.
        
        Computes a round number for axis ticks based on the data span, preferring multiples of 1, 2, 5, or 10.
        
        Args:
            span (int): Range of values (max - min).
            target_ticks (int): Desired approximate number of ticks. Defaults to 5.
        
        Returns:
            int: Nice integer major unit for axis tick spacing.
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
    def add_hyperlinks_to_sheet(
        writer: pd.ExcelWriter,
        source_sheet: str,
        target_sheet: str,
        df_source: pd.DataFrame,
        df_target: pd.DataFrame,
        link_column: str = "exp_prop_id",
        has_subtotals: bool=True) -> None:
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
            excel_row = idx + ModelToExcel.get_data_start_row(has_subtotals=has_subtotals) + 1  # Accounts for header and optional subtotal row
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
                excel_row = src_idx + ModelToExcel.get_data_start_row(has_subtotals=has_subtotals) + 1  # Accounts for header and optional subtotal row
                
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

                        ws_source.write_url(cell_range, url=address, string=src_val_str, tip=screen_tip)

                        hyperlinks_added += 1
                    except Exception as e:
                        logging.warning(f"Failed to write hyperlink for {source_sheet} row {excel_row}: {e}")
                else:
                    unmatched_count += 1
        
        except Exception as e:
            logging.error(f"Error adding hyperlinks: {e}")
        
        logging.info(f"Added {hyperlinks_added} hyperlinks from {source_sheet} to {target_sheet} ({unmatched_count} unmatched)")

    
    @staticmethod
    def compute_equal_axis_bounds(
        x_values: Iterable[float],
        y_values: Iterable[float],
        pad_ratio: float=0.02,
        integer_ticks: bool=True,
        log_plot: bool=True,
        target_ticks: int=5,
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calculate equal bounds for both axes of a scatter plot with padding.
        
        Ensures x and y axes have identical ranges for better visual comparison on prediction plots. Computes nice integer tick spacing if requested.
        
        Args:
            x_values (Iterable[float]): X-axis values (experimental values).
            y_values (Iterable[float]): Y-axis values (predicted values).
            pad_ratio (float): Fraction of total span to pad axes. Defaults to 0.02 (2%).
            integer_ticks (bool): Whether to use integer-based tick spacing. Defaults to True.
            log_plot (bool): Whether this is a log-scale plot. Affects major unit calculation. Defaults to True.
            target_ticks (int): Desired approximate number of axis ticks. Defaults to 5.
        
        Returns:
            Tuple[float, float, Optional[float]]: (min_value, max_value, major_unit) where major_unit is None for non-integer ticks.
        
        Raises:
            ValueError: If no numeric values are provided.
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
            # For integer ticks, always use major_unit=1 to show ticks at every integer
            # This is important for log-log plots where integer differences represent significant changes
            if log_plot:
                major_unit = 1
            else:
                major_unit = ModelToExcel.nice_integer_major_unit(int_max - int_min, target_ticks=target_ticks)
            
            return float(int_min), float(int_max), float(major_unit)
    
        return mn, mx, None
    

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
        has_subtotals: bool = False) -> None:
        """
        Add a prediction vs experimental scatter plot to the worksheet.
        
        Creates a scatter plot with predicted (y-axis) vs experimental (x-axis) values, including a y=x reference line. 
        Data points are plotted as circles, and axes are scaled equally with integer tick marks.
        
        Args:
            writer: pandas ExcelWriter object for accessing worksheets.
            workbook: xlsxwriter workbook object for creating the chart.
            sheet_name (str): Name of the sheet containing data.
            sheet_name_plot (str): Name of the sheet where chart will be inserted.
            df (pd.DataFrame): Dataframe with data to plot.
            x_col (str): Name of the column to use for x-axis. Defaults to "exp".
            y_col (str): Name of the column to use for y-axis. Defaults to "pred".
            chart_size_px (int): Square chart size in pixels. Defaults to 520.
            pad_ratio (float): Axis padding as fraction of data span. Defaults to 0.02.
            integer_ticks (bool): Whether to use integer-based tick spacing. Defaults to True.
            log_plot (bool): Whether this is a log-scale plot. Defaults to True.
            yx_offset_rows (int): Empty rows between data and y=x helper points. Defaults to 3.
            property_name (str, optional): Name of the property being modeled. Used for chart labeling. If None, uses y_col name.
            property_units (str, optional): Units of the property (e.g., 'mg/L', 'log scale'). Appended to y-axis label if provided.
            has_subtotals (bool): Whether subtotals are present above headers. Defaults to False.
        """
        worksheet = writer.sheets[sheet_name]
        worksheet_plot = writer.sheets[sheet_name_plot]
        # Hide worksheet gridlines (screen + print)
        worksheet.hide_gridlines(2)
        nrows = len(df)
        
        # Calculate the actual data row position based on subtotals and headers
        data_start_row = ModelToExcel.get_data_start_row(has_subtotals)
        
        # Get column indices for x and y columns (0-based)
        if x_col is None:
            x_col = "exp"
            x_col_name = "Experimental"
        else:
            x_col_name = x_col
        if y_col is None:
            y_col = "pred"
            y_col_name = "Predicted"
        else:
            y_col_name = y_col
        x_col_idx = df.columns.get_loc(x_col)
        y_col_idx = df.columns.get_loc(y_col)
        
        # Compute unified bounds (also used for y=x line)
        mn, mx, major_unit = ModelToExcel.compute_equal_axis_bounds(
            df[x_col], df[y_col], pad_ratio=pad_ratio, integer_ticks=integer_ticks, log_plot=log_plot, target_ticks=5)
        # Create scatter chart with markers
        chart = workbook.add_chart({"type":"scatter", "subtype":"straight_with_markers"})
        title = f"{sheet_name} for {property_name}" if property_name is not None else f"{y_col.capitalize()} vs {x_col.capitalize()}"
        series_name = f"{y_col.capitalize()} vs {x_col.capitalize()}"
        chart.set_title({"name": title})
        chart.set_style(10)
        # Series 1: data points (markers only)
        # Use the actual column indices from the dataframe, with dynamic row calculation
        data_end_row = data_start_row + nrows - 1
        chart.add_series({
                "name":series_name,
                "categories":[sheet_name, data_start_row, x_col_idx, data_end_row, x_col_idx],
                "values":[sheet_name, data_start_row, y_col_idx, data_end_row, y_col_idx],
                "marker":{"type":"circle", "size":6},
                "line":{"none":True}})
        # Write y = x helper points a few rows below the data, using the same columns as the data
        yx_row_start = data_end_row + 1 + yx_offset_rows  # 0-based row index after header and data
        worksheet.write_number(yx_row_start, x_col_idx, mn)  # Write min value to x column
        worksheet.write_number(yx_row_start, y_col_idx, mn)  # Write min value to y column
        worksheet.write_number(yx_row_start + 1, x_col_idx, mx)  # Write max value to x column
        worksheet.write_number(yx_row_start + 1, y_col_idx, mx)  # Write max value to y column
        # Series 2: y = x line (dark blue, solid, no markers)
        # Reference the helper points in the same columns as the data
        chart.add_series({
                "name":"y = x",
                "categories":[sheet_name, yx_row_start, x_col_idx, yx_row_start + 1, x_col_idx],
                "values":[sheet_name, yx_row_start, y_col_idx, yx_row_start + 1, y_col_idx],
                "marker":{"type":"none"},
                "line":{"color":"#1f4e79", "width":2.25}})  # dark blue
        # Axes with same bounds, integer labels, no gridlines
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
                
        # Make the chart square
        chart.set_size({"width":chart_size_px, "height":chart_size_px})
        # Plot area: border + margins so axis titles/labels aren't overlapped
        chart.set_plotarea({
                "fill":{"none":True},
                "border":{"color":"#666666", "width":1.0},
                "layout":{
                    "x":0.12,  # left padding for y-axis title + labels
                    "y":0.10,  # top padding for chart title
                    "width":0.84,  # reduce width so right edge doesn't crowd labels/legend
                    "height":0.80}})  # reduce height to leave room for x-axis title
        # Legend: bottom-right, overlaid, explicit size to avoid warnings
        chart.set_legend({
                "overlay":True,
                "layout":{"x":0.7, "y":0.75, "width":0.25, "height":0.1}})
        
        if sheet_name == sheet_name_plot:
            # Position chart to the right of the data
            # Estimate: each column is ~20 characters wide, chart starts after all data columns
            # Add some buffer columns for spacing (typically 1-2 columns)
            chart_start_col = len(df.columns)  # + 1?
            chart_position = xl_rowcol_to_cell(data_start_row, chart_start_col)  # Convert to Excel cell reference
            worksheet_plot.insert_chart(chart_position, chart, {"x_offset":20, "y_offset":10})
        else:
            # If chart is on a separate sheet, place it at top-left
            worksheet_plot.insert_chart(0, 0, chart, {'x_scale': 1.0, 'y_scale': 1.0})
    
    
    @staticmethod
    def getEngine(connect_args: Optional[dict] = None) -> Any:
        connect_url = URL.create(
            drivername='postgresql+psycopg2',
            username=os.getenv('DEV_QSAR_USER'),
            password=os.getenv('DEV_QSAR_PASS'),
            host=os.getenv('DEV_QSAR_HOST', 'localhost'),
            port=os.getenv('DEV_QSAR_PORT', 5432),
            database=os.getenv('DEV_QSAR_DATABASE')
        )
                
        engine = create_engine(connect_url, echo=False)
        return engine


    @staticmethod
    def getSession(engine: Any=None) -> Any:
        if engine is None:
            engine = ModelToExcel.getEngine()
        Session = sessionmaker(bind=engine)
        session = Session()
        return session
    

    def query_model(self) -> Optional[Model]:
        """
        Query the database for the model.
        """
        try:            
            if self.model is None:
                logging.info(f"Loading model {self.model_id} from database")
                initializer: ModelInitializer = ModelInitializer()
                model: Model = initializer.initModel(self.model_id)
                self.model = model
                logging.info(f"Done loading model {self.model_id} from database")
            else:
                model = self.model
            
            if model is None:
                logging.error(f"Failed to load model {self.model_id}")
                return None
                        
            return model
        
        except Exception as e:
            logging.error(f"Error retrieving model {self.model_id}: {e}")
            traceback.print_exc()
            return None
    

    def query_df_pv(self) -> Optional[pd.DataFrame]:
        """
        Query df_pv from the database using getMappedDatapoints
        """
        try:
            if self.df_pv is None:
                logging.info("Loading df_pv from database")
                
                if self.dataset_name is None:
                    try:
                        model = self.query_model()
                        self.dataset_name = model.datasetName
                    except Exception as e:
                        logging.warning("dataset_name not set and failed to query from database. Cannot query records without dataset_name. Please set dataset_name or ensure it can be queried from the database with the provided model_id.")
                        raise e
                
                if self.snapshot_id is None:
                    self.snapshot_id = 4  # TODO: Determine how to set snapshot_id/what it is? Is also a TODO in run_model_building_db...

                if self.duplicate_strategy is None:
                    self.duplicate_strategy = "id_suffix"  # TODO: Determine how to set duplicate_strategy and what the options should be. Is a hardcoded constant in run_model_building_db...
                
                edg = ExpDataGetter()
                df_pv, _ = edg.get_mapped_property_values(self.session, self.dataset_name, self.snapshot_id, duplicate_strategy=self.duplicate_strategy)

                self.df_pv = df_pv

                logging.info(f"Done loading df_pv from database")
            else:
                df_pv = self.df_pv
            
            if df_pv is None:
                logging.error(f"Failed to query df_pv")
                return None

            return df_pv
        
        except Exception as e:
            logging.error(f"Error retrieving df_pv: {e}")
            traceback.print_exc()
            return None


    def query_df_gmd(self) -> Optional[pd.DataFrame]:
        """
        Query df_gmd from the database using getMappedDatapoints
        """
        try:
            if self.df_gmd is None:
                logging.info("Loading df_gmd from database")

                engine = ModelToExcel.getEngine() if self.engine is None else self.engine
                session = ModelToExcel.getSession(engine) if self.session is None else self.session

                if self.dataset_name is None:
                    try:
                        model = self.query_model()
                        self.dataset_name = model.datasetName
                    except Exception as e:
                        logging.warning("dataset_name not set and failed to query from database. Cannot query records without dataset_name. Please set dataset_name or ensure it can be queried from the database with the provided model_id.")
                        raise e
                dataset_name = self.dataset_name
                
                df_gmd = getMappedDatapoints(session, dataset_name)

                self.df_gmd = df_gmd

                logging.info(f"Done loading df_gmd from database")
            else:
                df_gmd = self.df_gmd
            
            if df_gmd is None:
                logging.error(f"Failed to query df_gmd")
                return None

            return df_gmd
        
        except Exception as e:
            logging.error(f"Error retrieving df_gmd: {e}")
            traceback.print_exc()
            return None
    

    def query_df_gmd_external(self) -> Optional[pd.DataFrame]:
        """
        Query df_gmd_external from the database using getMappedDatapoints
        """
        try:
            if self.df_gmd_external is None:
                logging.info("Loading df_gmd_external from database")

                engine = ModelToExcel.getEngine() if self.engine is None else self.engine
                session = ModelToExcel.getSession(engine) if self.session is None else self.session

                if self.dataset_name_external is None:
                    try:
                        model = self.query_model()
                        self.dataset_name_external = model.external_dataset_name
                    except Exception as e:
                        logging.warning("dataset_name_external not set and failed to query from database. Cannot query records without dataset_name_external. Please set dataset_name_external or ensure it can be queried from the database with the provided model_id.")
                        raise e
                dataset_name_external = self.dataset_name_external
                
                df_gmd_external = getMappedDatapoints(session, dataset_name_external)

                self.df_gmd_external = df_gmd_external

                logging.info(f"Done loading df_gmd_external from database")
            else:
                df_gmd_external = self.df_gmd_external
            
            if df_gmd_external is None:
                logging.error(f"Failed to query df_gmd_external")
                return None

            return df_gmd_external
        
        except Exception as e:
            logging.error(f"Error retrieving df_gmd_external: {e}")
            traceback.print_exc()
            return None
    

    def query_model_coefficients(self) -> Optional[pd.DataFrame]:
        """
        Query the database for model coefficients.
        """
        try:            
            model = self.query_model()
            
            if model is None:
                logging.error(f"Failed to load model {self.model_id}")
                return None
            elif model.qsar_method not in ["reg", "las", "gcm"]:
                logging.warning(f"Model {self.model_id} has QSAR method that does not support coefficient retrieval: {model.qsar_method}")
                return None
                        
            df_training = model.df_training

            y = df_training[df_training.columns[1]]
            X = df_training[model.embedding]

            # Step 3: Call the method that computes coefficients
            coefficients_json = model.getOriginalRegressionCoefficients2(X, y)
            
            # Step 4: Parse JSON and return
            coefficients_dict = json.loads(coefficients_json)
            
            return coefficients_dict
            
        except Exception as e:
            logging.error(f"Error retrieving coefficients: {e}")
            traceback.print_exc()
            return None


    def create_excel(
            self,
            x_col: str=None,
            y_col: str=None,
            chart_size_px: int=520,  # square chart size
            pad_ratio: float=0.02,
            integer_ticks: bool=True,
            yx_offset_rows: int=3,  # empty rows between data and y=x helper points
            col_width_pad: int=4,
            min_col_width: int=5
        ) -> None:
        """
        Generate complete Excel workbook with all model summary sheets and charts.
        
        Creates an Excel file with multiple sheets containing model summary information, statistics, experimental records, 
        descriptor information, and prediction plots for training, test, and external validation sets. Hyperlinks are added 
        as a post-processing step after the file is written.
        
        Args:
            x_col (str): Column name to use for x-axis in prediction plots. If None, defaults to 'exp'.
            y_col (str): Column name to use for y-axis in prediction plots. If None, defaults to 'pred'.
            chart_size_px (int): Square chart size in pixels for prediction plots. Defaults to 520.
            pad_ratio (float): Axis padding as fraction of data span. Defaults to 0.02.
            integer_ticks (bool): Whether to use integer-based tick spacing on charts. Defaults to True.
            yx_offset_rows (int): Empty rows between chart data and y=x reference line helper points. Defaults to 3.
            col_width_pad (int): Padding added to calculated column widths. Defaults to 4.
            min_col_width (int): Minimum column width. Defaults to 5.
        """
        logging.info("Creating detailed Excel...")
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

            logging.info("Creating External Predictions...")
            df = self.external_predictions(writer, self.external_predictions_df, add_subtotals=self.add_subtotals, x_col=x_col, y_col=y_col, chart_size_px=chart_size_px, pad_ratio=pad_ratio, integer_ticks=integer_ticks, yx_offset_rows=yx_offset_rows, col_width_pad=col_width_pad, min_col_width=min_col_width, property_name=property_name, property_units=property_units)

            # logging.info("Done creating detailed Excel!")
            # logging.info("Done with initial passthrough of all sheets!")

            # Add Hyperlinks
            logging.info("Adding hyperlinks...")
            try:
                ModelToExcel.add_hyperlinks_to_sheet(writer, "Records", "Training CV Predictions", self.records_df, self.training_cv_predictions_df, has_subtotals=self.add_subtotals)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ModelToExcel.add_hyperlinks_to_sheet(writer, "Records", "Test Set Predictions", self.records_df, self.test_set_predictions_df, has_subtotals=self.add_subtotals)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ModelToExcel.add_hyperlinks_to_sheet(writer, "Training CV Predictions", "Records", self.training_cv_predictions_df, self.records_df, has_subtotals=self.add_subtotals)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            try:
                ModelToExcel.add_hyperlinks_to_sheet(writer, "Test Set Predictions", "Records", self.test_set_predictions_df, self.records_df, has_subtotals=self.add_subtotals)
            except Exception as e:
                logging.error(f"Error adding hyperlinks: {e}")
            
            logging.info("Done creating detailed Excel!")


def custom_encoder(obj):
    if isinstance(obj, PMMLPipeline):
        return obj.__dict__
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')


def main():
    engine = ModelToExcel.getEngine()
    session = ModelToExcel.getSession(engine)
    model_id = 1746
    excel_path = "test_summary.xlsx"
    add_subtotals = True
    exclude_blank_columns = True
    test = ModelToExcel(engine=engine, session=session, model_id=model_id, excel_path=excel_path, add_subtotals=add_subtotals, exclude_blank_columns=exclude_blank_columns)
    test.create_excel()


def main2():
    engine = ModelToExcel.getEngine()
    session = ModelToExcel.getSession(engine)
    model_id = 1746
    excel_path = "test_summary.xlsx"
    test = ModelToExcel(engine=engine, session=session, model_id=model_id, excel_path=excel_path)
    model = test.query_model()
    print(model.__dict__)

    with open("test_model_details.json", "w") as f:
        json.dump(model.__dict__, f, indent=4, default=custom_encoder)
    with open("test_model.pkl", "wb") as f:
        f.write(pickle.dumps(model))
    
    df_pv = test.query_df_pv()
    with open("test_df_pv.pkl", "wb") as f:
        pickle.dump(df_pv, f)


def main3():
    engine = ModelToExcel.getEngine()
    session = ModelToExcel.getSession(engine)
    dataset_name = "KOC v1 modeling"
    df_gmd = getMappedDatapoints(session, dataset_name)

    with open("test_df_gmd.pkl", "wb") as f:
        pickle.dump(df_gmd, f)
    
    dataset_name_external = "KOC v1 external"
    df_gmd_external = getMappedDatapoints(session, dataset_name_external)

    with open("test_df_gmd_external.pkl", "wb") as f:
        pickle.dump(df_gmd_external, f)


if __name__ == "__main__":
    main()
    # main2()
    # main3()
