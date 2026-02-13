from model_ws_db_utilities import getEngine, getSession
import pandas as pd
from sqlalchemy import text
from pathlib import Path

import logging
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, force=True)


class ModelToExcel:
    def __init__(
            self,
            excel_path: str="summary.xlsx",
            cover_sheet_df = None,
            statistics_df = None,
            training_set_df = None,
            test_set_df = None,
            records_df = None,
            records_field_descriptions_df = None,
            test_set_predictions_df = None,
            model_descriptors_df = None,
            model_descriptor_values_df = None,
            engine=None,
            session=None,
            model_id: int=1065
        ):
        # TODO: Determine correct default for excel_path given model_id
        #       Maybe write a new method that can be ran post init to
        #       change the excel_path based on a query of the database?
        if engine is None:
            engine = getEngine()
        if session is None:
            session = getSession()
        
        self.excel_path = excel_path
        self.cover_sheet_df = cover_sheet_df
        self.statistics_df = statistics_df
        self.training_set_df = training_set_df
        self.test_set_df = test_set_df
        self.records_df = records_df
        self.records_field_descriptions_df = records_field_descriptions_df
        self.test_set_predictions_df = test_set_predictions_df
        self.model_descriptors_df = model_descriptors_df
        self.model_descriptor_values_df = model_descriptor_values_df
        self.engine = engine
        self.session = session
        self.model_id = model_id
    

    @staticmethod
    def get_cover_sheet_df(results_dict):
        cover_sheet_df = {
            "Property Name": [results_dict["model_details"].get("propertyName", None)],
            "Property Description": [results_dict["model_details"].get("propertyDescription", None)],
            "Property Units": [results_dict["model_details"].get("unitsModel", None)],
            "Dataset Name": [results_dict["model_details"].get("datasetName", None)],
            "Dataset Description": [results_dict["model_details"].get("datasetDescription", None)],
            "nTraining": [results_dict["model_details"].get("numTraining", None)],
            "nTest": [results_dict["model_details"].get("numPrediction", None)],
            "Method Name": [results_dict["model_details"].get("qsar_method", None)],
            "Method Description": [results_dict["model_details"].get("qsar_method_description", None)],
            "Applicability Domain": [results_dict["model_details"].get("applicabilityDomainName", None)],
            "Applicability Domain Cutoff": [results_dict["model_details"].get("applicabilityDomainCutoff", None)] # TODO
        }
        cover_sheet_df = pd.DataFrame.from_dict(cover_sheet_df)
        return cover_sheet_df
    

    def query_cover_sheet_df(self):
        # TODO: Determine how to obtain nTraining and nTest (1, 2)
        # TODO: Determine how to obtain AD data (3, 4, 5, 6)
        sql = text(f"""
        SELECT distinct prop.name as "Property Name", prop.description as "Property Description",
            u.abbreviation_ccd as "Property Units", m.dataset_name as "Dataset Name",
            d.description as "Dataset Description",
            1 as "nTraining", 2 as "nTest", -- replace with sql to get nTraining and nTest
            meth.name as "Method Name", meth.description as "Method Description",
            3 as "Applicability Domain", 4 as "Applicability Domain Cutoff",
            5 as "Applicability Domain Descriptors",
            6 as "Applicability Domain Distance Measure" -- replace with sql to get AD results
        FROM qsar_models.models AS m
            JOIN qsar_datasets.datasets AS d ON d.name = m.dataset_name
            JOIN qsar_models.predictions AS p ON p.fk_model_id = m.id
            JOIN qsar_models.methods AS meth ON meth.id = m.fk_method_id
            JOIN qsar_datasets.properties AS prop ON prop.id = d.fk_property_id
            JOIN qsar_datasets.units AS u ON u.id = d.fk_unit_id
            JOIN qsar_models.ad_methods AS ad ON ad.id = m.fk_ad_method
        WHERE m.id = {self.model_id} -- input model_id here
        """)
        logging.info("Querying database for Cover Sheet")
        summary = pd.read_sql(sql, self.engine)
        logging.info("Finished querying database for Cover Sheet")
        return summary
    

    def cover_sheet(self, writer, cover_sheet=None):
        if cover_sheet is None:
            cover_sheet = self.query_cover_sheet_df() if self.cover_sheet_df is None else self.cover_sheet_df
        
        cover_sheet = cover_sheet.transpose().round(2)
        cover_sheet.reset_index(inplace=True, names=[""])
        cover_sheet.to_excel(writer, sheet_name="Summary", index=False, header=False)

        workbook = writer.book
        worksheet = writer.sheets["Summary"]
        worksheet.freeze_panes(0, 1)
        bold_format = workbook.add_format({"bold": True})

        ModelToExcel.set_column_width(writer, "Summary", cover_sheet, how="full", first_col_format=bold_format)

        return cover_sheet


    @staticmethod
    def get_statistics_df(results_dict):
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


    def query_statistics_df(self):
        # TODO: Determine if RSQ_Training should be PearsonRSQ_Training as is or R2_Training
        #       Values ARE different in the database for these stats on the same models,
        #       even if only slightly
        sql = text(f"""
        select
            1 as "nTraining", 2 as "nTest",
            MAX(case when s.name = 'PearsonRSQ_Training' then ms.statistic_value end) as "RSQ_Training",
            MAX(case when s.name = 'RMSE_Training' then ms.statistic_value end) as "RMSE_Training",
            MAX(case when s.name = 'MAE_Training' then ms.statistic_value end) as "MAE_Training",
            MAX(case when s.name = 'PearsonRSQ_CV_Training' then ms.statistic_value end) as "RSQ_CV_Training",
            MAX(case when s.name = 'RMSE_CV_Training' then ms.statistic_value end) as "RMSE_CV_Training",
            MAX(case when s.name = 'MAE_CV_Training' then ms.statistic_value end) as "MAE_CV_Training",
            MAX(case when s.name = 'PearsonRSQ_Test' then ms.statistic_value end) as "RSQ_Test",
            MAX(case when s.name = 'RMSE_Test' then ms.statistic_value end) as "RMSE_Test",
            MAX(case when s.name = 'MAE_Test' then ms.statistic_value end) as "MAE_Test",
            MAX(case when s.name = 'Q2_Test' then ms.statistic_value end) as "Q2_Test",
            MAX(case when s.name = 'MAE_Test_inside_AD' then ms.statistic_value end) as "MAE_Test_Inside_AD",
            MAX(case when s.name = 'MAE_Test_outside_AD' then ms.statistic_value end) as "MAE_Test_Outside_AD",
            MAX(case when s.name = 'Coverage_Test' then ms.statistic_value end) as "Coverage_Test"
        from
            qsar_models.models as m
        join qsar_models.methods as meth on
            meth.id = m.fk_method_id
        join qsar_models.model_statistics as ms on
            ms.fk_model_id = m.id
        join qsar_models.statistics as s on
            ms.fk_statistic_id = s.id
        where
            m.id = {self.model_id} -- input model_id here
        group by
            m.id;
        """)

        logging.info("Querying database for Statistics")
        statistics = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Statistics")
        return statistics
    

    def statistics(self, writer, statistics=None):
        if statistics is None:
            statistics = self.query_statistics_df() if self.statistics_df is None else self.statistics_df
        
        # statistics.to_excel(writer, sheet_name="Statistics", index=False)

        workbook = writer.book
        worksheet = workbook.add_worksheet("Statistics")
        # worksheet.freeze_panes(1, 0)

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
        worksheet.merge_range("A1:C1", f"Training Set ({statistics.at[0, "nTraining"]})", merge_format_training)
        worksheet.merge_range("D1:F1", f"5-Fold CV ({statistics.at[0, "nTraining"]})", merge_format_cv)
        worksheet.merge_range("A5:C5", f"Test Set ({statistics.at[0, "nTest"]})", merge_format_test)
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
        # ModelToExcel.add_filter(writer, "Statistics", statistics)
        worksheet.insert_image("A8", Path("resources") / "equations.png", {"x_scale": 0.5, "y_scale": 0.5, "x_offset": 10, "y_offset": 2})

        return statistics


    def query_training_set_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Training Set")
        training_set = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Training Set")
        return training_set
    

    def training_set(self, writer, training_set=None):
        # TODO: Write Function
        if training_set is None:
            training_set = self.query_training_set_df() if self.training_set_df is None else self.training_set_df
        
        training_set.to_excel(writer, sheet_name="Training Set", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Training Set"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Training Set", training_set, how="full")
        ModelToExcel.add_filter(writer, "Training Set", training_set)

        return training_set


    def query_test_set_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Test Set")
        test_set = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Test Set")
        return test_set
    

    def test_set(self, writer, test_set=None):
        # TODO: Write Function
        if test_set is None:
            test_set = self.query_test_set_df() if self.test_set_df is None else self.test_set_df
        
        test_set.to_excel(writer, sheet_name="Test Set", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Test Set"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Test Set", test_set, how="full")
        ModelToExcel.add_filter(writer, "Test Set", test_set)

        return test_set


    def query_records_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Records")
        records = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Records")
        return records
    

    def records(self, writer, records=None):
        # TODO: Write Function
        if records is None:
            records = self.query_records_df() if self.records_df is None else self.records_df
        
        records.to_excel(writer, sheet_name="Records", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Records"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Records", records, how="full")
        ModelToExcel.add_filter(writer, "Records", records)

        return records


    def query_records_field_descriptions_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Records Field Descriptions")
        records_field_descriptions = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Records Field Descriptions")
        return records_field_descriptions
    

    def records_field_descriptions(self, writer, records_field_descriptions=None):
        # TODO: Write Function
        if records_field_descriptions is None:
            records_field_descriptions = self.query_records_field_descriptions_df() if self.records_field_descriptions_df is None else self.records_field_descriptions_df
        
        records_field_descriptions.to_excel(writer, sheet_name="Records Field Descriptions", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Records Field Descriptions"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Records Field Descriptions", records_field_descriptions, how="full")
        ModelToExcel.add_filter(writer, "Records Field Descriptions", records_field_descriptions)

        return records_field_descriptions


    def query_test_set_predictions_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Test Set Predictions")
        test_set_predictions = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Test Set Predictions")
        return test_set_predictions
    

    def test_set_predictions(self, writer, test_set_predictions=None):
        # TODO: Write Function
        if test_set_predictions is None:
            test_set_predictions = self.query_test_set_predictions_df() if self.test_set_predictions_df is None else self.test_set_predictions_df
        
        test_set_predictions.to_excel(writer, sheet_name="Test Set Predictions", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Test Set Predictions"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Test Set Predictions", test_set_predictions, how="full")
        ModelToExcel.add_filter(writer, "Test Set Predictions", test_set_predictions)

        return test_set_predictions


    def query_model_descriptors_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Model Descriptors")
        model_descriptors = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Model Descriptors")
        return model_descriptors
    

    def model_descriptors(self, writer, model_descriptors=None):
        # TODO: Write Function
        if model_descriptors is None:
            model_descriptors = self.query_model_descriptors_df() if self.model_descriptors_df is None else self.model_descriptors_df
        
        model_descriptors.to_excel(writer, sheet_name="Model Descriptors", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptors"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Model Descriptors", model_descriptors, how="full")
        ModelToExcel.add_filter(writer, "Model Descriptors", model_descriptors)

        return model_descriptors


    def query_model_descriptor_values_df(self):
        sql = text(f"""
        
        """)

        logging.info("Querying database for Model Descriptor Values")
        model_descriptor_values = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Model Descriptor Values")
        return model_descriptor_values
    

    def model_descriptor_values(self, writer, model_descriptor_values=None):
        # TODO: Write Function
        if model_descriptor_values is None:
            model_descriptor_values = self.query_model_descriptor_values_df() if self.model_descriptor_values_df is None else self.model_descriptor_values_df
        
        model_descriptor_values.to_excel(writer, sheet_name="Model Descriptor Values", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptor Values"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Model Descriptor Values", model_descriptor_values, how="full")
        ModelToExcel.add_filter(writer, "Model Descriptor Values", model_descriptor_values)

        return model_descriptor_values
    
    
    @staticmethod
    def set_column_width(
        writer,
        sheet_name: str,
        df: pd.DataFrame,
        col_width_pad: int=4,
        min_col_width: int=5,
        how: str="header",
        first_col_format = None):
        worksheet = writer.sheets[sheet_name]
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
    def add_filter(writer, sheet_name, df):
        # Get the worksheet object
        ws = writer.sheets[sheet_name]

        # Determine the range (0-indexed for xlsxwriter)
        nrows, ncols = df.shape

        # Add an auto-filter to the full range (header row is 0; data ends at row nrows)
        ws.autofilter(0, 0, nrows, ncols - 1)
    

    def create_excel(
            self,
            chart_size_px: int=520,  # square chart size
            pad_ratio: float=0.02,
            integer_ticks: bool=True,
            yx_offset_rows: int=3,  # empty rows between data and y=x helper points
            col_width_pad: int=4,
            min_col_width: int=5
        ):
        logging.info("Creating detailed Excel...")
        with pd.ExcelWriter(self.excel_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            logging.info("Creating Cover Sheet...")
            self.cover_sheet(writer, self.cover_sheet_df)

            logging.info("Creating Statistics...")
            self.statistics(writer, self.statistics_df)

            # TODO: Write each sheet method below

            # logging.info("Creating Training Set...")
            # df = self.training_set(writer, self.training_set_df)
            # logging.info(f"training_set:\n\t{df.head(2)}")

            # logging.info("Creating Test Set...")
            # df = self.test_set(writer, self.test_set_df)
            # logging.info(f"test_set:\n\t{df.head(2)}")

            # logging.info("Creating Records...")
            # df = self.records(writer, self.records_df)
            # logging.info(f"records:\n\t{df.head(2)}")

            # logging.info("Creating Records Field Descriptions...")
            # df = self.records_field_descriptions(writer, self.records_field_descriptions_df)
            # logging.info(f"records_field_descriptions:\n\t{df.head(2)}")

            # logging.info("Creating Test Set Predictions...")
            # df = self.test_set_predictions(writer, self.test_set_predictions_df)
            # logging.info(f"test_set_predictions:\n\t{df.head(2)}")

            # logging.info("Creating Model Descriptors...")
            # df = self.model_descriptors(writer, self.model_descriptors_df)
            # logging.info(f"model_descriptors:\n\t{df.head(2)}")

            # logging.info("Creating Model Descriptor Values...")
            # df = self.model_descriptor_values(writer, self.model_descriptor_values_df)
            # logging.info(f"model_descriptor_values:\n\t{df.head(2)}")


def main():
    engine = getEngine()
    session = getSession()
    model_id = 1065
    excel_path = "summary.xlsx"
    test = ModelToExcel(engine, session, model_id, excel_path)
    test.create_excel()


if __name__ == "__main__":
    main()
