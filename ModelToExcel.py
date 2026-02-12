from model_ws_db_utilities import getEngine, getSession
import pandas as pd
from sqlalchemy import text

import logging
logging.getLogger('sqlalchemy').setLevel(logging.ERROR)

class ModelToExcel:
    def __init__(
            self,
            engine,
            session,
            model_id: int=1065,
            excel_path: str="summary.xlsx"):
        # TODO: Determine correct default for excel_path given model_id
        #       Maybe write a new method that can be ran post init to
        #       change the excel_path based on a query of the database?
        if engine is None:
            engine = getEngine()
        if session is None:
            session = getSession()
        self.engine = engine
        self.session = session
        self.model_id = model_id
        self.excel_path = excel_path
    
    def cover_sheet(self, writer):
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
        WHERE m.id = {self.model_id} -- input model_id here
        """)

        logging.info("Querying database for Cover Sheet")
        summary = pd.read_sql(sql, self.engine)
        logging.info("Finished querying database for Cover Sheet")
        summary = summary.transpose().round(2)
        summary.reset_index(inplace=True, names=[""])
        summary.to_excel(writer, sheet_name="Summary", index=False, header=False)

        workbook = writer.book
        worksheet = writer.sheets["Summary"]
        worksheet.freeze_panes(0, 1)
        bold_format = workbook.add_format({"bold": True})

        ModelToExcel.set_column_width(writer, "Summary", summary, how="full", first_col_format=bold_format)

        return summary

    def statistics(self, writer):
        # TODO: Determine if RSQ_Training should be PearsonRSQ_Training as is or R2_Training
        #       Values ARE different in the database for these stats on the same models,
        #       even if only slightly
        # TODO: Add explanatory image below data
        sql = text(f"""
        select
            m.dataset_name as "Dataset Name",
            m.descriptor_set_name as "Descriptor Set",
            meth.name as "Method Name",
            MAX(case when s.name = 'PearsonRSQ_Training' then ms.statistic_value end) as "RSQ_Training",
            MAX(case when s.name = 'PearsonRSQ_CV_Training' then ms.statistic_value end) as "RSQ_CV_Training",
            MAX(case when s.name = 'PearsonRSQ_Test' then ms.statistic_value end) as "RSQ_Test",
            MAX(case when s.name = 'Q2_Test' then ms.statistic_value end) as "Q2_Test",
            MAX(case when s.name = 'RMSE_Test' then ms.statistic_value end) as "RMSE_Test",
            MAX(case when s.name = 'MAE_CV_Training' then ms.statistic_value end) as "MAE_CV_Training",
            MAX(case when s.name = 'MAE_Test' then ms.statistic_value end) as "MAE_Test",
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
            m.id,
            m.dataset_name,
            m.descriptor_set_name,
            meth.name;
        """)

        logging.info("Querying database for Statistics")
        statistics = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Statistics")
        statistics.to_excel(writer, sheet_name="Statistics", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Statistics"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Statistics", statistics, how="full")
        ModelToExcel.add_filter(writer, "Statistics", statistics)

        return statistics

    def training_set(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2

    def test_set(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2

    def records(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2

    def records_field_descriptions(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2

    def test_set_predictions(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2

    def model_descriptors(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for Model Descriptors")
        model_descriptors = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for Model Descriptors")
        model_descriptors.to_excel(writer, sheet_name="Model Descriptors", index=False)

        workbook = writer.book
        worksheet = writer.sheets["Model Descriptors"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "Model Descriptors", model_descriptors, how="full")
        ModelToExcel.add_filter(writer, "Model Descriptors", model_descriptors)

        return model_descriptors

    def model_descriptor_values(self, writer):
        # TODO: Write Function
        sql = text(f"""
        
        """)

        logging.info("Querying database for _1")
        _2 = pd.read_sql(sql, self.engine).round(2)
        logging.info("Finished querying database for _1")
        _2.to_excel(writer, sheet_name="_1", index=False)

        workbook = writer.book
        worksheet = writer.sheets["_1"]
        worksheet.freeze_panes(1, 0)

        ModelToExcel.set_column_width(writer, "_1", _2, how="full")
        ModelToExcel.add_filter(writer, "_1", _2)

        return _2
    
    
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
        with pd.ExcelWriter(self.excel_path, engine="xlsxwriter") as writer:
            workbook = writer.book

            self.cover_sheet(writer)
            self.statistics(writer)
            self.training_set(writer)
            self.test_set(writer)
            self.records(writer)
            self.records_field_descriptions(writer)
            self.test_set_predictions(writer)
            self.model_descriptors(writer)
            self.model_descriptor_values(writer)
