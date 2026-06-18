'''
Created on Feb 27, 2026

@author: TMARTI02
'''

from util.database_utilities import getSession
import os
from sqlalchemy import text

import webbrowser
from model_ws_db_utilities import ModelInitializer
from models import make_test_plots as mtp 
import traceback
from pathlib import Path

from models.db_utilities.model_files import upload_or_update_model_file_in_db

from dotenv import load_dotenv
load_dotenv('../../personal.env')  # be sure to set PROJECT_ROOT so can write files to hard drive for checking
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

        
def display_image_from_db(fk_model_id, fk_file_type_id, session):
    try:
        # Prepare the SQL query to retrieve the image using foreign keys
        select_query = text("""
        SELECT file FROM qsar_models.model_files 
        WHERE fk_model_id = :fk_model_id AND fk_file_type_id = :fk_file_type_id
        """)

        # <PROJECT_ROOT>/data/plots/db
        folder_path = Path(PROJECT_ROOT) / "data" / "plots" / "db"
        print("folder_path", folder_path)

        # Create the exact folder where the file will be saved (and its parents)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Execute the query
        result = session.execute(select_query, {
            'fk_model_id': fk_model_id,
            'fk_file_type_id': fk_file_type_id
        })
        row = result.fetchone()

        if row and row[0]:
            # Some drivers return memoryview; coerce to bytes
            binary_data = row[0]
            if isinstance(binary_data, memoryview):
                binary_data = binary_data.tobytes()

            # Define file path
            temp_file_path = folder_path / f"{fk_model_id}_{fk_file_type_id}.png"

            # Write the binary data to file
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(binary_data)

            # Open the image in the default web browser
            webbrowser.open('file://' + str(temp_file_path.resolve()))
        else:
            print("No image found with the specified foreign keys.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()


def loadPlotsForModelsForDataset(session, dataset_name, write_to_db=False, write_to_harddisk=True):
    username = "tmarti02"
    mi = ModelInitializer()

    if not PROJECT_ROOT:
        print("set PROJECT_ROOT in env file")
        return

    try:

        if not dataset_name:
            print("dataset_name must be provided")
            return

        folder_path = Path(PROJECT_ROOT) / "data" / "plots"
        print("folder_path", folder_path)

        if write_to_harddisk:
            folder_path.mkdir(parents=True, exist_ok=True)

        # Parameterized query (avoids SQL injection; uses function arg)
        sql = text("""
            SELECT m.id
            FROM qsar_models.models m
            WHERE m.dataset_name = :dataset_name
        """)

        results = session.execute(sql, {"dataset_name": dataset_name}).fetchall()

        for row in results:
            model_id = row[0] if isinstance(row, (tuple, list)) else row.id
            create_and_load_plots(session, write_to_db, write_to_harddisk, username, mi, folder_path, model_id)
            
        # Following works if only need the model ids:
        # for model_id in session.scalars(sql, {"dataset_name": dataset_name}):
        #     create_and_load_plots(session, write_to_db, write_to_harddisk, username, mi, folder_path, model_id)

    except Exception as ex:
        traceback.print_exc()
        print(f"Exception occurred: {ex}")
    finally:
        session.close()


def create_and_load_plots(session, write_to_db, write_to_harddisk, username, mi, folder_path, model_id):
    model = mi.init_model(model_id)
    mpsTraining = model.df_preds_training_cv.to_dict(orient='records')
    mpsTest = model.df_preds_test.to_dict(orient='records')
# Scatter plot path (create folder beforehand)
    if write_to_harddisk:
        filePathOutScatter = str(folder_path / f"scatter_plot_{model.modelId}.png")
    else:
        filePathOutScatter = None
    title = f"{model.modelName} results for {model.propertyName}"
    bytes_scatter = mtp.generateScatterPlot2(
        filePathOut=filePathOutScatter,
        title=title,
        unitName=model.unitsModel,
        mpsTraining=mpsTraining,
        mpsTest=mpsTest,
        seriesNameTrain="Training set (CV)",
        seriesNameTest="Test set")
    
    if write_to_db and bytes_scatter is not None:
        upload_or_update_model_file_in_db(bytes_scatter, username, model.modelId, 3, session)
# Histogram path (same folder)
    if write_to_harddisk:
        filePathOutHistogram = str(folder_path / f"histogram_{model.modelId}.png")
    else:
        filePathOutHistogram = None
    bytes_histogram = mtp.generateHistogram2(
        fileOutHistogram=filePathOutHistogram,
        property_name=model.propertyName,
        unit_name=model.unitsModel,
        mpsTraining=mpsTraining,
        mpsTest=mpsTest,
        seriesNameTrain="Training set",
        seriesNameTest="Test set")
        
    if write_to_db and bytes_histogram is not None:
        upload_or_update_model_file_in_db(bytes_histogram, username, model.modelId, 4, session)
        
    print(model.modelId, len(bytes_scatter), len(bytes_histogram))


def createTrainingTestPlotsForReports(session, write_to_db=False, write_to_harddisk=True):
    """
    Adds reports for models in database
    """
    username = "tmarti02"
    mi = ModelInitializer()
    
    try:
        # <PROJECT_ROOT>/data/plots
        folder_path = Path(PROJECT_ROOT) / "data" / "plots"
        print("folder_path", folder_path)

        # Create the exact folder where files will be saved (and its parents)
        if write_to_harddisk:
            folder_path.mkdir(parents=True, exist_ok=True)

        sql = text("""
            SELECT m.id
            FROM qsar_models.models m
            WHERE m.fk_source_id = 3 AND m.is_public = true
            ORDER BY m.id;
        """)
        results = session.execute(sql).fetchall()

        for row in results:
            # row may be a tuple or Row; adjust as needed
            model_id = row[0] if isinstance(row, (tuple, list)) else row.id
            create_and_load_plots(session, write_to_db, write_to_harddisk, username, mi, folder_path, model_id)

    except Exception as ex:
        traceback.print_exc()
        print(f"Exception occurred: {ex}")
    finally:
        session.close()


if __name__ == '__main__':

    session = getSession()  
      
    createTrainingTestPlotsForReports(session, write_to_db=False, write_to_harddisk=True)
    createTrainingTestPlotsForReports(session, write_to_db=True, write_to_harddisk=True)
    display_image_from_db(1065, 3, session)
    display_image_from_db(1065, 4, session)
    
    # loadPlotsForModelsForDataset(session, dataset_name='ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling', write_to_db=True, write_to_harddisk=True)
    # display_image_from_db(1887, 3, session)
    # display_image_from_db(1887, 4, session)

