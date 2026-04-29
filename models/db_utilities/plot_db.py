'''
Created on Feb 27, 2026

@author: TMARTI02
'''

from util.database_utilities import getSession
import os
from sqlalchemy import text
from datetime import datetime    
import webbrowser
from model_ws_db_utilities import ModelInitializer
from models import make_test_plots as mtp 
import traceback
from pathlib import Path

def upload_or_update_model_file_in_db(file_bytes, username, fk_model_id, fk_file_type_id, session):
    """
    Insert or update a model file blob for (fk_model_id, fk_file_type_id).
    Requires a unique constraint or index on (fk_model_id, fk_file_type_id).
    """
    try:
        now = datetime.now()

        upsert_query = text("""
            INSERT INTO qsar_models.model_files
                (created_at, created_by, file, updated_at, updated_by, fk_file_type_id, fk_model_id)
            VALUES
                (:now, :username, :file, :now, :username, :fk_file_type_id, :fk_model_id)
            ON CONFLICT (fk_model_id, fk_file_type_id)
            DO UPDATE SET
                file       = EXCLUDED.file,
                updated_at = EXCLUDED.updated_at,
                updated_by = EXCLUDED.updated_by
        """)

        params = {
            'now': now,
            'username': username,
            'file': file_bytes,
            'fk_file_type_id': fk_file_type_id,
            'fk_model_id': fk_model_id
        }

        session.execute(upsert_query, params)
        session.commit()

    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
        
def display_image_from_db(fk_model_id, fk_file_type_id, session):
    try:
        # Prepare the SQL query to retrieve the image using foreign keys
        select_query = text("""
        SELECT file FROM qsar_models.model_files 
        WHERE fk_model_id = :fk_model_id AND fk_file_type_id = :fk_file_type_id
        """)

        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        if not PROJECT_ROOT:
            print("set PROJECT_ROOT in env file")
            return

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



def createTrainingTestPlotsForReports(session, write_to_db=False, write_to_harddisk=True):
    """
    Adds reports for models in database
    """
    username = "tmarti02"
    mi = ModelInitializer()
    
    try:
        PROJECT_ROOT = os.getenv("PROJECT_ROOT")
        if PROJECT_ROOT is None:
            print("set PROJECT_ROOT in env file")
            return

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
            model = mi.init_model(model_id)
            print(model.modelId)

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
                seriesNameTest="Test set"
            )

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
                seriesNameTest="Test set"
            )

            if write_to_db and bytes_histogram is not None:
                upload_or_update_model_file_in_db(bytes_histogram, username, model.modelId, 4, session)

    except Exception as ex:
        traceback.print_exc()
        print(f"Exception occurred: {ex}")
    finally:
        session.close()

if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv('../../personal.env') # be sure to set PROJECT_ROOT so can write files to hard drive for checking

    session = getSession()  
      
    # createTrainingTestPlotsForReports(session, write_to_db=False, write_to_harddisk=True)
    createTrainingTestPlotsForReports(session, write_to_db=True, write_to_harddisk=True)

    display_image_from_db(1065, 3, session)
    display_image_from_db(1065, 4, session)
