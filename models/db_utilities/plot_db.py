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

def upload_image_to_db(file_path, username, fk_model_id, fk_file_type_id, session):

    try:
    # Read the image file as binary
        with open(file_path, 'rb') as file:
            binary_data = file.read()
    
        # Prepare the SQL query
        insert_query = text("""
        INSERT INTO qsar_models.model_files (created_at, created_by, file, updated_at, updated_by, fk_file_type_id, fk_model_id)
        VALUES (:created_at, :created_by, :file, :updated_at, :updated_by, :fk_file_type_id, :fk_model_id)
        """)
    
        # Data to insert
        data = {
            'created_at': datetime.now(),
            'created_by': username,
            'file': binary_data,
            'updated_at': datetime.now(),
            'updated_by': username,
            'fk_file_type_id': fk_file_type_id,  # Example foreign key value
            'fk_model_id': fk_model_id  # Example foreign key value
        }
    
        session.execute(insert_query, data)
        session.commit()
        
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")

def display_image(fk_model_id, fk_file_type_id, session):
    try:
        # Prepare the SQL query to retrieve the image using foreign keys
        select_query = text("""
        SELECT file FROM qsar_models.model_files 
        WHERE fk_model_id = :fk_model_id AND fk_file_type_id = :fk_file_type_id
        """)

        # Execute the query
        result = session.execute(select_query, {'fk_model_id': fk_model_id, 'fk_file_type_id': fk_file_type_id})
        row = result.fetchone()
        
        if row and row[0]:  # Access the first element of the tuple
            # Get the binary data from the result
            binary_data = row[0]
            
            # Define a temporary file path
            temp_file_path = "data/plots/db/" + str(fk_model_id) + "_" + str(fk_file_type_id) + '.png'
            
            # Write the binary data to a temporary file
            with open(temp_file_path, 'wb') as temp_file:
                temp_file.write(binary_data)
            
            # Open the image in the default web browser
            webbrowser.open('file://' + os.path.realpath(temp_file_path))
        else:
            print("No image found with the specified foreign keys.")
    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()

def createTrainingTestPlotsForReports(session):

        username = "tmarti02"
        mi = ModelInitializer()
        
        write_to_db = False

        try:
                        
            PROJECT_ROOT = os.getenv("PROJECT_ROOT")
            # script_dir = os.path.dirname(os.path.abspath(__file__))
            folder_path = os.path.join(PROJECT_ROOT, "data","plots")
            
            
            
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            
            sql = text("select m.id from qsar_models.models m WHERE m.fk_source_id = 3 and m.is_public=true order by m.id;")  # fk_source_id=3 => cheminformatics modules
            results = session.execute(sql).fetchall()
    
            # Process the result
            for row in results:
                model = mi.init_model(row.id)
                print(model.modelId)              

                mpsTraining = model.df_preds_training_cv.to_dict(orient='records')
                mpsTest = model.df_preds_test.to_dict(orient='records')
                
                filePathOutScatter = os.path.join(folder_path, "scatter_plot_" + str(model.modelId) + ".png")
                title = model.modelName + " results for " + model.propertyName

                mtp.generateScatterPlot2(filePathOut=filePathOutScatter, title=title, unitName=model.unitsModel,
                                          mpsTraining=mpsTraining, mpsTest=mpsTest,
                                          seriesNameTrain="Training set (CV)", seriesNameTest="Test set")
                
                print(filePathOutScatter)
                
                if write_to_db:
                    upload_image_to_db(filePathOutScatter, username, model.modelId, 3, session)
                    display_image(model.modelId, 3, session)
                
                filePathOutHistogram = os.path.join(folder_path, "histogram_" + str(model.modelId) + ".png")
                
                mtp.generateHistogram2(fileOutHistogram=filePathOutHistogram, property_name=model.propertyName, unit_name=model.unitsModel,
                                       mpsTraining=mpsTraining, mpsTest=mpsTest,
                                       seriesNameTrain="Training set", seriesNameTest="Test set")
                if write_to_db:
                    upload_image_to_db(filePathOutHistogram, username, model.modelId, 4, session)
                    display_image(model.modelId, 4, session)
                
        except Exception as ex:
            traceback.print_exc()
            print(f"Exception occurred: {ex}")
        
        finally:
            session.close()

if __name__ == '__main__':

    from dotenv import load_dotenv
    load_dotenv('../../personal.env')

    session=getSession()    
    createTrainingTestPlotsForReports(session)
    display_image(1065, 3, session)
    display_image(1065, 4, session)
