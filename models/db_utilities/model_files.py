'''
Created on May 6, 2026

@author: TMARTI02
'''
from datetime import datetime    
from sqlalchemy import text
import os
from util.database_utilities import getSession


def update_tester_page():
    
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    file_path= os.path.join(PROJECT_ROOT, "resources","test_api_882.html")

    model_id=1670
    fk_file_type_id=5
    upload_to_db=True
    username='tmarti02'
    
    
    with open(file_path, 'rb') as file:
        file_bytes = file.read()
        print(f'model# {model_id}, length of summary={len(file_bytes)}')

    if len(file_bytes) == 0:
        print(f'{model_id}, file has 0 bytes')
        return

    if upload_to_db:
        session=getSession()
        upload_or_update_model_file_in_db(file_bytes, username, model_id, fk_file_type_id, session)  # inserts or updates model_file if exists in db


    print(file_path)



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


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    update_tester_page()
