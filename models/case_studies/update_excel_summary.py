'''
Created on Apr 28, 2026

@author: TMARTI02
'''

import os
import sys
from dotenv import load_dotenv

# Get the absolute path to the .env file relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(script_dir, '../../personal.env')
env_file = os.path.abspath(env_file)  # Resolve .. properly
load_dotenv(env_file)

PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise ValueError("PROJECT_ROOT not found in personal.env file")
sys.path.insert(0, PROJECT_ROOT)

from models.ModelToExcel import ModelDataObjects, ModelToExcel
from models.db_utilities.plot_db import getSession
from models.db_utilities.model_files import upload_or_update_model_file_in_db

def update_excel_summaries():
    
    # username = 'tmarti02'
    username = "weston.murdock"
    # model_ids = [1763]
    # model_ids = [1754, 1756, 1757, 1758, 1763]
    model_ids = [1753]
    
    # upload_to_db = False
    upload_to_db = True
    
    session = getSession()

    for model_id in model_ids:
        file_path = os.path.join(PROJECT_ROOT, "data", "excel_summaries", f"{model_id}_summary.xlsx")
        mdo = ModelDataObjects(model_id=model_id)
        mte = ModelToExcel(mdo, file_path)
        mte.create_excel()
        mte.update_statistics(upload_to_db=upload_to_db, session=session, user_id=username)
    
        with open(file_path, 'rb') as file:
            file_bytes = file.read()
            print(f'model# {model_id}, length of summary={len(file_bytes)}')

        if len(file_bytes) == 0:
            print(f'{model_id}, file has 0 bytes')
            continue
    
        if upload_to_db:
            upload_or_update_model_file_in_db(file_bytes, username, model_id, 2, session)  # inserts or updates model_file if exists in db
        
        
if __name__ == '__main__':
    update_excel_summaries()
