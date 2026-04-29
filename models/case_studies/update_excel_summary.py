'''
Created on Apr 28, 2026

@author: TMARTI02
'''

import os
from dotenv import load_dotenv
load_dotenv('../../personal.env')
PROJECT_ROOT = os.getenv("PROJECT_ROOT")

from models.ModelToExcel import ModelDataObjects, ModelToExcel
from models.db_utilities.plot_db import upload_or_update_model_file_in_db, getSession


def update_excel_summaries():
    
    username = 'tmarti02'
    # model_ids = [1763]
    model_ids = [1754, 1756, 1757, 1758, 1763]
    
    # upload_to_db = False
    upload_to_db = True
    
    session = getSession()

    for model_id in model_ids:
        file_path = os.path.join(PROJECT_ROOT, "data", "excel_summaries", f"{model_id}_summary.xlsx")
        mdo = ModelDataObjects(model_id=model_id)
        mte = ModelToExcel(mdo, file_path)
        mte.create_excel()
    
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
