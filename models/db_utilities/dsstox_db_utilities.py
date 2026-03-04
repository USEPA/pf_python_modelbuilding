'''
Created on Feb 27, 2026

@author: TMARTI02
'''


from util.database_utilities import getSession
import os
from sqlalchemy import text, bindparam
# import traceback

from sqlalchemy.engine import URL
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def getSessionDsstox():
    
    connect_url = URL.create(
        drivername='mysql+pymysql',
        username=os.getenv('DSSTOX_USER'),
        password=os.getenv('DSSTOX_PASS'),
        host=os.getenv('DSSTOX_HOST'),
        port=int(os.getenv('DSSTOX_PORT', '3306')),
        database=os.getenv('DSSTOX_DATABASE'),
        query={'charset': 'utf8mb4'}  # recommended for full Unicode
    )
    
    # print(connect_url)
    engine = create_engine(connect_url, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    return session



def runRandomSample():

    OUTPUT_JSON = "dsstox smiles sample.json"
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        smiles_list = json.load(f)

    
    from model_ws_db_utilities import ModelPredictor
    mp = ModelPredictor()
    
    out_path = "dsstox smiles sample.txt"
    
    start_smiles = "CC(=O)OC1CCN2C1=NC1=C2C(=O)C(C)=C(N2CC2)C1=O"
    
    start = False
    
    with open(out_path, "a", encoding="utf-8") as f:
        for idx, smiles in enumerate(smiles_list):
            
            if smiles == start_smiles:
                start = True
                
            if not start:
                continue
            
            for model_id in range(1065, 1071):
                _, code = mp.predict_model_smiles(model_id, smiles)
                print(idx, model_id, smiles, code, file=f, sep="\t")
                f.flush()




def exportRandomDsstoxSample():
    """Export sample smiles from dsstox"""
    
    sessionDsstox = getSessionDsstox()
    
    sql = text("""SELECT c.smiles
                FROM compounds AS c
                JOIN (SELECT FLOOR(RAND() * (SELECT MAX(id) FROM compounds)) AS start_id) AS r
                WHERE c.id >= r.start_id
                ORDER BY c.id
                LIMIT 1000;       
    """)

    results = sessionDsstox.execute(sql)

    smiles = []
    
    for result in results:
        smiles.append(result[0])
    
    smiles_list = [s for s in smiles if s is not None]
    
    OUTPUT_JSON = "dsstox smiles sample.json"
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(smiles_list, f, ensure_ascii=False, indent=2)


def findMissingDsstoxRecordsInPhyschemModelDatasets():
    """Some datapoints are missing records in qsar_models.dsstox_records table in postgresql due to changes in dsstox over time"""
    
    session = getSession()
    sessionDsstox = getSessionDsstox()
    
    fk_dsstox_snapshot_id = 1

    sql1 = text("""
            SELECT DISTINCT dp.canon_qsar_smiles, split_part(dp.qsar_dtxcid, '|', 1) AS cid 
            from qsar_models.models m                
            join qsar_datasets.datasets d on d.name = m.dataset_name 
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            LEFT JOIN qsar_models.dsstox_records dr ON dr.dtxcid = split_part(dp.qsar_dtxcid, '|', 1) AND dr.fk_dsstox_snapshot_id = :fk_dsstox_snapshot_id
            WHERE d.name LIKE :name_pattern and m.is_public =true AND dr.dtxcid IS NULL;
        """)
    
    results = session.execute(sql1, {"name_pattern": "% v1 modeling", "fk_dsstox_snapshot_id": fk_dsstox_snapshot_id})
    rows = results.mappings().all()  # list of dict-like rows
    cids = [r["cid"] for r in rows]
    
    print(len(cids))         
    print (cids, "\n")        

    # there are 8 dtxcids not in my dsstox_records table
    # DTXCID001783033 doesnt have a generic substance in dsstox 
            
    sql2a = text("""SELECT dsstox_compound_id as cid,  c.smiles, gs.dsstox_substance_id as sid, gs.casrn, gs.preferred_name 
    FROM compounds c
    join generic_substance_compounds gsc on gsc.fk_compound_id =c.id
    join generic_substances gs on gs.id=gsc.fk_generic_substance_id
     WHERE dsstox_compound_id IN :cids;        
    """)

    sql2 = sql2a.bindparams(bindparam("cids", expanding=True))
    res = sessionDsstox.execute(sql2, {"cids": cids})
    # print(list(res.keys()))
    results = res.fetchall()
    
    cid_to_info = {}
    for row in results:
        cid, smiles, sid, casrn, name = row
        cid_to_info[cid] = {"smiles":smiles, "sid":sid, "casrn":casrn, "name": name}
                
    print(json.dumps(cid_to_info))
        
        


if __name__ == '__main__':
    
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    
    
    exportRandomDsstoxSample()
    runRandomSample()

    findMissingDsstoxRecordsInPhyschemModelDatasets()
    # exportRandomDsstoxSample()

