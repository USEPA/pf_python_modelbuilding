'''
Created on Feb 27, 2026

@author: TMARTI02
'''

import logging
from util.database_utilities import getSession
import os
from pathlib import Path
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


def _load_missing_dsstox_records_from_resources():
    """
    Load resources/dict_missing_dsstox_records.json (under PROJECT_ROOT)
    into the global dict_missing_dsstox_records.

    - overwrite=True: replace the global dict with the file contents.
    - overwrite=False: shallow-merge file contents into the global dict.

    Returns the loaded/merged dictionary.
    """
    
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        logging.warn("Error: PROJECT_ROOT environment variable is not set.")
        return None

    resources_dir = Path(project_root) / "resources"
    json_path = resources_dir / "dict_missing_dsstox_records.json"

    if not json_path.exists():
        logging.warn(f"Missing file: {json_path}")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logging.warn(f"Failed to read {json_path}: {e}")
        return None

def findMissingDsstoxRecordsInDatasets(dataset_names, fk_dsstox_snapshot_id=1):
    
    session = getSession()
    sessionDsstox = getSessionDsstox()

    sql1 = text("""
            SELECT DISTINCT dp.canon_qsar_smiles, split_part(dp.qsar_dtxcid, '|', 1) AS cid
            FROM qsar_datasets.data_points dp 
            JOIN qsar_datasets.datasets d ON d.id = dp.fk_dataset_id
            LEFT JOIN qsar_models.dsstox_records dr
                   ON dr.dtxcid = split_part(dp.qsar_dtxcid, '|', 1)
                  AND dr.fk_dsstox_snapshot_id = :fk_dsstox_snapshot_id
            WHERE d.name IN :dataset_names AND dr.dtxcid IS NULL;
        """).bindparams(bindparam("dataset_names", expanding=True))
        
    # print (sql1)

    results = session.execute(sql1, {"dataset_names": dataset_names, "fk_dsstox_snapshot_id": fk_dsstox_snapshot_id})
    rows = results.mappings().all()  # list of dict-like rows
    cids = [r["cid"] for r in rows]
    
    print(len(cids))         
    # print (cids, "\n")
    
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
    
        
    sql_hits = text("""
        SELECT DISTINCT c.dsstox_compound_id AS cid
        FROM generic_substance_compounds gsc
        JOIN compounds c ON c.id = gsc.fk_compound_id
        WHERE c.dsstox_compound_id IN :cids
    """).bindparams(bindparam("cids", expanding=True))
    

    hits = set(sessionDsstox.execute(sql_hits, {"cids": cids}).scalars().all())
    missing_cids = [cid for cid in cids if cid not in hits]
    print("cids not in live dsstox", missing_cids)
    
    for missing_cid in missing_cids:
        sql2a = text("""
            SELECT DISTINCT dpc.dtxsid
            FROM qsar_datasets.data_points dp 
            JOIN qsar_datasets.datasets d ON dp.fk_dataset_id = d.id
            JOIN qsar_datasets.data_point_contributors dpc ON dpc.fk_data_point_id = dp.id
            WHERE COALESCE(dp.qsar_dtxcid, '') LIKE :cid_pattern
        """)
        sids = session.execute(sql2a, {"cid_pattern": f"%{missing_cid}%"}).scalars().all()
        print(missing_cid, sids)  # sids is a Python list
        
        # Guard against empty/None SID list
        if sids and sids[0]:
            sql2a = text("""
                SELECT
                    gs.dsstox_substance_id AS sid,
                    gs.casrn,
                    gs.preferred_name AS name,
                    c.smiles
                FROM generic_substances gs
                JOIN generic_substance_compounds gsc on gsc.fk_generic_substance_id = gs.id
                JOIN compounds c on c.id = gsc.fk_compound_id
                WHERE gs.dsstox_substance_id = :sid
            """)
        
            # Use mappings() so we can access by column aliases
            row = sessionDsstox.execute(sql2a, {"sid": sids[0]}).mappings().first()
        
            if row:
                sid  = row["sid"]
                casrn = row["casrn"]
                name = row["name"]
                smiles = row["smiles"]
                print(sid, casrn, name, smiles)
                cid_to_info[missing_cid] = {"sid": sid, "casrn": casrn, "name": name, "smiles":smiles}
            else:
                print(f"No generic_substances row for SID {sids[0]}")
                cid_to_info[missing_cid] = {"sid": sids[0], "casrn": None, "name": None, "smiles": None}
        else:
            print("No SID available for this CID")
            cid_to_info[missing_cid] = {"sid": None, "casrn": None, "name": None}
    
    print(json.dumps(cid_to_info, indent=4))
    
    write_missing_dsstox_records(cid_to_info)
    
    # dict_missing_dsstox_records



# cid_to_info should be a plain dict, e.g. {"DTXCID...": {"sid": "...", "casrn": "...", "name": "..."}}
def write_missing_dsstox_records(cid_to_info):
    project_root = os.getenv("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT environment variable is not set.")
        return False

    resources_dir = Path(project_root) / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)

    out_path = resources_dir / "dict_missing_dsstox_records.json"

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cid_to_info, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Wrote {len(cid_to_info)} entries to {out_path}")
        return True
    except Exception as e:
        print(f"Failed to write JSON to {out_path}: {e}")
        return False

# Example usage:
# write_missing_dsstox_records(cid_to_info)


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
    
    
    # exportRandomDsstoxSample()
    # runRandomSample()

    # findMissingDsstoxRecordsInPhyschemModelDatasets()
    # exportRandomDsstoxSample()
    
    dataset_names = ["HLC v1 modeling", "WS v1 modeling", "VP v1 modeling", "BP v1 modeling", "LogP v1 modeling", "MP v1 modeling",
                     "KOC v1 modeling", "exp_prop_RBIODEG_RIFM_CHEMREG", "exp_prop_RBIODEG_301F v1 modeling", "ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling"
                     ]
    
    findMissingDsstoxRecordsInDatasets(dataset_names, 1)

