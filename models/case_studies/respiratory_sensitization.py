'''
Created on Jan 29, 2026

@author: TMARTI02
'''

import pandas as pd
import os
import json

# serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
serverAPIs = "https://cim-dev.sciencedataexperts.com/"

from model_ws_db_utilities import ModelPredictor


def standardize_smiles_csv():
    
    folder = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\Comptox\000 scientists\keith salazar"
    file_name="Class_RI.csv"
    qsarReadyRuleSet="qsar-ready_04242025_0"
    omitSalts=False
    
    file_path = os.path.join(folder,file_name)
    
    df = pd.read_csv(file_path)
    print("read ",file_name)

    # print(df.head())
    
    mp=ModelPredictor()
    
    
    chemicals=[]
    
    for row in df.itertuples(index=True):  # index=False to skip index
        
        smiles = row.SMILES
        
        # Standardize smiles:
        chemical, code = mp.standardizeStructure2(serverAPIs, smiles, qsarReadyRuleSet, omitSalts)
        # print(json.dumps(chemical, indent=4))
        if code != 200:
            print(row.Index, row.CID, row.SMILES,"Error")
            continue
        else:
            print(row.Index, row.CID, row.SMILES,"OK")
            # print(json.dumps(chemical))
            chemicals.append(chemical)
            
    with open(os.path.join(folder,"chemicals.json"), "w", encoding="utf-8") as f:
        json.dump(chemicals, f, indent=2, ensure_ascii=False)
    
    
        # if code != 200:
        #     error = chemical
        #     chemical = {}
        #     chemical["chemId"] = smiles  # TODO add inchiKey
        #     chemical["smiles"] = smiles
        #     img_base64 = self.smiles_to_base64(chemical["smiles"])
        #
        #     if img_base64: 
        #         chemical["imageSrc"] = f'data:image/png;base64,{img_base64}'
        #     else:
        #         chemical["imageSrc"] = "N/A"
        #
        #     print(smiles,"error generating qsar smiles")
        #     continue
        #
        # if "smiles" in chemical and "cid" not in chemical:
        #     img_base64 = self.smiles_to_base64(chemical["smiles"])
        #     chemical["imageSrc"] = f'data:image/png;base64,{img_base64}'
        # else:
        #     chemical["imageSrc"] = imgURLCid + chemical["cid"]



if __name__ == '__main__':
    standardize_smiles_csv()