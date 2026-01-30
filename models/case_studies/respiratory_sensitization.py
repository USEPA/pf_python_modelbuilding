'''
Created on Jan 29, 2026

@author: TMARTI02
'''

import pandas as pd
import os
import json
import csv 
from utils import print_first_row
from applicability_domain import applicability_domain_utilities as adu
from models import df_utilities as dfu
from predict_constants import PredictConstants as pc

from dominate import document
from dominate.tags import *

# serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
serverAPIs = "https://cim-dev.sciencedataexperts.com/"

from model_ws_db_utilities import ModelPredictor, DescriptorsAPI
from utils import to_json_safe

folder = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\Comptox\000 scientists\keith salazar"

from dataclasses import dataclass, asdict

from typing import Optional, Dict, Any

box_style_true  = "display:inline-block; padding:4px 8px; border:2px solid #2e7d32; border-radius:6px; background:#e8f5e9; color:#2e7d32; font-weight:600;"
box_style_false = "display:inline-block; padding:4px 8px; border:2px solid #c62828; border-radius:6px; background:#ffebee; color:#c62828; font-weight:600;"


mp=ModelPredictor()


def get_neighbors(row, df_train_ids):
    
    neighbors = []
        
    for smiles in row.ids:
        
        train_rows = df_train_ids[df_train_ids['canonicalSmiles'] == smiles]
        if train_rows.empty:
            continue

        r = train_rows.iloc[0]  # r is a Series with scalar values
        
        if r.Property == 1:
            Property = "Active"
        elif r.Property ==0:
            Property = "Inactive"
        else:
            Property="N/A"
        
        neighbor = {"canonicalSmiles": r['canonicalSmiles'],"CID": r['CID'],"Property":Property} 
        neighbors.append(neighbor)        

    return neighbors







def addStructureImage(cid, smiles):
    # Structure Image:
    img_base64 = mp.smiles_to_base64(smiles,800)
    if img_base64:
        img(src=f'data:image/png;base64,{img_base64}', alt=f'CID {cid} | {smiles}')
    else:
        span(f"(render failed) {smiles}")
# CID in the same cell (linked to PubChem)
    br()
    span(str(cid))


def add_ad_test(ad):

    
    ad_val = ad["AD"]

    p(
        # strong(str(ad["adMethod"]["name"]) + ": "),
        strong("AD based on average distance to closest training analogs:"),
        span("True" if ad_val else "False", style=box_style_true if ad_val else box_style_false)
    )    
    
        
    with table():
        with thead():
            with tr():
                for i, _ in enumerate(ad["neighbors"], start=1):
                        th(f"Analog{i}")
        with tr():
            for neighbor in ad["neighbors"]:
                with td():
                    addStructureImage(neighbor["CID"], neighbor["canonicalSmiles"])
                    br()

                    toxval = str(neighbor["Property"])
                    is_active = toxval.strip().lower() == "active"
                    span(
                        toxval,
                        style="font-weight:700; color:#c62828;" if is_active else "font-weight:700; color:#2e7d32;"
                    )
        
def add_ad_frag(ad):
    
    ad_val = ad["AD"]

    p(
        # strong(str(ad["adMethod"]["name"]) + ": "),
        strong("AD based on whether the TEST fragments are within the range for the training set:"),
        span("True" if ad_val else "False", style=box_style_true if ad_val else box_style_false)
    )    
    
    with table():
    
        # Header
        with thead():
            tr(
                th('Fragment'),
                th('Test value'),
                th('Training min'),
                th('Training max'),
                th('Training count')
            )
        # Body
        tb = tbody()
        for row in ad["fragment_table"]:
            # Safely fetch values; format numbers using :g to drop trailing .0
            frag = row.get('fragment', '')
            test_val = row.get('test_value', '')
            tmin = row.get('training_min', '')
            tmax = row.get('training_max', '')
            tcount = row.get('training_count', '')

            with tb:
                tr(
                    td(frag),
                    td(f"{test_val:g}" if isinstance(test_val, (int, float)) else str(test_val)),
                    td(f"{tmin:g}" if isinstance(tmin, (int, float)) else str(tmin)),
                    td(f"{tmax:g}" if isinstance(tmax, (int, float)) else str(tmax)),
                    td(str(tcount))
                )



def build_fragment_table(fragment_table):
    t = table(cls='frag-table')
    with t:
        # Header
        with thead():
            tr(
                th('Fragment'),
                th('Test value'),
                th('Training min'),
                th('Training max'),
                th('Training count')
            )
        # Body
        tb = tbody()
        for row in fragment_table:
            # Safely fetch values; format numbers using :g to drop trailing .0
            frag = row.get('fragment', '')
            test_val = row.get('test_value', '')
            tmin = row.get('training_min', '')
            tmax = row.get('training_max', '')
            tcount = row.get('training_count', '')

            with tb:
                tr(
                    td(frag),
                    td(f"{test_val:g}" if isinstance(test_val, (int, float)) else str(test_val)),
                    td(f"{tmin:g}" if isinstance(tmin, (int, float)) else str(tmin)),
                    td(f"{tmax:g}" if isinstance(tmax, (int, float)) else str(tmax)),
                    td(str(tcount))
                )
    return t
    
    # with table():
    #     with tr():
    #         th("AD")
    #     with tr():
    #         td(ad["AD"])

def create_webpage(results, out_path='results.html', title='Results'):
    # Initialize Indigo
    # Optional renderer settings:
    # renderer.setOption("render-bond-length", 12)
    # renderer.setOption("render-image-size", 300, 200)  # width, height

    doc = document(title=title)
    with doc.head:
        style("""
        body { font-family: sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; }
        thead { background: #f7f7f7; }
        th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }
        img { max-height: 300px; display: block; }
        small { color: #555; }
        """)

    with doc:
        h3(title)

        t = table()
        with t:
            # Single-column header: Structure + CID in the same cell
            
            # result0=results[0]
            
            with thead():
                with tr():
                    th('Chemical')                
                    th('Applicability Domains')
                    
                    # for applicability_domain in result0["applicability_domains"]:
                    #     th(applicability_domain["adMethod"]["name"]) 

            # Body rows
            tbody()
            
            for result in results:
                cid = result["CID"]
                smiles = result["canonicalSmiles"]
                
                ads = result["applicability_domains"]                
                
                with tr():
                    with td():
                        addStructureImage(cid, smiles)
                    
                    with td():
                        with table():
                            for ad in ads:
                                with tr():
                                        with td():
                                            if ad["adMethod"]["name"] == pc.Applicability_Domain_TEST_All_Descriptors_Euclidean:
                                                add_ad_test(ad)
                                            elif ad["adMethod"]["name"] == pc.Applicability_Domain_TEST_Fragment_Counts:
                                                add_ad_frag(ad)                                    

    html = doc.render()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return html




def get_ad_test(canonicalSmiles, df_train_ids, df_ad):
    
    df_ad_rows = df_ad[df_ad['idTest'] == canonicalSmiles]
    df_ad_row=df_ad_rows.iloc[0]

    adResultsTest = {}
    adResultsTest["adMethod"] = {}
    adResultsTest["adMethod"]["name"] = pc.Applicability_Domain_TEST_All_Descriptors_Euclidean
    neighbors = get_neighbors(df_ad_row, df_train_ids)
    adResultsTest["AD"] = bool(df_ad_row.AD)
    adResultsTest["neighbors"] = neighbors
    return adResultsTest

def get_ad_frag(canonicalSmiles, df_ad):
    
    df_ad_rows = df_ad[df_ad['idTest'] == canonicalSmiles]
    df_ad_row=df_ad_rows.iloc[0]
    
    adResultsTest = {}
    adResultsTest["adMethod"] = {}
    adResultsTest["adMethod"]["name"] = pc.Applicability_Domain_TEST_Fragment_Counts
    
    adResultsTest["AD"] = bool(df_ad_row.AD)
    adResultsTest["fragment_table"] = df_ad_row.fragment_table
    return adResultsTest


def determine_ad():
    
    file_name_test = "test_set2.tsv"
    path_test = os.path.join(folder, file_name_test)    
    df_test = pd.read_csv(path_test, delimiter='\t')
    df_test_ids = df_test.iloc[:,:3].copy()
    df_test.drop(df_test.columns[0], axis=1, inplace=True)
    
    file_name_train = "training_set.tsv"
    path_train = os.path.join(folder, file_name_train)    
    df_train = pd.read_csv(path_train, delimiter='\t')
    df_train_ids = df_train.iloc[:,:3].copy()
    df_train.drop(df_train.columns[0], axis=1, inplace=True)

    start = "As [+5 valence, one double bond]"
    stop = "-N=S=O"
    features_frag = dfu.keep_columns_between(df_train, start, stop, True)
    cols_frag = features_frag.columns.tolist()    
                
    df_ad_frag = runAD(df_train, df_test, cols_frag, pc.Applicability_Domain_TEST_Fragment_Counts)    
    df_ad_neighbors = runAD(df_train, df_test, None, pc.Applicability_Domain_TEST_All_Descriptors_Euclidean)
        
    results=[]
    
    for _, row in df_test_ids.iterrows():
        
        result=row.to_dict()
        result["applicability_domains"]=[]
        
        adResultsNeighbors = get_ad_test(row.canonicalSmiles, df_train_ids, df_ad_neighbors)        
        result["applicability_domains"].append(adResultsNeighbors)
        
        adResultsFrag = get_ad_frag(row.canonicalSmiles, df_ad_frag)
        result["applicability_domains"].append(adResultsFrag)
        
        results.append(result)     
        
    # print(json.dumps(results))
    
    print(json.dumps(results,indent=4))
    out_path = os.path.join(folder, "results.html")
    create_webpage(results, out_path,"Applicability Domain Results for Sophorolipids for Respiratory Irritation Model by Chushak et al.(2025)")


def runAD(df_training, df_prediction, embedding, ad_measure):
    
    df_ad_output, _ = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
        train_df=df_training.copy(), test_df=df_prediction.copy(),
        remove_log_p=False,
        embedding=embedding, applicability_domain=ad_measure,
        filterColumnsInBothSets=False,
        returnTrainingAD=False)
    
    print_first_row(df_ad_output)
    
    return df_ad_output 


def create_df():
        
    # file_name_input = "Class_RI.csv"
    # file_name_output = "training_set.tsv"
    
    # file_name_input = "surfactants.csv"
    # file_name_output = "test_set.tsv"

    file_name_input = "surfactants2.csv"
    file_name_output = "test_set2.tsv"
    
    qsarReadyRuleSet = "qsar-ready_04242025_0"
    omitSalts = False
    descriptorAPI = DescriptorsAPI()
    descriptorService = "webtest"
    
    file_path = os.path.join(folder, file_name_input)
        
    df = pd.read_csv(file_path)
    
    # print(df.head())

    print_first_row(df)
    
    mp = ModelPredictor()
        
    out_path = os.path.join(folder, file_name_output)
    
    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t", lineterminator="\n")
        header_written = False
        header_prefix = ["CID", "canonicalSmiles", "Property"]
    
        for row in df.itertuples(index=True):
            smiles = row.SMILES
            chemical, code = mp.standardizeStructure2(serverAPIs, smiles, qsarReadyRuleSet, omitSalts)
            
            if code != 200:
                print(row.Index, row.CID, row.SMILES, "Smiles Error")
                continue
    
            if row.Classifier == "Active":
                chemical["Property"] = 1
            elif row.Classifier == "Inactive":
                chemical["Property"] = 0
            else:
                chemical["Property"] = None
    
            chemical["CID"] = row.CID
            qsarSmiles = chemical.get("canonicalSmiles")
            
            df_prediction, code = descriptorAPI.calculate_descriptors(serverAPIs, qsarSmiles, descriptorService)
            
            if code != 200 or df_prediction is None or df_prediction.empty:
                print(row.Index, row.CID, row.SMILES, "Descriptors Error")
                continue
            
            df_prediction = df_prediction.drop(columns=df_prediction.columns[[0, 1]])  # uses the ones in prefix
    
            if not header_written:
                descriptor_cols = list(df_prediction.columns)
                writer.writerow(header_prefix + descriptor_cols)
                header_written = True
    
            for _, desc_row in df_prediction.iterrows():
                values_prefix = [chemical["CID"], qsarSmiles, chemical["Property"]]
                values_desc = [v.item() if hasattr(v, "item") else v for v in desc_row.tolist()]
                writer.writerow(values_prefix + values_desc)

            # if row.Index == 3:
            #     break
            
    # with open(os.path.join(folder, "chemicals.json"), "w", encoding="utf-8") as f:
    #     json.dump(chemicals, f, indent=2, ensure_ascii=False)
    
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
    # create_df()
    determine_ad()
    
