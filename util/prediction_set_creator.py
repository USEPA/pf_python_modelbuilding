'''
Created on Aug 21, 2026

@author: TMARTI02
'''
from API_Utilities import DescriptorsAPI,QsarSmilesAPI
import pandas as pd
import os
import logging

from model_ws_db_utilities import ModelInitializer

from model_service_common.config import get_env as _get_env
from dotenv import load_dotenv
load_dotenv('../personal.env')
STDIZER_API = _get_env('stdizer.url', 'STDIZER_API', default='http://stdizer-api:8200/api/stdizer')
DESCRIPTORS_API = _get_env('descriptors.url', 'DESCRIPTORS_API', default='http://descriptors-api:8804/api/descriptors')
# DESCRIPTORS_API = "https://hcd.rtpnc.epa.gov/api/descriptors"

def standardizeStructure(stdizer_api, smiles, qsarReadyRuleSet, omitSalts):
    useFullStandardize = False
    try:
        chemicals, code = QsarSmilesAPI.call_qsar_ready_standardize_post(stdizer_api=stdizer_api, smiles=smiles, full=useFullStandardize,
                                                           workflow=qsarReadyRuleSet)
    except Exception as exc:
        logging.exception("Standardization request failed for %s", smiles)
        return f"{smiles}: standardization request failed: {exc}", 500
    logging.debug(chemicals)
    
    if code >= 400:
        return chemicals, code
            
    if len(chemicals) == 0:
        # logging.debug('Standardization failed')
        return f"{smiles} failed standardization" if smiles else 'No Structure', 400

    if len(chemicals) > 1 and omitSalts:
        # print('qsar smiles indicates mixture')
        return f"{smiles}: model can't run mixtures", 400

    chemical = chemicals[0]
    qsarSmiles = chemical["canonicalSmiles"]
    if not qsarSmiles:
        logging.warning("Standardization returned empty canonicalSmiles for %s", smiles)
        return f"{smiles} failed standardization", 400
    logging.debug(f"qsarSmiles: {qsarSmiles}")
    return chemical, 200


def prediction_tsv_from_excel(
    excel_file_path, output_tsv_path,
    sheetName,
    qsarReadyRuleset,
    descriptorService,
    smiles_column='Smiles',
    exp_column=None,
    use_qsar_smiles=False
):
    """
    Reads SMILES from Excel, standardizes them, calculates descriptors,
    and saves output TSV with columns:
      ID, Property, descriptor columns...
    """

    descriptorAPI = DescriptorsAPI()

    df = pd.read_excel(excel_file_path, sheet_name=sheetName)

    if smiles_column not in df.columns:
        raise KeyError(f"'{smiles_column}' column not found in Excel sheet")

    if exp_column is not None and exp_column not in df.columns:
        raise KeyError(f"'{exp_column}' column not found in Excel sheet")

    if exp_column is not None:
        
        if exp_column == 'Multiclass':
            df[exp_column] = pd.to_numeric(df[exp_column], errors='coerce')
            df = df[df[exp_column].isin([0, 1, 2])].copy()
            df["Property"] = df[exp_column]
        else:
            df = df[df[exp_column].isin(["Y", "N"])].copy()
            df["Property"] = df[exp_column].map({"Y": 1, "N": 0})
    else:
        df["Property"] = None

    id_values = []
    property_values = []
    descriptor_rows = []

    counter = 1

    for _, row in df.iterrows():
        smiles = row[smiles_column]
        property_value = row["Property"]

        try:
            chemical, code = standardizeStructure(
                STDIZER_API,
                smiles,
                qsarReadyRuleSet=qsarReadyRuleset,
                omitSalts=True
            )

            if code != 200 or chemical is None:
                id_values.append(None)
                property_values.append(property_value)
                descriptor_rows.append({})
                continue

            qsarSmiles = chemical["canonicalSmiles"]

            id_values.append(qsarSmiles if use_qsar_smiles else smiles)
            property_values.append(property_value)

            desc_df, code = descriptorAPI.calculate_descriptors(
                DESCRIPTORS_API,
                qsarSmiles,
                descriptorService
            )

            if code != 200 or desc_df is None or desc_df.empty:
                descriptor_rows.append({})
                continue

            desc_row = desc_df.iloc[0].to_dict()
            descriptor_rows.append(desc_row)

            print(counter, smiles, qsarSmiles)
            counter += 1

        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            id_values.append(None)
            property_values.append(property_value)
            descriptor_rows.append({})

    # Base output
    df_out = pd.DataFrame({
        "ID": id_values,
        "Property": property_values
    })

    # Descriptor dataframe
    df_desc = pd.DataFrame(descriptor_rows)

    # Remove any conflicting columns from descriptors
    df_desc = df_desc.drop(columns=["ID", "Property"], errors="ignore")

    # Concatenate
    df_out = pd.concat([df_out.reset_index(drop=True), df_desc.reset_index(drop=True)], axis=1)

    # Remove any duplicated column names just in case
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    # Make sure ID and Property are first
    cols = ["ID", "Property"] + [c for c in df_out.columns if c not in ["ID", "Property"]]
    df_out = df_out[cols]

    # Save TSV
    df_out.to_csv(output_tsv_path, sep="\t", index=False)

    return df_out, 200

def prediction_tsv_from_excel_old(
    excel_file_path, output_tsv_path,
    sheetName,
    qsarReadyRuleset,
    descriptorService,
    smiles_column='Smiles',
    exp_column=None,
    use_qsar_smiles=False
):
    """
    Reads SMILES from Excel, standardizes them, calculates descriptors,
    and saves output TSV with columns:
      ID, Property, descriptor columns...

    If use_qsar_smiles=True, ID will be canonical QSAR SMILES.
    Otherwise, ID will be the original Excel SMILES.
    """

    descriptorAPI = DescriptorsAPI()

    df = pd.read_excel(excel_file_path, sheet_name=sheetName)

    if smiles_column not in df.columns:
        raise KeyError(f"'{smiles_column}' column not found in Excel sheet")

    if exp_column is not None and exp_column not in df.columns:
        raise KeyError(f"'{exp_column}' column not found in Excel sheet")

    # If exp_column is provided, keep only Y/N rows and map to 0/1
    if exp_column is not None:
        df = df[df[exp_column].isin(["Y", "N"])].copy()
        df["Property"] = df[exp_column].map({"Y": 1, "N": 0})
    else:
        df["Property"] = None

    smiles_list = df[smiles_column].tolist()

    # Build base output dataframe
    df_out = df[[smiles_column, "Property"]].copy()
    df_out = df_out.rename(columns={smiles_column: "ID"})

    descriptor_rows = []
    id_values = []

    counter = 1

    for smiles in smiles_list:
        try:
            chemical, code = standardizeStructure(
                STDIZER_API,
                smiles,
                qsarReadyRuleSet=qsarReadyRuleset,
                omitSalts=True
            )

            if code != 200 or chemical is None:
                id_values.append(None)
                descriptor_rows.append({})
                continue

            qsarSmiles = chemical["canonicalSmiles"]

            # choose ID value
            if use_qsar_smiles:
                id_values.append(qsarSmiles)
            else:
                id_values.append(smiles)

            desc_df, code = descriptorAPI.calculate_descriptors(
                DESCRIPTORS_API,
                qsarSmiles,
                descriptorService
            )

            if code != 200 or desc_df is None or desc_df.empty:
                descriptor_rows.append({})
                continue

            desc_row = desc_df.iloc[0].to_dict()
            descriptor_rows.append(desc_row)

            print(counter, smiles, qsarSmiles)
            counter += 1

        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            id_values.append(None)
            descriptor_rows.append({})

    # Overwrite ID column with chosen values
    df_out["ID"] = id_values

    # Add descriptors
    df_desc = pd.DataFrame(descriptor_rows)
    df_out = pd.concat([df_out.reset_index(drop=True), df_desc.reset_index(drop=True)], axis=1)

    # Save TSV
    df_out.to_csv(output_tsv_path, sep="\t", index=False)

    return df_out, 200


import pandas as pd


def addinchiKeyColumn(input_excel_path, output_excel_path):

    # Read Excel file
    df = pd.read_excel(input_excel_path)

    from util.indigo_utils import IndigoUtils 
    
    iu = IndigoUtils()
        
    out_df = pd.DataFrame()
    out_df["SMILES"] = df["SMILES"]
    out_df["InChIKey"] = df["SMILES"].apply(iu.inchi_key_from_smiles)
    
    # Write new Excel file
    out_df.to_excel(output_excel_path, index=False)

def create_RBIODEG_external_set_tsvs():
    
    from pathlib import Path
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    folder = Path(PROJECT_ROOT) / "data" / "datasets" / "RBIODEG external"
    excel_file_path =  folder / "SMILES_OECD 301F_RASD_NON CBI_v2.xlsx"
    
    # descriptorService='webtest'
    descriptorService='padel'

    # descriptorServices=["webtest"]
    descriptorServices=["webtest","padel", "rdkit","mordred"]
    # descriptorServices=["rdkit","mordred"]
    # descriptorServices=["mordred"]
    
    sheetName='Sheet1'

    for descriptorService in descriptorServices:
        
        if descriptorService =='mordred':#for some reason in cim-dev it's down
            DESCRIPTORS_API = 'https://cim.epa.gov/api/descriptors' 
    
        output_tsv_path = folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_{descriptorService}.tsv"
        prediction_tsv_from_excel(excel_file_path, output_tsv_path, sheetName=sheetName, qsarReadyRuleset='qsar-ready_04242025_0', 
                                         descriptorService=descriptorService, smiles_column='SMILES', exp_column='Passes 60%?', use_qsar_smiles=True)
        
        output_tsv_path = folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day_{descriptorService}.tsv"
        prediction_tsv_from_excel(excel_file_path, output_tsv_path, sheetName=sheetName, qsarReadyRuleset='qsar-ready_04242025_0', 
                                         descriptorService=descriptorService, smiles_column='SMILES', exp_column='Meets 10-day window criteria?', use_qsar_smiles=True)

        output_tsv_path = folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_multiclass_{descriptorService}.tsv"
        prediction_tsv_from_excel(excel_file_path, output_tsv_path, sheetName=sheetName, qsarReadyRuleset='qsar-ready_04242025_0', 
                                         descriptorService=descriptorService, smiles_column='SMILES', exp_column='Multiclass', use_qsar_smiles=True)

def descriptor_tsv_from_excel(
    excel_file_path,
    output_tsv_path,
    sheetName,
    qsarReadyRuleset,
    descriptorService,
    smiles_column='Smiles',
    use_qsar_smiles=True
):
    descriptorAPI = DescriptorsAPI()
    df = pd.read_excel(excel_file_path, sheet_name=sheetName)

    if smiles_column not in df.columns:
        raise KeyError(f"'{smiles_column}' column not found in Excel sheet")

    unique_rows = {}  # qsarSmiles -> dict with ID/QSAR_SMILES/descriptors
    counter = 1

    for _, row in df.iterrows():
        smiles = row[smiles_column]

        try:
            chemical, code = standardizeStructure(
                STDIZER_API,
                smiles,
                qsarReadyRuleSet=qsarReadyRuleset,
                omitSalts=True
            )

            if code != 200 or chemical is None:
                continue

            qsarSmiles = chemical["canonicalSmiles"]
            if not qsarSmiles:
                continue

            # Skip already processed standardized structures
            if qsarSmiles in unique_rows:
                continue

            desc_df, code = descriptorAPI.calculate_descriptors(
                DESCRIPTORS_API,
                qsarSmiles,
                descriptorService
            )

            if code != 200 or desc_df is None or desc_df.empty:
                continue

            desc_row = desc_df.iloc[0].to_dict()

            unique_rows[qsarSmiles] = {
                "ID": qsarSmiles if use_qsar_smiles else smiles,
                "QSAR_SMILES": qsarSmiles,
                **desc_row
            }

            print(counter, smiles, qsarSmiles)
            counter += 1

        except Exception as e:
            print(f"Error processing {smiles}: {e}")

    df_out = pd.DataFrame(unique_rows.values())
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]
    df_out.to_csv(output_tsv_path, sep="\t", index=False)

    return df_out, 200


def merge_descriptor_tsv_with_experimental(
    descriptor_tsv_path,
    excel_file_path,
    output_tsv_path,
    sheetName,
    exp_column,
    smiles_column='Smiles',
    qsarReadyRuleset='qsar-ready_04242025_0',
    skip_existing_output_file=True
):
    """
    Merge precomputed descriptors with experimental values.
    Assumes descriptor TSV has columns:
      ID, QSAR_SMILES, descriptor columns...
    """
    
    if os.path.exists(output_tsv_path) and skip_existing_output_file:
        print(f"Skipping merge; output file already exists: {output_tsv_path}")
        return pd.read_csv(output_tsv_path, sep="\t"), 200
    

    df_desc = pd.read_csv(descriptor_tsv_path, sep="\t")
    df_exp = pd.read_excel(excel_file_path, sheet_name=sheetName)

    if smiles_column not in df_exp.columns:
        raise KeyError(f"'{smiles_column}' column not found in Excel sheet")

    if exp_column not in df_exp.columns:
        raise KeyError(f"'{exp_column}' column not found in Excel sheet")

    # Keep only relevant experimental rows and create Property
    if exp_column == 'Multiclass':
        df_exp[exp_column] = pd.to_numeric(df_exp[exp_column], errors='coerce')
        df_exp = df_exp[df_exp[exp_column].isin([0, 1, 2])].copy()
        df_exp["Property"] = df_exp[exp_column].astype(int)
    else:
        df_exp = df_exp[df_exp[exp_column].isin(["Y", "N"])].copy()
        df_exp["Property"] = df_exp[exp_column].map({"Y": 1, "N": 0})

    if "Property" not in df_exp.columns:
        raise KeyError("'Property' column not found in experimental dataframe")

    # Standardize experimental SMILES to QSAR_SMILES
    qsar_smiles_list = []
    for _, row in df_exp.iterrows():
        smiles = row[smiles_column]
        try:
            chemical, code = standardizeStructure(
                STDIZER_API,
                smiles,
                qsarReadyRuleSet=qsarReadyRuleset,
                omitSalts=True
            )
            if code != 200 or chemical is None:
                qsar_smiles_list.append(None)
            else:
                qsar_smiles_list.append(chemical["canonicalSmiles"])
        except Exception:
            logging.exception("Error standardizing SMILES %s", smiles)
            qsar_smiles_list.append(None)

    df_exp["QSAR_SMILES"] = qsar_smiles_list
    df_exp = df_exp[df_exp["QSAR_SMILES"].notna()].copy()

    # IMPORTANT: remove duplicate QSAR_SMILES rows here
    df_exp = df_exp.drop_duplicates(subset=["QSAR_SMILES"]).copy()

    # Build output base
    df_out = df_exp[["QSAR_SMILES", "Property"]].copy()
    df_out = df_out.rename(columns={"QSAR_SMILES": "ID"})

    # Merge descriptors
    df_desc2 = df_desc.drop(columns=["ID"], errors="ignore") # smiles column dropped 
    
    df_out = df_out.merge(
        df_desc2,
        left_on="ID",
        right_on="QSAR_SMILES",
        how="left"
    )

    # Remove merge helper key if present
    df_out = df_out.drop(columns=["QSAR_SMILES"], errors="ignore")

    # Remove duplicated columns if any
    df_out = df_out.loc[:, ~df_out.columns.duplicated()]

    # Put ID/Property first, safely
    preferred_cols = ["ID", "Property"] + [c for c in df_out.columns if c not in ["ID", "Property"]]
    df_out = df_out[[c for c in preferred_cols if c in df_out.columns]]

    # Save TSV
    df_out.to_csv(output_tsv_path, sep="\t", index=False)
    return df_out, 200


def create_RBIODEG_external_set_tsvs_no_recalc():
    from pathlib import Path
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    folder = Path(PROJECT_ROOT) / "data" / "datasets" / "RBIODEG external"
    excel_file_path = folder / "SMILES_OECD 301F_RASD_NON CBI_v2.xlsx"

    descriptorServices = ["webtest", "padel", "rdkit", "mordred"]
    sheetName = "Sheet1"

    # One descriptor TSV per service
    descriptor_tsv_paths = {}
    
    redo_descriptor_file = False

    for descriptorService in descriptorServices:
        if descriptorService == "mordred":
            global DESCRIPTORS_API
            DESCRIPTORS_API = "https://cim.epa.gov/api/descriptors"

        descriptor_tsv_path = folder / f"descriptor_only_{descriptorService}.tsv"
        
        if descriptor_tsv_path.exists() and not redo_descriptor_file:
            print(f"Skipping descriptor calculation for {descriptorService}; file exists: {descriptor_tsv_path}")
        else:
            descriptor_tsv_from_excel(
                excel_file_path=excel_file_path,
                output_tsv_path=descriptor_tsv_path,
                sheetName=sheetName,
                qsarReadyRuleset="qsar-ready_04242025_0",
                descriptorService=descriptorService,
                smiles_column="SMILES",
                use_qsar_smiles=True
            )

        descriptor_tsv_paths[descriptorService] = descriptor_tsv_path

        # Now create endpoint-specific TSVs by merging with experimental values
        merge_descriptor_tsv_with_experimental(
            descriptor_tsv_path=descriptor_tsv_path,
            excel_file_path=excel_file_path,
            output_tsv_path=folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_{descriptorService}.tsv",
            sheetName=sheetName,
            exp_column="Passes 60%?",
            smiles_column="SMILES"
        )

        merge_descriptor_tsv_with_experimental(
            descriptor_tsv_path=descriptor_tsv_path,
            excel_file_path=excel_file_path,
            output_tsv_path=folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day_{descriptorService}.tsv",
            sheetName=sheetName,
            exp_column="Meets 10-day window criteria?",
            smiles_column="SMILES"
        )
        
        merge_descriptor_tsv_with_experimental(
            descriptor_tsv_path=descriptor_tsv_path,
            excel_file_path=excel_file_path,
            output_tsv_path=folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day2_{descriptorService}.tsv",
            sheetName=sheetName,
            exp_column="Meets 10-day window criteria2?",
            smiles_column="SMILES"
        )

        

        merge_descriptor_tsv_with_experimental(
            descriptor_tsv_path=descriptor_tsv_path,
            excel_file_path=excel_file_path,
            output_tsv_path=folder / f"SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_multiclass_{descriptorService}.tsv",
            sheetName=sheetName,
            exp_column="Multiclass",
            smiles_column="SMILES"
        )

if __name__ == '__main__':
    
    # create_RBIODEG_external_set_tsvs()
    create_RBIODEG_external_set_tsvs_no_recalc()
    

    # input_excel_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2.xlsx"
    # output_excel_path= r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2_inchikey_from_smiles.xlsx"
    # addinchiKeyColumn(input_excel_path, output_excel_path)

