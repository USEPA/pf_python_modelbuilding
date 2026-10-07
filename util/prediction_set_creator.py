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
DESCRIPTORS_API = "https://hcd.rtpnc.epa.gov/api/descriptors"

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



if __name__ == '__main__':
    
    # excel_file_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI.xlsx"
    
    excel_file_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2.xlsx"
    # descriptorService='webtest'
    descriptorService='padel'

    # descriptorServices=["webtest","padel", "rdkit","mordred"]
    # descriptorServices=["rdkit","mordred"]
    descriptorServices=["mordred"]

    for descriptorService in descriptorServices:
    
        output_tsv_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_"+descriptorService+".tsv"
        prediction_tsv_from_excel(excel_file_path, output_tsv_path, sheetName='Original', qsarReadyRuleset='qsar-ready_04242025_0', 
                                         descriptorService=descriptorService, smiles_column='SMILES', exp_column='Passes 60%?', use_qsar_smiles=True)
        
        
        output_tsv_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2_RBIODEG_10_day_"+descriptorService+".tsv"
        prediction_tsv_from_excel(excel_file_path, output_tsv_path, sheetName='Original', qsarReadyRuleset='qsar-ready_04242025_0', 
                                         descriptorService=descriptorService, smiles_column='SMILES', exp_column='Meets 10-day window criteria?', use_qsar_smiles=True)


    # input_excel_path = r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2.xlsx"
    # output_excel_path= r"C:\Users\tmarti02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\ghs-data-gathering\data\experimental\RIFM_2026_08_12\excel files\SMILES_OECD 301F_RASD_NON CBI_v2_inchikey_from_smiles.xlsx"
    # addinchiKeyColumn(input_excel_path, output_excel_path)

