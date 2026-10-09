'''
Created on Oct 7, 2026

@author: TMARTI02
'''

import os
import json
from io import StringIO
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging
import pandas as pd
from pathlib import Path

from model_ws_utilities import call_do_predictions_from_df
from StatsCalculator import calculate_binary_statistics, calculate_continuous_statistics

from models.case_studies.run_model_building_db import (
    run_dataset,
)

from models.ModelToExcel import ModelDataObjects, ModelToExcel
from model_ws_db_utilities import getSession, ModelInitializer
from openpyxl.reader.excel import load_workbook
from util import predict_constants as pc

PROJECT_ROOT = os.getenv("PROJECT_ROOT")


def _run_test_set(df_external, model, folder_path, df_smiles_subset=None):
    '''
    :param df_external: dataframe for external set
    :param model: the model used to run external set
    :param folder_path: model file folder
    :param df_smiles_subset: smiles that appear in RIFM test set (fragrances)
    '''
    
    json_predictions = call_do_predictions_from_df(df_external, model)
    df_external = pd.read_json(StringIO(json_predictions), orient="records")
    
    # TODO add check for is_binary and generate continuous or binary stats

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    filePathOutTsv = os.path.join(folder_path, "external set predictions.tsv")    
    df_external.to_csv(filePathOutTsv, index=False, sep='\t')
    
    filePathOutJson = os.path.join(folder_path, "external set statistics.json")    
    external_stats["N"] = df_external.shape[0]
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
        
    filePathTestSet = os.path.join(folder_path, "test set predictions.csv")
    df_test = pd.read_csv(filePathTestSet)
    df_test = df_test[["canon_qsar_smiles", "exp", "pred"]].copy()
    df_test = df_test.rename(columns={"canon_qsar_smiles": "id"})
    
    if df_smiles_subset is not None:
        # print(df_smiles_subset)
        # print(df_test)
        df_test = df_test.merge(
            df_smiles_subset[["canon_qsar_smiles"]].drop_duplicates().rename(
                columns={"canon_qsar_smiles": "id"}
            ),
            on="id",
            how="inner"
        )
        
    filePathOutTsv = os.path.join(folder_path, "RIFM test set predictions.tsv")
    df_test.to_csv(filePathOutTsv, sep='\t', index=False)

    test_stats = calculate_binary_statistics(df_test, 0.5, "_Test")
    test_stats["N"] = df_test.shape[0]

    filePathOutJson = os.path.join(folder_path, "RIFM test set statistics.json")
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(test_stats, f, indent=4)

    df_all = pd.concat([df_test, df_external], ignore_index=True)
    # print(df_all)
    
    test_stats_all = calculate_binary_statistics(df_all, 0.5, "_Test")
    
    test_stats_all["N"] = df_all.shape[0]

    filePathOutJson = os.path.join(folder_path, "test and external sets statistics.json")   
    
    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(test_stats_all, f, indent=4)
    
    # print(filePathOutJson) 
    # print("Exists after write:", os.path.exists(filePathOutJson))
    # print("Size:", os.path.getsize(filePathOutJson))
    # folder = os.path.dirname(filePathOutJson)
    # os.startfile(folder)
    
    return external_stats

def _predictSetFromDB_SmilesFromExcel(model, smilesCol, excel_file_path, sheetName):
        """
        Runs whole workflow: standardize, descriptors, prediction, applicability domain using smiles in an excel file
        Stores results in tsv file in same folder as excel file
        Runs one at a time since standardizer and descriptors are slow if not cached in mongo (qsar predictions are fast- could aggregate dataframe to run at the end though)
        :param model_id:
        :param smiles:
        :param mwu:
        :return:
        """

        from API_Utilities import QsarSmilesAPI, DescriptorsAPI
        from model_ws_db_utilities import ModelPredictor
        from model_service_common.config import get_env as _get_env
        STDIZER_API = _get_env('stdizer.url', 'STDIZER_API', default='http://stdizer-api:8200/api/stdizer')
        DESCRIPTORS_API = _get_env('descriptors.url', 'DESCRIPTORS_API', default='http://descriptors-api:8804/api/descriptors')
    
        print(DESCRIPTORS_API)
        descriptorAPI = DescriptorsAPI()

        mp = ModelPredictor()

        # initialize model bytes and all details from db:

        df = pd.read_excel(excel_file_path, sheet_name=sheetName)
        smiles_list = df[smilesCol].tolist()  # Extract the 'Smiles' column into a list

        directory = os.path.dirname(excel_file_path)

        # Create a text file path in the same directory
        text_file_path = os.path.join(directory, "output.txt")
        print(text_file_path)

        with open(text_file_path, 'w') as file:
            file.write("smiles\tqsarSmiles\tpred_value\tpred_AD\n")

            # for smiles, predOld in zip(smiles_list, pred_list):
            for smiles in smiles_list:
                chemical, code = mp.standardizeStructure(STDIZER_API, smiles, model)

                qsarSmiles = chemical["canonicalSmiles"]

                if code != 200:
                    print(smiles, qsarSmiles)
                    file.write(smiles + "\terror smiles")
                    continue

                if model.descriptorSetName == 'WebTEST-default':
                    descriptorSet = 'webtest'
                else:
                    print('couldnt assign descriptorSet for descriptor API')
                    return

                df_prediction, code = descriptorAPI.calculate_descriptors(DESCRIPTORS_API, qsarSmiles, descriptorSet)
                if code != 200:
                    print(smiles, 'error descriptors')
                    file.write(smiles + "\terror descriptors\n")

                    continue

                import model_ws_utilities as mwu

                pred_results = json.loads(mwu.call_do_predictions_from_df(df_prediction, model))
                pred_value = pred_results[0]['pred']

                line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\n"

                # ad_results = mp.determineApplicabilityDomain(model, model.applicabilityDomainName, df_prediction)
                # pred_AD = ad_results["AD"]

                # line = smiles + "\t" + qsarSmiles + "\t" + str(pred_value) + "\t" + str(pred_AD) + "\n"
                print(line)
                file.write(line)
                file.flush()

        return "OK", 200

def _full_test_mte():
    # Need to test local and database models
    # Need to test binary and continuous models
    # Need to test models with and without an external set

    write_to_db = False
    append_to_models_folder = "_mte_testing"
    ad_measure_model = [pc.Applicability_Domain_TEST_Embedding_Euclidean, pc.Applicability_Domain_TEST_Fragment_Counts]

    # LOCAL/BINARY/NO EXTERNAL
    dataset_name = "exp_prop_RBIODEG_301F v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/BINARY/EXTERNAL
    dataset_name = "exp_prop_RBIODEG_RIFM_CHEMREG"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/CONTINUOUS/NO EXTERNAL
    dataset_name = "HLC v1 modeling"
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK

    # LOCAL/CONTINUOUS/EXTERNAL
    dataset_name = "KOC v1 modeling"    
    run_dataset(dataset_name=dataset_name, qsar_method='gcm', feature_selection=False,
                ad_measure_model=ad_measure_model, write_to_db=write_to_db,
                append_to_models_folder=append_to_models_folder)  # OK
    
    # DATABASE/BINARY/NO EXTERNAL
    # Model id no longer exists in database
    # model_id = 1567
    # file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")
    # mdo = ModelDataObjects(model_id=model_id)
    # mte = ModelToExcel(mdo, file_path)
    # mte.create_excel()

    # DATABASE/BINARY/EXTERNAL
    model_id = 1831
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"binary_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/NO EXTERNAL
    model_id = 1065
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()

    # DATABASE/CONTINUOUS/EXTERNAL (external stats not saved in database)
    model_id = 1753
    file_path = os.path.join(PROJECT_ROOT, "data", f"models{append_to_models_folder}", "database_models", f"continuous_no_external.xlsx")  # type: ignore
    mdo = ModelDataObjects(model_id=model_id)
    mte = ModelToExcel(mdo, file_path)
    mte.create_excel()



def _summarize_fragrance_results(dataset_name, append_to_models_folder):
    '''
    Aggregates the fragrance results from the different json files

    :param dataset_name:
    :param append_to_models_folder:
    :return: DataFrame with summary rows
    '''
    folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name

    rows = []

    for item in folder.iterdir():
        if item.is_file():
            continue

        test_stats = None
        external_stats = None
        test_and_external_stats = None

        for json_file in item.rglob("*.json"):
            if json_file.name == "RIFM test set statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    test_stats = json.load(f)

            elif json_file.name == "external set statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    external_stats = json.load(f)

            elif json_file.name == "test and external sets statistics.json":
                with open(json_file, "r", encoding="utf-8") as f:
                    test_and_external_stats = json.load(f)

        n_test = test_stats["N"] if test_stats else None
        ba_test = test_stats["BA_Test"] if test_stats and "BA_Test" in test_stats else None

        n_external = external_stats["N"] if external_stats else None
        ba_external = external_stats["BA_Test"] if external_stats and "BA_Test" in external_stats else None

        n_both = test_and_external_stats["N"] if test_and_external_stats else None
        ba_both = test_and_external_stats["BA_Test"] if test_and_external_stats and "BA_Test" in test_and_external_stats else None

        rows.append({
            "Run": item.name,
            "Ntest": n_test,
            "BA_Test_Fragrances": ba_test,
            "Nexternal": n_external,
            "BA_External_Fragrances": ba_external,
            "Nboth": n_both,
            "BA_Both_Fragrances": ba_both,
        })

    # Sort descending by BA_Both_Fragrances, putting missing values at the end
    rows.sort(
        key=lambda r: (
            r["BA_Both_Fragrances"] is None,
            -(r["BA_Both_Fragrances"] or float("-inf"))
        )
    )

    df = pd.DataFrame(rows)

    # Optional: print to console in a readable way
    print("Run\tNtest\tBA_Test_Fragrances\tNexternal\tBA_External_Fragrances\tN_both\tBA_Both_Fragrances")
    for _, r in df.iterrows():
        ba_test_str = f"{r['BA_Test_Fragrances']:.3f}" if pd.notna(r["BA_Test_Fragrances"]) else "NA"
        ba_external_str = f"{r['BA_External_Fragrances']:.3f}" if pd.notna(r["BA_External_Fragrances"]) else "NA"
        ba_both_str = f"{r['BA_Both_Fragrances']:.3f}" if pd.notna(r["BA_Both_Fragrances"]) else "NA"

        print(
            f"{r['Run']}\t{r['Ntest']}\t{ba_test_str}\t"
            f"{r['Nexternal']}\t{ba_external_str}\t{r['Nboth']}\t{ba_both_str}"
        )

    return df


def _getStatsFromDatasets(endpoint_abbrevs, append_to_models_folder, run, stat, stat_dict):
    '''
    Prints stats for different run folders
    
    :param endpoint_abbrevs:
    :param append_to_models_folder:
    :param run:
    :param stat:
    :param stat_dict:
    '''
    for endpoint_abbrev in endpoint_abbrevs:
        results_path = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / f"TEST_{endpoint_abbrev}" / run / "results.json"
        if results_path.exists():
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
                model_statistics = results.get("model_statistics", {})
                test_val = model_statistics.get(stat_dict).get(stat)
                test_coverage = model_statistics.get(stat_dict).get("Coverage_Test")
                print(f"{endpoint_abbrev}\t{test_val:.3f}\t{test_coverage:.3f}")
        else:
            print(endpoint_abbrev, "missing")



def _get_qsar_smiles_from_fragrance_spreadsheet():
    excel_path = Path(os.getenv("PROJECT_ROOT")) / "data" / "models" / "exp_prop_RBIODEG_301F v1 modeling" / "DSSTox_FRAGRANCEBB_20260413_qsar_ready.xlsx"  # type: ignore
    df = pd.read_excel(excel_path)
    df = df.rename(columns={'Structure_qsar_ready': 'canon_qsar_smiles'})
    unique_df = df[['canon_qsar_smiles']].dropna().drop_duplicates().reset_index(drop=True)
    # print(unique_df)
    return unique_df


def _recalc_stats(session, dataset_name, subset, append_to_models_folder, dataset_to_exclude):
    
    folder = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset

    smiles_to_exclude = _get_smiles_for_training_set(session, dataset_to_exclude)
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())
    
    file_path_rifm = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset / "RIFM test set predictions.tsv"

    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm = df_test_set_rifm[~df_test_set_rifm["id"].isin(exclude_set)]
        # print('df_test_set_rifm',df_test_set_rifm)

    file_path_external = Path(PROJECT_ROOT) / "data" / f"models{append_to_models_folder}" / dataset_name / subset / "external set predictions.tsv"

    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external = df_external[~df_external["id"].isin(exclude_set)]
        # print('df_test_set_external',df_test_set_external)

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    external_stats["N"] = df_external.shape[0]

    filePathOutJson = os.path.join(folder, f"external set statistics exclude {dataset_to_exclude}.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
    
    df_both = pd.concat([df_test_set_rifm, df_external], ignore_index=True).drop_duplicates(subset=["id"])    
    # print(df_both)
    
    both_stats = calculate_binary_statistics(df_both, 0.5, "_Test")
    both_stats["N"] = df_both.shape[0]
    filePathOutJson = os.path.join(folder, f"test and external sets statistics2.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(both_stats, f, indent=4)



def _summarize_fragrance_results_as_excel(session, folder, dataset_to_exclude):
    smiles_to_exclude = _get_smiles_for_training_set(session, dataset_to_exclude) # this may not be needed if we were successful in removing external set compounds before the dataset splitting was created
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())

    frames = []

    file_path_rifm = folder / "RIFM test set predictions.tsv"
    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm["set"] = "test"
        df_test_set_rifm.loc[df_test_set_rifm["id"].isin(exclude_set), "set"] = "training"
        frames.append(df_test_set_rifm)

    file_path_external = folder / "external set predictions.tsv"
    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external["set"] = "external"
        df_external.loc[df_external["id"].isin(exclude_set), "set"] = "training"
        frames.append(df_external)

    if not frames:
        raise FileNotFoundError("No input prediction files were found.")

    df_all = pd.concat(frames, ignore_index=True)

    # Optional: drop duplicates by id, keeping the first occurrence
    # If the same id appears in both files, the first one in concat order wins.
    df_all = df_all.drop_duplicates(subset=["id"], keep="first")

    # Write a single-tab spreadsheet
    output_path = folder / f"fragrance_results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="results", index=False)

    return df_all


def _recalc_stats2(session, folder, dataset_to_exclude):
    """
    Recalculates pooled statistics in test set and external set to exclude chemicals in the training set. This method isnt needed if the external set is excluded from the overall set prior to dataset splitting
    """
    smiles_to_exclude = _get_smiles_for_training_set(session, dataset_to_exclude)
    exclude_set = set(smiles_to_exclude["canon_qsar_smiles"].dropna())
    
    file_path_rifm = folder / "RIFM test set predictions.tsv"

    if file_path_rifm.exists():
        df_test_set_rifm = pd.read_csv(file_path_rifm, sep="\t")
        df_test_set_rifm = df_test_set_rifm[~df_test_set_rifm["id"].isin(exclude_set)]
        # print('df_test_set_rifm',df_test_set_rifm)

    file_path_external = folder / "external set predictions.tsv"

    if file_path_external.exists():
        df_external = pd.read_csv(file_path_external, sep="\t")
        df_external = df_external[~df_external["id"].isin(exclude_set)]
        # print('df_test_set_external',df_test_set_external)

    external_stats = calculate_binary_statistics(df_external, 0.5, "_Test")
    external_stats["N"] = df_external.shape[0]

    filePathOutJson = os.path.join(folder, f"external set statistics exclude {dataset_to_exclude}.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(external_stats, f, indent=4)
    
    df_both = pd.concat([df_test_set_rifm, df_external], ignore_index=True).drop_duplicates(subset=["id"])    
    # print(df_both)
    
    both_stats = calculate_binary_statistics(df_both, 0.5, "_Test")
    both_stats["N"] = df_both.shape[0]
    filePathOutJson = os.path.join(folder, f"test and external sets statistics2.json")    
    with open(filePathOutJson, "w", encoding="utf-8") as f:
        json.dump(both_stats, f, indent=4)        



def _get_model_ids(session, dataset_name: str):
    """
    Return a list of model IDs for the given dataset name.
    
    :param session:
    :param dataset_name:
    """
    sql = text("SELECT id FROM qsar_models.models WHERE dataset_name = :dataset_name")
    try:
        result = session.execute(sql, {"dataset_name": dataset_name})
        # Extract the single selected column as a list
        return result.scalars().all()
    except SQLAlchemyError:
        logging.exception("Failed to fetch model ids for dataset_name=%r", dataset_name)
        return []    

def _lookAtModelCoefficients(model_id):
    '''
    Initializes model and gets the regression coefficients
    :param model_id:
    '''
    
    # model_id = 1847 #GCM RBIODEG ECHA+RIFM
    # model_id = 1843   #REG RBIODEG ECHA+RIFM
    # model_id = 1763 #GCM KOC
    
    # model_id = 1877  #GCM RBIODEG RIFM, redone
    
    mi = ModelInitializer()
    model = mi.initModel(model_id)

    if model is not None:
        y = model.df_training[model.df_training.columns[1]]
        X = model.df_training[model.embedding]

        modelCoefficients = json.loads(model.getOriginalRegressionCoefficients2(X, y))
        
        print(json.dumps(modelCoefficients, indent=4))


def _testCoefficientFromScratch():

    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from models.ModelBuilder import Model
    
    X, y = make_classification(n_samples=500, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        # ("clf", LogisticRegression(max_iter=1000))
        # ("clf", LogisticRegression(penalty="l2", solver="liblinear", random_state=0))
        ("clf", LogisticRegression(solver='liblinear', max_iter=1000, dual=False))        
    ]).fit(df, y)
    
    # pipe = Pipeline([
    #     ("scaler", StandardScaler()),
    #     ("clf", LogisticRegression(random_state=0))  # doesnt give std_errors
    # ]).fit(df, y)
    
    self_like = type("T", (), {})()
    self_like.get_model = lambda: pipe  # type: ignore
    self_like.embedding = list(df.columns)  # type: ignore
    
    print(Model.getOriginalRegressionCoefficients2(self_like, df, y))  # type: ignore
    


def _calculate_fragrance_stats(dataset_name, append_to_models_folder, dataset_name_subset):
    '''
    Calculates stats for the RIFM fragrances and for DSSTox list fragrances from results in test set predictions.csv
    :param dataset_name:
    :param append_to_models_folder:
    :param dataset_name_subset:
    '''
    folder = Path(os.getenv("PROJECT_ROOT")) / "data" / ("models" + append_to_models_folder) / dataset_name
    session = getSession()
    df_smiles_subset = _fetch_set_qsar_smiles(session, dataset_name_subset, 1)
    print('\n\nSubset stats')
    for entry in folder.iterdir():
        if entry.is_dir():
            _calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, entry.name)

def _find_model_folder():
    '''
    Iterate through the different model folders to print the model_ids for the given dataset
    '''
    
    # dataset_name = 'KOC v1 modeling'
    dataset_name = 'exp_prop_RBIODEG_10_day_RIFM_2026_08_12_CHEMREG'
        
    data_folder = Path(PROJECT_ROOT) / "data"
    
    for folder in data_folder.iterdir():
        
        if folder.is_dir() and "models" in folder.name:
            
            koc_folder = folder / dataset_name
            
            if koc_folder.is_dir():
                
                for folder2 in koc_folder.iterdir():
                    
                    if folder2.is_dir():
                        
                        # results_file = folder2 / "results.json"
                        #
                        # if results_file.is_file():
                        #     with results_file.open("r", encoding="utf-8") as f:
                        #         results = json.load(f)
                        #     # print(results)
                        #     params = results["params"]
                        #     model_details = results["model_details"]
                        #     print(folder.name, params['qsar_method'], model_details["modelId"])
                        # else:
                        #     print("results.json not found")
                            
                        results_file = folder2 / "detailed_summary.xlsx"
                        
                        # print(results_file)
                                                
                        if results_file.is_file():
                            wb = load_workbook(results_file, data_only=True)
                            ws = wb["Summary"]
                            print(folder.name, folder2.name, ws["B1"].value)
                            
                        else:
                            # print("detailed_summary.xlsx not found")
                            pass
                        
                print('\n')



def _saveMergedStats(df_stats, df_stats_fragrance, excel_path):
    
    col_width_pad=4
    min_col_width=5

    sheet_name = "Statistics"
    
    df_merged = df_stats.merge(
        df_stats_fragrance,
        on="Run",
        how="left"
    )
    
    df_merged = df_merged.drop(columns=["BA_External", "Metric", "Ntest","Nexternal","Nboth"])
    
    print_first_row(df_merged)
    
    from models.case_studies.run_model_building_db import ExcelCreator
    
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df_merged.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.3f")
    
            ws = writer.sheets[sheet_name]
            nrows, ncols = df_merged.shape
            ws.autofilter(0, 0, nrows, ncols - 1)
            ws.freeze_panes(1, 0)
    
            ExcelCreator.set_column_width(
                writer,
                sheet_name=sheet_name,
                df=df_merged,
                col_width_pad=col_width_pad,
                min_col_width=min_col_width,
                how="full"
            )
            
def _calc_stats_training_cv_fragrances():
    """
    Recalculates the training CV stats for just the fragrances 
    """
    
    model_ids = [1964, 1965, 1966, 1967, 1968, 1976]
    dataset_name_subset = 'exp_prop_RBIODEG_RIFM_2026_08_12_CHEMREG'
    session=getSession()
    df_smiles_training_fragrances = _fetch_set_qsar_smiles(session, dataset_name_subset, 0)
    
    for model_id in model_ids:
        mi=ModelInitializer()
        model=mi.init_model(model_id)
        # Filter model training CV predictions to only those in the subset smiles list
        subset_smiles_training_fragrances = set(df_smiles_training_fragrances["canon_qsar_smiles"].dropna().astype(str))
        
        df_preds_training_cv_fragrances = model.df_preds_training_cv[
            model.df_preds_training_cv["id"].isin(subset_smiles_training_fragrances)
        ].copy()

        stats_training_cv_fragrances = calculate_binary_statistics(df_preds_training_cv_fragrances, 0.5, "_Training_CV")
        print(model_id, stats_training_cv_fragrances["BA_Training_CV"],df_preds_training_cv_fragrances.shape[0])



def _calculate_stats_for_binary_test_set(df_prediction, model, log_path="binary_test_stats.log"):
    """
    Calculates binary statistics for a model that was trained using continuous model (e.g. dependent variable is % biodeg)
    """
    json_predictions = call_do_predictions_from_df(df_prediction, model)
    df_predictions_test = pd.read_json(StringIO(json_predictions), orient="records")
    df_predictions_test["pred"] = (df_predictions_test["pred"] >= 60).astype(int)

    test_stats = calculate_binary_statistics(df_predictions_test, 0.5, "_Test")

    # Compose the same line as printed
    line = f"{model.subfolder}\t{test_stats['BA_Test']:.3f}\n"
    print(line.rstrip("\n"))

    # Ensure directory exists (if a path with directories was provided)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Append to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)

    return test_stats


def _get_smiles_for_training_set(session, dataset_name):
    """
    Gets the qsar smiles in the training set for dataset_name
    """
    sql = text("""
        SELECT dp.canon_qsar_smiles
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp
          ON dp.fk_dataset_id = d.id

        join qsar_datasets.data_points_in_splittings dpis on dpis.fk_data_point_id =dp.id
        WHERE d.name = :dataset_name
        and dpis.split_num =0 and dpis.fk_splitting_id =1;
    """)

    result = session.execute(sql, {"dataset_name": dataset_name})
    rows = result.fetchall()

    return pd.DataFrame(rows, columns=["canon_qsar_smiles"])

def _calculate_stats_for_subset(dataset_name, df_smiles_subset, append_to_models_folder, run_folder):
    """
    Calculates stats for RIFM fragrances and Fragrances in Tony's fragrance list
    
    :param dataset_name:
    :param df_smiles_subset:
    :param append_to_models_folder:
    :param run_folder:
    """    
    
    PROJECT_ROOT = os.getenv("PROJECT_ROOT")
    path_segments = [PROJECT_ROOT, "data", "models" + append_to_models_folder, dataset_name, run_folder, "test set predictions.csv"]
    output_csv_path = os.path.join(*path_segments)
    df_pred = pd.read_csv(output_csv_path)
    
    # TODO: it's not necessary to filter predictions if "RIFM test set predictions.tsv" exists in the folder (already filtered to those chemicals)
    
    df_pred_in_rifm_test = df_pred.merge(
        df_smiles_subset[["canon_qsar_smiles"]].drop_duplicates(),
        on="canon_qsar_smiles",
        how="inner")
    test_stats_RIFM = calculate_binary_statistics(df_pred_in_rifm_test, 0.5, "_Test")
    
    df_fragrances_list = _get_qsar_smiles_from_fragrance_spreadsheet()
    df_fragrances_list_only = df_fragrances_list[~df_fragrances_list['canon_qsar_smiles'].isin(df_smiles_subset['canon_qsar_smiles'])]

    # filter to only fragrances in Tony's list
    df_pred_fragrances = df_pred.merge(
        df_fragrances_list_only[["canon_qsar_smiles"]].drop_duplicates(),
        on="canon_qsar_smiles",
        how="inner")
    
    print(df_pred_fragrances)
    frac_exp_1 = (df_pred_fragrances["exp"] == 1).mean()
    print(f"Fraction of rows where exp = 1: {frac_exp_1:.3f}")
    
    test_stats_other_fragrances = calculate_binary_statistics(df_pred_fragrances, 0.5, "_Test")
    # print('Fragrances test chemical stats', json.dumps(test_stats,indent=4))
    # print(df_pred_fragrances.shape[0])
    # print('')

    # print(json.dumps(test_stats_RIFM,indent=4))    
    print(f"{run_folder}\tRIFM_BA_TEST={test_stats_RIFM['BA_Test']:.3f}\tn={df_pred_in_rifm_test.shape[0]}\tOther_Fragrances_BA_TEST={test_stats_other_fragrances['BA_Test']:.3f}\tn={df_pred_fragrances.shape[0]}")




def _fetch_set_qsar_smiles(session, dataset_name, split_num, descriptor_set_name='WebTEST-default',  outer_splitting_name='RND_REPRESENTATIVE'):
    
    
    """
    Gets the qsar smiles for dataset for the given split_num (0=training, 1=test)
    Only includes rows where the descriptor values are present
    """
    sql = text("""
        SELECT dp.canon_qsar_smiles
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
        JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
        JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
        JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
        JOIN qsar_datasets.splittings s ON s.id = dpis.fk_splitting_id
        WHERE d.name = :datasetName
          AND ds.name = :descriptorSetName
          AND s.name = :outerSplittingName and dpis.split_num=:split_num
        ORDER BY dp.id
    """)
    rows = session.execute(
        sql,
        {
            "datasetName": dataset_name,
            "descriptorSetName": descriptor_set_name,
            "outerSplittingName": outer_splitting_name,
            "split_num": split_num,
        },
    ).fetchall()

    if not rows:
        raise ValueError(
            f"No rows found for dataset='{dataset_name}', descriptorSet='{descriptor_set_name}', "
            f"outerSplitting='{outer_splitting_name}'."
        )

    smiles_list = [r[0] for r in rows]
    return pd.DataFrame({"canon_qsar_smiles": smiles_list})



if __name__ == '__main__':
    pass