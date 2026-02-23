'''
Created on Jan 22, 2026

@author: TMARTI02
'''

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
import logging
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, mean_absolute_error
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


from utils import print_first_row
from models.df_utilities import remove_log_p_descriptors
    
def _load_df(tsv_string):
    """Loads data from TSV/CSV into a pandas dataframe"""
    if "\t" in tsv_string:
        separator = '\t'
    else:
        separator = ','
        
    # print('separator*',separator,'*')
        
    df = pd.read_csv(StringIO(tsv_string), sep=separator, na_values="null")
    return df


def get_training_cv_instances(session, dataset_name, descriptor_set_name):
    folds = {}
    for i in range(1, 6):
        fold_splitting_name = f'RND_REPRESENTATIVE_CV{i}'
            
        df_training, df_prediction = get_training_prediction_instances(
            session, dataset_name, descriptor_set_name, fold_splitting_name
        )
        folds[i] = {"train": df_training, "pred": df_prediction}
    return folds

def getSqlMappedDataPoints():
        """
        Useful for creating a lookup between canon_qsar_smiles and the dsstox info
        """
        
        return text("""
        WITH filtered_dp AS (
          SELECT
            dp.canon_qsar_smiles,
            dp.qsar_exp_prop_property_values_id,
            dp.qsar_dtxcid,
            TRIM(SPLIT_PART(dp.qsar_exp_prop_property_values_id, '|', 1)) AS qsar_exp_prop_property_values_id_first,
            TRIM(SPLIT_PART(dp.qsar_dtxcid, '|', 1)) AS dtxcid
          FROM qsar_datasets.data_points AS dp
          JOIN qsar_datasets.datasets AS d
            ON dp.fk_dataset_id = d.id
          WHERE d.name = :dataset_name
        )
        SELECT
          fdp.canon_qsar_smiles,
          --fdp.qsar_exp_prop_property_values_id,
          fdp.qsar_exp_prop_property_values_id_first,
          --fdp.qsar_dtxcid,
          fdp.dtxcid,
          r.dtxsid,
          r.casrn,
          r.preferred_name,
          r.smiles,
          r.mol_weight
        FROM filtered_dp AS fdp
        LEFT JOIN qsar_models.dsstox_records AS r
          ON r.dtxcid = fdp.dtxcid
         AND r.fk_dsstox_snapshot_id = 4;
        """)
    
def getMappedDatapoints(session, dataset_name):
    logging.info(f"Getting mapped datapoints for {dataset_name}")
    df_pv = pd.read_sql(getSqlMappedDataPoints(), con=session.get_bind(), params={"dataset_name": dataset_name})
    df_pv = df_pv.replace('', np.nan).dropna(axis=1, how='all')  # drop the columns with no data
    logging.info(f"Done")
    return df_pv



def getSqlPropertyValuesForDataset():
        
        # TODO could get additional mapped identifiers from dsstox by doing a join to qsar_models.dsstox_records

        return text("""
        SELECT
            pv.id AS prop_value_id,
            dp.canon_qsar_smiles,
            sc.source_dtxsid,
            sc.source_casrn,
            sc.source_chemical_name,
            sc.source_smiles,
            --dpc.dtxsid AS mapped_dtxsid,
            --dpc.dtxcid AS mapped_dtxcid,
            --dpc.smiles AS mapped_smiles,
            dr.dtxsid AS mapped_dtxsid, 
            dr.dtxcid AS mapped_dtxcid,
            dr.casrn AS mapped_smiles,
            dr.preferred_name AS mapped_chemical_name, -- get from dsstox records instead of from dpc so that can get the mapped name
            dr.mol_weight as mapped_mol_weight,
            p.name_ccd AS prop_name,
            'experimental' AS prop_type,
            pc.name AS prop_category,
            d."name" AS dataset,
            dpc.property_value AS prop_value,--use the dpc value rather than the pv value fields because it's in the right units
            u.abbreviation_ccd AS prop_unit,
            dp.qsar_property_value,
            u2.abbreviation_ccd AS qsar_property_unit,
            pv.value_original AS prop_value_original,
            pv.value_text AS prop_value_text,
            pvT.value_point_estimate AS exp_details_temperature_c,
            pvP.value_point_estimate AS exp_details_pressure_mmHg,
            pvpH.value_point_estimate AS exp_details_pH, -- note: will convert to lower case
            pvRS.value_text AS exp_details_response_site, -- for BCF, fish tox
            pvSL.value_text AS exp_details_species_latin, -- for BCF, fish tox
            pvSC.value_text AS exp_details_species_common,
            pvSS.value_text AS exp_details_species_supercategory,
            --CASE WHEN ps.name IS NOT NULL THEN ps.name ELSE ls.name END AS source_name,
            --CASE WHEN ps.name IS NOT NULL THEN ps.description ELSE ls.citation END AS source_description,
            --CASE WHEN ps.name IS NOT NULL THEN ps.url ELSE ls.doi END AS source_url,
            ps."name" AS public_source_name,
            ps.description AS public_source_description,
            ps.url AS public_source_url,
            pv.page_url AS direct_url,
            ls."name" AS literature_source_name,
            ls.citation AS literature_source_citation,
            ls.doi AS literature_source_doi,
            pv.document_name AS brief_citation, -- From OPERA2.9 usually
            ps2."name" AS public_source_original_name, -- For sources like toxval, pubchem, sander
            ps2.description AS public_source_original_description,
            ps2.url AS public_source_original_url
        FROM qsar_datasets.data_points AS dp
        JOIN qsar_datasets.data_point_contributors AS dpc
            ON dpc.fk_data_point_id = dp.id
        JOIN exp_prop.property_values AS pv
            ON dpc.exp_prop_property_values_id = pv.id
        LEFT JOIN exp_prop.literature_sources AS ls
            ON pv.fk_literature_source_id = ls.id
        LEFT JOIN exp_prop.public_sources AS ps
            ON pv.fk_public_source_id = ps.id
        LEFT JOIN exp_prop.public_sources AS ps2
            ON pv.fk_public_source_original_id = ps2.id
        LEFT JOIN exp_prop.parameter_values AS pvT
            ON pvT.fk_property_value_id = pv.id AND pvT.fk_parameter_id = 2
        LEFT JOIN exp_prop.parameter_values AS pvP
            ON pvP.fk_property_value_id = pv.id AND pvP.fk_parameter_id = 1
        LEFT JOIN exp_prop.parameter_values AS pvpH
            ON pvpH.fk_property_value_id = pv.id AND pvpH.fk_parameter_id = 3
        LEFT JOIN exp_prop.parameter_values AS pvRS
            ON pvRS.fk_property_value_id = pv.id AND pvRS.fk_parameter_id = 22
        LEFT JOIN exp_prop.parameter_values AS pvSS
            ON pvSS.fk_property_value_id = pv.id AND pvSS.fk_parameter_id = 38
        LEFT JOIN exp_prop.parameter_values AS pvSL
            ON pvSL.fk_property_value_id = pv.id AND pvSL.fk_parameter_id = 21
        LEFT JOIN exp_prop.parameter_values AS pvSC
            ON pvSC.fk_property_value_id = pv.id AND pvSC.fk_parameter_id = 11
        JOIN exp_prop.source_chemicals AS sc
            ON sc.id = pv.fk_source_chemical_id
        JOIN qsar_datasets.datasets AS d
            ON dp.fk_dataset_id = d.id
        JOIN qsar_datasets.properties AS p
            ON d.fk_property_id = p.id
        JOIN qsar_datasets.units AS u
            ON u.id = d.fk_unit_id_contributor
        JOIN qsar_datasets.units AS u2
            ON u2.id = d.fk_unit_id
        LEFT JOIN qsar_datasets.properties_in_categories AS pic
            ON p.id = pic.fk_property_id
        LEFT JOIN qsar_datasets.property_categories AS pc
            ON pic.fk_property_category_id = pc.id
        LEFT JOIN qsar_models.dsstox_records AS dr
          ON dr.dtxcid = dpc.dtxcid
        WHERE d.name = :dataset_name AND dr.fk_dsstox_snapshot_id = 4
          AND keep = TRUE
        """)



def getMappedPropertyValues(session, dataset_name):
    """
    Gets detailed property value records from the database
    """
    
    logging.info(f"Getting mapped property values for {dataset_name}")
    df_pv = pd.read_sql(getSqlPropertyValuesForDataset(), con=session.get_bind(), params={"dataset_name": dataset_name})
    df_pv = df_pv.replace('', np.nan).dropna(axis=1, how='all')  # drop the columns with no data
    df_pv = df_pv.sort_values(["canon_qsar_smiles", "qsar_property_value"]) #sql query could handle the sorting, but easier to handle here
    logging.info(f"Done")
    return df_pv

    
    
def get_dataset_details(session, dataset_name):
    """
    Gets m meta data (except training and test set tsvs).
    TODO Should this info be stored directly in m object and then for new models we won't need to query the db since will be already in the pickled object?
    """
    try:
        
        query = """
        SELECT 
            d.id,
            d.name as dataset_name,
            d.description as dataset_description, 
            u.abbreviation_ccd AS units_model,
            u2.abbreviation_ccd AS units_display,
            d.dsstox_mapping_strategy,
            p.name_ccd as property_name,
            p.description as property_description
        FROM qsar_datasets.datasets AS d
        LEFT JOIN qsar_datasets.units AS u ON d.fk_unit_id = u.id
        LEFT JOIN qsar_datasets.units AS u2 ON d.fk_unit_id_contributor = u2.id
        LEFT JOIN qsar_datasets.properties AS p ON d.fk_property_id = p.id
        """
                    # SQL query to retrieve m details
        sql = text(query + "\nWHERE d.name = :dataset_name")

        # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for m)
        # print(sql)

        # Execute the query
        row = session.execute(sql, {'dataset_name': dataset_name}).fetchone()
        row_dict = dict(row._mapping) if row is not None else None

        # Process the result
        return row_dict

    except Exception as ex:
        ex.with_traceback()
        print(f"Exception occurred: {ex}")



def get_training_ids(session, datasetName, descriptorSetName, splittingName):


    sql = text("""
        SELECT dp.id as fk_data_point_id, dp.qsar_property_value
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp on dp.fk_dataset_id = d.id
        JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
        JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
        JOIN qsar_descriptors.descriptor_sets ds on ds.id = dv.fk_descriptor_set_id
        join qsar_datasets.splittings s on s.id = dpis.fk_splitting_id
        WHERE d.name = :datasetName
        AND ds.name = :descriptorSetName
        AND s.name = :splittingName
        AND dpis.split_num = 0;
        """)

    try:
        
        res = session.execute(sql, {'datasetName': datasetName, 'descriptorSetName': descriptorSetName,
                                        'splittingName': splittingName})
        
        df = pd.DataFrame(res.fetchall(), columns=res.keys())
        return df
    
    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None     


# @timer    
def get_training_prediction_instances(session, datasetName, descriptorSetName, splittingName):


    sql = text("""
        select headers_tsv from qsar_descriptors.descriptor_sets ds
        where ds.name=:descriptorSetName
        """)
    
    try:
        results = session.execute(sql, {'descriptorSetName': descriptorSetName}).fetchone()
        instance_header = "ID\tProperty\t" + results[0] + "\r\n"
        # print(instance_header)
        
        sql = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv, dpis.split_num
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp on dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
            JOIN qsar_datasets.data_points_in_splittings dpis ON dpis.fk_data_point_id = dp.id
            JOIN qsar_descriptors.descriptor_sets ds on ds.id = dv.fk_descriptor_set_id
            join qsar_datasets.splittings s on s.id = dpis.fk_splitting_id
            WHERE d.name = :datasetName
            AND ds.name = :descriptorSetName
            AND s.name = :splittingName
            ORDER BY dp.canon_qsar_smiles;
            """)
        
        sb_training = [instance_header]
        sb_prediction = [instance_header]
    
        results = session.execute(sql, {'datasetName': datasetName, 'descriptorSetName': descriptorSetName,
                                        'splittingName': splittingName})
    
        for row in results:
            chemical_id, qsar_property_value, descriptors, split_num = row
            instance = _generate_instance(chemical_id, qsar_property_value, descriptors)
            
            # print(len(instance.split("\t")))
                        
            if instance is None:
                logging.debug(f"{id}\tnull instance\tdatasetName={datasetName}\tdescriptorSetName={descriptorSetName}")
                continue
        
            if split_num == 0:
                sb_training.append(instance)
        
            elif split_num == 1:
                sb_prediction.append(instance)
                
        df_training = _load_df(''.join(sb_training))
        df_prediction = _load_df(''.join(sb_prediction))
        
        # print(df_training.shape)
        # first_row_dict = df_training.loc[0].to_dict()
        # print(json.dumps(first_row_dict, indent=4))
                
        logging.debug('trainingSet shape' + str(df_training.shape))
        logging.debug('predictionSet shape' + str(df_prediction.shape))
    
        return df_training, df_prediction
    
    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None, None     



from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import logging

def get_instances_excluding(
    session,
    datasetName,          # the dataset to load
    datasetName2,         # exclude any canon_qsar_smiles present in this dataset
    descriptorSetName
):
    """
    Return a single DataFrame of instances for datasetName/descriptorSetName,
    excluding any canon_qsar_smiles that appear in datasetName2.
    """
    sql_headers = text("""
        SELECT headers_tsv
        FROM qsar_descriptors.descriptor_sets ds
        WHERE ds.name = :descriptorSetName
    """)

    try:
        # Build header
        res_header = session.execute(sql_headers, {'descriptorSetName': descriptorSetName}).fetchone()
        instance_header = "ID\tProperty\t" + res_header[0] + "\r\n"

        # Main query without splittings; exclude rows found in datasetName2
        sql_main = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp
              ON dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv
              ON dv.canon_qsar_smiles = dp.canon_qsar_smiles
            JOIN qsar_descriptors.descriptor_sets ds
              ON ds.id = dv.fk_descriptor_set_id
            WHERE d.name = :datasetName
              AND ds.name = :descriptorSetName
              AND NOT EXISTS (
                    SELECT 1
                    FROM qsar_datasets.datasets d2
                    JOIN qsar_datasets.data_points dp2
                      ON dp2.fk_dataset_id = d2.id
                    WHERE d2.name = :datasetName2
                      AND dp2.canon_qsar_smiles = dp.canon_qsar_smiles
              )
            ORDER BY dp.canon_qsar_smiles
        """)

        sb = [instance_header]

        results = session.execute(sql_main, {
            'datasetName': datasetName,
            'descriptorSetName': descriptorSetName,
            'datasetName2': datasetName2
        })

        for row in results:
            chemical_id, qsar_property_value, descriptors = row
            instance = _generate_instance(chemical_id, qsar_property_value, descriptors)

            if instance is None:
                logging.debug(f"{chemical_id}\tnull instance\tdatasetName={datasetName}\tdescriptorSetName={descriptorSetName}")
                continue

            sb.append(instance)

        df = _load_df(''.join(sb))

        logging.debug('instances shape ' + str(df.shape))
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None
    
    

def _generate_instance(chemical_id, qsar_property_value, descriptors):
        return f"{chemical_id}\t{qsar_property_value}\t{descriptors}\r\n"


def get_descriptor_headers(session, descriptorSetName):
    """
    Fetch and parse descriptor headers for the given descriptor set.

    Returns:
      list[str]: descriptor column names in order.

    Raises:
      ValueError if no headers are found or headers_tsv is NULL/empty.
    """
    headers_sql = text("""
        SELECT headers_tsv
        FROM qsar_descriptors.descriptor_sets ds
        WHERE ds.name = :descriptorSetName
    """)
    row = session.execute(headers_sql, {'descriptorSetName': descriptorSetName}).fetchone()
    if not row or row[0] is None:
        msg = f"No headers found for descriptor set '{descriptorSetName}'"
        logging.error(msg)
        raise ValueError(msg)

    headers_tsv = str(row[0]).strip().rstrip('\r\n')
    # Filter out empty header tokens (in case of trailing tabs)
    descriptor_cols = [h for h in headers_tsv.split('\t') if h != '']
    if not descriptor_cols:
        msg = f"Descriptor headers are empty for descriptor set '{descriptorSetName}'"
        logging.error(msg)
        raise ValueError(msg)

    return descriptor_cols


def get_instances(session, datasetName, descriptorSetName):
    """
    Returns a pandas DataFrame with columns:
      ID, Property, <descriptor_1>, <descriptor_2>, ...

    Behavior:
      - Strict check: if any row's descriptor value count does not match the header count,
        raises ValueError (no padding/truncation).

    Returns:
      pd.DataFrame on success

    On SQLAlchemyError:
      returns (None, None)
    """
    try:
        descriptor_cols = get_descriptor_headers(session, descriptorSetName)

        data_sql = text("""
            SELECT dp.canon_qsar_smiles, dp.qsar_property_value, dv.values_tsv
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            JOIN qsar_descriptors.descriptor_values dv ON dp.canon_qsar_smiles = dv.canon_qsar_smiles
            JOIN qsar_descriptors.descriptor_sets ds ON ds.id = dv.fk_descriptor_set_id
            WHERE d.name = :datasetName
              AND ds.name = :descriptorSetName
            ORDER BY dp.canon_qsar_smiles
        """)

        rows = []
        params = {'datasetName': datasetName, 'descriptorSetName': descriptorSetName}
        for chem_id, prop_val, values_tsv in session.execute(data_sql, params):
            vals = [] if values_tsv is None else str(values_tsv).rstrip('\r\n').split('\t')

            expected = len(descriptor_cols)
            got = len(vals)
            if got != expected:
                msg = (
                    f"Descriptor value count mismatch for ID '{chem_id}': "
                    f"expected {expected}, got {got}."
                )
                logging.error(msg)
                raise ValueError(msg)

            rows.append([chem_id, prop_val] + vals)

        columns = ['ID', 'Property'] + descriptor_cols
        df = pd.DataFrame(rows, columns=columns)

        # Convert numeric columns where possible
        to_convert = ['Property'] + descriptor_cols
        df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')

        logging.debug(f"Number of datapoints: {df.shape}")
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None



def getDatapointsLookup(session, datasetName):
    """
    Returns a pandas DataFrame with columns:
      ID, Property, <descriptor_1>, <descriptor_2>, ...

    Behavior:
      - Strict check: if any row's descriptor value count does not match the header count,
        raises ValueError (no padding/truncation).

    Returns:
      pd.DataFrame on success

    On SQLAlchemyError:
      returns (None, None)
    """
    try:

        data_sql = text("""
            SELECT dp.id, dp.canon_qsar_smiles, dp.qsar_property_value
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp ON dp.fk_dataset_id = d.id
            WHERE d.name = :datasetName
            ORDER BY dp.canon_qsar_smiles
        """)

        rows = []
        params = {'datasetName': datasetName}
        
        for fk_data_point_id, qsar_smiles, exp in session.execute(data_sql, params):
            rows.append([fk_data_point_id, qsar_smiles, exp])

        columns = ['fk_data_point_id', 'qsar_smiles',"exp"] 
        df = pd.DataFrame(rows, columns=columns)

        # Convert numeric columns where possible
        # to_convert = ['Property'] + descriptor_cols
        # df[to_convert] = df[to_convert].apply(pd.to_numeric, errors='coerce')

        logging.debug(f"Number of datapoints: {df.shape}")
        return df

    except SQLAlchemyError:
        logging.exception("An exception was thrown!")
        return None


# @deprecated("use make_representative_and_inner_cv_splits")
def find_representative_split(datasetName, descriptorSetName, remove_log_p_descriptors, user, n_threads, n_splits=5, 
                              random_state=0, write_to_db=False):
    
    print('enter find_representative_split')

    from util.database_utilities import getSession
    session = getSession()
    data = get_instances(session, datasetName, descriptorSetName) # all the datapoints no splitting
    # print_first_row(data)

    print(f"Shape of input df:{data.shape}")


    # Columns: col0 = ID, col1 = label/target, remaining = features
    y = data[data.columns[1]].to_numpy()
    X = data.drop(columns=[data.columns[0], data.columns[1]])
    ids = data[data.columns[0]].to_numpy()

    # Optionally remove logP descriptors from the feature set
    if remove_log_p_descriptors:
        X = X.loc[:, ~X.columns.str.contains(r'log[_ ]?p', case=False, regex=True)]

    # Binary classification if labels are strictly in {0, 1}; otherwise regression
    is_binary = pd.Series(y).isin([0, 1]).all()
    print(f"is_binary:{is_binary}")

    # Choose CV strategy
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if is_binary else
        KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )

    # Choose a baseline estimator (replace with your own if needed)
    if is_binary:
        base_model = RandomForestClassifier(
            n_estimators=100, min_impurity_decrease=1e-5,
            random_state=random_state, n_jobs=n_threads, class_weight='balanced'
        )
    else:
        base_model = RandomForestRegressor(
            n_estimators=100, min_impurity_decrease=1e-5,
            random_state=random_state, n_jobs=n_threads
        )

    # Build folds using sklearn and run CV
    folds = list(splitter.split(X, y if is_binary else None))
    scores = []
    y_pred = np.empty_like(y, dtype=np.float64)  # holds out-of-fold predictions

    print(f"Fold\tRMSE\tMAE")
    for i, (train_idx, test_idx) in enumerate(folds):
        model = clone(base_model)
        model.fit(X.iloc[train_idx], y[train_idx])
        preds = model.predict(X.iloc[test_idx])
        y_pred[test_idx] = preds

        if is_binary:
            scores.append(balanced_accuracy_score(y[test_idx], preds))
        else:
            rmse = root_mean_squared_error(y[test_idx], preds)
            mae = mean_absolute_error(y[test_idx], preds)
            scores.append(rmse)
            print(f"{i+1}\t{rmse:.3f}\t{mae:.3f}")

    # Compute pooled score across all out-of-fold predictions
    if is_binary:
        pooled_score = balanced_accuracy_score(y, y_pred)
    else:
        pooled_score = root_mean_squared_error(y, y_pred)

    # Find the fold whose score is closest to pooled score
    scores_arr = np.asarray(scores)
    representative_split = int(np.argmin(np.abs(scores_arr - pooled_score)))
    prediction_idx = folds[representative_split][1]  # test indices for representative split

    metric_name = "balanced accuracy" if is_binary else "RMSE"
    print(f"\npooled {metric_name} over all {n_splits} folds = {pooled_score}")
    print("The split with the most representative score uses fold " +
          str(representative_split + 1) + " for prediction set")

    # Build results aligned to original row order
    # t_p: 0 for training, 1 for prediction
    t_p = np.zeros(len(y), dtype=np.int8)
    t_p[prediction_idx] = 1

    results = pd.DataFrame({
        'qsar_smiles': ids,
        'exp': y,
        'pred': y_pred,
        'split_num': t_p
    })
    
    df_lookup = getDatapointsLookup(session, datasetName)
    # print_first_row(df_lookup)
    # print(df_lookup.shape)

    merged = (
        results[["qsar_smiles", "split_num"]]
        .merge( df_lookup[["qsar_smiles", "fk_data_point_id"]], on="qsar_smiles", how="inner")
        [["fk_data_point_id", "split_num"]]
    )

    # print(merged.to_dict(orient="records"))
    
    created_at = datetime.now()
    
    # Inject uniform constants; updated_at uses the same value as created_at
    merged = merged.assign(
        created_at=created_at,
        updated_at=created_at,
        created_by=user,
        updated_by=user,
        fk_splitting_id=1
    )

    print_first_row(merged)
    print(merged.shape)
    
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None) # make a new session so will commit correctly
        records = merged.replace({np.nan: None}).to_dict(orient="records")
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Count splittings loaded for RND_REPRESENTATIVE splitting: {count}")        


# @deprecated("use make_representative_and_inner_cv_splits")
def find_representative_cv_splits(datasetName, descriptorSetName, user, n_splits=5, 
                              random_state=0, write_to_db=False, shuffle=True):
    
    print('enter find_representative_cv_splits')

    from util.database_utilities import getSession
    session = getSession()
    # data = getDatapoints(session, datasetName, descriptorSetName) # all the datapoints no splitting
    
    import util.predict_constants as pc
    
    splittingName = pc.SPLITTING_RND_REPRESENTATIVE

    df = get_training_ids(session, datasetName, descriptorSetName,splittingName=splittingName)
    
    # print(df.shape)
    # print_first_row(df)
    
    required = {'fk_data_point_id', 'qsar_property_value'}
    if not required.issubset(df.columns):
        raise ValueError("df must contain 'fk_data_point_id' and 'qsar_property_value' columns")

    # Ensure integer fk_data_point_id values
    ids_num = pd.to_numeric(df['fk_data_point_id'], errors='coerce')
    if ids_num.isna().any():
        raise ValueError("Found null/invalid 'fk_data_point_id' values")

    if not np.all(np.isclose(ids_num.values, np.round(ids_num.values))):
        raise ValueError("Non-integer 'fk_data_point_id' values detected (found fractional parts)")

    ids_int = ids_num.astype(np.int64).to_numpy()

    # Target
    y = df['qsar_property_value'].to_numpy()
    n = len(df)

    # Choose CV strategy
    is_binary = pd.Series(y).isin([0, 1]).all()
    if is_binary:
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_iter = splitter.split(np.zeros(n), y)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_iter = splitter.split(np.zeros(n), None)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        split_iter = splitter.split(np.zeros(n), None)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        # Training rows
        for idx in train_idx:
            rows.append({'fk_data_point_id': int(ids_int[idx]), 'fold': fold_idx, 'split_num': 0})
        # Prediction (test) rows
        for idx in test_idx:
            rows.append({'fk_data_point_id': int(ids_int[idx]), 'fold': fold_idx, 'split_num': 1})

    assign_df = pd.DataFrame(rows, columns=['fk_data_point_id', 'fold', 'split_num'])
    assign_df['fk_data_point_id'] = assign_df['fk_data_point_id'].astype('int64')
    assign_df['split_num'] = assign_df['split_num'].astype('int8')
    
    created_at = datetime.now()
    
    merged = (
        assign_df
        .assign(
            created_at=created_at,
            updated_at=created_at,
            created_by=user,
            updated_by=user,
        )
        .assign(fk_splitting_id=lambda d: (d['fold'] + 2).astype('int64')) # add 2 to get the right foreign key from the fold #
        .drop(columns=['fold'])
    )

    print_first_row(merged)
    print(merged.shape)
    
    records = merged.replace({np.nan: None}).to_dict(orient="records")

    # for record in records:
    #     print(record["fk_data_point_id"], record["fk_splitting_id"])
    # print(records)
    
    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None) # make a new session so will commit correctly
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Count splittings loaded for RND_REPRESENTATIVE_CV# splittings: {count}")        


def preview_and_delete_splittings_by_dataset(
    dataset_name,
    split_ids=None,      # e.g., [1] or [2,3,4,5,6]; if None, target all split IDs for the dataset
    head_n=50,           # how many rows to print in preview
    dry_run=True,        # True = preview only; False = perform DELETE
):
    """
    Preview and optionally delete rows in qsar_datasets.data_points_in_splittings for a given dataset,
    using a JOIN on datasets -> data_points -> data_points_in_splittings.

    Arguments:
      - dataset_name: exact name in qsar_datasets.datasets.name
      - split_ids: optional list of fk_splitting_id values to restrict the operation
      - head_n: number of rows to show in preview
      - dry_run: if True, only preview; if False, perform the deletion (irreversible)

    Returns dict with counts and, if deleted, the reported rowcount.
    """
    from sqlalchemy import text, bindparam
    from util.database_utilities import getSession

    session = getSession()

    # Build optional filter for split_ids
    filter_split = ""
    params = {"dataset_name": dataset_name}
    if split_ids is not None and len(split_ids) > 0:
        filter_split = " AND dpis.fk_splitting_id IN :split_ids"
        params["split_ids"] = split_ids
        split_bind = {"split_ids": bindparam("split_ids", expanding=True)}
    else:
        split_bind = {}

    # 1) Count
    count_sql = text(f"""
        SELECT COUNT(dpis.id)
        FROM qsar_datasets.datasets d
        JOIN qsar_datasets.data_points dp
          ON d.id = dp.fk_dataset_id
        JOIN qsar_datasets.data_points_in_splittings dpis
          ON dpis.fk_data_point_id = dp.id
        WHERE d.name = :dataset_name
        {filter_split}
    """).bindparams(**split_bind)

    total = session.execute(count_sql, params).scalar() or 0
    print(f"Total matching rows for dataset '{dataset_name}'"
          f"{' and split_ids=' + str(split_ids) if split_ids else ''}: {total}")

    # 2) Preview rows
    if head_n and head_n > 0:
        preview_sql = text(f"""
            SELECT dpis.id,
                   dpis.fk_data_point_id,
                   dpis.fk_splitting_id,
                   dpis.split_num,
                   dpis.created_at,
                   dpis.created_by
            FROM qsar_datasets.datasets d
            JOIN qsar_datasets.data_points dp
              ON d.id = dp.fk_dataset_id
            JOIN qsar_datasets.data_points_in_splittings dpis
              ON dpis.fk_data_point_id = dp.id
            WHERE d.name = :dataset_name
            {filter_split}
            ORDER BY dpis.fk_splitting_id, dpis.split_num, dpis.fk_data_point_id
            LIMIT :head_n
        """).bindparams(**split_bind)

        for row in session.execute(preview_sql, {**params, "head_n": head_n}):
            print(
                f"id={row.id}, "
                f"fk_data_point_id={row.fk_data_point_id}, "
                f"fk_splitting_id={row.fk_splitting_id}, "
                f"split_num={row.split_num}, "
                f"created_at={row.created_at}, "
                f"created_by={row.created_by}"
            )

    if dry_run:
        print("Dry run: no deletions performed.")
        return {
            "dataset_name": dataset_name,
            "split_ids": split_ids,
            "total_matching": int(total),
            "deleted": 0
        }

    # 3) DELETE with JOIN (PostgreSQL USING clause)
    delete_sql = text(f"""
        DELETE FROM qsar_datasets.data_points_in_splittings dpis
        USING qsar_datasets.data_points dp,
              qsar_datasets.datasets d
        WHERE dpis.fk_data_point_id = dp.id
          AND dp.fk_dataset_id = d.id
          AND d.name = :dataset_name
          {filter_split}
    """).bindparams(**split_bind)

    res = session.execute(delete_sql, params)
    session.commit()

    # rowcount is typically supported on Postgres; fallback to recount if needed
    deleted = res.rowcount if (res.rowcount is not None and res.rowcount >= 0) else None
    if deleted is None:
        # Fallback recount after delete
        remaining = session.execute(count_sql, params).scalar() or 0
        deleted = int(total) - int(remaining)

    print(f"Deleted {deleted} rows for dataset '{dataset_name}'.")
    return {
        "dataset_name": dataset_name,
        "split_ids": split_ids,
        "total_matching": int(total),
        "deleted": int(deleted)
    }

def create_splittings(datasetName, descriptorSetName = 'WebTEST-default'):
    """
    Creates entries in data_points_in_splittings table in database
    """
    
    write_to_db = True
    
    user="tmarti02"
    remove_log_p_descriptors = False
    n_outer_splits = 5
    n_inner_splits = 5
    random_state = 0
    n_threads = 8
    shuffle = True

    # preview_and_delete_splittings_by_dataset(
    #     dataset_name="KOC v1 modeling",
    #     split_ids=None,
    #     head_n=10,
    #     dry_run=False
    # )

    # old way:
    # find_representative_split(datasetName, descriptorSetName, remove_log_p_descriptors, user, n_threads=4, n_splits=5, random_state=42,
    #                           write_to_db=write_to_db)
    #
    # find_representative_cv_splits(datasetName, descriptorSetName, user, n_splits=5, random_state=42,
    #                         write_to_db=write_to_db)

    # all in one:
    make_representative_and_inner_cv_splits(datasetName, descriptorSetName, remove_log_p_descriptors, user, 
                                            n_outer_splits, n_inner_splits, random_state, n_threads, write_to_db, shuffle)

    

    
def make_representative_and_inner_cv_splits(
    datasetName,
    descriptorSetName,
    remove_log_p_descriptors,
    user,
    n_outer_splits=5,
    n_inner_splits=5,
    random_state=0,
    n_threads=None,
    write_to_db=False,
    shuffle=True,
):
    """
    Computes:
      1) A representative outer split (1 fold used as prediction/holdout) chosen by closeness to pooled OOF performance.
      2) Inner CV folds on the remaining training portion.

    Writes both assignments in one go if write_to_db=True.

    Produces rows:
      fk_data_point_id, split_num (0=train, 1=prediction), fk_splitting_id
      - fk_splitting_id == 1 for the representative outer split
      - fk_splitting_id == 2 + inner_fold_idx for the inner CV folds
    """

    # ---- Helpers ----
    def build_splitter(y, n_splits, shuffle, random_state):
        # Binary classification if strictly {0,1}; else regression
        is_binary = pd.Series(y).isin([0, 1]).all()
        if is_binary:
            classes, counts = np.unique(y, return_counts=True)
            # Fallback to KFold if not enough samples per class for StratifiedKFold
            if counts.min() >= n_splits:
                return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), True
            else:
                return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), True
        else:
            return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state), False

    def build_model(is_binary, n_threads, random_state):
        if is_binary:
            return RandomForestClassifier(
                n_estimators=100,
                min_impurity_decrease=1e-5,
                random_state=random_state,
                n_jobs=n_threads,
                class_weight='balanced'
            )
        else:
            return RandomForestRegressor(
                n_estimators=100,
                min_impurity_decrease=1e-5,
                random_state=random_state,
                n_jobs=n_threads
            )

    def rows_for_assignment(idx_array, split_num, fk_splitting_id):
        # Vectorized construction of assignment rows
        return pd.DataFrame(
            {
                "fk_data_point_id": idx_array.astype("int64"),
                "split_num": np.full(len(idx_array), split_num, dtype="int8"),
                "fk_splitting_id": np.full(len(idx_array), fk_splitting_id, dtype="int64"),
            }
        )

    # ---- Load data once ----
    from util.database_utilities import getSession
    session = getSession()

    data = get_instances(session, datasetName, descriptorSetName)
    
    print(f"Input shape = {data.shape}")
    
    # Columns: col0=ID (qsar_smiles), col1=label/target, remaining=features
    y = data[data.columns[1]].to_numpy()
    X = data.drop(columns=[data.columns[0], data.columns[1]])
    ids = data[data.columns[0]].to_numpy()

    if remove_log_p_descriptors:
        X = X.loc[:, ~X.columns.str.contains(r'log[_ ]?p', case=False, regex=True)]

    outer_splitter, is_binary = build_splitter(y, n_outer_splits, shuffle, random_state)
    base_model = build_model(is_binary, n_threads, random_state)

    # ---- Outer CV to pick representative fold ----
    folds = list(outer_splitter.split(X, y if is_binary else None))
    y_oof = np.empty_like(y, dtype=float)
    fold_scores = []

    print("fold\tRMSE\tMAE")

    for i, (tr_idx, te_idx) in enumerate(folds):
        m = clone(base_model)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        preds = m.predict(X.iloc[te_idx])
        y_oof[te_idx] = preds
        if is_binary:
            fold_scores.append(balanced_accuracy_score(y[te_idx], preds))
        else:
            RMSE = root_mean_squared_error(y[te_idx], preds)
            MAE = mean_absolute_error(y[te_idx], preds)
            fold_scores.append(RMSE)
            print(f"{i}\t{RMSE:.3f}\t{MAE:.3f}")

    # Pooled score across all OOF predictions
    pooled = balanced_accuracy_score(y, y_oof) if is_binary else root_mean_squared_error(y, y_oof)

    # Representative fold: closest to pooled
    rep_fold = int(np.argmin(np.abs(np.asarray(fold_scores) - pooled)))
    rep_test_idx = folds[rep_fold][1]
    rep_train_idx = folds[rep_fold][0]
    
    print(f"Best fold:{rep_fold} with RMSE = {fold_scores[rep_fold]:.3f}, pooled value = {pooled:.3f}")
    

    # ---- Map to fk_data_point_id ----
    df_lookup = getDatapointsLookup(session, datasetName)  # must contain qsar_smiles, fk_data_point_id
    map_df = pd.DataFrame({"qsar_smiles": ids})
    map_df = map_df.merge(df_lookup[["qsar_smiles", "fk_data_point_id"]], on="qsar_smiles", how="inner")

    # Build arrays aligned to original order (same as ids/y rows)
    fk_ids_all = map_df["fk_data_point_id"].astype("int64").to_numpy()

    # Outer representative split (fk_splitting_id=1)
    outer_rows = pd.concat(
        [
            rows_for_assignment(fk_ids_all[rep_train_idx], split_num=0, fk_splitting_id=1),
            rows_for_assignment(fk_ids_all[rep_test_idx], split_num=1, fk_splitting_id=1),
        ],
        ignore_index=True,
    )

    # ---- Inner CV on training portion only ----
    if n_inner_splits and n_inner_splits > 1 and len(rep_train_idx) >= n_inner_splits:
        inner_splitter, _ = build_splitter(y[rep_train_idx], n_inner_splits, shuffle, random_state)
        inner_rows = []
        # Each inner fold gets fk_splitting_id = 2 + fold_idx
        for inner_fold_idx, (inner_tr_rel, inner_te_rel) in enumerate(inner_splitter.split(
            np.zeros(len(rep_train_idx)), y[rep_train_idx] if is_binary else None
        )):
            # Convert relative indices (within rep_train_idx) to absolute indices in full dataset
            inner_tr_abs = rep_train_idx[inner_tr_rel]
            inner_te_abs = rep_train_idx[inner_te_rel]
            fk_tr = fk_ids_all[inner_tr_abs]
            fk_te = fk_ids_all[inner_te_abs]
            inner_rows.append(rows_for_assignment(fk_tr, split_num=0, fk_splitting_id=2 + inner_fold_idx))
            inner_rows.append(rows_for_assignment(fk_te, split_num=1, fk_splitting_id=2 + inner_fold_idx))
        inner_rows = pd.concat(inner_rows, ignore_index=True)
    else:
        inner_rows = pd.DataFrame(columns=["fk_data_point_id", "split_num", "fk_splitting_id"])

    # ---- Final assembly and optional write ----
    created_at = datetime.now()
    all_rows = pd.concat([outer_rows, inner_rows], ignore_index=True)
    all_rows = all_rows.assign(
        created_at=created_at,
        updated_at=created_at,
        created_by=user,
        updated_by=user,
    )
    
    import os
    from pathlib import Path
    excel_path = Path(os.getenv("PROJECT_ROOT")) / "data" / "models" / datasetName /"splittings.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows.to_excel(excel_path, index=False)
    print(excel_path)


    if write_to_db:
        from util.database_utilities import DatabaseUtilities
        dbl = DatabaseUtilities(schema="qsar_datasets", session=None)  # independent session for commit
        records = all_rows.replace({np.nan: None}).to_dict(orient="records")
        count = dbl.create_many_chunked(table="data_points_in_splittings", records=records)
        print(f"Wrote {count} rows across representative split and inner CV folds")

    # Return both the outer and inner subsets (useful for testing)
    return {
        "outer_assignments": outer_rows.copy(),
        "inner_assignments": inner_rows.copy(),
        "representative_fold_index": rep_fold,
        "pooled_score": float(pooled),
        "per_fold_scores": list(map(float, fold_scores)),
        "is_binary": bool(is_binary),
    }

if __name__ == '__main__':
    
    # test getting the detailed property data:
    from dotenv import load_dotenv
    load_dotenv()
    
    # dataset_name = 'KOC v1 modeling'
    dataset_name = 'ECOTOX_2024_12_12_96HR_Fish_LC50_v3a modeling'
    create_splittings(dataset_name)


# *************************************************************************************************
    # from util.database_utilities import getSession
    # session = getSession()
    
    # df_pv = getMappedPropertyValues(session, dataset_name)
    #
    # print('\nMapped property values:')
    # for i in range(1, 3):
    #     print_first_row(df_pv, row=i)
    #
    # print('\nMapped datapoints:')
    # df_dps = getMappedDatapoints(session, dataset_name)
    #
    # for i in range(1, 3):
    #     print_first_row(df_dps, row=i)
        
    # descriptorSetName = 'WebTEST-default'
    # df_dps = getDatapoints(session, dataset_name, descriptorSetName)
    # print_first_row(df_dps)
    # print(df_dps.shape)
    
    # df_lookup = getDatapointsLookup(session, dataset_name)
    # print_first_row(df_lookup)
    # print(df_lookup.shape)

    
    # print_first_row(df_dps, row=1)

