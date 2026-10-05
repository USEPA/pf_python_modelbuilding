'''
Created on Sep 28, 2026

@author: TMARTI02
'''

from model_ws_db_utilities import getSession
from sqlalchemy import text
from sqlalchemy.orm.session import Session
from typing import Any, Dict, List

def determine_analog_performance_continuous_external():

    # dataset_name = "KOC v1 modeling"
    dataset_name = "ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling"
    dataset_name_external = "QSAR_Toolbox_96HR_Fish_LC50_v3b modeling" 
    
    from models.case_studies.run_model_building_db import run_model_embedding_as_knn_external
    session = getSession()
    
    rows = fetch_continuous_model_metrics_by_dataset(session, dataset_name)
    for r in rows:
        stats, embedding = run_model_embedding_as_knn_external(r["id"], dataset_name, dataset_name_external, session)
        # print(r["id"], r["method_name"], r["variables"], r["rmse_test"], r["rmse_cv"], r["rmse_external"],stats["RMSE_Test"])
        print(r["id"], r["method_name"], len(embedding), stats["RMSE_Test"])




def fetch_binary_model_metrics_by_dataset(session: Session, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Run the aggregated RMSE query for a given dataset_name.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Active SQLAlchemy session.
    dataset_name : str
        Dataset name to filter on (m.dataset_name = :dataset_name).

    Returns
    -------
    List[Dict[str, Any]]
        Rows with keys: id, created_at, method_name, variables, rmse_test, rmse_cv, rmse_external.
    """
    sql = text("""
        select
          m.id,
          m.created_at,
          m2.name as method_name,
          length(de.embedding_tsv) - length(replace(de.embedding_tsv, E'\\t', '')) + 1 as variables,
          max(case when s.name = 'BA_Test'          then ms.statistic_value end) as ba_test,
          max(case when s.name = 'BA_CV_Training'  then ms.statistic_value end) as ba_cv,
          max(case when s.name = 'BA_External'      then ms.statistic_value end) as ba_external
        from qsar_models.models m
        join qsar_models.methods m2 on m2.id = m.fk_method_id
        join qsar_models.descriptor_embeddings de on de.id = m.fk_descriptor_embedding_id
        left join qsar_models.model_statistics ms on ms.fk_model_id = m.id
        left join qsar_models."statistics" s on s.id = ms.fk_statistic_id
        where m.dataset_name = :dataset_name
        group by
          m.id,
          m.created_at,
          m2.name,
          length(de.embedding_tsv) - length(replace(de.embedding_tsv, E'\\t', '')) + 1
        order by method_name asc, variables desc
    """)

    result = session.execute(sql, {"dataset_name": dataset_name})
    return result.mappings().all()

def determine_analog_performance_continuous():

    # dataset_name = "KOC v1 modeling"
    dataset_name = "ECOTOX_2024_12_12_96HR_Fish_LC50_v3b modeling"
    
    from models.case_studies.run_model_building_db import run_model_embedding_as_knn
    session = getSession()
    
    rows = fetch_continuous_model_metrics_by_dataset(session, dataset_name)
    for r in rows:
        stats, embedding = run_model_embedding_as_knn(r["id"], dataset_name, session)
        # print(r["id"], r["method_name"], r["variables"], r["rmse_test"], r["rmse_cv"], r["rmse_external"],stats["RMSE_Test"])
        print(r["id"], r["method_name"], len(embedding), stats["RMSE_Test"])


def fetch_continuous_model_metrics_by_dataset(session: Session, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Run the aggregated RMSE query for a given dataset_name.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        Active SQLAlchemy session.
    dataset_name : str
        Dataset name to filter on (m.dataset_name = :dataset_name).

    Returns
    -------
    List[Dict[str, Any]]
        Rows with keys: id, created_at, method_name, variables, rmse_test, rmse_cv, rmse_external.
    """
    sql = text("""
        select
          m.id,
          m.created_at,
          m2.name as method_name,
          length(de.embedding_tsv) - length(replace(de.embedding_tsv, E'\\t', '')) + 1 as variables,
          max(case when s.name = 'RMSE_Test'          then ms.statistic_value end) as rmse_test,
          max(case when s.name = 'RMSE_CV_Training'  then ms.statistic_value end) as rmse_cv,
          max(case when s.name = 'RMSE_External'      then ms.statistic_value end) as rmse_external
        from qsar_models.models m
        join qsar_models.methods m2 on m2.id = m.fk_method_id
        join qsar_models.descriptor_embeddings de on de.id = m.fk_descriptor_embedding_id
        left join qsar_models.model_statistics ms on ms.fk_model_id = m.id
        left join qsar_models."statistics" s on s.id = ms.fk_statistic_id
        where m.dataset_name = :dataset_name
        group by
          m.id,
          m.created_at,
          m2.name,
          length(de.embedding_tsv) - length(replace(de.embedding_tsv, E'\\t', '')) + 1
        order by method_name asc, variables desc
    """)

    result = session.execute(sql, {"dataset_name": dataset_name})
    return result.mappings().all()

def determine_analog_performance_binary():
    from models.case_studies.run_model_building_db import run_model_embedding_as_knn
    session = getSession()
    dataset_name = "exp_prop_RBIODEG_301F v1 modeling"
    # dataset_name = "exp_prop_RBIODEG_RIFM_CHEMREG"
    rows = fetch_binary_model_metrics_by_dataset(session, dataset_name)
    
    print(dataset_name)
    for r in rows:
        stats, embedding = run_model_embedding_as_knn(r["id"], dataset_name, session)
        print(r["id"], r["method_name"], r["variables"], r["ba_test"], r["ba_cv"], r["ba_external"], stats["BA_Test"])
        # print(r["id"], r["method_name"], len(embedding), stats["RMSE_Test"])



if __name__ == '__main__':
    determine_analog_performance_binary()
    # determine_analog_performance_continuous()
    # determine_analog_performance_continuous_external()

