'''
Created on Feb 27, 2026

@author: TMARTI02
'''


from model_ws_db_utilities import ModelInitializer, ModelPredictor
from models.ModelBuilder import  Model
import util.predict_constants as pc
import json

from sqlalchemy import text
from datetime import datetime
import pandas as pd
import StatsCalculator as stats
from util.database_utilities import getSession
from applicability_domain import applicability_domain_utilities as adu

import traceback
    
def redo_cv_stats(session, model: Model):

    mi = ModelInitializer()
    df_preds_training_cv = mi.get_cv_predictions(session, model)

    stats_training_cv = stats.calculate_continuous_statistics(df_preds_training_cv, 0,
        pc.TAG_CV + pc.TAG_TRAINING)

    statistic_name = "RMSE_CV_Training"
    statistic_value = stats_training_cv[statistic_name]
    print(model.propertyName, stats_training_cv["RMSE_CV_Training"])
    update_statistic_value(session, model.modelId, statistic_name, statistic_value, "tmarti02")
    compare_stats(model, stats_training_cv)

def recalculate_test_set_stats(session, model):
    """Recalculate stats using predictions stored in the db"""

    mi = ModelInitializer()
    mi.get_model_statistics(model, session)

    df_preds_training = mi.get_predictions(session, model=model, split_num=0, fk_splitting_id=1)
    df_preds_test = mi.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
    mean_exp_training = stats.calculate_mean_exp_training(df_preds_training)
    stats_test_set = stats.calculate_continuous_statistics(df_preds_test, mean_exp_training,
        pc.TAG_TEST)

    compare_stats(model, stats_test_set)


def determineApplicabilityDomainBatch(model: Model, applicabilityDomainName, df_prediction):
    """
    Calculate the applicability domain using the model's training set and the AD measure assigned to the model in the DB
    TODO make sure this works when a model doesnt have a set embedding object
    
    output is a dataframe with rows as follows:
    {
        "idTest": "BrC1C(Br)=C(Br)C(Br)=C(Br)C=1Br",
        "ids": [
            "BrC1C(OC2=C(Br)C(Br)=C(Br)C(Br)=C2Br)=C(Br)C(Br)=C(Br)C=1Br",
            "ClC(Br)Br",
            "BrC(Br)Br"
        ],
        "distances": [
            2.811670273467691,
            3.2020696523378205,
            3.7474088837637805
        ],
        "AD": false
    }
    
    
    :param model:
    :param df_prediction:
    :return:
    """

    output, ad_cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
        train_df=model.df_training,
        test_df=df_prediction,
        # test_df=model.df_prediction,  #for testing running batch type ad calc
        remove_log_p=model.remove_log_p_descriptors,
        embedding=model.embedding,
        # applicability_domain=model.applicabilityDomainName,
        applicability_domain=applicabilityDomainName,
        filterColumnsInBothSets=True)
    return output


def calculate_ad_stats(session, model:Model):

    mi = ModelInitializer()
    mi.get_training_prediction_instances(session, model)

    df_preds_test = mi.get_predictions(session, model=model, split_num=1, fk_splitting_id=1)
    df_ad = determineApplicabilityDomainBatch(model, model.applicabilityDomainName, model.df_prediction)
    
    # from utils import print_first_row
    # print_first_row(df_ad)
    
    calculate_AD_stats(df_ad, df_preds_test, pc.TAG_TEST, model.modelId, session)

def updateStatsPredictModuleModels():

    try:

        session = getSession()
        # print(dbl.session == None)
        # import os
        # print(os.getenv('DEV_QSAR_PASS'))

        mi = ModelInitializer()

        # Use left joins so can still get a result if something is missing (like fk_ad_method was not set for model)
        sql = text("select m.id from qsar_models.models m WHERE m.fk_source_id = 3 and m.is_public=true order by m.id;")  # fk_source_id=3 => cheminformatics modules
        results = session.execute(sql).fetchall()

        # Process the result
        for row in results:
            model = mi.init_model(row.id)
            # print(json.dumps(json.loads(model.get_model_description()), indent=4))
            # recalculate_test_set_stats(session, model)
            # redo_cv_stats(session, model)
            calculate_ad_stats(session, model)
            # if True:
            #     break # stop after first model for testing

    except Exception as ex:
        traceback.print_exc()
        print(f"Exception occurred: {ex}")
    finally:
        # Close the session - close it later after get training/test sets
        session.close()

def calculate_AD_stats(df_ad, df_preds, tag, model_id, session):

    merged_df = pd.merge(df_ad, df_preds, left_on='idTest', right_on='id')
    # print(merged_df.columns)

    df_inside_ad = merged_df[merged_df['AD']].loc[:, ['id', 'exp', 'pred']]
    stats_test_inside_AD = stats.calculate_continuous_statistics(df_inside_ad, 0, tag + "_inside_AD")
    # print(stats_test_inside_AD)
    MAE_inside_AD = stats_test_inside_AD ["MAE" + tag + "_inside_AD"]
    # print(MAE_Test_inside_AD)

    df_outside_ad = merged_df[~merged_df['AD']].loc[:, ['id', 'exp', 'pred']]
    stats_test_outside_AD = stats.calculate_continuous_statistics(df_outside_ad, 0, tag + "_outside_AD")
    # print(stats_test_inside_AD)
    MAE_outside_AD = stats_test_outside_AD ["MAE" + tag + "_outside_AD"]

    count_inside = df_inside_ad.shape[0]
    count_outside = df_outside_ad.shape[0]
    coverage = count_inside / (count_inside + count_outside)

    dict_stats = {}
    dict_stats["MAE" + tag + "_inside_AD"] = MAE_inside_AD
    dict_stats["MAE" + tag + "_outside_AD"] = MAE_outside_AD
    dict_stats["Coverage" + tag] = coverage

    for statistic_name in dict_stats:
        new_statistic_value = dict_stats[statistic_name]
        print(f"model_id:{model_id}, statistic_name:{statistic_name}, new_statistic_value:{new_statistic_value:.3f}")
        update_statistic_value(session, model_id, statistic_name, new_statistic_value, "tmarti02")

    # print(dict_stats)

    return MAE_inside_AD, MAE_outside_AD, coverage

def update_statistic_value(session, model_id: int, statistic_name: str, new_statistic_value: float, user_id: str):
    try:
        # Query to find the statistic ID
        statistic_id_result = session.execute(
            text("SELECT id FROM qsar_models.\"statistics\" WHERE name = :name"),
            {"name": statistic_name}
        ).fetchone()

        if statistic_id_result is None:
            raise ValueError(f"Statistic '{statistic_name}' does not exist.")

        statistic_id = statistic_id_result[0]

        # Query to check if the statistic already exists for the model
        existing_statistic_result = session.execute(
            text("""
                SELECT 1 FROM qsar_models.model_statistics
                WHERE fk_model_id = :model_id AND fk_statistic_id = :statistic_id
            """),
            {"model_id": model_id, "statistic_id": statistic_id}

        ).fetchone()

        # print(existing_statistic_result, statistic_id)
        #
        # if True:
        #     return

        if existing_statistic_result is None:
            # Insert if not exists
            session.execute(
                text("""
                    INSERT INTO qsar_models.model_statistics (
                        fk_model_id, fk_statistic_id, statistic_value, created_at, updated_at, created_by, updated_by
                    ) VALUES (
                        :model_id, :statistic_id, :statistic_value, :created_at, :updated_at, :created_by, :updated_by
                    )
                """),
                {
                    "model_id": model_id,
                    "statistic_id": statistic_id,
                    "statistic_value": new_statistic_value,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now(),
                    "created_by": user_id,
                    "updated_by": user_id
                }
            )
        else:
            # Update if exists
            session.execute(
                text("""
                    UPDATE qsar_models.model_statistics
                    SET statistic_value = :statistic_value,
                        updated_at = :updated_at,
                        updated_by = :updated_by
                    WHERE fk_model_id = :model_id AND fk_statistic_id = :statistic_id
                """),
                {
                    "model_id": model_id,
                    "statistic_id": statistic_id,
                    "statistic_value": new_statistic_value,
                    "updated_at": datetime.now(),
                    "updated_by": user_id
                }
            )

        # Commit the transaction
        session.commit()
    except Exception as e:
        e.with_traceback()
        session.rollback()
        raise e

def compare_stats(model, stats_new):

    data = []

    for stat_name in stats_new:

        if stat_name == "Coverage_CV_Training":
            continue

        stat_value_old = model.modelStatistics.get(stat_name, "N/A")
        stat_value_new = stats_new[stat_name]
        data.append({
            'Statistic': stat_name,
            'Old Value': stat_value_old,
            'New Value': stat_value_new
        })

    # Create a DataFrame
    df = pd.DataFrame(data, columns=['Statistic', 'Old Value', 'New Value'])

    # Print the DataFrame in a readable format
    print("\n" + model.propertyName + "\n")
    print(df.to_string(index=False, float_format='{:.3f}'.format))


if __name__ == '__main__':
    
    from dotenv import load_dotenv
    load_dotenv('../../personal.env')
    
    # print(username)
    updateStatsPredictModuleModels()
