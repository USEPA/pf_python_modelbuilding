import pandas as pd
import math
from typing import Dict

class PredictConstants:
    COVERAGE = "Coverage"
    MAE = "MAE"
    PEARSON_RSQ = "PearsonRSQ"
    RMSE = "RMSE"
    Q2 = "Q2"
    R2 = "R2"

    TAG_TEST = "_Test"
    TAG_TRAINING = "_Training"
    TAG_CV = "_CV"

    Q2_TEST = Q2 + TAG_TEST
    R2_TRAINING = R2+TAG_TRAINING


def calculate_mean_exp_training(df_training: pd.DataFrame):
    # Filter out rows where 'exp' or 'pred' are NaN
    valid_df = df_training.dropna(subset=['exp', 'pred'])

    # Calculate means
    return valid_df['exp'].mean()


def calculate_continuous_statistics(df: pd.DataFrame, mean_exp_training: float, tag: str) -> Dict[str, float]:

    # Filter out rows where 'exp' or 'pred' are NaN
    valid_df = df.dropna(subset=['exp', 'pred'])

    # Total counts
    count_total = len(df.dropna(subset=['exp']))
    count_predicted = len(valid_df)

    if count_predicted == 0:
        raise ValueError("No valid predictions available for calculation.")

    # Calculate means
    mean_exp = valid_df['exp'].mean()
    mean_pred = valid_df['pred'].mean()

    # Calculate MAE
    mae = (valid_df['exp'] - valid_df['pred']).abs().mean()

    # Calculate terms for Pearson RSQ
    term_xy = ((valid_df['exp'] - mean_exp) * (valid_df['pred'] - mean_pred)).sum()
    term_xx = ((valid_df['exp'] - mean_exp) ** 2).sum()
    term_yy = ((valid_df['pred'] - mean_pred) ** 2).sum()

    # Calculate sums for coefficient of determination
    ss = ((valid_df['exp'] - valid_df['pred']) ** 2).sum()
    ss_total = ((valid_df['exp'] - mean_exp_training) ** 2).sum()

    # Calculate statistics
    coverage = count_predicted / count_total
    pearson_rsq = (term_xy ** 2) / (term_xx * term_yy) if term_xx != 0 and term_yy != 0 else float('nan')
    coeff_det = 1 - ss / ss_total if ss_total != 0 else float('nan')
    rmse = math.sqrt(ss / count_predicted)

    model_statistic_values = {
        PredictConstants.COVERAGE + tag: coverage,
        PredictConstants.MAE + tag: mae,
        PredictConstants.PEARSON_RSQ + tag: pearson_rsq,
        PredictConstants.RMSE + tag: rmse
    }

    if tag == PredictConstants.TAG_TEST:
        model_statistic_values[PredictConstants.Q2_TEST] = coeff_det
    elif tag == PredictConstants.TAG_TRAINING:
        model_statistic_values[PredictConstants.R2_TRAINING] = coeff_det

    return model_statistic_values