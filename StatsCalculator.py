import pandas as pd
import math
from typing import Dict

# class StatsConstants:
#     COVERAGE = "Coverage"
#     MAE = "MAE"
#     PEARSON_RSQ = "PearsonRSQ"
#     RMSE = "RMSE"
#     Q2 = "Q2"
#     R2 = "R2"
#
#     TAG_TEST = "_Test"
#     TAG_TRAINING = "_Training"
#     TAG_CV = "_CV"
#
#     Q2_TEST = Q2 + TAG_TEST
#     R2_TRAINING = R2+TAG_TRAINING

from util import predict_constants as pc


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
        pc.COVERAGE + tag: coverage,
        pc.MAE + tag: mae,
        pc.PEARSON_RSQ + tag: pearson_rsq,
        pc.RMSE + tag: rmse
    }

    if tag == pc.TAG_TEST:
        model_statistic_values[pc.Q2_TEST] = coeff_det
    elif tag == pc.TAG_TRAINING:
        model_statistic_values[pc.R2_TRAINING] = coeff_det

    return model_statistic_values


def calculate_binary_statistics(df: pd.DataFrame, cutoff: float, tag: str) -> Dict[str, float]:
    # Keep only rows with a known expected label
    valid = df.dropna(subset=['exp'])
    count_total = len(valid)

    # Among those, keep only rows with a prediction
    predicted = valid.dropna(subset=['pred'])
    count_predicted = len(predicted)

    # If there are no predicted rows, return coverage and NaNs for other metrics
    if count_predicted == 0:
        coverage = (count_predicted / count_total) if count_total else float('nan')
        return {
            pc.COVERAGE + tag: coverage,
            pc.CONCORDANCE + tag: float('nan'),
            pc.SENSITIVITY + tag: float('nan'),
            pc.SPECIFICITY + tag: float('nan'),
            pc.BALANCED_ACCURACY + tag: float('nan'),
        }

    # Binary predictions using the cutoff
    pred_bin = (predicted['pred'] >= cutoff).astype(int)

    # Use exp values from the same (predicted) subset
    exp_vals = predicted['exp']

    # Java logic counts positives/negatives only among rows that have predictions
    pos_mask = (exp_vals == 1)
    neg_mask = (exp_vals == 0)

    count_positive = int(pos_mask.sum())
    count_negative = int(neg_mask.sum())

    tp = int((pos_mask & (pred_bin == 1)).sum())
    tn = int((neg_mask & (pred_bin == 0)).sum())
    count_true = tp + tn

    # Safe divisions (match Java behavior but avoid ZeroDivisionError)
    def safe_div(n, d):
        return n / d if d else float('nan')

    coverage = safe_div(count_predicted, count_total)
    concordance = safe_div(count_true, count_predicted)
    sensitivity = safe_div(tp, count_positive)
    specificity = safe_div(tn, count_negative)
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        pc.COVERAGE + tag: coverage,
        pc.CONCORDANCE + tag: concordance,
        pc.SENSITIVITY + tag: sensitivity,
        pc.SPECIFICITY + tag: specificity,
        pc.BALANCED_ACCURACY + tag: balanced_accuracy,
    }