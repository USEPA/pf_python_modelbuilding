import pandas as pd
import math
from typing import Dict, Optional
import numpy as np

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


# Safe divisions (match Java behavior but avoid ZeroDivisionError)
def safe_div(n, d):
    try:
        return n / d if d else float('nan')
    except ZeroDivisionError:
        return float('nan')


def calculate_mean_exp_training(df_training: pd.DataFrame):
    # Filter out rows where 'exp' or 'pred' are NaN
    valid_df = df_training.dropna(subset=['exp', 'pred'])

    # Calculate means
    return valid_df['exp'].mean()


def calculate_continuous_statistics(df: pd.DataFrame, mean_exp_training: float, tag: str, ad_measure_final: Optional[str] = None) -> Dict[str, float]:
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
    if ad_measure_final is None:
        coverage = safe_div(count_predicted, count_total)
    else:
        # Build list of AD columns
        colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]

        # Inside/outside consensus AD masks
        mask_all_true = valid_df[colsAD].eq(True).all(axis=1)
        mask_outside = ~mask_all_true

        # Coverage of consensus AD
        total_rows = len(valid_df)
        coverage = safe_div(mask_all_true.sum(), total_rows)
    
    pearson_rsq = safe_div(term_xy ** 2, term_xx * term_yy)
    # coeff_det = 1 - ss / ss_total if ss_total != 0 else float('nan')
    coeff_det = 1 - safe_div(ss, ss_total)
    rmse = math.sqrt(safe_div(ss, count_predicted))

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
    elif tag == pc.TAG_EXTERNAL:
        model_statistic_values[pc.Q2_EXTERNAL] = coeff_det

    return model_statistic_values


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_continuous_statistics_sklearn(
    df: pd.DataFrame,
    mean_exp_training: float,
    tag: str,
    ad_measure_final: Optional[str] = None
) -> Dict[str, float]:

    valid_df = df.loc[df["exp"].notna() & df["pred"].notna()].copy()

    count_total = df["exp"].notna().sum()
    count_predicted = len(valid_df)

    if count_predicted == 0:
        raise ValueError("No valid predictions available for calculation.")

    y_true = valid_df["exp"].to_numpy()
    y_pred = valid_df["pred"].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    pearson_rsq = np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_true) > 1 else float("nan")

    # Standard sklearn R2
    r2 = r2_score(y_true, y_pred)

    # Original Q2-style metric using training mean
    ss = ((y_true - y_pred) ** 2).sum()
    ss_total = ((y_true - mean_exp_training) ** 2).sum()
    q2 = 1 - safe_div(ss, ss_total)

    if ad_measure_final is None:
        coverage = safe_div(count_predicted, count_total)
    else:
        colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]
        mask_all_true = valid_df[colsAD].eq(True).all(axis=1)
        coverage = safe_div(mask_all_true.sum(), len(valid_df))

    model_statistic_values = {
        pc.COVERAGE + tag: coverage,
        pc.MAE + tag: mae,
        pc.PEARSON_RSQ + tag: pearson_rsq,
        pc.RMSE + tag: rmse,
    }

    if tag == pc.TAG_TEST:
        model_statistic_values[pc.Q2_TEST] = q2
    elif tag == pc.TAG_TRAINING:
        model_statistic_values[pc.R2_TRAINING] = r2
    elif tag == pc.TAG_EXTERNAL:
        model_statistic_values[pc.Q2_EXTERNAL] = q2

    return model_statistic_values


from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, confusion_matrix

def calculate_binary_statistics_sklearn(
    df: pd.DataFrame,
    cutoff: float,
    tag: str,
    ad_measure_final: Optional[str] = None
) -> Dict[str, float]:

    # Keep only rows with known expected label
    valid = df.dropna(subset=["exp"])
    count_total = len(valid)

    # Keep only rows with prediction
    predicted = valid.dropna(subset=["pred"])
    count_predicted = len(predicted)

    coverage = safe_div(count_predicted, count_total)

    # If no predictions, return early
    if count_predicted == 0:
        return {
            pc.COVERAGE + tag: coverage,
            pc.CONCORDANCE + tag: float("nan"),
            pc.SENSITIVITY + tag: float("nan"),
            pc.SPECIFICITY + tag: float("nan"),
            pc.BALANCED_ACCURACY + tag: float("nan"),
        }

    # Binary predictions and true labels
    y_true = predicted["exp"].astype(int).to_numpy()
    y_pred = (predicted["pred"] >= cutoff).astype(int).to_numpy()

    # Confusion matrix gives tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    count_positive = int((y_true == 1).sum())
    count_negative = int((y_true == 0).sum())

    if ad_measure_final is None:
        coverage = safe_div(count_predicted, count_total)
    else:
        # Build list of AD columns
        colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]

        # Consensus AD coverage
        total_rows = len(valid)
        mask_all_true = valid[colsAD].eq(True).all(axis=1)
        coverage = safe_div(mask_all_true.sum(), total_rows)

    concordance = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = safe_div(tn, count_negative)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    return {
        pc.COVERAGE + tag: coverage,
        pc.CONCORDANCE + tag: concordance,
        pc.SENSITIVITY + tag: sensitivity,
        pc.SPECIFICITY + tag: specificity,
        pc.BALANCED_ACCURACY + tag: balanced_accuracy,
    }
    
    
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score

def calculate_ternary_statistics_sklearn(
    df: pd.DataFrame,
    tag: str,
    ad_measure_final: Optional[str] = None
) -> Dict[str, float]:

    valid = df.dropna(subset=["exp", "pred"])
    count_total = len(df.dropna(subset=["exp"]))
    count_predicted = len(valid)

    coverage = safe_div(count_predicted, count_total)

    if count_predicted == 0:
        return {
            pc.COVERAGE + tag: coverage,
            pc.CONCORDANCE + tag: float("nan"),
            pc.RECALL_CLASS_0 + tag: float("nan"),
            pc.RECALL_CLASS_1 + tag: float("nan"),
            pc.RECALL_CLASS_2 + tag: float("nan"),
            pc.BALANCED_ACCURACY + tag: float("nan"),
        }

    y_true = valid["exp"].astype(int).to_numpy()
    y_pred = valid["pred"].astype(int).to_numpy()

    if ad_measure_final is not None:
        colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]
        mask_all_true = valid[colsAD].eq(True).all(axis=1)
        coverage = safe_div(mask_all_true.sum(), len(valid))

    concordance = accuracy_score(y_true, y_pred)

    recalls = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0
    )
    recall_0, recall_1, recall_2 = recalls

    # Balanced accuracy only if all three classes are present in y_true
    unique_classes = set(np.unique(y_true))
    if unique_classes == {0, 1, 2}:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    else:
        balanced_accuracy = float("nan") #dont set a value if one of the classes isnt present in observed values

    return {
        pc.COVERAGE + tag: coverage,
        pc.CONCORDANCE + tag: concordance,
        pc.RECALL_CLASS_0 + tag: recall_0,
        pc.RECALL_CLASS_1 + tag: recall_1,
        pc.RECALL_CLASS_2 + tag: recall_2,
        pc.BALANCED_ACCURACY + tag: balanced_accuracy,
    }
    
def calculate_binary_statistics(df: pd.DataFrame, cutoff: float, tag: str, ad_measure_final: Optional[str] = None) -> Dict[str, float]:
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

    # Calculate statistics
    if ad_measure_final is None:
        coverage = safe_div(count_predicted, count_total)
    else:
        # Build list of AD columns
        colsAD = [f"AD_{ad.replace(' ', '_')}" for ad in ad_measure_final]

        # Inside/outside consensus AD masks
        mask_all_true = valid[colsAD].eq(True).all(axis=1)
        mask_outside = ~mask_all_true

        # Coverage of consensus AD
        total_rows = len(valid)
        coverage = safe_div(mask_all_true.sum(), total_rows)
    
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