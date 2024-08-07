import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats


def calcStats(predictions, df_prediction, excelPath):
    df_preds = pd.DataFrame(predictions, columns=['Prediction'])
    df_pred = df_prediction[['ID', 'Property']]
    df_pred = pd.merge(df_pred, df_preds, how='left', left_index=True, right_index=True)

    if excelPath:
        df_pred.to_excel(excelPath, index=False)

    # a scatter plot comparing num_children and num_pets
    # myplot=df_pred.plot(kind='scatter', x='Property', y='Prediction', color='black')
    m, b, r_value, p_value, std_err = scipy.stats.linregress(df_pred['Property'], df_pred['Prediction'])
    strR2 = str("{:.3f}".format(r_value ** 2))

    y_true, predictions = np.array(df_pred['Property']), np.array(df_pred['Prediction'])

    # print (y_true)
    # print(predictions)

    MAE = np.mean(np.abs(y_true - predictions))
    strMAE = str("{:.3f}".format(MAE))
    # print(strR2,MAE)

    return strR2, strMAE


def calc_pearson_r2(predictions, targets):
    score = stats.pearsonr(list(predictions), list(targets))[0]
    score = score * score
    return score


def calc_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def calc_MAE(predictions, targets):
    return np.mean(np.abs(targets - predictions))


def generatePlot(property_name, title, exp, pred):
    m, b, r_value, p_value, std_err = scipy.stats.linregress(exp, pred)

    r2 = r_value * r_value

    fig, ax = plt.subplots()

    strR2 = '$r^2$=' + str("{:.2f}".format(r2))
    title += ' (' + strR2 + ')'

    plt.title(title)
    plt.xlabel('experimental ' + property_name)
    plt.ylabel('predicted ' + property_name)
    ax.scatter(exp, pred, label='exp vs pred')
    ax.plot(exp, exp, label='Y=X',color="black")

    yreg = [m * x + b for x in exp]

    ax.plot(exp, yreg, label='Regression',color='red')

    plt.legend(loc="lower right")

    fig.show()
    # plt.show(block=False)
    plt.show()
