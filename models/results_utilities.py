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


def generatePlot(fileOut, property_name, title, exp, pred):
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

    fileOutPNG = fileOut.replace(".csv", ".png")

    # plt.savefig(fileOutPNG)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(6, 6)
    # when saving, specify the DPI
    plt.savefig(fileOutPNG, dpi=300)


    fig.show()
    # plt.show(block=False)
    plt.show()






def generateTrainingPredictionPlot(fileOut, property_name, title, figtitle, exp_training, pred_training,exp_prediction, pred_prediction):

    #    fig, ax = plt.subplots()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 6))

#    plt.title(title)
    fig.suptitle(title)

    createSubplot(exp_training, pred_training, property_name,ax1,'Training')
    createSubplot(exp_prediction, pred_prediction, property_name, ax2,'Prediction')

    fig.show()
    plt.show()
    plt.close()

    figtrain = fileOut.replace('.csv','_train.png')
    generatePlot2(figtrain, property_name, title, exp_training, pred_training)

    figpred = fileOut.replace('.csv','_pred.png')
    generatePlot2(figpred,property_name,title,exp_prediction, pred_prediction)


def createSubplot(exp, pred, property_name, ax1, set):

    m, b, r_value, p_value, std_err = scipy.stats.linregress(exp, pred)
    r2 = r_value * r_value
    strR2 = '$r^2$=' + str("{:.2f}".format(r2))

    ax1.set_xlabel('experimental ' + property_name)
    ax1.set_ylabel('predicted ' + property_name)
    ax1.set_title(set+' Set')

    ax1.scatter(exp, pred, label='exp vs pred')
    ax1.plot(exp, exp, label='Y=X', color="black")

    yreg = [m * x + b for x in exp]
    ax1.plot(exp, yreg, label='Regression ('+strR2+')', color='red')

    ax1.legend(loc="lower right")


def generatePlot2(fileOut, property_name, title, exp, pred):

    m, b, r_value, p_value, std_err = scipy.stats.linregress(exp, pred)
    r2 = r_value * r_value
    strR2 = '$r^2$=' + str("{:.2f}".format(r2))

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')

    plt.xlabel('experimental ' + property_name)
    plt.ylabel('predicted ' + property_name)
    plt.title(title)

    ax.scatter(exp, pred, label='exp vs pred')
    ax.plot(exp, exp, label='Y=X',color="black")

    yreg = [m * x + b for x in exp]
    ax.plot(exp, yreg, label='Regression ('+strR2+')',color='red')

    plt.legend(loc="lower right")

    # plt.savefig(fileOutPNG)
    figure = plt.gcf()  # get current figure
    # figure.set_size_inches(6, 6)
    # when saving, specify the DPI

    # print(fileOut)
    plt.savefig(fileOut, dpi=300)
    # plt.close()