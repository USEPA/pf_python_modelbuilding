"""
Class to make plots for TEST5.1.3 for web page reports
@author: Todd Martin
"""

import json
# import random
from math import floor, ceil
import numpy as np
import matplotlib.pyplot as plt



def getBin(exps, property_name,unit_name):
    maxVal, minVal = getMinMax(exps)
    # print(minVal, maxVal)

    if property_name == 'Density' or property_name == 'Viscosity':
        bins = np.linspace(minVal, maxVal, (maxVal - minVal) * 2 + 1)
    elif property_name == 'Surface tension':
        maxVal = round(maxVal, -2) + 10
        minVal = round(minVal, -2) - 10
        bins = int((maxVal - minVal) / 10)
    elif '°C' in unit_name or property_name == 'Thermal conductivity':
        maxVal = round(maxVal, -2) + 100
        minVal = round(minVal, -2) - 100
        bins = int((maxVal - minVal) / 100)
    elif 'log' in unit_name:
        bins = range(minVal, maxVal)
    else:
        print('else', property_name)
        # bins = np.linspace(-10,-10,10)

    print(minVal, maxVal, bins)

    return bins

def getBin2(expsTrain,expsTest, property_name, unit_name):

    maxVal, minVal = getMinMax2(expsTrain,expsTest)
    # print(minVal, maxVal)



    if property_name == 'Viscosity':
        bins = np.linspace(minVal, maxVal, (maxVal - minVal) * 2 + 1)

    elif property_name == 'Density':
        bins = np.linspace(0.5, 4, 15)

    elif property_name == 'Liquid Chromatography Retention Time':

        bins = np.linspace(0, 50, 11)

    elif property_name == 'Surface Tension':
        maxVal = round(maxVal, -2) + 10
        minVal = round(minVal, -2) - 10
        if minVal < 0:
            minVal = 0
        # bins = int((maxVal - minVal) / 10)
        # bins = range(minVal,maxVal, )
        bins = np.linspace(minVal, maxVal, int((maxVal - minVal) / 10)+1)

    elif property_name == 'Thermal Conductivity':
        maxVal = round(maxVal, -2) + 25
        minVal = round(minVal, -2) - 25
        if minVal < 0:
            minVal = 0
        # bins = int((maxVal - minVal) / 100)
        bins = np.linspace(minVal, maxVal, (int((maxVal - minVal) / 25))+1)

    elif '°C' in unit_name:
        maxVal = round(maxVal, -2) + 50
        minVal = round(minVal, -2) - 50
        # bins = int((maxVal - minVal) / 100)
        bins = np.linspace(minVal, maxVal, (int((maxVal - minVal) / 50))+1)

    elif property_name == 'Fraction Unbound in Human Plasma':
        bins = np.linspace(0, 1, 11)

    elif 'log' in unit_name.lower():
        maxVal = maxVal+1
        minVal = minVal-1
        nbins = maxVal-minVal+1

        if nbins < 12:
            bins = np.linspace(minVal, maxVal, 2*(maxVal-minVal)+1)
        else:
            bins = range(minVal, maxVal)
    else:
        print('\n*** else', property_name,unit_name)
        bins = range(minVal, maxVal)


        # bins = np.linspace(-10,-10,10)

    print(property_name, minVal, maxVal, bins)


    return bins


def generateHistogram(file_path_json, property_name, unit_name, mpsTraining, mpsTest, seriesNameTrain, seriesNameTest):
    expsTraining = getExpArray(mpsTraining)

    expsTest = getExpArray(mpsTest)

    # print(expsTraining)

    # binsTraining=getBin(expsTraining,property_name,unit_name)
    # binsTest = getBin(expsTest, property_name, unit_name)
    # plt.hist(expsTraining, binsTraining, alpha=0.5, label=seriesNameTrain, color='darkblue', edgecolor='black')
    # plt.hist(expsTest, binsTest, alpha=1.0, label=seriesNameTest, color='white', edgecolor='black')

    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')

    if len(expsTraining) == 0 and len(expsTest) == 0:
        return

    bins = getBin2(expsTraining,expsTest, property_name, unit_name)



    if len(expsTraining) > 0:
        plt.hist(expsTraining, bins, alpha=0.5, label=seriesNameTrain, color='black', edgecolor='black')

    if len(expsTest) > 0:
        plt.hist(expsTest, bins, alpha=1.0, label=seriesNameTest, color='red', edgecolor='black')

    plt.xlabel('Experimental ' + unit_name)
    plt.ylabel('Count')
    plt.title("Histogram of " + property_name + " Data")

    # plt.legend(loc='upper left')

    if property_name == 'Vapor Pressure' \
            or property_name == 'In Vitro Intrinsic Hepatic Clearance' \
            or (property_name == 'Oral Rat LD50' and '-log' not in unit_name) \
            or (property_name == 'Water Solubility' and '-log' not in unit_name) \
            or property_name == 'Atmos. Hydroxylation Rate':
        plt.legend(loc="upper left")
    else:
        plt.legend(loc="upper right")


    # plt.show()

    fileOutHistogram = file_path_json.replace(".json", "_histogram.png")
    plt.savefig(fileOutHistogram, dpi=300)
    plt.close()
    
def generateHistogram2(fileOutHistogram, property_name, unit_name, mpsTraining, mpsTest, seriesNameTrain, seriesNameTest):
    expsTraining = getExpArray(mpsTraining)

    expsTest = getExpArray(mpsTest)

    # print(expsTraining)

    # binsTraining=getBin(expsTraining,property_name,unit_name)
    # binsTest = getBin(expsTest, property_name, unit_name)
    # plt.hist(expsTraining, binsTraining, alpha=0.5, label=seriesNameTrain, color='darkblue', edgecolor='black')
    # plt.hist(expsTest, binsTest, alpha=1.0, label=seriesNameTest, color='white', edgecolor='black')

    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained',dpi=200)

    if len(expsTraining) == 0 and len(expsTest) == 0:
        return

    bins = getBin2(expsTraining,expsTest, property_name, unit_name)


    if len(expsTraining) > 0:
        plt.hist(expsTraining, bins, alpha=0.5, label=seriesNameTrain, color='black', edgecolor='black')

    if len(expsTest) > 0:
        plt.hist(expsTest, bins, alpha=1.0, label=seriesNameTest, color='red', edgecolor='black')

    plt.xlabel('Experimental ' + unit_name)
    plt.ylabel('Count')
    plt.title("Histogram of " + property_name + " Data in Sets")

    # plt.legend(loc='upper left')

    if property_name == 'Vapor Pressure' \
            or property_name == 'In Vitro Intrinsic Hepatic Clearance' \
            or (property_name == 'Oral Rat LD50' and '-log' not in unit_name) \
            or (property_name == 'Water Solubility' and '-log' not in unit_name) \
            or property_name == 'Atmos. Hydroxylation Rate':
        plt.legend(loc="upper left")
    else:
        plt.legend(loc="upper right")


    # plt.show()

    plt.savefig(fileOutHistogram, dpi=200)
    plt.close()


def getMinMax2(exp1, exp2):
    minVal = 9999
    maxVal = -9999
    for val in exp1:
        if val < minVal:
            minVal = val
        if val > maxVal:
            maxVal = val

    # print('exp2',exp2)

    if exp2 is not None:
        for val in exp2:
            if val < minVal:
                minVal = val
            if val > maxVal:
                maxVal = val

    # print(minVal, maxVal)
    minVal = floor(minVal)
    maxVal = ceil(maxVal)

    return maxVal, minVal

def getMinMax(exps):
    minVal = 9999
    maxVal = -9999
    for val in exps:
        if val < minVal:
            minVal = val
        if val > maxVal:
            maxVal = val

    # print(minVal, maxVal)
    minVal = floor(minVal)
    maxVal = ceil(maxVal)

    return maxVal, minVal



def setAxisBounds(unitName, expsTraining, predsTraining, expsTest, predsTest, ax):
    min_value = min(min(expsTraining), min(predsTraining), min(expsTest), min(predsTest))
    max_value = max(max(expsTraining), max(predsTraining), max(expsTest), min(predsTest))
# Check if "log" is in unitName
    if "log" in unitName.lower():
        min_int = int(np.floor(min_value))
        max_int = int(np.ceil(max_value))
        # Determine if padding is needed
        if (min_value - min_int) < 0.5:
            min_value = min_int - 1
        else:
            min_value = min_int
        if (max_int - max_value) < 0.5:
            max_value = max_int + 1
        else:
            max_value = max_int
        
        # ax.set_xticks(range(min_value, max_value + 1))
        # ax.set_yticks(range(min_value, max_value + 1))
        
        ax.set_xticks(range(min_value, max_value + 1, 2))
        ax.set_yticks(range(min_value, max_value + 1, 2))

    elif unitName == "°C":
        min_value = (np.floor(min_value / 50) * 50) - 50
        max_value = (np.ceil(max_value / 50) * 50) + 50
    else:
        padding = (max_value - min_value) * 0.05
        min_value -= padding
        max_value += padding
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    



def generateScatterPlot2(filePathOut, title, unitName, mpsTraining, mpsTest, seriesNameTrain,
                    seriesNameTest):
    
        
    expsTraining, predsTraining = getArraysOmitNullPreds(mpsTraining)
    expsTest, predsTest = getArraysOmitNullPreds(mpsTest)

    fig, ax = plt.subplots(figsize=(5, 5), layout='constrained', dpi=150)

    plt.xlabel('Experimental ' + unitName)
    plt.ylabel('Predicted  '+ unitName)
    plt.title(title)

    if len(expsTraining) == 0 and len(predsTraining) == 0 and len(expsTest) == 0 and len(predsTest) == 0:
        return

    if len(expsTraining)>0 and len(predsTraining)>0:
        ax.scatter(expsTraining, predsTraining, label=seriesNameTrain, color="black", edgecolor='black')

    if len(expsTest) > 0 and len(predsTest) > 0:
        ax.scatter(expsTest, predsTest, label=seriesNameTest, color="red", edgecolor='black')

    ax.plot(expsTraining, expsTraining, label='Y=X', color="black")
    
    setAxisBounds(unitName, expsTraining, predsTraining, expsTest, predsTest, ax)
    
    # yreg = [m * x + b for x in exp]
    # ax.plot(exp, yreg, label='Regression ('+strR2+')',color='red')

    plt.legend(loc="lower right")

    # plt.savefig(fileOutPNG)
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(6, 6)
    # when saving, specify the DPI

    # print(fileOut)
    plt.savefig(filePathOut, dpi=200)
    plt.close()

def generateScatterPlot(file_path_json, property_name, unit_name, mpsTraining, mpsTest, seriesNameTrain,
                        seriesNameTest):
    fileOutScatter = file_path_json.replace(".json", "_scatter_plot.png")

    expsTraining, predsTraining = getArraysOmitNullPreds(mpsTraining)
    expsTest, predsTest = getArraysOmitNullPreds(mpsTest)



    # m, b, r_value, p_value, std_err = scipy.stats.linregress(exp, pred)
    # r2 = r_value * r_value
    # strR2 = '$r^2$=' + str("{:.2f}".format(r2))

    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')

    plt.xlabel('Experimental ' + unit_name)
    plt.ylabel('Predicted  '+ unit_name)
    plt.title(property_name + " Model Results")

    if len(expsTraining) == 0 and len(predsTraining) == 0 and len(expsTest) == 0 and len(predsTest) == 0:
        return

    if len(expsTraining)>0 and len(predsTraining)>0:
        ax.scatter(expsTraining, predsTraining, label=seriesNameTrain, color="black", edgecolor='black')

    if len(expsTest) > 0 and len(predsTest) > 0:
        ax.scatter(expsTest, predsTest, label=seriesNameTest, color="red", edgecolor='black')

    ax.plot(expsTraining, expsTraining, label='Y=X', color="black")

    # yreg = [m * x + b for x in exp]
    # ax.plot(exp, yreg, label='Regression ('+strR2+')',color='red')

    plt.legend(loc="lower right")

    # plt.savefig(fileOutPNG)
    figure = plt.gcf()  # get current figure
    # figure.set_size_inches(6, 6)
    # when saving, specify the DPI

    # print(fileOut)
    plt.savefig(fileOutScatter, dpi=300)
    plt.close()


# x = [random.gauss(3,1) for _ in range(400)]
# y = [random.gauss(4,2) for _ in range(400)]
#
# bins = np.linspace(-10, 10, 10)
#
# pyplot.hist(x, bins, alpha=0.5, label='x')
# pyplot.hist(y, bins, alpha=0.5, label='y')
# pyplot.legend(loc='upper right')
# pyplot.show()


def getArraysOmitNullPreds(mps):
    exps = []
    preds = []

    for mp in mps:
        if 'pred' not in mp:
            continue
        exps.append(mp['exp'])
        preds.append(mp['pred'])

    exps = np.array(exps)
    preds = np.array(preds)

    return exps, preds


def getExpArray(mps):
    exps = []
    for mp in mps:
        exps.append(mp['exp'])
    exps = np.array(exps)
    return exps


def run(filepath):
    with open(filepath) as json_file:
        json_data = json.load(json_file)

        propertyName = json_data['propertyName']
        unitName = json_data['units']

        mpsTest = json_data['mpsTest']
        mpsTraining = json_data['mpsTraining']

        if '°C' in unitName:
            unitName = '°C'

        # print(propertyName, unitName)

        generateScatterPlot(file_path_json=filepath, property_name=propertyName, unit_name=unitName,
                            mpsTraining=mpsTraining, mpsTest=mpsTest, seriesNameTrain="Training set",
                            seriesNameTest="Test set")

        generateHistogram(file_path_json=filepath, property_name=propertyName, unit_name=unitName,
                          mpsTraining=mpsTraining, mpsTest=mpsTest, seriesNameTrain="Training set",
                          seriesNameTest="Test set")


if __name__ == "__main__":

    folder = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\hibernate_qsar_model_building\data\TEST5.1.3\reports\plots"
    # folder = r"C:\Users\TMARTI02\OneDrive - Environmental Protection Agency (EPA)\0 java\0 model_management\hibernate_qsar_model_building\data\OPERA2.8\reports\plots"

    # filepath = folder + "/96 hour fathead minnow LC50_consensus_preds.json"
    # filepath = folder + "/Boiling point_consensus_preds.json"

    import os

    directory = os.fsencode(folder)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if 'from_sdf' in filename:
            continue

        if filename.endswith(".json"):
            run(folder + '/' + filename)
