from concurrent.futures import thread
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import time
import threading
import re
from datetime import date
from matplotlib import colors as mcolors
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

def containsDigit(string):
    for i in string:
        if i.isdigit():
            return True;
    return False;

def intToBooleans(number):
   binaryString = format(number, '12b');
   return [x == '1' for x in binaryString[::-1]];



def getCleanedData( yearData, fileName):

    countryNames =['United Kingdom', 'France', 'USA', 'Belgium', 'Australia', 'EIRE', 'Germany', 'Portugal', 'Japan', 'Denmark',
     'Nigeria', 'Netherlands', 'Poland', 'Spain', 'Channel Islands', 'Italy', 'Cyprus', 'Greece', 'Norway', 'Austria',
     'Sweden', 'United Arab Emirates', 'Finland', 'Switzerland', 'Malta', 'Bahrain', 'RSA', 'Bermuda',
     'Hong Kong', 'Singapore', 'Thailand', 'Israel', 'Lithuania', 'West Indies', 'Lebanon', 'Korea', 'Brazil', 'Canada',
     'Iceland']

    mediaPostageCosts =[5.0,18.0,135.5,15.0,305.0,25.0,18.0,28.0,180.0,20.0,75.0,15.0,40.0,28.0,20.0,28.0,15.0,50.0,40.0,40.0,
     40.0,37.5,40.0,40.0,52.5,40.0,150.0,130.0,160.0,170.0,170.0,45.0,40.0,130.0, 39.67,180.0,170.0,130.0,40.0]

    data = []
    customerIDs = [];
    for i in yearData:
        if i[5] > 0.0 and i[7] in countryNames and not np.isnan(i[6]) and containsDigit(str(i[1])):
            data.append(i);
            if (int)(i[6]) not in customerIDs and i[3] != 0:
                customerIDs.append(i[6]);

    customerPostage = [0.0] * len(customerIDs);
    customerQuantity = [0.0] * len(customerIDs);
    customerSubtotal = [0.0] * len(customerIDs);
    customerAverageQuantity =[0.0] * len(customerIDs);
    customerAverageSubtotal = [0.0] * len(customerIDs);
    customerAverageItemCost = [0.0] * len(customerIDs);
    customerReturnRates = [0.0] * len(customerIDs);
    customerReturnCost = [0.0] * len(customerIDs);
    customerReturnQuantity = [0.0] * len(customerIDs);
    customerReturnQuantityPercentage = [0.0] * len(customerIDs);
    customerReturnCostPercentage = [0.0] * len(customerIDs);
    customerOrderCount = [0.0] * len(customerIDs);
    customerRecency = [0.0] * len(customerIDs);

    for i in range(0, len(customerIDs)):
        customerOrderNumbers = [];
        customerOrderDates = [];
        customerReturnOrders = [];

        # return order is not same ID as purchase order
        for j in data:
            if j[6] == customerIDs[i]:
                if not str(j[0]).startswith("C"):
                    customerQuantity[i] += j[3];
                    customerSubtotal[i] += j[5] * j[3];

                    if j[0] not in customerOrderNumbers:
                        customerOrderNumbers.append(j[0]);
                        customerPostage[i] = mediaPostageCosts[countryNames.index(j[7])];
                        customerOrderDates.append(j[4]);

                elif str(j[0]).startswith("C"):
                    customerReturnQuantity[i] += abs(j[3]);
                    customerReturnCost[i] += abs(j[5]) * abs(j[3]);

                    if j[0] not in customerReturnOrders:
                        customerReturnOrders.append(j[0]);
                        # customerOrderNumbers.append(j[0]);

        customerPostageCosts = []
        for j in yearData:
            if j[6] == customerIDs[i] and str(j[1]).startswith("POST"):
                customerPostage[i] += np.abs(j[3]);
        if len(customerPostageCosts) > 0:
            customerPostage[i] /= len(customerPostageCosts);

        if len(customerOrderDates) >= 2:
            for j in range(0, len(customerOrderDates) - 1):
                daysDelta = (customerOrderDates[j + 1].to_pydatetime() - customerOrderDates[j].to_pydatetime()).days;
                customerRecency[i] = customerRecency[i] + daysDelta;
            customerRecency[i] = customerRecency[i] / (len(customerOrderDates) - 1);
        else:
            customerRecency[i] = 366;

        customerOrderCount[i] = len(customerOrderNumbers)

        if len(customerOrderNumbers) > 0:
            customerReturnRates[i] = float(len(customerReturnOrders)) / float(len(customerOrderNumbers));
            customerOrderCount[i] = float(len(customerOrderNumbers));
            customerAverageQuantity[i] = customerQuantity[i] / float(len(customerOrderNumbers));
            customerAverageSubtotal[i] = customerSubtotal[i] / float(len(customerOrderNumbers));
            customerAverageItemCost[i] = customerSubtotal[i] / float(len(customerQuantity));

            customerReturnQuantityPercentage[i] = customerReturnQuantity[i] / customerQuantity[i];
            customerReturnCostPercentage[i] = customerReturnCost[i] / customerSubtotal[i] if customerSubtotal[i] > 0 else 0;

        else:
            customerReturnRates[i] = 1.0;
            customerOrderCount[i] = float(len(customerReturnOrders));
            customerAverageQuantity[i] = customerReturnQuantity[i] / float(len(customerReturnOrders));
            customerAverageSubtotal[i] = customerReturnCost[i] / float(len(customerReturnOrders));
            customerAverageItemCost[i] = customerReturnCost[i] / customerReturnQuantity[i];

            customerReturnQuantityPercentage[i] = 1.0;
            customerReturnCostPercentage[i] = 1.0;

    newData = np.asarray([
        np.asarray(customerIDs),
        np.asarray(customerPostage),
        np.asarray(customerQuantity),
        np.asarray(customerSubtotal),
        np.asarray(customerAverageQuantity),
        np.asarray(customerAverageSubtotal),
        np.asarray(customerAverageItemCost),
        np.asarray(customerReturnRates),
        np.asarray(customerReturnCost),
        np.asarray(customerReturnQuantity),
        np.asarray(customerReturnQuantityPercentage),
        np.asarray(customerReturnCostPercentage),
        np.asarray(customerOrderCount),
        np.asarray(customerRecency)]);

    data_df = pd.DataFrame(newData.transpose())  # Key 1, Convert ndarray format to DataFrame

    # Change the index of the table
    data_df.columns = ['CustomerID', 'Postage', 'Quantity', 'Subtotal', 'Average Quantity','Average Subtotal',
                       'Average Item Cost', 'Order Return Rate','Return Cost','Return Quantity','Return Quantity Percentage','Return Cost Percentage', 'OrderCount','Recency'];

    # Write the file into the excel table
    writer = pd.ExcelWriter(fileName + '.xlsx')
    data_df.to_excel(writer, fileName, float_format='%.5f')
    writer.save()

def createCleanedModelData():

    year1 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2009-2010');
    year2 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2010-2011');

    year1.sort_values(by=["InvoiceDate"],ascending=True);
    year2.sort_values(by=["InvoiceDate"],ascending=True);

    year1 = year1.to_numpy();
    year2 = year2.to_numpy();

    print(f'cleaning data')

    print(len(year1));
    print(len(year2));

    start = time.time();
    getCleanedData(year1,"2009");
    getCleanedData(year2,"2010");
    end = time.time();
    print(end - start);

def trainModels(data):

    linear = SVC(kernel='linear', cache_size=1000,max_iter= 10000000, probability=True)
    poly = SVC(kernel='poly',cache_size=1000,max_iter= 10000000,probability=True)
    rbf = SVC(kernel='rbf',cache_size=1000,max_iter= 10000000,probability=True)
    sigmoid = SVC(kernel='sigmoid',cache_size=1000,max_iter= 10000000, probability=True)
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier(n_estimators=150,min_samples_split=20,min_samples_leaf=10)
    knn = KNeighborsClassifier(n_neighbors=59)

    linearScores = cross_val_score(linear, data[1], data[2], cv=5, scoring='roc_auc')
    polyScores = cross_val_score(poly, data[1], data[2], cv=5, scoring='roc_auc')
    rbfScores = cross_val_score(rbf, data[1], data[2], cv=5, scoring='roc_auc')
    sigmoidScores = cross_val_score(sigmoid, data[1], data[2], cv=5, scoring='roc_auc')
    treeScores = cross_val_score(tree, data[1], data[2], cv=5, scoring='roc_auc')
    forestScores = cross_val_score(forest, data[1], data[2], cv=5, scoring='roc_auc')
    knnScores = cross_val_score(knn, data[1], data[2], cv=5, scoring='roc_auc')

    linearPred = cross_val_predict(linear, data[1], data[2], cv=5)
    polyPred = cross_val_predict(poly, data[1], data[2], cv=5)
    rbfPred = cross_val_predict(rbf, data[1], data[2], cv=5)
    sigmoidPred = cross_val_predict(sigmoid, data[1], data[2], cv=5)
    treePred = cross_val_predict(tree, data[1], data[2], cv=5)
    forestPred = cross_val_predict(forest, data[1], data[2], cv=5)
    knnPred = cross_val_predict(knn, data[1], data[2], cv=5)

    linearConf = confusion_matrix(data[2], linearPred)
    polyConf = confusion_matrix(data[2], polyPred)
    rbfConf = confusion_matrix(data[2], rbfPred)
    sigmoidConf = confusion_matrix(data[2], sigmoidPred)
    treeConf = confusion_matrix(data[2], treePred)
    forestConf = confusion_matrix(data[2], forestPred)
    knnConf = confusion_matrix(data[2], knnPred)

    linearTN, linearFP, linearFN, linearTP = linearConf.ravel()
    polyTN, polyFP, polyFN, polTP = polyConf.ravel()
    rbfTN, rbfFP, rbfFN, rbfTP = rbfConf.ravel()
    sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP = sigmoidConf.ravel()
    treeTN, treeFP, treeFN, treeTP = treeConf.ravel()
    forestTN, forestFP, forestFN, forestTP = forestConf.ravel()
    knnTN, knnFP, knnFN, knnTP = knnConf.ravel()

    linearAccuracy = accuracy_score(data[2], linearPred)
    polyAccuracy = accuracy_score(data[2], polyPred)
    rbfAccuracy = accuracy_score(data[2], rbfPred)
    sigmoidAccuracy = accuracy_score(data[2], sigmoidPred)
    treeAccuracy = accuracy_score(data[2], treePred)
    forestAccuracy = accuracy_score(data[2], forestPred)
    knnAccuracy = accuracy_score(data[2], knnPred)

    linearRow = [data[0], linearAccuracy,np.mean(linearScores), linearTN, linearFP, linearFN, linearTP];
    polyRow = [data[0],polyAccuracy,np.mean(polyScores),polyTN, polyFP, polyFN, polTP]
    rbfRow = [data[0], rbfAccuracy,np.mean(rbfScores),rbfTN, rbfFP, rbfFN, rbfTP ]
    sigmoidRow = [data[0], sigmoidAccuracy, np.mean(sigmoidScores),sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP]
    treeRow = [data[0], treeAccuracy,np.mean(treeScores), treeTN, treeFP, treeFN, treeTP]
    forestRow = [data[0], forestAccuracy,np.mean(forestScores), forestTN, forestFP, forestFN, forestTP]
    knnRow = [data[0], knnAccuracy,np.mean(knnScores), knnTN, knnFP, knnFN, knnTP]

    return [linearRow,polyRow,rbfRow,sigmoidRow,treeRow,forestRow,knnRow]


def main():

    needsDataCleaned = False;

    if needsDataCleaned:
        createCleanedModelData();

    print(f'loading Segmentation Data')
    year1 = pd.read_excel(r'2009.xlsx', sheet_name='2009').to_numpy()
    year2 = pd.read_excel(r'2010.xlsx', sheet_name='2010').to_numpy()

    target = [];
    retention = [];
    for i in year1:
        if i[1] in year2[:,1]:
            retention.append(i[1]);
            target.append(1);
        else:
            target.append(0);

    for i in range(0,len(year1)):
        if year1[i,1] in year2:
            target[i] = 1;

    retentionCount= float( len(retention));

    print( f"Remaining Customers: " + str(retentionCount));
    print( f"Remaining Customers: " + str((retentionCount / float(len(year1)) * 100.0)) + "%");
    print( f"New Customers: " + str( len(year2) - len(retention)) + "");

    scaler = MinMaxScaler()

    dataSet = year1[:,3:15];
    scaler.fit(dataSet);

    pool = ThreadPool(16)
    permutations = [];
    for i in range(1, 4096):
        permutation = []
        test = intToBooleans(i)
        binaryString = ""

        if test[0]:
            permutation.append( dataSet[:,0])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[1]:
            permutation.append( dataSet[:,1])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[2]:
            permutation.append( dataSet[:,2])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[3]:
            permutation.append(  dataSet[:,3])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[4]:
            permutation.append( dataSet[:,4])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[5]:
            permutation.append( dataSet[:,5])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[6]:
            permutation.append( dataSet[:,6])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[7]:
            permutation.append( dataSet[:,7])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[8]:
            permutation.append(dataSet[:, 8])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[9]:
            permutation.append(dataSet[:, 9])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[10]:
            permutation.append(dataSet[:, 10])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[11]:
            permutation.append(dataSet[:, 11])
            binaryString += "1"
        else:
            binaryString += "0"

        permutation = normalize(permutation, axis=0, norm='max').transpose();
        permutations.append( (binaryString[::-1],permutation,target));


    results = pool.map( trainModels, permutations);
    pool.close()
    pool.join()

    names = [];
    indices = []
    accuracies = []
    aucs = []
    tn = []
    fp = []
    fn = []
    tp = []

    #data[0], knnAccuracy, knnTN, knnFP, knnFN, knnTP
    classifierNames = ['linear','poly','rbf','sigmoid','tree','forest','knn']

    for i in range(0, len(results)):
        for j in range(0,len(results[i])):
            names.append(classifierNames[j])
            indices.append(results[i][j][0])
            accuracies.append(results[i][j][1])
            aucs.append(results[i][j][2])
            tn.append(results[i][j][3])
            fp.append(results[i][j][4])
            fn.append(results[i][j][5])
            tp.append(results[i][j][6])

    newData = np.asarray( [
        np.asarray(names),
        np.asarray(indices),
        np.asarray(accuracies),
        np.asarray(aucs ),
        np.asarray(tn ),
        np.asarray(fp),
        np.asarray(fn),
        np.asarray(tp)]);

    data_df = pd.DataFrame(newData.transpose())

    data_df.columns = ['Classifier','Index','Accuracy','AUC','TN', 'FP','FN','TP'];

    writer = pd.ExcelWriter("TrainingData2.xlsx")
    data_df.to_excel(writer, "Training Data", float_format='%.5f')
    writer.save();

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
