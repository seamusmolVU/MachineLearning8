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
   binaryString = format(number, '08b');
   return [x == '1' for x in binaryString[::-1]];

def getCleanedData( yearData, fileName):

    countryNames =['United Kingdom', 'France', 'USA', 'Belgium', 'Australia', 'EIRE', 'Germany', 'Portugal', 'Japan', 'Denmark',
     'Nigeria', 'Netherlands', 'Poland', 'Spain', 'Channel Islands', 'Italy', 'Cyprus', 'Greece', 'Norway', 'Austria',
     'Sweden', 'United Arab Emirates', 'Finland', 'Switzerland', 'Malta', 'Bahrain', 'RSA', 'Bermuda',
     'Hong Kong', 'Singapore', 'Thailand', 'Israel', 'Lithuania', 'West Indies', 'Lebanon', 'Korea', 'Brazil', 'Canada',
     'Iceland']

    mediaPostageCosts =[5.0,18.0,135.5,15.0,305.0,25.0,18.0,28.0,180.0,20.0,75.0,15.0,40.0,28.0,20.0,28.0,15.0,50.0,40.0,40.0,
     40.0,37.5,40.0,40.0,52.5,40.0,150.0,130.0,160.0,170.0,170.0,45.0,40.0,130.0, 39.67,180.0,170.0,130.0,40.0]

    itemIDs = []
    itemDescriptions = []

    data = []
    customerIDs = [];
    for i in yearData:
        if i[5] > 0.0 and i[7] in countryNames and not np.isnan( i[6]) and containsDigit( str(i[1])):
            data.append(i);
            if (int)(i[6]) not in customerIDs and i[3] > 0:
                customerIDs.append(i[6]);
        itemID = str(i[0]);
        if containsDigit( str(i[0])) and itemID not in itemIDs:
            itemIDs.append(itemID);
            itemDescriptions.append( ' '.join( [word for word in str(i[2]).replace("/"," ").split() if word.lower() not in mcolors.cnames.keys()]));

    customerPostage = np.array( [0.0] * len(customerIDs));
    customerQuantity = np.array([0.0] * len(customerIDs));
    customerSubtotal = np.array([0.0] * len(customerIDs));
    customerAverageQuantity = np.array([0.0] * len(customerIDs));
    customerAverageSubtotal = np.array([0.0] * len(customerIDs));
    customerAverageItemCost = np.array([0.0] * len(customerIDs));
    customerReturnRates = np.array([0.0] * len(customerIDs));
    customerOrderCount = np.array([0.0] * len(customerIDs));
    customerRecency = np.array([0.0] * len(customerIDs));
    orderNumber = 0;

    for i in range(0, len(customerIDs)):
        customerOrderNumbers = [];
        customerOrderDates = [];
        customerReturnOrders = [];

        #return order is not same ID as purchase order
        for j in data:
            if j[6] == customerIDs[i]:
                if not str(j[0]).startswith("C"):
                    customerQuantity[i] += j[3];
                    customerSubtotal[i] += j[5] * j[3];

                    if j[0] not in customerOrderNumbers:
                        customerOrderNumbers.append(orderNumber);
                        customerPostage[i] = mediaPostageCosts[countryNames.index(j[7])];
                        customerOrderDates.append(j[4]);
                else:
                    if j[0] not in customerReturnOrders:
                        customerReturnOrders.append(j[0]);
                        #customerOrderNumbers.append(j[0]);

        customerPostageCosts = []
        for j in yearData:
            if j[6] == customerIDs[i] and str(j[1]).startswith("POST"):
                customerPostage[i] += abs(j[3]);
        if len(customerPostageCosts) > 0:
            customerPostage[i] /= len(customerPostageCosts);


        if len(customerOrderDates) >= 2:
            for j in range(0,len(customerOrderDates)-1):
                daysDelta = (customerOrderDates[j+1].to_pydatetime() - customerOrderDates[j].to_pydatetime()).days;
                customerRecency[i] = customerRecency[i] + daysDelta;
            customerRecency[i] = customerRecency[i] / (len(customerOrderDates)-1);
        else:
            customerRecency[i] = 366;

        customerReturnRates[i] = len(customerReturnOrders) / len(customerOrderNumbers);
        customerOrderCount[i] = len(customerOrderNumbers);
        customerAverageQuantity[i] = customerQuantity[i] / len(customerOrderNumbers);
        customerAverageSubtotal[i] = customerSubtotal[i] / len(customerOrderNumbers);
        customerAverageItemCost[i] = customerSubtotal[i] / len(customerQuantity);

    dataSize = len(customerIDs);

    newData = np.asarray([
        customerIDs,
        customerPostage,
        customerQuantity,
        customerSubtotal,
        customerAverageQuantity,
        customerAverageSubtotal,
        customerAverageItemCost,
        customerReturnRates,
        customerOrderCount,
        customerRecency]);

    data_df = pd.DataFrame(newData.transpose())  # Key 1, Convert ndarray format to DataFrame

    # Change the index of the table
    data_df.columns = ['CustomerID', 'Postage', 'Quantity', 'Subtotal', 'Average Quantity','Average Subtotal',
                       'Average Item Cost', 'Order Return Rate', 'OrderCount','Recency'];

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

    #x_train, x_test, y_train, y_test = train_test_split(data[1], data[2], test_size=0.2);

    linear = SVC(kernel='linear',tol=0.025,probability=True)
    poly = SVC(kernel='poly',tol=0.025,probability=True)
    rbf = SVC(kernel='rbf',tol=0.025,probability=True)
    sigmoid = SVC(kernel='sigmoid', tol=0.025, probability=True)
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, min_samples_split=10, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=25)

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

    customerRetention = list( set(year1[:,1]) & set(year2[:,1]));
    retentionCount= float( len(customerRetention));

    print( f"Remaining Customers: " + str(retentionCount));
    print( f"Remaining Customers: " + str((retentionCount / float(len(year1)) * 100.0)) + "%");
    print( f"New Customers: " + str( len(year2) - len(customerRetention)) + "");

    target = np.zeros( len(year1));
    for i in range(0,len(target)):
        if year1[i,1] in customerRetention:
            target[i] = 1;


    pool = ThreadPool(16)
    permutations = [];
    for i in range(1, 16):
        permutation = []
        test = intToBooleans(i)
        binaryString = ""

        if test[0]:
            permutation.append( year1[:,3])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[1]:
            permutation.append( year1[:, 4])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[2]:
            permutation.append( year1[:,5])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[3]:
            permutation.append(  year1[:,6])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[4]:
            permutation.append( year1[:,7])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[5]:
            permutation.append(year1[:,8])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[6]:
            permutation.append( year1[:,9])
            binaryString += "1"
        else:
            binaryString += "0"
        if test[7]:
            permutation.append( year1[:,10])
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
            for k in range(0,7):
                names.append(classifierNames[k])
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

    writer = pd.ExcelWriter("TrainingData3.xlsx")
    data_df.to_excel(writer, "Training Data3", float_format='%.5f')
    writer.save();

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
