from concurrent.futures import thread

import numpy as np
import pandas as pd
import time
import threading
import re
from datetime import date
import queue as queue
from matplotlib import colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from mlxtend.plotting import plot_decision_regions

from multiprocessing.dummy import Pool as ThreadPool

from sklearn.tree import DecisionTreeClassifier


def containsDigit(string):
    for i in string:
        if i.isdigit():
            return True;
    return False;


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

    year1.sort_values(by=["Customer ID","InvoiceDate"],ascending=False);
    year2.sort_values(by=["Customer ID","InvoiceDate"],ascending=False);

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


def trainLinearModel( data ):
    # index, dataSet, target
    x_train, x_test, y_train, y_test = train_test_split(data[1], data[2], test_size=0.5);
    # linear test
    linear = SVC(kernel='linear')
    linear.fit(x_train, y_train);
    y_predicted = linear.predict(x_test)

    print(str(data[0]) + str(accuracy_score(y_test, y_predicted)));
    return (data[0], accuracy_score(y_test, y_predicted))

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

    dataSet = year1[:,3:10];
    linearResults = queue.Queue();

    """"
    pool = ThreadPool(16)
    permutations = [];
    for i in range(0, 512):
        permutation = []
        if i % 1 == 0:
            permutation.append( year1[:,3])
        if i % 2 == 0:
            permutation.append( year1[:, 4])
        if i % 4 == 0:
            permutation.append( year1[:,5])
        if i % 8 == 0:
            permutation.append( year1[:,6])
        if i % 16 == 0:
            permutation.append( year1[:,7])
        if i % 32 == 0:
            permutation.append( year1[:, 8])
        if i % 64 == 0:
            permutation.append( year1[:, 9])
        if i % 128 == 0:
            permutation.append( year1[:,10])

        permutation = np.asarray(permutation).transpose();

        permutations.append( (i,permutation,target));

    results = pool.map( trainLinearModel, permutations);
    pool.close()
    pool.join()

    for i in results:
        print(i);

    """

    # index, dataSet, target
    x_train, x_test, y_train, y_test = train_test_split(dataSet, target, test_size=0.5);
    # linear test
    linear = SVC(kernel='linear')
    linear.fit(x_train, y_train);
    y_predicted = linear.predict(x_test)

    print( accuracy_score(y_test, y_predicted));
    #plot_decision_regions(x_test[:500], y_test.astype(np.integer)[:500], clf=linear, res=0.1);

    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)

    y_predicted = tree.predict(x_test)
    print(accuracy_score(y_test, y_predicted))

    knn = KNeighborsClassifier(15)  # We set the number of neighbors to 15
    knn.fit(x_train, y_train)

    y_predicted = knn.predict(x_test)
    print('accuracy ', accuracy_score(y_test, y_predicted))


    #plot_decision_regions(x_test[:500], y_test.astype(np.integer)[:500], clf=linear, res=0.1);


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
