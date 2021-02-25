
import numpy as np
import pandas as pd
import time
import re
from matplotlib import colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import xlrd
import sklearn
import matplotlib.pyplot as plt
from pandas import read_excel

def containsDigit(string):
    for i in string:
        if i.isdigit():
            return True;
    return False;


def getCategoryClassification():

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
            if (int)(i[6]) not in customerIDs:
                customerIDs.append(i[6]);
        itemID = str(i[0]);
        if containsDigit( str(i[0])) and itemID not in itemIDs:
            itemIDs.append(itemID);
            itemDescriptions.append( ' '.join( [word for word in str(i[2]).replace("/"," ").split() if word.lower() not in mcolors.cnames.keys()]));

    #train K-Means model for item category classification
    #clusterCount = 12;
    #wordVectorizor = TfidfVectorizer(stop_words='english');
    #X = wordVectorizor.fit_transform(itemDescriptions);
    #model = KMeans( n_clusters=clusterCount, init='k-means++', max_iter=250, n_init=1);
    #model.fit(X);

    #print("Top terms per cluster:")
    #order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    #terms = wordVectorizor.get_feature_names()
    #for i in range(clusterCount):
    #    print("Cluster %d:" % i),
    #    for ind in order_centroids[i, :10]:
    #        print(' %s' % terms[ind]),
    #    print
    #print("\n")


    customerPostage = np.array( [0] * len(customerIDs));
    customerQuantity = np.array([0] * len(customerIDs));
    customerSubtotal = np.array([0.0] * len(customerIDs));
    customerAverageQuantity = np.array([0] * len(customerIDs));
    customerAverageSubtotal = np.array([0.0] * len(customerIDs));
    customerAverageItemCost = np.array([0.0] * len(customerIDs));
    customerReturnRates = np.array([0.0] * len(customerIDs));
    customerOrderCount = np.array([0] * len(customerIDs));
    #customerItemCategory = np.array([0] * len(customerIDs));

    orderNumber = 0;

    for i in range(0, len(customerIDs)):
        customerOrderNumbers = [];
        customerReturnOrder = [];
        #customerItemCategories = [];
        for j in data:
            if j[6] == customerIDs[i]:
                if not str(j[0]).startswith("C"):
                    customerQuantity[i] += j[3];
                    customerSubtotal[i] += j[5] * j[3];
                    #description = [' '.join( [word for word in str(j[2]).replace("/", " ") if word.lower() not in mcolors.cnames.keys() ])];
                    #itemDescriptionVector = wordVectorizor.transform(description);
                    #customerItemCategories.append(model.predict(itemDescriptionVector))

                    if j[0] not in customerOrderNumbers:
                        customerOrderNumbers.append(orderNumber);
                        customerPostage[i] = mediaPostageCosts[countryNames.index(j[7])];
                else:
                    if j[0] not in customerReturnOrder:
                        customerReturnOrder.append(j[0]);
                    orderNumber = (int)(re.sub("[^0-9]", "", str(j[0])));
                    if orderNumber not in customerOrderNumbers:
                        customerQuantity[i] += abs( j[3]);
                        customerSubtotal[i] += abs(j[5]) * abs(j[3]);
                        customerPostage[i] = mediaPostageCosts[countryNames.index(j[7])];
                        #description = [' '.join( [word for word in str(j[2]).replace("/", " ") if word.lower() not in mcolors.cnames.keys()])];

                        #itemDescriptionVector = wordVectorizor.transform(description);
                        #customerItemCategories.append(model.predict(itemDescriptionVector))

        for j in customerReturnOrder:
            if j not in customerOrderNumbers:
                customerOrderNumbers.append( orderNumber);

        customerReturnRates[i] = len(customerReturnOrder) / len(customerOrderNumbers);
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
        customerOrderCount]);

    data_df = pd.DataFrame(newData.transpose())  # Key 1, Convert ndarray format to DataFrame

    # Change the index of the table
    data_df.columns = ['CustomerID', 'Postage', 'Quantity', 'Subtotal', 'Average Quantity','Average Subtotal',
                       'Average Item Cost', 'Order Return Rate', 'OrderCount'];

    # Write the file into the excel table
    writer = pd.ExcelWriter(fileName + '.xlsx')  # Key 2, create an excel sheet named hhh
    data_df.to_excel(writer, fileName, float_format='%.5f')  # Key 3, float_format controls the accuracy, write data_df to the first page of the hhh form. If there are multiple files, you can write in page_2
    writer.save()  # Key 4

def createCleanedModelData():

    year1 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2009-2010');
    year2 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2010-2011');

    year1.sort_values(by="InvoiceDate");
    year2.sort_values(by="InvoiceDate");

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

def main():

    needsDataCleaned = True;

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





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
