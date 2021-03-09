
from multiprocessing.pool import ThreadPool
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

#TODO Summary of project
#

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

    # Create list of transactions with valid Customer ID, Country, and Invoice number.
    # Convert transactions to list of costumers
    data = []
    customerIDs = [];
    for i in yearData:
        if i[5] > 0.0 and i[7] in countryNames and not np.isnan(i[6]) and containsDigit(str(i[1])):
            data.append(i);
            if (int)(i[6]) not in customerIDs and i[3] != 0:
                customerIDs.append(i[6]);

    # Lists of all features obtained from transaction data.
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

    # Find all transactions relating to a valid customer ID.
    for i in range(0, len(customerIDs)):
        customerOrderNumbers = [];
        customerOrderDates = [];
        customerReturnOrders = [];
        for j in data:
            if j[6] == customerIDs[i]:
                # Return orders start with C in invoice number.
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

        # Calculate features from combination of other features
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

    # Set the headers for the Columns.
    data_df.columns = ['CustomerID', 'Postage', 'Quantity', 'Subtotal', 'Average Quantity','Average Subtotal',
                       'Average Item Cost', 'Order Return Rate','Return Cost','Return Quantity','Return Quantity Percentage','Return Cost Percentage', 'OrderCount','Recency'];

    # Write the file into the excel file
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

    linearTN, linearFP, linearFN, linearTP = confusion_matrix(data[2], linearPred).ravel()
    polyTN, polyFP, polyFN, polTP = confusion_matrix(data[2], polyPred).ravel()
    rbfTN, rbfFP, rbfFN, rbfTP = confusion_matrix(data[2], rbfPred).ravel()
    sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP = confusion_matrix(data[2], sigmoidPred).ravel()
    treeTN, treeFP, treeFN, treeTP = confusion_matrix(data[2], treePred).ravel()
    forestTN, forestFP, forestFN, forestTP = confusion_matrix(data[2], forestPred).ravel()
    knnTN, knnFP, knnFN, knnTP = confusion_matrix(data[2], knnPred).ravel()

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


def tuneModels(data):
    # Split data into 80% train, 20% test
    x_train, x_test, y_train, y_test = train_test_split(data[1], data[2], test_size= 0.2)

    # Create normal Models
    linear = SVC(kernel='linear', cache_size=1000, max_iter=10000000, probability=True)
    poly = SVC(kernel='poly', cache_size=1000, max_iter=10000000, probability=True)
    rbf = SVC(kernel='rbf', cache_size=1000, max_iter=10000000, probability=True)
    sigmoid = SVC(kernel='sigmoid', cache_size=1000, max_iter=10000000, probability=True)
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier(n_estimators=150, min_samples_split=20, min_samples_leaf=10)
    knn = KNeighborsClassifier(n_neighbors=59)

    # Obtain AUC score using 5 fold validation on Test data.
    linearScores = cross_val_score(linear, x_train, y_train, cv=5, scoring='roc_auc')
    polyScores = cross_val_score(poly, x_train, y_train, cv=5, scoring='roc_auc')
    rbfScores = cross_val_score(rbf, x_train, y_train, cv=5, scoring='roc_auc')
    sigmoidScores = cross_val_score(sigmoid, x_train, y_train, cv=5, scoring='roc_auc')
    treeScores = cross_val_score(tree, x_train, y_train, cv=5, scoring='roc_auc')
    forestScores = cross_val_score(forest, x_train, y_train, cv=5, scoring='roc_auc')
    knnScores = cross_val_score(knn, x_train, y_train, cv=5, scoring='roc_auc')

    # Predict using normal models
    linearPred = linear.predict(x_test);
    polyPred = poly.predict(x_test);
    rbfPred = rbf.predict(x_test);
    sigmoidPred = sigmoid.predict(x_test);
    treePred = tree.predict(x_test);
    forestPred = forest.predict(x_test);
    knnPred = knn.predict(x_test);

    # Get ROC for all normal models
    linearTN, linearFP, linearFN, linearTP = confusion_matrix(y_test, linearPred).ravel()
    polyTN, polyFP, polyFN, polTP = confusion_matrix(y_test, polyPred).ravel()
    rbfTN, rbfFP, rbfFN, rbfTP = confusion_matrix(y_test, rbfPred).ravel()
    sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP = confusion_matrix(y_test, sigmoidPred).ravel()
    treeTN, treeFP, treeFN, treeTP = confusion_matrix(y_test, treePred).ravel()
    forestTN, forestFP, forestFN, forestTP = confusion_matrix(y_test, forestPred).ravel()
    knnTN, knnFP, knnFN, knnTP = confusion_matrix(y_test, knnPred).ravel()

    # Get normal models
    linearAccuracy = accuracy_score(y_test, linearPred)
    polyAccuracy = accuracy_score(y_test, polyPred)
    rbfAccuracy = accuracy_score(y_test, rbfPred)
    sigmoidAccuracy = accuracy_score(y_test, sigmoidPred)
    treeAccuracy = accuracy_score(y_test, treePred)
    forestAccuracy = accuracy_score(y_test, forestPred)
    knnAccuracy = accuracy_score(y_test, knnPred)

    # HyperParameters for fine tuning
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    svcParams = {'C': Cs, 'gamma': gammas}
    knnParams = {'knn__n_neighbors': [x % 2 == 1 for x in range(3, 99)]}

    # Get parameters for tree,forest,knn hyperparameter tuning
    treeParams = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150]
    }
    forestParams = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 250, 500, 1000]
    }

    # Create Grid Search Models
    linearGridSearch = GridSearchCV(sklearn.svm.SVC(kernel='linear', cache_size=1000, max_iter=10000000, probability=True), param_grid=svcParams, scoring='roc_auc',cv=5)
    polyGridSearch = GridSearchCV(sklearn.svm.SVC(kernel='poly', cache_size=1000, max_iter=10000000, probability=True),param_grid=svcParams,scoring='roc_auc', cv=5)
    rbfGridSearch = GridSearchCV(sklearn.svm.SVC(kernel='rbf', cache_size=1000, max_iter=10000000, probability=True),param_grid=svcParams,scoring='roc_auc', cv=5)
    sigmoidGridSearch = GridSearchCV(sklearn.svm.SVC(kernel='sigmoid', cache_size=1000, max_iter=10000000, probability=True), param_grid=svcParams,scoring='roc_auc',cv=5)
    treeGridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid=treeParams,scoring='roc_auc', cv=5)
    forestGridSearch = GridSearchCV(RandomForestClassifier(), param_grid=forestParams,scoring='roc_auc', cv=5)
    knnGridSearch = GridSearchCV(KNeighborsClassifier(), param_grid=knnParams,scoring='roc_auc', cv=5)

    # Fit all GridSearch Models
    linearGridSearch.fit(x_train, y_train);
    polyGridSearch.fit(x_train, y_train);
    rbfGridSearch.fit(x_train, y_train);
    sigmoidGridSearch.fit(x_train, y_train);
    treeGridSearch.fit(x_train, y_train);
    forestGridSearch.fit(x_train, y_train);
    knnGridSearch.fit(x_train, y_train);

    # Get Predictions from all GridSearch Models
    linearSearchPred = linearGridSearch.best_estimator_.predict(x_test);
    polySearchPred = polyGridSearch.best_estimator_.predict(x_test);
    rbfSearchPred = rbfGridSearch.best_estimator_.predict(x_test);
    sigmoidSearchPred = sigmoidGridSearch.best_estimator_.predict(x_test);
    treeSearchPred = treeGridSearch.best_estimator_.predict(x_test);
    forestSearchPred = forestGridSearch.best_estimator_.predict(x_test);
    knnSearchPred = knnGridSearch.best_estimator_.predict(x_test);

    # Get ROC for all GridSearch Models
    linearSearchTN, linearSearchFP, linearSearchFN, linearSearchTP = confusion_matrix(y_test, linearSearchPred).ravel()
    polySearchTN, polySearchFP, polySearchFN, polSearchTP = confusion_matrix(y_test, polySearchPred).ravel()
    rbfSearchTN, rbfSearchFP, rbfSearchFN, rbSearchfTP = confusion_matrix(y_test, rbfSearchPred).ravel()
    sigmoidSearchTN, sigmoidSearchFP, sigmoidSearchFN, sigmoidSearchTP = confusion_matrix(y_test, sigmoidSearchPred).ravel()
    treeSearchTN, treeSearchFP, treeSearchFN, treeSearchTP = confusion_matrix(y_test, treeSearchPred).ravel()
    forestSearchTN, forestSearchFP, forestSearchFN, forestSearchTP = confusion_matrix(y_test, forestSearchPred).ravel()
    knnSearchTN, knnSearchFP, knnSearchFN, knnSearchTP = confusion_matrix(y_test, knnSearchPred).ravel()

    # Get Accuracy Scores for all GridSearch Models
    linearSearchAccuracy = accuracy_score(y_test,linearSearchPred);
    polySearchAccuracy = accuracy_score(y_test, polySearchPred);
    rbfSearchAccuracy = accuracy_score(y_test, rbfSearchPred);
    sigmoidSearchAccuracy = accuracy_score(y_test, sigmoidSearchPred);
    treeSearchAccuracy = accuracy_score(y_test, treeSearchPred);
    forestSearchAccuracy = accuracy_score(y_test, forestPred);
    knnSearchAccuracy = accuracy_score(y_test, knnSearchPred);

    # Get AUC scores for all GridSearch Models
    linearSearchAUC = linearGridSearch.best_score_;
    polySearchAUC = polyGridSearch.best_score_;
    rbfSearchAUC = rbfGridSearch.best_score_;
    sigmoidSearchAUC = sigmoidGridSearch.best_score_;
    treeSearchAUC = treeGridSearch.best_score_;
    forestSearchAUC = forestGridSearch.best_score_;
    knnSearchAUC = knnGridSearch.best_score_;

    linearRow = [data[0], linearAccuracy, np.mean(linearScores), linearTN, linearFP, linearFN, linearTP, linearSearchAccuracy,linearSearchAUC,linearSearchTN, linearSearchFP, linearSearchFN, linearSearchTP];
    polyRow = [data[0], polyAccuracy, np.mean(polyScores), polyTN, polyFP, polyFN, polTP,polySearchAccuracy,polySearchAUC,polySearchTN, polySearchFP, polySearchFN, polSearchTP]
    rbfRow = [data[0], rbfAccuracy, np.mean(rbfScores), rbfTN, rbfFP, rbfFN, rbfTP,rbfSearchAccuracy,rbfSearchAUC,rbfSearchTN, rbfSearchFP, rbfSearchFN, rbSearchfTP]
    sigmoidRow = [data[0], sigmoidAccuracy, np.mean(sigmoidScores), sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP,sigmoidSearchAccuracy,sigmoidSearchAUC,sigmoidSearchTN, sigmoidSearchFP, sigmoidSearchFN, sigmoidSearchTP]
    treeRow = [data[0], treeAccuracy, np.mean(treeScores), treeTN, treeFP, treeFN, treeTP,treeSearchAccuracy,treeSearchAUC,treeSearchTN, treeSearchFP, treeSearchFN, treeSearchTP]
    forestRow = [data[0], forestAccuracy, np.mean(forestScores), forestTN, forestFP, forestFN, forestTP,forestSearchAccuracy,forestSearchAUC,forestSearchTN, forestSearchFP, forestSearchFN, forestSearchTP]
    knnRow = [data[0], knnAccuracy, np.mean(knnScores), knnTN, knnFP, knnFN, knnTP,knnSearchAccuracy,knnSearchAUC,knnSearchTN, knnSearchFP, knnSearchFN, knnSearchTP]

    return [linearRow, polyRow, rbfRow, sigmoidRow, treeRow, forestRow, knnRow]

def main():
    needsDataCleaned = False;
    needsFeatureSelection = False;
    needsHyperTuning = True;

    if needsDataCleaned:
        createCleanedModelData();

    #Loads Excel sheets from cleaning process
    print(f'loading Cleaned Data')
    year1 = pd.read_excel(r'2009.xlsx', sheet_name='2009').to_numpy()
    year2 = pd.read_excel(r'2010.xlsx', sheet_name='2010').to_numpy()

    if needsFeatureSelection:
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

        # Apply Preprocessing to reduce training time. Normalizes each feature to the Range [0,1]
        scaler = MinMaxScaler();
        dataSet = year1[:,3:15];
        scaler.fit(dataSet);

        # Create all different combinations of the 12 features.
        permutations = [];
        for i in range(1, 4096):
            permutation = []
            test = intToBooleans(i)
            binaryString = ""

            for j in range(0,12):
                if test[j]:
                    permutation.append(dataSet[:, j])
                    binaryString += "1"
                else:
                    binaryString += "0"

            permutation = normalize(permutation, axis=0, norm='max').transpose();
            permutations.append( (binaryString[::-1],permutation,target));

        # Create ThreadPool for MultiThreading
        pool = ThreadPool(16)
        results = pool.map( trainModels, permutations);
        pool.close()
        pool.join()

        # Convert Feature Selection data to 2D array and export into Excel File
        names = [];
        indices = []
        accuracies = []
        aucs = []
        tn = []
        fp = []
        fn = []
        tp = []

        # data[0], knnAccuracy, knnTN, knnFP, knnFN, knnTP
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
        writer = pd.ExcelWriter("TrainingData.xlsx")
        data_df.to_excel(writer, "Training Data", float_format='%.5f')
        writer.save();

    if needsHyperTuning:
        # Top 16 features obtained from Feature selection
        features = ['011000111010',
            '011000001010',
            '011110000101',
            '010110101010',
            '011100001010',
            '011000011010',
            '010011101010',
            '010010111010',
            '010000111110',
            '011111001010',
            '010001111010',
            '011101111010',
            '011111101010',
            '011010111010',
            '010000100101',
            '011110001010'];

        tuningFeatures = [];
        binaryString = "";
        for i in features:
            permutation = [];
            hasFeatures = [i == '1' for i in features[i]];
            for j in range(0,len(hasFeatures)):
                if hasFeatures[j]:
                    permutation.append(dataSet[:, j])
                    binaryString += "1"
                else:
                    binaryString += "0"
            tuningFeatures.append((binaryString[::-1],permutation,target));
        print(binaryString);

        # run 16 threads for hyperparameter tuning
        pool = ThreadPool(16)
        results = pool.map(tuneModels, tuningFeatures);
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
        tunedAccuracies = []
        tunedAUCS = []
        tunedTN = []
        tunedFP = []
        tunedFN = []
        tunedTP = []

        # Convert Results to 2D numpy array
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
                tunedAccuracies.append(results[i][j][7]);
                tunedAUCS.append(results[i][j][8]);
                tunedTN.append(results[i][j][9]);
                tunedFP.append(results[i][j][10]);
                tunedFN.append(results[i][j][11]);
                tunedTP.append(results[i][j][12]);

        newData = np.asarray( [
            np.asarray(names),
            np.asarray(indices),
            np.asarray(accuracies),
            np.asarray(aucs ),
            np.asarray(tn ),
            np.asarray(fp),
            np.asarray(fn),
            np.asarray(tp),
            np.asarray(tunedAccuracies),
            np.asarray(tunedAUCS),
            np.asarray(tunedTN),
            np.asarray(tunedFP),
            np.asarray(tunedFN),
            np.asarray(tunedTP)
        ]);

        data_df = pd.DataFrame(newData.transpose())

        data_df.columns = ['Classifier','Index','Accuracy','AUC','TN', 'FP','FN','TP', 'Tuned Accuracy','Tuned AUC', 'Tuned TN','TunedFP','Tuned FN','Tuned TP'];
        writer = pd.ExcelWriter("TunedFeatures.xlsx")
        data_df.to_excel(writer, "Tuned Features", float_format='%.5f')
        writer.save();


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
