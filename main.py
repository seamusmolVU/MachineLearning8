
from multiprocessing.pool import ThreadPool
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

# This project consists of several components:
# The project loads customers segmentation data of individual transactions and creates in return an excel sheet
# containing a dataset of valid customers with the following features:
# Postage
# Quantity
# Subtotal
# Average Quantity
# Average Subtotal
# Average Item Cost
# Order Return Rate
# Return Cost
# Return Quantity
# Return Quantity Percentage
# Return Cost Percentage
# OrderCount
# Recency

# Next the intersection of the first year and second year datasets are used to obtain the retention of customers.
# The first year and retention datasets are used to conduct Feature selection.
# This consists of running everyone combination of the features(except for postage due to 99% class imbalance)

# The 16 optimal feature combinations are used for hyperparameter tuning.
# The data is broken into 60%,20%,20% for training,validation,testing
# Hypermeter tuning uses training and tests on validation
# The final run uses training+validation and tests on the test data.

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

    # Split into 60% 20% 20% training,validation,test sets

    # split into test and train
    x_train, x_test, y_train, y_test = train_test_split(data[1], data[2], test_size=0.2, random_state=1)

    # split into train and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)

    linear = SVC(kernel='linear', cache_size=1000,max_iter= 10000000, probability=True)
    poly = SVC(kernel='poly',cache_size=1000,max_iter= 10000000,probability=True)
    rbf = SVC(kernel='rbf',cache_size=1000,max_iter= 10000000,probability=True)
    sigmoid = SVC(kernel='sigmoid',cache_size=1000,max_iter= 10000000, probability=True)
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier(n_estimators=150,min_samples_split=20,min_samples_leaf=10)
    knn = KNeighborsClassifier(n_neighbors=59)

    linear.fit(x_train, y_train)
    poly.fit(x_train, y_train)
    rbf.fit(x_train, y_train)
    sigmoid.fit(x_train, y_train)
    tree.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    linearScores = roc_auc_score(y_val, linear.predict_proba(x_val)[:,1]);
    polyScores = roc_auc_score(y_val, poly.predict_proba(x_val)[:,1]);
    rbfScores = roc_auc_score(y_val, rbf.predict_proba(x_val)[:,1]);
    sigmoidScores = roc_auc_score(y_val, sigmoid.predict_proba(x_val)[:,1]);
    treeScores = roc_auc_score(y_val, tree.predict_proba(x_val)[:,1]);
    forestScores = roc_auc_score(y_val, forest.predict_proba(x_val)[:,1]);
    knnScores = roc_auc_score(y_val, knn.predict_proba(x_val)[:,1]);

    linearPred = linear.predict(x_val)
    polyPred = poly.predict(x_val)
    rbfPred = rbf.predict(x_val)
    sigmoidPred = sigmoid.predict(x_val)
    treePred = tree.predict(x_val)
    forestPred = forest.predict(x_val)
    knnPred = knn.predict(x_val)

    linearTN, linearFP, linearFN, linearTP = confusion_matrix(y_val, linearPred).ravel()
    polyTN, polyFP, polyFN, polTP = confusion_matrix(y_val, polyPred).ravel()
    rbfTN, rbfFP, rbfFN, rbfTP = confusion_matrix(y_val, rbfPred).ravel()
    sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP = confusion_matrix(y_val, sigmoidPred).ravel()
    treeTN, treeFP, treeFN, treeTP = confusion_matrix(y_val, treePred).ravel()
    forestTN, forestFP, forestFN, forestTP = confusion_matrix(y_val, forestPred).ravel()
    knnTN, knnFP, knnFN, knnTP = confusion_matrix(y_val, knnPred).ravel()

    linearAccuracy = accuracy_score(y_val, linearPred)
    polyAccuracy = accuracy_score(y_val, polyPred)
    rbfAccuracy = accuracy_score(y_val, rbfPred)
    sigmoidAccuracy = accuracy_score(y_val, sigmoidPred)
    treeAccuracy = accuracy_score(y_val, treePred)
    forestAccuracy = accuracy_score(y_val, forestPred)
    knnAccuracy = accuracy_score(y_val, knnPred)

    linearRow = [data[0], linearAccuracy,np.mean(linearScores), linearTN, linearFP, linearFN, linearTP];
    polyRow = [data[0],polyAccuracy,np.mean(polyScores),polyTN, polyFP, polyFN, polTP]
    rbfRow = [data[0], rbfAccuracy,np.mean(rbfScores),rbfTN, rbfFP, rbfFN, rbfTP ]
    sigmoidRow = [data[0], sigmoidAccuracy, np.mean(sigmoidScores),sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP]
    treeRow = [data[0], treeAccuracy,np.mean(treeScores), treeTN, treeFP, treeFN, treeTP]
    forestRow = [data[0], forestAccuracy,np.mean(forestScores), forestTN, forestFP, forestFN, forestTP]
    knnRow = [data[0], knnAccuracy,np.mean(knnScores), knnTN, knnFP, knnFN, knnTP]

    return [linearRow,polyRow,rbfRow,sigmoidRow,treeRow,forestRow,knnRow]


def tuneModelsAndGetFinalRun(data):
    # Split data into 60% train,20% validation, 20% test
    # split into test and train
    x_whole_train, x_test, y_whole_train, y_test = train_test_split( data[1], data[2], test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_whole_train, y_whole_train, test_size=0.25, random_state=1)

    # Create normal Models
    linear = SVC(kernel='linear', cache_size=1000, max_iter=10000000, probability=True)
    poly = SVC(kernel='poly', cache_size=1000, max_iter=10000000, probability=True)
    rbf = SVC(kernel='rbf', cache_size=1000, max_iter=10000000, probability=True)
    sigmoid = SVC(kernel='sigmoid', cache_size=1000, max_iter=10000000, probability=True)
    tree = DecisionTreeClassifier()
    forest = RandomForestClassifier(n_estimators=150, min_samples_split=20, min_samples_leaf=10)
    knn = KNeighborsClassifier(n_neighbors= 59)

    linear.fit(x_train, y_train)
    poly.fit(x_train, y_train)
    rbf.fit(x_train, y_train)
    sigmoid.fit(x_train, y_train)
    tree.fit(x_train, y_train)
    forest.fit(x_train, y_train)
    knn.fit(x_train, y_train)

    linearPred = linear.predict(x_val)
    polyPred = poly.predict(x_val)
    rbfPred = rbf.predict(x_val)
    sigmoidPred = sigmoid.predict(x_val)
    treePred = tree.predict(x_val)
    forestPred = forest.predict(x_val)
    knnPred = knn.predict(x_val)

    # Obtain AUC score
    linearScores = roc_auc_score (y_val, linear.predict_proba(x_val)[:,1]);
    polyScores = roc_auc_score (y_val, poly.predict_proba(x_val)[:,1]);
    rbfScores = roc_auc_score (y_val, rbf.predict_proba(x_val)[:,1]);
    sigmoidScores = roc_auc_score (y_val, sigmoid.predict_proba(x_val)[:,1]);
    treeScores = roc_auc_score (y_val, tree.predict_proba(x_val)[:,1]);
    forestScores = roc_auc_score (y_val, forest.predict_proba(x_val)[:,1]);
    knnScores = roc_auc_score (y_val, knn.predict_proba(x_val)[:,1]);

    # Get ROC for all normal models
    linearTN, linearFP, linearFN, linearTP = confusion_matrix(y_val, linearPred).ravel()
    polyTN, polyFP, polyFN, polTP = confusion_matrix(y_val, polyPred).ravel()
    rbfTN, rbfFP, rbfFN, rbfTP = confusion_matrix(y_val, rbfPred).ravel()
    sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP = confusion_matrix(y_val, sigmoidPred).ravel()
    treeTN, treeFP, treeFN, treeTP = confusion_matrix(y_val, treePred).ravel()
    forestTN, forestFP, forestFN, forestTP = confusion_matrix(y_val, forestPred).ravel()
    knnTN, knnFP, knnFN, knnTP = confusion_matrix(y_val, knnPred).ravel()

    # Get normal models
    linearAccuracy = accuracy_score(y_val, linearPred)
    polyAccuracy = accuracy_score(y_val, polyPred)
    rbfAccuracy = accuracy_score(y_val, rbfPred)
    sigmoidAccuracy = accuracy_score(y_val, sigmoidPred)
    treeAccuracy = accuracy_score(y_val, treePred)
    forestAccuracy = accuracy_score(y_val, forestPred)
    knnAccuracy = accuracy_score(y_val, knnPred)

    print("Finished Base Models")

    # HyperParameters for fine tuning
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.0001, 0.001, 0.01, 0.1, 1]
    linearParams = {'C': Cs, 'gamma': gammas, 'max_iter':[10000000],'probability': [True], 'cache_size':[1000]}
    polyParams = {'C': Cs, 'gamma': gammas, 'max_iter':[10000000],'probability': [True], 'cache_size':[1000]}
    rbfParams = {'C': Cs, 'gamma': gammas, 'max_iter': [10000000], 'probability': [True],  'cache_size':[1000]}
    sigmoidParams = {'C': Cs, 'gamma': gammas, 'max_iter': [10000000], 'probability': [True],  'cache_size':[1000]}

    neighborRange = []
    for x in range(5, 99):
        if x % 2 == 1:
            neighborRange.append(x);

    knnParams = {
        'n_neighbors': neighborRange,
        'weights':['uniform','distance'],
        'algorithm':['ball_tree','kd_tree','brute'],
        'leaf_size':[10,20,30,40,50,60],
        'p':[1,2]
    }

    # Get parameters for tree,forest,knn hyperparameter tuning
    treeParams = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5,8, 10,12, 15],
        'class_weight': ['balanced'],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [3, 4, 5, 10, 20],
        'min_samples_split': [5, 10, 15, 20],
    }
    forestParams = {
        'bootstrap': [True],
        'max_depth': [5,8, 10,12, 15],
        'min_samples_leaf': [3, 4, 5, 10, 20],
        'min_samples_split': [5, 10, 15, 20],
        'n_estimators': [150],
        'class_weight':['balanced','balanced_subsample']
    }

    # Create Grid Search Models
    linearGridSearch = GridSearchCV(sklearn.svm.SVC(kernel="linear"), param_grid=linearParams, scoring='roc_auc')
    polyGridSearch = GridSearchCV(sklearn.svm.SVC(kernel="poly"),param_grid=polyParams,scoring='roc_auc')
    rbfGridSearch = GridSearchCV(sklearn.svm.SVC(kernel="rbf"),param_grid=rbfParams,scoring='roc_auc')
    sigmoidGridSearch = GridSearchCV(sklearn.svm.SVC(kernel="sigmoid"), param_grid=sigmoidParams,scoring='roc_auc')
    treeGridSearch = GridSearchCV(DecisionTreeClassifier(), param_grid=treeParams,scoring='roc_auc')
    forestGridSearch = GridSearchCV(RandomForestClassifier(), param_grid=forestParams,scoring='roc_auc')
    knnGridSearch = GridSearchCV(KNeighborsClassifier(), param_grid=knnParams,scoring='roc_auc')

    # Fit all GridSearch Models
    linearGridSearch.fit(x_train, y_train);
    polyGridSearch.fit(x_train, y_train);
    rbfGridSearch.fit(x_train, y_train);
    sigmoidGridSearch.fit(x_train, y_train);
    print("Finished SVC models")
    treeGridSearch.fit(x_train, y_train);
    print("Finished Tree Grid")
    forestGridSearch.fit(x_train, y_train);
    print("Finished Forest Grid")
    knnGridSearch.fit(x_train, y_train);
    print("Done fitting Grid search models")

    # Get Predictions from all GridSearch Models
    linearSearchPred = linearGridSearch.best_estimator_.predict(x_val);
    polySearchPred = polyGridSearch.best_estimator_.predict(x_val);
    rbfSearchPred = rbfGridSearch.best_estimator_.predict(x_val);
    sigmoidSearchPred = sigmoidGridSearch.best_estimator_.predict(x_val);
    treeSearchPred = treeGridSearch.best_estimator_.predict(x_val);
    forestSearchPred = forestGridSearch.best_estimator_.predict(x_val);
    knnSearchPred = knnGridSearch.best_estimator_.predict(x_val);

    # Get ROC for all GridSearch Models
    linearSearchTN, linearSearchFP, linearSearchFN, linearSearchTP = confusion_matrix(y_val, linearSearchPred).ravel()
    polySearchTN, polySearchFP, polySearchFN, polSearchTP = confusion_matrix(y_val, polySearchPred).ravel()
    rbfSearchTN, rbfSearchFP, rbfSearchFN, rbSearchfTP = confusion_matrix(y_val, rbfSearchPred).ravel()
    sigmoidSearchTN, sigmoidSearchFP, sigmoidSearchFN, sigmoidSearchTP = confusion_matrix(y_val, sigmoidSearchPred).ravel()
    treeSearchTN, treeSearchFP, treeSearchFN, treeSearchTP = confusion_matrix(y_val, treeSearchPred).ravel()
    forestSearchTN, forestSearchFP, forestSearchFN, forestSearchTP = confusion_matrix(y_val, forestSearchPred).ravel()
    knnSearchTN, knnSearchFP, knnSearchFN, knnSearchTP = confusion_matrix(y_val, knnSearchPred).ravel()

    # Get Accuracy Scores for all GridSearch Models
    linearSearchAccuracy = accuracy_score(y_val,linearSearchPred);
    polySearchAccuracy = accuracy_score(y_val, polySearchPred);
    rbfSearchAccuracy = accuracy_score(y_val, rbfSearchPred);
    sigmoidSearchAccuracy = accuracy_score(y_val, sigmoidSearchPred);
    treeSearchAccuracy = accuracy_score(y_val, treeSearchPred);
    forestSearchAccuracy = accuracy_score(y_val, forestSearchPred);
    knnSearchAccuracy = accuracy_score(y_val, knnSearchPred);

    # Get AUC scores for all GridSearch Models
    linearSearchAUC = linearGridSearch.best_score_;
    polySearchAUC = polyGridSearch.best_score_;
    rbfSearchAUC = rbfGridSearch.best_score_;
    sigmoidSearchAUC = sigmoidGridSearch.best_score_;
    treeSearchAUC = treeGridSearch.best_score_;
    forestSearchAUC = forestGridSearch.best_score_;
    knnSearchAUC = knnGridSearch.best_score_;

    #create new estimator
    linearEstimator = SVC(**linearGridSearch.best_params_);
    polyEstimator = SVC(**polyGridSearch.best_params_);
    rbfEstimator = SVC(**rbfGridSearch.best_params_);
    sigmoidEstimator = SVC(**sigmoidGridSearch.best_params_);
    treeEstimator = DecisionTreeClassifier(**treeGridSearch.best_params_);
    forestEstimator = RandomForestClassifier(**forestGridSearch.best_params_);
    knnEstimator = KNeighborsClassifier(**knnGridSearch.best_params_);

    #final run
    linearEstimator.fit(x_whole_train, y_whole_train);
    polyEstimator.fit(x_whole_train, y_whole_train);
    rbfEstimator.fit(x_whole_train, y_whole_train);
    sigmoidEstimator.fit(x_whole_train, y_whole_train);
    treeEstimator.fit(x_whole_train, y_whole_train);
    forestEstimator.fit(x_whole_train, y_whole_train);
    knnEstimator.fit(x_whole_train, y_whole_train);

    linearFinalPred = linearEstimator.predict(x_test)
    polyFinalPred = polyEstimator.predict(x_test)
    rbfFinalPred = rbfEstimator.predict(x_test)
    sigmoidFinalPred = sigmoidEstimator.predict(x_test)
    treeFinalPred = treeEstimator.predict(x_test)
    forestFinalPred = forestEstimator.predict(x_test)
    knnFinalPred = knnEstimator.predict(x_test)

    # Get ROC for Final Run
    linearFinalTN, linearFinalFP, linearFinalFN, linearFinalTP = confusion_matrix(y_test, linearFinalPred).ravel()
    polyFinalTN, polyFinalFP, polyFinalFN, polyFinalTP = confusion_matrix(y_test, polyFinalPred).ravel()
    rbfFinalTN, rbfFinalFP, rbfFinalFN, rbFinalfTP = confusion_matrix(y_test, rbfFinalPred).ravel()
    sigmoidFinalTN, sigmoidFinalFP, sigmoidFinalFN, sigmoidFinalTP = confusion_matrix(y_test, sigmoidFinalPred).ravel()
    treeFinalTN, treeFinalFP, treeFinalFN, treeFinalTP = confusion_matrix(y_test, treeFinalPred).ravel()
    forestFinalTN, forestFinalFP, forestFinalFN, forestFinalTP = confusion_matrix(y_test, forestFinalPred).ravel()
    knnFinalTN, knnFinalFP, knnFinalFN, knnFinalTP = confusion_matrix(y_test, knnFinalPred).ravel()

    # Get Accuracy Scores for Final Run
    linearFinalAccuracy = accuracy_score(y_test, linearFinalPred);
    polyFinalAccuracy = accuracy_score(y_test, polyFinalPred);
    rbfFinalAccuracy = accuracy_score(y_test, rbfFinalPred);
    sigmoidFinalAccuracy = accuracy_score(y_test, sigmoidFinalPred);
    treeFinalAccuracy = accuracy_score(y_test, treeFinalPred);
    forestFinalAccuracy = accuracy_score(y_test, forestFinalPred);
    knnFinalAccuracy = accuracy_score(y_test, knnFinalPred);

    linearFinalAUC = roc_auc_score(y_test, linearEstimator.predict_proba(x_test)[:, 1]);
    polyFinalAUC = roc_auc_score(y_test, polyEstimator.predict_proba(x_test)[:, 1]);
    rbfFinalAUC = roc_auc_score(y_test, rbfEstimator.predict_proba(x_val)[:, 1]);
    sigmoidFinalAUC = roc_auc_score(y_test, sigmoidEstimator.predict_proba(x_test)[:, 1]);
    treeFinalAUC = roc_auc_score(y_test, treeEstimator.predict_proba(x_test)[:, 1]);
    forestFinalAUC = roc_auc_score(y_test, forestEstimator.predict_proba(x_test)[:, 1]);
    knnFinalAUC = roc_auc_score(y_test, knnEstimator.predict_proba(x_test)[:, 1]);

    linearRow = [data[0], linearAccuracy, linearScores, linearTN, linearFP, linearFN, linearTP,
                 linearSearchAccuracy,linearSearchAUC,linearSearchTN, linearSearchFP, linearSearchFN, linearSearchTP,
                 linearFinalAccuracy,linearFinalAUC,linearFinalTN, linearFinalFP, linearFinalFN, linearFinalTP];

    polyRow = [data[0], polyAccuracy, polyScores, polyTN, polyFP, polyFN, polTP,
               polySearchAccuracy,polySearchAUC,polySearchTN, polySearchFP, polySearchFN, polSearchTP,
               polyFinalAccuracy,polyFinalAUC,polyFinalTN, polyFinalFP, polyFinalFN, polyFinalTP]

    rbfRow = [data[0], rbfAccuracy, rbfScores, rbfTN, rbfFP, rbfFN, rbfTP,
              rbfSearchAccuracy,rbfSearchAUC,rbfSearchTN, rbfSearchFP, rbfSearchFN, rbSearchfTP,
              rbfFinalAccuracy,rbfFinalAUC,rbfFinalTN, rbfFinalFP, rbfFinalFN, rbFinalfTP ]

    sigmoidRow = [data[0], sigmoidAccuracy, sigmoidScores, sigmoidTN, sigmoidFP, sigmoidFN, sigmoidTP,
                  sigmoidSearchAccuracy,sigmoidSearchAUC,sigmoidSearchTN, sigmoidSearchFP, sigmoidSearchFN, sigmoidSearchTP,
                  sigmoidFinalAccuracy,sigmoidFinalAUC,sigmoidFinalTN, sigmoidFinalFP, sigmoidFinalFN, sigmoidFinalTP]

    treeRow = [data[0], treeAccuracy, treeScores, treeTN, treeFP, treeFN, treeTP,
               treeSearchAccuracy,treeSearchAUC,treeSearchTN, treeSearchFP, treeSearchFN, treeSearchTP,
               treeFinalAccuracy,treeFinalAUC,treeFinalTN, treeFinalFP, treeFinalFN, treeFinalTP]

    forestRow = [data[0], forestAccuracy, forestScores, forestTN, forestFP, forestFN, forestTP,
                 forestSearchAccuracy,forestSearchAUC,forestSearchTN, forestSearchFP, forestSearchFN, forestSearchTP,
                 forestFinalAccuracy,forestFinalAUC,forestFinalTN, forestFinalFP, forestFinalFN, forestFinalTP]

    knnRow = [data[0], knnAccuracy, knnScores, knnTN, knnFP, knnFN, knnTP,
              knnSearchAccuracy,knnSearchAUC,knnSearchTN, knnSearchFP, knnSearchFN, knnSearchTP,
              knnFinalAccuracy,knnFinalAUC,knnFinalTN, knnFinalFP, knnFinalFN, knnFinalTP]

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
    dataSet = year1[:, 3:15];
    scaler.fit(dataSet);

    if needsFeatureSelection:
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

        print(f'loading Cleaned Data')
        year1 = pd.read_excel(r'2009.xlsx', sheet_name='2009').to_numpy()
        year2 = pd.read_excel(r'2010.xlsx', sheet_name='2010').to_numpy()

        # Top 16 features obtained from Feature selection

        features = [
            '100010100101',
            '101010100101',
            '101110100101',
            '110110010101',
            '100010110101',
            '100110110101',
            '101010110101',
            '101110110101',
            '100010010101',
            '100110010101',
            '101010010101',
            '101110010101',
            '010001111010',
            '110010010101',
            '110100110101',
            '111110010101'
            ]

        tuningFeatures = [];
        for i in features:
            binaryString = "";
            permutation = [];
            hasFeatures = [ str(j) == '1' for j in i];

            for j in range(0,12):
                if hasFeatures[j]:
                    permutation.append(dataSet[:, j])
                    binaryString += "1"
                else:
                    binaryString += "0"
            permutation = normalize(permutation, axis=0, norm='max').transpose();
            tuningFeatures.append((binaryString[::-1],permutation,target));

        # run 16 threads for hyperparameter tuning
        pool = ThreadPool(16);
        results = pool.map(tuneModelsAndGetFinalRun, tuningFeatures);
        pool.close();
        pool.join();

        names = [];
        indices = [];
        accuracies = [];
        aucs = [];
        tn = [];
        fp = [];
        fn = [];
        tp = [];
        tunedAccuracies = [];
        tunedAUCS = [];
        tunedTN = [];
        tunedFP = [];
        tunedFN = [];
        tunedTP = [];

        finalAccuracies = [];
        finalAUCS = [];
        finalTN = [];
        finalFP = [];
        finalFN = [];
        finalTP = [];

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

                finalAccuracies.append(results[i][j][13]);
                finalAUCS.append(results[i][j][14]);
                finalTN.append(results[i][j][15]);
                finalFP.append(results[i][j][16]);
                finalFN.append(results[i][j][17]);
                finalTP.append(results[i][j][18]);

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
            np.asarray(tunedTP),
            np.asarray(finalAccuracies),
            np.asarray(finalAUCS),
            np.asarray(finalTN),
            np.asarray(finalFP),
            np.asarray(finalFN),
            np.asarray(finalTP)
        ]);

        data_df = pd.DataFrame(newData.transpose())

        data_df.columns = ['Classifier','Index','Accuracy','AUC','TN', 'FP','FN','TP',
                           'Tuned Accuracy','Tuned AUC', 'Tuned TN','Tuned FP','Tuned FN','Tuned TP',
                           'Final Accuracy','Final AUC','Final TN','Final FP', 'Final FN', 'Final TP'];
        writer = pd.ExcelWriter("TunedFeatures.xlsx")
        data_df.to_excel(writer, "Tuned Features", float_format='%.5f')
        writer.save();


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
