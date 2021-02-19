
import numpy as np
import pandas as pd
import xlrd
import sklearn
import matplotlib.pyplot as plt
from pandas import read_excel

def containsDigit(string):
    for i in string:
        if i.isdigit():
            return True;
    return False;

def main():
    print(f'loading Segmentation Data')

    year1 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2009-2010').to_numpy()
    year2 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2010-2011').to_numpy()

    print(f'cleaning data')

    #remove cost = 0
    #remove stockCode without numbers



    ordersY1 = np.empty([0,8]);
    cleanYear1 = np.empty([0,8]);
    for i in year1:
        if i[5] != 0 and containsDigit(i[1]):
            cleanYear1 = np.append(cleanYear1, i);




    #print(year1)

    #create list of features
    #description
    #

    #research questions

    #1. Predict customer retention, Customer makes purchase next year
    #customer return rate, any item in order
    #postage cost
    #average order quantity
    #average order subtotal
    #average price of item in an order
    #number of orders
    #most common item category
    #most common time that orders are made
    #most common day of the week that orders are made


    #Plotting



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
