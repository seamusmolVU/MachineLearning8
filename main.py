
import numpy as np
import pandas as pd
import xlrd
import sklearn
import matplotlib.pyplot as plt
from pandas import read_excel

def main():
    print(f'loading Segmentation Data')

    year1 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2009/2010')
    year2 = pd.read_excel(r'online_retail_II.xlsx', sheet_name='Year 2010/2011')

    print(f'cleaning data')





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main();

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
