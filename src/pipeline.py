import pandas as pd
import kagglehub
import os
import shutil
import statsmodels
from statsmodels.tsa.api import adfuller


def load_and_prepare():

    """
    Load the data from Kagglehub, download it into the current directory.
    Format the data, set date as index.
    """

    #step1: get the current working directory
    current_dir = os.getcwd()

    # Step 2: Download dataset (returns temp path where it's stored)
    dataset_path = kagglehub.dataset_download("arashnic/time-series-forecasting-with-yahoo-stock-price")

    # Step 3: Move file(s) to current working directory
    source_file = os.path.join(dataset_path, "yahoo_stock.csv")
    destination_file = os.path.join(current_dir, "yahoo_stock_data.csv")

    # Copy to current working directory. shutil is for copying and archiving files. 
    shutil.copy(source_file, destination_file)
    print("File copied to the current working directory.")

    #Read the data
    data = pd.read_csv(destination_file)

    #drop last column
    data = data.iloc[:,:-1]

    #Format date column
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

    #set Date as index for time series data
    data.set_index('Date',inplace=True)

    data = data.asfreq('D') 

    print("Data has been loaded and pre-processed.")
    return data 

def stationarity_check(data, col):
    """
    Stationarity check for the data. Returns result
    """
    result = adfuller(data[col])
    if result[1]<0.05:
        print(f'Feature {col} is stationary')
    else:
        print(f'Feature {col} is not stationary')
        #convert data into stationary
        data[col] = data[col].pct_change(periods=1).mul(100)
        data = data.iloc[1:,:]
    return result[1]

def split_train_test(data, target_column):

    """
    Splits the data into train, test basis the split ratio.
    Creates exogeneous variables 
    Returns train and test data, train and test exogeneous data
    """

    if target_column not in data.columns:
        raise ValueError (f'{target_column} not found in the dataset.')

    #define training and test size
    training_size = int(len(data)*0.70)

    #split the data into train, test basis size
    train_data = data[:training_size]
    test_data = data[training_size:]

    #define the exogeneous variables
    #define the exogeneous variables
    # train_data_exog= train_data.drop(columns=[target_column])
    # train_data_exog = train_data_exog.loc[train_data_exog.index]

    # test_data_exog = test_data.drop(columns=[target_column])
    # test_data_exog = test_data_exog.loc[test_data_exog.index]
    print("Data has been split into train, test")
    return train_data, test_data