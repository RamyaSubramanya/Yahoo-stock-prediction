from src.pipeline import load_and_prepare, split_train_test
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math
import pandas as pd
import os
import matplotlib.pyplot as mp
import mlflow
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def build_model(train_data, test_data, target_column):
    
    model = ARIMA(endog=train_data[target_column], order=(4,0,0))
    model = model.fit()

    #define index
    start=len(train_data)
    end=len(train_data)+len(test_data)-1

    mlflow.start_run()

    #make predictions
    predictions = model.predict(start=start, end=end)

    #check error metric
    mae = round(mean_absolute_error(test_data[target_column], predictions),2)
    rmse = round(math.sqrt(mean_squared_error(test_data[target_column], predictions)),2)
    mape = round(mean_absolute_percentage_error(test_data[target_column], predictions),2)*100

    # print(f'MAE:{mae}, RMSE:{rmse}, predictions are off by {mape}%')
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE':mape}
    mlflow.log_metrics(metrics)
    mlflow.end_run()

    return predictions, mae, rmse, mape, metrics
