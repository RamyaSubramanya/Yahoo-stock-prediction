import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.pipeline import load_and_prepare, split_train_test
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import math
import pandas as pd
import os
import matplotlib.pyplot as mp
import mlflow
import sys

def build_model(train_data, test_data, target_column):
    """
    Build Arima model and fit the model on training data
    Make predictions on test data
    Calculate error metric mae, rmse, mape
    Log the metrics
    Returns predictions, mae, rmse, mape
    """
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
    
    print("Model is built and predictions are made.")
    print("Error metrics has been calculated.")

    # print(f'MAE:{mae}, RMSE:{rmse}, predictions are off by {mape}%')
    metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE':mape}
    mlflow.log_metrics(metrics)
    mlflow.end_run()
    
    print("Metrics has been logged.")

    return predictions, mae, rmse, mape, metrics
