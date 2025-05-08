import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import load_and_prepare, split_train_test
from src.modelling import build_model
import kagglehub
import pandas as pd


def test_model():
    data = load_and_prepare()

    train_data, test_data = split_train_test(data, target_column='Close')

    predictions, mae, rmse, mape, metrics = build_model(train_data, test_data, target_column='Close')

    assert len(predictions)==len(test_data)
    assert isinstance(mae, float)
    assert isinstance(rmse, float) 
    assert isinstance(mape, float) 
    print("Testing has been completed.")

