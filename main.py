import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.pipeline import load_and_prepare, split_train_test
from src.modelling import build_model
from tests.test_model import test_model

data = load_and_prepare()

train_data, test_data = split_train_test(data, target_column='Close')

predictions, mae, rmse, mape, metrics = build_model(train_data, test_data, target_column='Close')