import pandas as pd
import numpy as np

DATA_TRAIN   = "Dataset\\train_final.csv"
DATA_TRAIN_Y = "Dataset\\train_label.csv"
DATA_TEST    = "Dataset\\test_final.csv"

feature_train_drop = ['ID', 'reservation_status', 'reservation_status_date', 'concat_date']
feature_test_drop  = ['ID', 'concat_date']


def read_file(path):
    return pd.read_csv(path)


def get_data():
    return read_file(DATA_TRAIN).drop(feature_train_drop), read_file(DATA_TEST).drop(feature_test_drop)



    