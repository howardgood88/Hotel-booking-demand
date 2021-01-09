import pandas as pd
import numpy as np

DATA_TRAIN = 'Dataset/train_final.csv'
LABEL_DATA_TRAIN = 'Dataset/train_label.csv'

DATA_TEST  = 'Dataset/test_final.csv'
LABEL_DATA_TEST  = 'Dataset/test_nolabel.csv'

feature_drop = ['ID']

def read_file(path):
    return pd.read_csv(path)


def get_data():
    return read_file(DATA_TRAIN), read_file(LABEL_DATA_TRAIN), \
            read_file(DATA_TEST), read_file(LABEL_DATA_TEST)



    