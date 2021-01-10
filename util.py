import pandas as pd
import numpy as np

DATA_TRAIN = 'Dataset/train_final.csv'
DATA_TRAIN2 = 'Dataset/train_final2.csv'
LABEL_DATA_TRAIN = 'Dataset/train_label.csv'

DATA_TEST  = 'Dataset/test_final.csv'
DATA_TEST2 = 'Dataset/test_final2.csv'
LABEL_DATA_TEST  = 'Dataset/test_nolabel.csv'

def read_file(path):
    return pd.read_csv(path)


def get_data():
    return read_file(DATA_TRAIN), read_file(LABEL_DATA_TRAIN)['label'], \
            read_file(DATA_TEST), read_file(LABEL_DATA_TEST)

def get_data_forest():
    return read_file(DATA_TRAIN2), read_file(LABEL_DATA_TRAIN)['label'], \
            read_file(DATA_TEST2), read_file(LABEL_DATA_TEST)