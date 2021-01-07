import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# pd.set_option('display.max_colwidth', None)

def read_file(path):
    return pd.read_csv(path)

if __name__ == '__main__':
    pd.set_option('max_columns', None)
    # Read data
    input_path = 'Dataset/train.csv'
    data = read_file(input_path)

    # Split data
    is_canceled = data['is_canceled']
    adr = data['adr']
    X = data.drop(['is_canceled', 'adr'], axis=1)
    print(X)