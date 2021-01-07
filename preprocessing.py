import numpy as np
import pandas as pd



def count_features(df:pd.DataFrame):
    """
        This function computes the valid entries for specified column.
    """
    columns = df.columns
    #rows, cols = df.shape
    d = {}
    for column in columns:
        c = df[column].count()
        d[column] = c
    for key in d:
        print("{0} has {1} entities".format(key, d[key]))

def drop_useless(df:pd.DataFrame):
    '''
        This function drops the invalid entries for dataframe.
    '''
    rows, cols = df.shape
    drop_index = []
    for row in range(rows):
        c = df.iloc[row].count()
        if c != cols:
            drop_index.append(row)
    this_df = df.drop(drop_index)
    return this_df


def messager(a:str):
    print("##########################################{0}##############################".format(a))

    
if __name__ == '__main__':
    train_df = pd.read_csv("Dataset/train.csv")
    test_df  = pd.read_csv("Dataset/test.csv")

    messager("Train_DF")
    count_features(train_df)

    messager("Test_DF")
    count_features(test_df)

    messager("Train_DF after Dropping")
    train_df = drop_useless(train_df)
    count_features(train_df)
    