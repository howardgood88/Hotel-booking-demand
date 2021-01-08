import numpy as np
import pandas as pd

feature_predict_cancel = ['is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'deposit_type']
feature_del            = ['company']
feature_fillna         = ['agent']
feature_one_hot        = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type',
                        'assigned_room_type', 'deposit_type', 'agent', 'customer_type', 'day_of_the_week']


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


def drop_useless_entry(df:pd.DataFrame):
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


def fill_na(df:pd.DataFrame, number:int = 0):
    '''
        This function replace the NaN with the speicified number for the columns listed in the feature_fillna.
    '''
    return df[feature_fillna].fillna(number)
    

def drop_feature(df:pd.DataFrame):
    '''
        This function removes the columns listed in the feature_del.
    '''
    return df.drop(feature_del)


def one_hot(df:pd.DataFrame):
    return pd.get_dummies(df[feature_one_hot])


def concatenate(df1:pd.DataFrame, df2:pd.DataFrame):
    '''
        This function concates two dataframes.
    '''
    return pd.concat((df1, df2))


def messager(a:str):
    print("#" * 20, a, "#" * 20)

    
if __name__ == '__main__':
    train_df = pd.read_csv("Dataset/train.csv")
    test_df  = pd.read_csv("Dataset/test.csv")

    '''
        drop the features listed on *feature_del*
    '''
    train_df = drop_feature(train_df)
    test_df  = drop_feature(test_df)

    '''
        replace NaN with a specified number
    '''
    train_df = fill_na(train_df, number=0)
    test_df  = fill_na(test_df,  number=0)


    train_df = drop_useless_entry(train_df)
    test_df  = drop_useless_entry(test_df)

