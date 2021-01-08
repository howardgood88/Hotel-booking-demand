import numpy as np
import pandas as pd


############################################################################################################################################
#                                                             Utilities
############################################################################################################################################


def messager(a:str):
    print("#" * 20, a, "#" * 20)


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
        print("{} \thas \t{} \tentities".format(key, d[key]))


def print_shape_change(f):
    '''
        Keep track of the shape of obj/args[0].
    '''
    def decorator(*args, **kwargs):
        assert(len(args) == 1)
        before_shape = args[0].shape
        obj = f(*args, **kwargs)
        print('[Function {}] Shape change from {} to {}'.format(f.__name__, before_shape, obj.shape))
        return obj
    return decorator


def print_nan_count(f):
    '''
        Keep track of the number of NaN in obj/args[0].
    '''
    def decorator(*args, **kwargs):
        assert(len(args) == 1)
        before_na = args[0][feature_fillna].isna().values.sum()
        obj = f(*args, **kwargs)
        print('[Function {}] NaN change from {} to {}'.format(f.__name__, before_na, obj[feature_fillna].isna().values.sum()))
        return obj
    return decorator


############################################################################################################################################
#                                                           Data processing
############################################################################################################################################
    

@print_shape_change
def drop_feature(df:pd.DataFrame):
    '''
        Remove the features listed in feature_del.
    '''
    return df.drop(feature_del, axis=1)


@print_nan_count
def fill_na(df:pd.DataFrame, number:int =0):
    '''
        Replace NaN with a specified number for the columns listed in the feature_fillna.
    '''
    df[feature_fillna] = df[feature_fillna].fillna(number)
    return df


@print_shape_change
def drop_useless_entry(df:pd.DataFrame):
    '''
        Remove rows that have NaN after drop_feature() and fill_na().
    '''
    rows, cols = df.shape
    drop_index = []
    for row in range(rows):
        c = df.iloc[row].count()
        if c != cols:
            drop_index.append(row)
    this_df = df.drop(drop_index)
    return this_df


@print_shape_change
def ont_hot(df:pd.DataFrame):
    '''
        One-hot encoding for the features listed in *feature_one_hot*
    '''
    df2 = pd.DataFrame()
    for feature in feature_one_hot:
        new_df = pd.get_dummies(df[feature])
        df2 = pd.concat([df2, new_df], axis=1)
    df = df.drop(feature_one_hot, axis=1)
    df = pd.concat([df, df2], axis=1)
    return df


def one_hot(df:pd.DataFrame):
    return pd.get_dummies(df[feature_one_hot])


def one_hot_encoding(df1:pd.DataFrame, df2:pd.DataFrame):
    '''
        One-hot encoding for the features listed in *feature_one_hot*
    '''
    features_df1 = df1[feature_one_hot]
    features_df2 = df2[feature_one_hot]

    num = features_df1.shape[0]
    one_hot_pd = pd.concat([features_df1, features_df2])
    one_hot_pd = one_hot(one_hot_pd)

    one_hot_df1 = one_hot_pd.iloc[:num]
    one_hot_df2 = one_hot_pd.iloc[num:]

    df1 = df1.drop(feature_one_hot, axis=1)
    df2 = df2.drop(feature_one_hot, axis=1)

    df1 = pd.concat([df1, one_hot_df1], axis=1)
    df2 = pd.concat([df2, one_hot_df2], axis=1)
    return df1, df2


############################################################################################################################################
#                                                           Main
############################################################################################################################################


feature_predict_cancel = ['is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'deposit_type']
feature_del            = ['company']
feature_fillna         = ['agent']
feature_one_hot        = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type',
                        'assigned_room_type', 'deposit_type', 'agent', 'customer_type', 'day_of_the_week']


if __name__ == '__main__':
    train_df = pd.read_csv("Dataset/train.csv")
    test_df  = pd.read_csv("Dataset/test.csv")
    
    train_df = drop_feature(train_df)
    test_df  = drop_feature(test_df)
    
    train_df = fill_na(train_df, number=0)
    test_df  = fill_na(test_df,  number=0)

    train_df = drop_useless_entry(train_df)
    test_df  = drop_useless_entry(test_df)

    train_df, test_df = one_hot_encoding(train_df, test_df)

    train_df.to_csv('train_df_out.csv', index=False)
    test_df.to_csv('test_df_out.csv', index=False)
