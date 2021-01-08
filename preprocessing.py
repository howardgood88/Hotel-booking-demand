import numpy as np
import pandas as pd


###################################################################
#                             Utilities
###################################################################


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
        [Decorator] Keep track of the shape of args.
    '''
    def decorator(*args, **kwargs):

        before_shape = []
        for arg in args:
            before_shape.append(arg.shape)

        args = f(*args, **kwargs)
        # To make single return value act the same with multi return value 
        if type(args) != tuple:
            args = [args]

        for idx, arg in enumerate(args):
            print('[Function {}] arg{} shape from {} to {}'.format(f.__name__, idx, before_shape[idx], arg.shape))

        # Recover the change on single return value
        if type(args) == list:
            args = args[0]

        return args
    return decorator


def print_nan_count(f):
    '''
        [Decorator] Keep track of the number of NaN in args.
    '''
    def decorator(*args, **kwargs):
        before_na = []
        for arg in args:
            na = args[0][feature_fillna].isna().values.sum()
            before_na.append(na)

        obj = f(*args, **kwargs)
        # To make single return value act the same with multi return value 
        if type(args) != tuple:
            args = [args]

        for idx, arg in enumerate(args):
            print('[Function {}] arg{} NaN from {} to {}'.format(f.__name__, idx, before_na[idx], obj[feature_fillna].isna().values.sum()))

        # Recover the change on single return value
        if type(args) == list:
            args = args[0]

        return obj
    return decorator


###################################################################
#                      Data processing
###################################################################
    

@print_shape_change
def drop_feature(df:pd.DataFrame):
    '''
        Remove the features listed in *feature_del*.
    '''
    return df.drop(feature_del, axis=1)


@print_nan_count
def fill_na(df:pd.DataFrame, number:int =0):
    '''
        Replace NaN with a specified number for the columns listed in the *feature_fillna*.
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
def add_room_change_feature(df:pd.DataFrame):
    '''
        Add the boolean feature by the change of feature 'reserved_room_type' and 'assigned_room_type'.
    '''
    df2 = df['reserved_room_type'].eq(df['assigned_room_type']).astype(int)
    df2 = pd.DataFrame(df2, columns=['room_change'])
    return pd.concat([df, df2], axis=1)


def one_hot(df:pd.DataFrame):
    return pd.get_dummies(df)


@print_shape_change
def one_hot_encoding(df1:pd.DataFrame, df2:pd.DataFrame):
    '''
        One-hot encoding for the features listed in *feature_one_hot*.
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


###################################################################
#                            Main
###################################################################


feature_predict_cancel = ['is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'deposit_type']
feature_del            = ['company']
feature_fillna         = ['agent']
feature_one_hot        = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type',
                        'assigned_room_type', 'deposit_type', 'agent', 'customer_type', 'day_of_the_week']


if __name__ == '__main__':
    train_df = pd.read_csv("Dataset/train_day_of_week.csv")
    test_df  = pd.read_csv("Dataset/test_day_of_week.csv")
    
    train_df = drop_feature(train_df)
    test_df  = drop_feature(test_df)
    
    train_df = fill_na(train_df, number=0)
    test_df  = fill_na(test_df,  number=0)

    train_df = drop_useless_entry(train_df)
    test_df  = drop_useless_entry(test_df)

    train_df = add_room_change_feature(train_df)
    test_df = add_room_change_feature(test_df)

    train_df, test_df = one_hot_encoding(train_df, test_df)

    train_df.to_csv('Dataset/train_final.csv', index=False)
    test_df.to_csv('Dataset/test_final.csv', index=False)
