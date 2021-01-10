import numpy as np
import pandas as pd
import util
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import SVR
from joblib import dump, load
import os
# from sklearn.model_selection import KFold
# pd.set_option('max_columns', None)


###################################################################
#                         Utilities
###################################################################


def drop_cancel(X, is_canceled:pd.Series):
    '''
        Drop is_cancel by boolean index.
    '''
    drop_is_canceled = pd.Series([i != 1 for i in is_canceled])
    return X[drop_is_canceled]


def get_daily_revenue(adr:pd.Series, stay_nights:pd.Series, days:pd.DataFrame):
    '''
        Calculate daily revenue.
    '''
    print('Calculating daily revenue...', end='')
    request_revenue = adr * stay_nights
    daily_revenue = 0
    daily_revenue_list = []
    for idx in range(len(days)):
        if idx > 0 and not days.iloc[idx].equals(days.iloc[idx-1]):
            daily_revenue_list.append(daily_revenue)
            daily_revenue = 0
        else:
            daily_revenue += request_revenue.iloc[idx]
    daily_revenue_list.append(daily_revenue)
    daily_revenue_list = np.array(daily_revenue_list)

    print(' Finished')
    return daily_revenue_list[:, np.newaxis]


###################################################################
#                           Predict
###################################################################


def predict(X, clf, clf2, clf3):
    print('Predicting...')
    is_canceled = clf.predict(X)
    adr = clf2.predict(X)

    adr = drop_cancel(adr, is_canceled)
    stay_nights = test_x['stays_in_weekend_nights'] + test_x['stays_in_week_nights']
    stay_nights = drop_cancel(stay_nights, is_canceled)
    days = test_x.filter(regex=('arrival_date_day_of_month_*'))
    days = drop_cancel(days, is_canceled)
    daily_revenue_list = get_daily_revenue(adr, stay_nights, days)

    scale = clf3.predict(daily_revenue_list)

    # Output result
    output_path = 'result.csv'
    result = pd.DataFrame({'label': scale})
    result = pd.concat([test_y, result], axis=1)
    result.to_csv(output_path, index=False)
    print('Predict finished. Result Saved as', output_path)


###################################################################
#                            Train
###################################################################


def train(X, y, model, task:str, verbose:bool=0):
    '''
        Train task by model.
    '''
    if os.path.isfile('Joblib/{}.joblib'.format(task)):
        print('Model {}.joblib detected, loading...'.format(task), end='')
        clf = load('Joblib/{}.joblib'.format(task))
        print(' Success')
    else:
        clf = make_pipeline(MinMaxScaler(), model(verbose=True), verbose=True)
        clf.fit(X, y)
        print('{} training finished...'.format(task))
        dump(clf, 'Joblib/{}.joblib'.format(task))
        print('Model Saved as Joblib/{}.joblib'.format(task))

    if verbose:
        print('Accuracy {}: {}'.format(task, clf.score(X, y)))

    print('--------------------------------------------')
    return clf


def train_main(X:pd.DataFrame, is_canceled:pd.Series, adr:pd.Series,
                train_y:pd.Series):
    '''
        Main function for training.
    '''
    # dir for saving model
    if not os.path.isdir('Joblib'):
        os.mkdir('Joblib')

    clf = train(X, is_canceled, SVC, 'is_canceled')
    clf2 = train(X, adr, SVR, 'adr')

    adr = drop_cancel(adr, is_canceled)
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
    stay_nights = drop_cancel(stay_nights, is_canceled)
    days = train_x.filter(regex=('arrival_date_day_of_month_*'))
    days = drop_cancel(days, is_canceled)
    daily_revenue_list = get_daily_revenue(adr, stay_nights, days)

    clf3 = train(daily_revenue_list, train_y, SVC, 'scale')

    return clf, clf2, clf3


###################################################################
#                            Main
###################################################################


drop_features_train = ['ID', 'is_canceled', 'arrival_date_week_number', 'adr'
    , 'reservation_status', 'reservation_status_date', 'concat_date']
drop_features_test = ['ID', 'arrival_date_week_number', 'concat_date']

# Read data
train_x, train_y, test_x, test_y = util.get_data()

if __name__ == '__main__':
    # Split data
    is_canceled = train_x['is_canceled']
    adr = train_x['adr']
    X_train = train_x.drop(drop_features_train, axis=1)
    X_test = test_x.drop(drop_features_test, axis=1)
    print('Input data shape:', X_train.shape)

    clf, clf2, clf3 = train_main(X_train, is_canceled, adr, train_y)
    predict(X_test, clf, clf2, clf3)