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


def get_daily_revenue(adr, stay_nights):
    '''
        Calculate daily revenue.
    '''
    request_revenue = adr * stay_nights
    days = train_x.filter(regex=('arrival_date_day_of_month_*'))
    daily_revenue = 0
    daily_revenue_list = []
    for idx in range(len(days)):
        if idx > 0 and not days.iloc[idx].equals(days.iloc[idx-1]):
            daily_revenue_list.append(daily_revenue)
            daily_revenue = 0
        else:
            daily_revenue += request_revenue[idx]
    daily_revenue_list.append(daily_revenue)
    daily_revenue_list = np.array(daily_revenue_list)
    assert(len(daily_revenue_list) == len(train_y))

    print('daily revenue calculate finished...')
    return daily_revenue_list[:, np.newaxis]


###################################################################
#                             Test
###################################################################


def predict(X, clf, clf2, clf3):
    is_canceled = clf.predict(X)
    adr = clf2.predict(X)

    adr = drop_cancel(adr, is_canceled)
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
    stay_nights = drop_cancel(stay_nights, is_canceled)
    daily_revenue_list = get_daily_revenue(adr, stay_nights)

    scale = clf3.predict(daily_revenue_list)
    result = pd.DataFrame({'label': scale})
    result = pd.concat([test_y, result], axis=1)
    result.to_csv('result.csv', index=False)


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
        print(' Success.')
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

    adr = adr[is_canceled]
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
    stay_nights = stay_nights[is_canceled]
    daily_revenue_list = get_daily_revenue(adr, stay_nights)

    clf3 = train(daily_revenue_list, train_y, SVC, 'scale')

    return clf, clf2, clf3


###################################################################
#                            Main
###################################################################


drop_features = ['ID', 'is_canceled', 'arrival_date_week_number', 'adr'
    , 'reservation_status', 'reservation_status_date', 'concat_date']

# Read data
train_x, train_y, test_x, test_y = util.get_data()

if __name__ == '__main__':
    # Split data
    is_canceled = train_x['is_canceled']
    adr = train_x['adr']
    X = train_x.drop(drop_features, axis=1)
    print('Input data shape:', X.shape)

    clf, clf2, clf3 = train_main(X, is_canceled, adr, train_y)
    predict(test_x, clf, clf2, clf3)