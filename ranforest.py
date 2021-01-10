import numpy as np
import pandas as pd
import util
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from joblib import dump, load
import os
from sklearn.metrics import mean_absolute_error, make_scorer


###################################################################
#                         Utilities
###################################################################


def MAE(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def drop_cancel(X, is_canceled:pd.Series):
    '''
        Drop is_cancel by boolean index.
    '''
    drop_is_canceled = pd.Series([i != 1 for i in is_canceled])
    return X[drop_is_canceled]


def get_daily_revenue(adr:pd.Series, stay_nights:pd.Series, days:pd.Series):
    '''
        Calculate daily revenue.
    '''
    print('Calculating daily revenue...', end='')
    request_revenue = adr * stay_nights
    daily_revenue = 0
    daily_revenue_list = []
    for idx in range(len(days)):
        if idx > 0 and not days.iloc[idx] == days.iloc[idx-1]:
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
    print('is_canceled predicting finished')
    adr = clf2.predict(X)
    print('adr predicting finished')

    adr = drop_cancel(adr, is_canceled)
    stay_nights = test_x['stays_in_weekend_nights'] + test_x['stays_in_week_nights']
    stay_nights = drop_cancel(stay_nights, is_canceled)
    days = test_x['arrival_date_day_of_month']
    days = drop_cancel(days, is_canceled)
    daily_revenue_list = get_daily_revenue(adr, stay_nights, days)

    scale = clf3.predict(daily_revenue_list)
    print('scale predicting finished')

    # Output result
    output_path = 'result.csv'
    result = pd.DataFrame({'label': scale})
    result = pd.concat([test_y, result], axis=1)
    result.to_csv(output_path, index=False)
    print('Predict finished. Result Saved as', output_path)


def eval(X, y, clf):
    y_pred = clf.predict(X)
    return MAE(y, y_pred)


###################################################################
#                            Train
###################################################################


def train(X, y, model, n_estimators_list:list, task:str):
    '''
        Train task by model.
    '''
    if os.path.isfile('Joblib/{}.joblib'.format(task)):
        print('Model {}.joblib detected, loading...'.format(task), end='')
        clf = load('Joblib/{}.joblib'.format(task))
        print(' Success')
    else:
        min_val_score = float('inf')
        for n_estimators in n_estimators_list:
            clf = make_pipeline(MinMaxScaler(), model(n_estimators, n_jobs=-1))
            score = make_scorer(mean_absolute_error)
            val_score = cross_val_score(clf, X, y, cv=10, scoring=score).mean()

            if val_score < min_val_score:
                min_val_score = val_score
                best_n_estimators = n_estimators
        print('The n_estimators be chose:', best_n_estimators)
        print('The minimum validation score:', min_val_score)
        clf = make_pipeline(MinMaxScaler(), model(best_n_estimators, n_jobs=-1))
        clf.fit(X, y)
        print('{} training finished...'.format(task))
        dump(clf, 'Joblib/{}.joblib'.format(task))
        print('Model Saved as Joblib/{}.joblib'.format(task))

    y_pred = clf.predict(X)
    print('{} MAE: {}'.format(task, MAE(y, y_pred)))

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

    n_estimators_list = [1500]
    clf = train(X, is_canceled, RandomForestClassifier, n_estimators_list, 'is_canceled')
    n_estimators_list = [525, 550]
    clf2 = train(X, adr, RandomForestRegressor, n_estimators_list, 'adr')

    adr = drop_cancel(adr, is_canceled)
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
    stay_nights = drop_cancel(stay_nights, is_canceled)
    days = train_x['arrival_date_day_of_month']
    days = drop_cancel(days, is_canceled)
    daily_revenue_list = get_daily_revenue(adr, stay_nights, days)

    n_estimators_list = [i for i in range(1, 51)]
    clf3 = train(daily_revenue_list, train_y, RandomForestClassifier, n_estimators_list, 'scale')

    return clf, clf2, clf3


###################################################################
#                            Main
###################################################################


drop_features_train = ['ID', 'is_canceled', 'arrival_date_week_number', 'adr'
    , 'reservation_status', 'reservation_status_date', 'concat_date']
drop_features_test = ['ID', 'arrival_date_week_number', 'concat_date']

# Read data
train_x, train_y, test_x, test_y = util.get_data_forest()


if __name__ == '__main__':
    # Split data
    is_canceled = train_x['is_canceled']
    adr = train_x['adr']
    X_train = train_x.drop(drop_features_train, axis=1)
    X_test = test_x.drop(drop_features_test, axis=1)
    print('Input data shape:', X_train.shape)

    clf, clf2, clf3 = train_main(X_train, is_canceled, adr, train_y)
    predict(X_test, clf, clf2, clf3)