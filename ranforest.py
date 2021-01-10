import numpy as np
import pandas as pd
import util
import os
from joblib import dump, load
from enum import Enum
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

drop_features = ['ID', 'is_canceled', 'arrival_date_week_number', 'adr'
    , 'reservation_status', 'reservation_status_date', 'concat_date']

class model(Enum):
    MODEL_IS_CANCELED = "ranforest_is_canceled.joblib"
    MODEL_ADR         = "ranforest_adr.joblib"
    MODEL_SCALE       = "ranforest_scale.joblib"

def model_exists(filename, path):
    if filename in os.listdir(path):
        return True
    return False

def get_model(model_name, path):
    assert(model_name in model)
    if model_exists(model_name, path):
        clf = load(os.path.join(path, model_name))
        print("model {} loading success".format(model_name))
        return clf
    return False


MODE = 0

if __name__ == '__main__' and MODE:
    # Read data
    train_x, train_y, test_x, test_y = util.get_data()
    scale_label = train_y['label']

    # Split data
    is_canceled = train_x['is_canceled']
    adr = train_x['adr']
    X = train_x.drop(drop_features, axis=1)
    print('Input data shape:', X.shape)

    # Train is_canceled
    if os.path.isfile('Joblib/is_canceled.joblib'):
        print('Model is_canceled.joblib detected, loading...')
        clf = load('Joblib/is_canceled.joblib')
        print('Model loaded success.')
        print('Accuracy is_canceled:', clf.score(X, is_canceled))
    else:
        clf = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=100, verbose=True), verbose=True)
        clf.fit(X, is_canceled)
        print('is_canceled training finished...')
        print('Accuracy is_canceled:', clf.score(X, is_canceled))
        dump(clf, 'Joblib/is_canceled.joblib')
        print('Model Saved as Joblib/is_canceled.joblib')
    print('--------------------------------------------')

    # Train adr
    if os.path.isfile('Joblib/adr.joblib'):
        print('Model adr.joblib detected, loading...')
        clf2 = load('Joblib/adr.joblib')
        print('Model loaded success.')
        print('Accuracy adr:', clf2.score(X, adr))
    else:
        clf2 = make_pipeline(MinMaxScaler(), RandomForestRegressor(n_estimators=100, verbose=True), verbose=True)
        clf2.fit(X, adr)
        print('adr training finished...')
        print('accuracy adr:', clf2.score(X, adr))
        dump(clf2, 'Joblib/adr.joblib')
        print('Model saved as Joblib/adr.joblib')
    print('--------------------------------------------')

    # Calculate daily revenue
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
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
    assert(len(daily_revenue_list) == len(scale_label))
    print('daily revenue calculate finished...')

    # Train scale
    if os.path.isfile('Joblib/scale.joblib'):
        print('Model scale.joblib detected, loading...')
        clf3 = load('Joblib/scale.joblib')
        print('Model loaded success.')
        print('Accuracy scale:', clf3.score(daily_revenue_list[:, np.newaxis], scale_label))
    else:
        clf3 = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=100, verbose=True), verbose=True)
        clf3.fit(daily_revenue_list[:, np.newaxis], scale_label)
        print('scale training finished...')
        print('accuracy scale:', clf3.score(daily_revenue_list[:, np.newaxis], scale_label))
        dump(clf3, 'Joblib/scale.joblib')
        print('Model saved as Joblib/scale.joblib')

if __name__ == '__main__' and MODE != 1:
    drop_features = ['ID', 'arrival_date_week_number','concat_date']
    test_x, test_y = util.get_test()

    clf1 = load('Joblib/is_canceled.joblib')
    clf2 = load('Joblib/adr.joblib')
    clf3 = load('Joblib/scale.joblib')

    test_x = test_x.drop(drop_features, axis=1)
    is_canceled = clf1.predict(test_x)

    is_canceled_drop = [i for i, x in enumerate(is_canceled) if x == 1]
    test_x = test_x.drop(is_canceled_drop, axis=0)

    adr = clf2.predict(test_x)

    stay_nights = test_x['stays_in_weekend_nights'] + test_x['stays_in_week_nights']
    request_revenue = adr * stay_nights
    days = test_x.filter(regex=('arrival_date_day_of_month_*'))

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
    assert(len(daily_revenue_list) == len(test_y))
    print('daily revenue calculate finished...')

    A = clf3.predict(daily_revenue_list[:, np.newaxis])
    result = pd.DataFrame({'label': A})
    r = pd.concat([test_y, result], axis=1)
    r.to_csv('result.csv', index=False)