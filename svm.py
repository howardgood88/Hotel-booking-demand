import numpy as np
import pandas as pd
import util
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.svm import SVR
# from sklearn.model_selection import KFold
# pd.set_option('max_columns', None)

drop_features = ['ID', 'is_canceled', 'arrival_date_week_number', 'adr'
    , 'reservation_status', 'reservation_status_date', 'concat_date']

if __name__ == '__main__':
    # Read data
    train_x, train_y, test_x, test_y = util.get_data()
    scale_label = train_y['label']

    # Split data
    is_canceled = train_x['is_canceled']
    adr = train_x['adr']
    X = train_x.drop(drop_features, axis=1)
    print('Input data shape:', X.shape)

    # Train is_canceled
    clf = make_pipeline(MinMaxScaler(), SVC(verbose=True), verbose=True)
    clf.fit(X, is_canceled)
    print('is_canceled training finished...')

    # Train adr
    clf2 = make_pipeline(MinMaxScaler(), SVR(verbose=True), verbose=True)
    clf2.fit(X, adr)
    print('adr training finished...')

    # Calculate daily revenue
    stay_nights = train_x['stays_in_weekend_nights'] + train_x['stays_in_week_nights']
    request_revenue = adr * stay_nights
    days = train_x.filter(regex=('arrival_date_day_of_month_*'))
    daily_revenue = 0
    daily_revenue_list = []
    for idx in range(len(days)):
        if idx > 0 and days[idx] != days[idx-1]:
            daily_revenue_list.append(daily_revenue)
            daily_revenue = 0
        else:
            daily_revenue += request_revenue[idx]
    assert(len(daily_revenue_list) == len(scale_label))
    print('daily revenue calculate finished...')

    # Train scale
    clf3 = make_pipeline(MinMaxScaler(), SVR(verbose=True), verbose=True)
    clf3.fit(daily_revenue_list, scale_label)
    print('scale training finished...')