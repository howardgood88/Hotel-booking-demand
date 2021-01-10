import numpy as np 
import pandas as pd
import torch
from nn_model import BinaryClassifier, linearRegression, TenClassClassifier
from torch.autograd import Variable


#################################################################
#                       Utilities
#################################################################


np.random.seed(987)
torch.manual_seed(987)


def drop_cancel(X, is_canceled):
    '''
        Drop is_cancel by boolean index.
    '''
    drop_is_canceled = pd.Series([i != 1 for i in is_canceled])
    return X[drop_is_canceled]


def get_daily_revenue(adr:pd.Series, stay_nights:pd.Series, days:pd.DataFrame):
    '''
        Calculate daily revenue.
    '''
    print('Calculating daily revenue...')
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


# test is_cancel

y_label ='is_canceled'
not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_test = pd.read_csv('Dataset/test_final.csv')

_ = [df_test.pop(x) for x in not_for_test]
x_df_test = df_test

print('Is_cancel x_df_test shape: ',x_df_test.values.shape)

inputDim = x_df_test.shape[1]
outputDim = 1
model = BinaryClassifier(inputDim, outputDim)
model.load_state_dict(torch.load('is_cancel_model.pth'))
model.eval()

x_test = Variable(torch.from_numpy(x_df_test.values).float())
prediction = model(x_test)
is_cancel_test = prediction.detach().numpy().round()

print('Is cancel test shape: ',is_cancel_test.shape)


# test adr

is_cancel_test = pd.DataFrame({'is_canceled': is_cancel_test.reshape(-1)})


not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_test = pd.read_csv('Dataset/test_final.csv')

_ = [df_test.pop(x) for x in not_for_test]
x_df_test = pd.concat([df_test,is_cancel_test], axis=1)

print('Adr x_df_test shape : ',x_df_test.shape)

inputDim = x_df_test.shape[1]
outputDim = 1
model = linearRegression(inputDim, outputDim)
model.load_state_dict(torch.load('adr_model.pth'))
model.eval()

x_test = Variable(torch.from_numpy(x_df_test.values).float())
prediction = model(x_test)
adr_test = prediction.detach().numpy()




# test scale

adr_test = pd.DataFrame({'adr': adr_test.reshape(-1)})

print('adr: ',adr_test)
print('adr shape: ',adr_test.shape)

df_test = pd.read_csv('Dataset/test_final.csv')

x_df_test = pd.concat([df_test,is_cancel_test,adr_test], axis=1)

# x_df_test = drop_cancel(x_df_test, x_df_test['is_canceled'])

print('x_df_test shape: ',x_df_test.shape)

np_test = get_daily_revenue(
    x_df_test['adr'],
    x_df_test['stays_in_weekend_nights'] + x_df_test['stays_in_week_nights'],
    x_df_test.filter(regex=('arrival_date_day_of_month_*')))

print('Scale np_test shape: ',np_test.shape)

inputDim = 1
outputDim = 1
model = TenClassClassifier(inputDim, outputDim)
model.load_state_dict(torch.load('label_model.pth'))
model.eval()

np_test = Variable(torch.from_numpy(np_test).float())

prediction = model(np_test)
result = torch.argmax(prediction, dim=1).reshape(-1).detach().numpy().round()

print('Result shape: ', result.shape)


no_label_test = pd.read_csv('Dataset/test_nolabel.csv')
result = pd.DataFrame({'label': result})
r = pd.concat([no_label_test, result], axis=1)
r.to_csv('result.csv', index=False)