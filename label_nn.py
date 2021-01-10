from nn_model import TenClassClassifier

import numpy as np 
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

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


#################################################################
#                       Reading Data
#################################################################


y_label ='label'
class_num = 10
# train_label = ['adr','is_cancel','stays_in_weekend_nights','stays_in_week_nights','arrival_date_day_of_month_*']

df_train = pd.read_csv('Dataset/train_final.csv')
df_train_label = pd.read_csv('Dataset/train_label.csv')

df_train = drop_cancel(df_train, df_train['is_canceled'])
np_train = get_daily_revenue(
    df_train['adr'],
    df_train['stays_in_weekend_nights'] + df_train['stays_in_week_nights'],
    df_train.filter(regex=('arrival_date_day_of_month_*')))

np_label = df_train_label[y_label].to_numpy()
print(np_label, np_label.shape)


x_np_train = np_train.reshape(-1,1)
y_np_train = np_label.reshape(-1,1)


# np_data = np.concatenate((np_train.reshape(-1,1), np_label.reshape(-1,1)), axis=1)
# np.random.shuffle(np_data)

# x_np_train = np_data[:576,0]
# x_np_valid = np_data[576:,0]

# y_np_train = np_data[:576,1]
# y_np_valid = np_data[576:,1]

# y_oh_train = np.eye(class_num)[y_np_train.astype(int)]
# y_oh_valid = np.eye(class_num)[y_np_valid.astype(int)]

# print(x_np_train.shape, y_oh_train.shape, x_np_valid.shape, y_oh_valid.shape)


#################################################################
#                       Training Label(Scale)
#################################################################


inputDim = 1
outputDim = 1
learningRate = 0.0001 
epochs = 100

model = TenClassClassifier(inputDim, outputDim)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_func = torch.nn.L1Loss()

loss_train = []
loss_valid = []
acc_train = []
acc_valid = []
for t in range(epochs):

    model.train()
    x_train = Variable(torch.from_numpy(x_np_train).float())
    y_train = Variable(torch.from_numpy(y_np_train).float())
    # x_train = x_train.view(-1,1)
    # y_train = y_train.view(-1,1)
    prediction = model(x_train)
    print(prediction.detach().numpy().round().reshape(1,-1))
    # print(y_train, y_train.shape, prediction.shape)
    loss = loss_func(prediction, y_train)
    tloss = loss.detach().numpy()
    loss_train.append(tloss)
    # print(torch.argmax(prediction, dim=1).reshape(-1).detach().numpy().shape, y_np_train.reshape(-1).shape)
    acc = (torch.argmax(prediction, dim=1).reshape(-1).detach().numpy().round() == y_np_train.reshape(-1)).mean()
    acc_train.append(acc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # model.eval()
    # x_valid = Variable(torch.from_numpy(x_np_valid).float())
    # y_valid = Variable(torch.from_numpy(y_oh_valid).float())
    # x_valid = x_valid.view(-1,1)
    # # y_valid = y_valid.view(-1,1)
    # prediction = model(x_valid)
    # print(prediction.detach().numpy())
    # vloss = loss_func(prediction, y_valid)
    # vloss = vloss.detach().numpy()
    # loss_valid.append(vloss)
    # vacc = (torch.argmax(prediction, dim=1).reshape(-1).detach().numpy() == y_np_valid.reshape(-1)).mean()
    # acc_valid.append(vacc)

    # if t % 50 == 0:
    #     print('epoch = {}, train_loss = {}, valid_loss = {}'.format(t,tloss,vloss))

    # if t % 50 == 0:
    print('epoch = {}, train_loss = {}'.format(t,tloss),end='\r')


torch.save(model.state_dict(),'label_model.pth')

print(loss_train, acc_train)
plt.plot(loss_train, label='train_loss')
plt.plot(acc_train, label='acc_train')
plt.legend(loc='best')
plt.show()