from nn_model import linearRegression

import numpy as np 
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#################################################################
#                       Utilities
#################################################################


def accuracy_(outputs, labels):
    outputs = outputs.detach().cpu()
    labels = labels.detach().cpu()
    _, preds = outputs.max(1)
    return float(preds.eq(labels).sum()) / outputs.size(0)


#################################################################
#                       Reading Data
#################################################################

y_label ='is_canceled'
not_for_train = ['ID','adr','reservation_status','reservation_status_date','concat_date',
                'arrival_date_year','arrival_date_week_number']
# not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_train = pd.read_csv('Dataset/train_final.csv')
# df_test = pd.read_csv('Dataset/test_final.csv')
# 
df_train.sample(frac=1)
_ = [df_train.pop(x) for x in not_for_train]
df_valid = df_train.iloc[81945:, :]
print(type(df_valid))
y_df_train = df_train.pop(y_label)
y_df_valid = df_valid.pop(y_label)
x_df_train = df_train
x_df_valid = df_valid

# _ = [df_test.pop(x) for x in not_for_test]
# x_df_test = df_test

print(x_df_train.values.shape, y_df_train.values.shape)
# print('========')
# print(x_df_test.shape[1])


#################################################################
#                       Training Settings
#################################################################


# if x_df_test.shape[1] != x_df_train.shape[1]:
#     import sys
#     sys.exit('Shape unmatch')

inputDim = x_df_train.shape[1]        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.0001 
epochs = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = linearRegression(inputDim, 1000, outputDim)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_func = torch.nn.MSELoss()

loss_train = []
loss_valid = []
for t in range(epochs):
    x_train = Variable(torch.from_numpy(x_df_train.values).float())
    y_train = Variable(torch.from_numpy(y_df_train.values).float())
    y_train = y_train.view(-1,1)

    model.train()
    optimizer.zero_grad()
    prediction = model(x_train)
    loss = loss_func(prediction, y_train)
    loss.backward()
    loss_train.append(loss.detach().numpy())
    optimizer.step()

    model.eval()
    x_valid = Variable(torch.from_numpy(x_df_valid.values).float())
    y_valid = Variable(torch.from_numpy(y_df_valid.values).float())
    y_valid = y_valid.view(-1,1)
    prediction = model(x_valid)
    vloss = loss_func(prediction, y_valid)
    loss_valid.append(vloss.detach().numpy())

    print('epoch = {}, train_loss = {}, valid_loss = {}'.format(t,loss.detach().numpy(),vloss.detach().numpy()),end='\r')




plt.plot(loss_train, label='train_loss')
plt.plot(loss_train, label='valid_loss')
plt.legend(loc='best')
plt.show()


