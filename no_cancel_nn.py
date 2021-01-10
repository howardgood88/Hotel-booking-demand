from nn_model import linearRegression

import numpy as np 
import pandas as pd
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#################################################################
#                       Utilities
#################################################################


from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self,x,y):
        # self.x = torch.tensor(x,dtype=torch.float32)
        self.x = Variable(torch.from_numpy(x.values).float())
        # self.y = torch.tensor(y,dtype=torch.float32)
        self.y = Variable(torch.from_numpy(y.values).float())
        self.length = self.x.shape[0]
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]

    def __len__(self):
        return self.length


#################################################################
#                       Reading Data
#################################################################

y_label ='is_canceled'
not_for_train = ['ID','adr','reservation_status','reservation_status_date','concat_date',
                'arrival_date_year','arrival_date_week_number']
# not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_train = pd.read_csv('Dataset/train_final.csv')
# df_test = pd.read_csv('Dataset/test_final.csv')

df_train.sample(frac=1)
_ = [df_train.pop(x) for x in not_for_train]
df_valid = df_train.iloc[81945:, :]
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

inputDim = x_df_train.shape[1]
outputDim = 1
learningRate = 0.0001 
epochs = 50

model = linearRegression(inputDim, 1000, outputDim)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_func = torch.nn.BCEWithLogitsLoss()

trainset = dataset(x_df_train,y_df_train)
trainloader = DataLoader(trainset,batch_size=64,shuffle=False)
validset = dataset(x_df_valid,y_df_valid)
validloader = DataLoader(validset,batch_size=64,shuffle=False)

loss_train = []
loss_valid = []
acc_train = []
acc_valid = []
for t in range(epochs):
    for (x_train, y_train) in trainloader:
        model.train()
        # x_train = Variable(torch.from_numpy(x_df_train.values).float())
        # y_train = Variable(torch.from_numpy(y_df_train.values).float())
        y_train = y_train.view(-1,1)
        prediction = model(x_train)
        loss = loss_func(prediction, y_train)
        loss_train.append(loss.detach().numpy())
        acc = (prediction.reshape(-1).detach().numpy().round() == y_train.reshape(-1).detach().numpy()).mean()
        acc_train.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for (x_valid, y_valid) in validloader:
        model.eval()
        # x_valid = Variable(torch.from_numpy(x_df_valid.values).float())
        # y_valid = Variable(torch.from_numpy(y_df_valid.values).float())
        y_valid = y_valid.view(-1,1)
        prediction = model(x_valid)
        vloss = loss_func(prediction, y_valid)
        loss_valid.append(vloss)
        vacc = (prediction.reshape(-1).detach().numpy().round() == y_valid.reshape(-1).detach().numpy()).mean()
        acc_valid.append(vacc)

    print('epoch = {}, train_loss = {}, train_acc = {}, valid_loss = {}, valid_acc = {}'.format(t,loss.detach().numpy(),acc,vloss,vacc),end='\r')


plt.plot(loss_train, label='train_loss')
plt.plot(loss_train, label='valid_loss')
plt.plot(acc_train, label='acc_train')
plt.plot(acc_valid, label='acc_valid')
plt.legend(loc='best')
plt.show()