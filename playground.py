from nn_model import ClassClassifier

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


# y_label ='is_canceled'
y_label ='label'
class_num = 10
not_for_train = ['ID','adr','reservation_status','reservation_status_date','concat_date',
                'arrival_date_year','arrival_date_week_number','is_canceled']
# not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_train = pd.read_csv('Dataset/train_final.csv')
# df_test = pd.read_csv('Dataset/test_final.csv')


_ = [df_train.pop(x) for x in not_for_train]


df_train_label = pd.read_csv('Dataset/train_label.csv')
np_label = df_train_label[y_label]
encode_label = np.eye(class_num)[np_label.astype(int)]

print(df_train, encode_label.shape)


inputDim = df_train.shape[1]
outputDim = 10
learningRate = 0.0001 
epochs = 100


model = ClassClassifier(inputDim, outputDim)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
loss_func = torch.nn.MSELoss()


loss_train = []
acc_train = []
for t in range(epochs):

    model.train()
    x_train = Variable(torch.from_numpy(df_train.values).float())
    y_train = Variable(torch.from_numpy(encode_label).float())
    # y_train = y_train.view(-1,1)
    prediction = model(x_train)
    loss = loss_func(prediction, y_train)
    tloss = loss.detach().numpy()
    loss_train.append(tloss)
    # print(torch.argmax(prediction, dim=1).reshape(-1).detach().numpy())
    acc = (torch.argmax(prediction, dim=1).reshape(-1).detach().numpy() == np_label.reshape(-1)).mean()
    acc_train.append(acc)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('epoch = {}, train_loss = {}, train_acc = {}'.format(t,tloss,acc),end='\r')


plt.plot(loss_train, label='train_loss')
plt.plot(acc_train, label='acc_train')
plt.legend(loc='best')
plt.show()