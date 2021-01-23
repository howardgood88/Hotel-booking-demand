from nn_model import linearRegression

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


#################################################################
#                       Reading Data
#################################################################


y_label ='label'
not_for_train = ['ID','reservation_status','reservation_status_date','concat_date',
                'arrival_date_year','arrival_date_week_number']
# not_for_test = ['ID','concat_date','arrival_date_year','arrival_date_week_number']
df_train = pd.read_csv('Dataset/train_final.csv')
# df_test = pd.read_csv('Dataset/test_final.csv')