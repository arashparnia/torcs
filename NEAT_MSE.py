import pandas as pd
import pickle

from random import random

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


import numpy as np

from pprint import pprint

import numpy
from matplotlib import pyplot
from pandas import scatter_matrix
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import copy
import pyrenn as prn
import pyESN as ESN


datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
import os
mypath = os.getcwd()
mypath += '/train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
data2 = pd.read_csv( mypath + datafile2, index_col=False)
data3 = pd.read_csv( mypath + datafile3, index_col=False)

# data = pd.concat([data3,data2,data1])
data = data1

data  = data.fillna(data.interpolate(),axis=0,inplace=False)
data.dropna(axis=0,inplace=True)
d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)


Y = pd.DataFrame(d1[['BRAKE']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])


X.ANGLE_TO_TRACK_AXIS = X.ANGLE_TO_TRACK_AXIS / np.math.pi
X.SPEED = [(max(0, min(x, 200)) / 100) - 1 for x in X.SPEED]

value = lambda x: (max(min(x / 2, 1), -1))
X['TRACK_POSITION'] = X['TRACK_POSITION'].apply(value)
value1 = lambda x: ((x + 1) / 100.5) - 1

X.TRACK_EDGE_0 = X['TRACK_EDGE_0'].apply(value1)
X.TRACK_EDGE_1 = X['TRACK_EDGE_1'].apply(value1)
X.TRACK_EDGE_2 = X['TRACK_EDGE_2'].apply(value1)
X.TRACK_EDGE_3 = X['TRACK_EDGE_3'].apply(value1)
X.TRACK_EDGE_4 = X['TRACK_EDGE_4'].apply(value1)
X.TRACK_EDGE_5 = X['TRACK_EDGE_5'].apply(value1)
X.TRACK_EDGE_6 = X['TRACK_EDGE_6'].apply(value1)
X.TRACK_EDGE_7 = X['TRACK_EDGE_7'].apply(value1)
X.TRACK_EDGE_8 = X['TRACK_EDGE_8'].apply(value1)
X.TRACK_EDGE_9 = X['TRACK_EDGE_9'].apply(value1)
X.TRACK_EDGE_10 = X['TRACK_EDGE_10'].apply(value1)
X.TRACK_EDGE_11 = X['TRACK_EDGE_11'].apply(value1)
X.TRACK_EDGE_12 = X['TRACK_EDGE_12'].apply(value1)
X.TRACK_EDGE_13 = X['TRACK_EDGE_13'].apply(value1)
X.TRACK_EDGE_14 = X['TRACK_EDGE_14'].apply(value1)
X.TRACK_EDGE_15 = X['TRACK_EDGE_15'].apply(value1)
X.TRACK_EDGE_16 = X['TRACK_EDGE_16'].apply(value1)
X.TRACK_EDGE_17 = X['TRACK_EDGE_17'].apply(value1)
X.TRACK_EDGE_18 = X['TRACK_EDGE_18'].apply(value1)
#

X['FEATURE1']=abs(X.TRACK_EDGE_9 - X.TRACK_EDGE_8)
X['FEATURE2']=abs(X.TRACK_EDGE_9 - X.TRACK_EDGE_10)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.20,random_state= 42,shuffle=False)



X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


P = X_train
Y = Y_train
Ptest = X_test
Ytest = Y_test
#
#
with open('network_break.pkl', 'rb') as input1:
    neat_str = pickle.load(input1)
#
predictions_br = []
for input in (X_test):
          output_pred = neat_str.activate(input)
          predictions_br.append(output_pred)
predictions_br = np.array(predictions_br)
print("Brake MSE :", np.sqrt(np.mean((predictions_br - Y_test)**2)))

trainlen= len(X_train)

future = len(X_test)


prediction = predictions_br
plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen+future),data.ix[:,1],'r',label="Training data")
plt.plot(range(trainlen,trainlen+future),prediction,'c--', label="NEAT output")
lo,hi = plt.ylim()
plt.title("NEAT- Brake predictions")
plt.xlabel("Number of entries")
plt.ylabel("Brake")
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
lgd=plt.legend(bbox_to_anchor=[1,1] ,loc=(1,2),fontsize='x-small')
plt.tight_layout()
# plt.savefig("NEAT_Brake.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


# Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
Y = pd.DataFrame(d1[['STEERING']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])

X.ANGLE_TO_TRACK_AXIS = X.ANGLE_TO_TRACK_AXIS / np.math.pi
X.SPEED = [(max(0, min(x, 200)) / 100) - 1 for x in X.SPEED]

value = lambda x: (max(min(x / 2, 1), -1))
X['TRACK_POSITION'] = X['TRACK_POSITION'].apply(value)

value1 = lambda x: ((x + 1) / 100.5) - 1
X.TRACK_EDGE_0 = X['TRACK_EDGE_0'].apply(value1)
X.TRACK_EDGE_1 = X['TRACK_EDGE_1'].apply(value1)
X.TRACK_EDGE_2 = X['TRACK_EDGE_2'].apply(value1)
X.TRACK_EDGE_3 = X['TRACK_EDGE_3'].apply(value1)
X.TRACK_EDGE_4 = X['TRACK_EDGE_4'].apply(value1)
X.TRACK_EDGE_5 = X['TRACK_EDGE_5'].apply(value1)
X.TRACK_EDGE_6 = X['TRACK_EDGE_6'].apply(value1)
X.TRACK_EDGE_7 = X['TRACK_EDGE_7'].apply(value1)
X.TRACK_EDGE_8 = X['TRACK_EDGE_8'].apply(value1)
X.TRACK_EDGE_9 = X['TRACK_EDGE_9'].apply(value1)
X.TRACK_EDGE_10 = X['TRACK_EDGE_10'].apply(value1)
X.TRACK_EDGE_11 = X['TRACK_EDGE_11'].apply(value1)
X.TRACK_EDGE_12 = X['TRACK_EDGE_12'].apply(value1)
X.TRACK_EDGE_13 = X['TRACK_EDGE_13'].apply(value1)
X.TRACK_EDGE_14 = X['TRACK_EDGE_14'].apply(value1)
X.TRACK_EDGE_15 = X['TRACK_EDGE_15'].apply(value1)
X.TRACK_EDGE_16 = X['TRACK_EDGE_16'].apply(value1)
X.TRACK_EDGE_17 = X['TRACK_EDGE_17'].apply(value1)
X.TRACK_EDGE_18 = X['TRACK_EDGE_18'].apply(value1)


X['FEATURE1']=abs(X.TRACK_EDGE_9 - X.TRACK_EDGE_8)
X['FEATURE2']=abs(X.TRACK_EDGE_9 - X.TRACK_EDGE_10)



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.20,random_state= 42,shuffle=False)


# for index, row in Y_test.iterrows():
#         if(row['STEERING'] <0):
#             print(row['STEERING'])

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


P = X_train
Y = Y_train

Ptest = X_test
Ytest = Y_test

with open('neat_torcs_winner_steering_new.pkl', 'rb') as input1:
    neat_str = pickle.load(input1)


predictions_str = []
for input in (X_test):
          output_pred = neat_str.activate(input)
          predictions_str.append(output_pred)

predictions_str = np.array(predictions_str)

print("Steering MSE :", np.sqrt(np.mean((predictions_str - Y_test)**2)))


trainlen= len(X_train)

future = len(X_test)





prediction = predictions_str
plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen+future),data.ix[:,2],'r',label="Training data")
plt.plot(range(trainlen,trainlen+future),prediction,'c--', label="NEAT output")
lo,hi = plt.ylim()
plt.title("NEAT- Steering predictions")
plt.xlabel("Number of entries")
plt.ylabel("Steering")
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
lgd=plt.legend(bbox_to_anchor=[1,1] ,loc=(1,2),fontsize='x-small')
plt.tight_layout()
# plt.savefig("NEAT_Steering.png",bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()


