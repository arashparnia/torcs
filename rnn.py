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
import pyESN as ESN
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
# from keras.layers.recurrent import LSTM
#
# in_out_neurons = 2
# hidden_neurons = 50
#
# model = Sequential()
# model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
# model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
# model.add(Activation("linear"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")


datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
import os
mypath = os.getcwd()
mypath += '/train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
# data2 = pd.read_csv( mypath + datafile2, index_col=False)
# data3 = pd.read_csv( mypath + datafile3, index_col=False)

# data = pd.concat([data3,data2,data1])
data = data1

data  = data.fillna(data.interpolate(),axis=0,inplace=False)
data.dropna(axis=0,inplace=True)
d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)
# Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
# Y = pd.DataFrame(d1[['STEERING']])
# Y = pd.DataFrame(d1[['STEERING']])
# X = pd.DataFrame(d2[['TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])

# X = X.values.tolist()
# Y = Y.values.tolist()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.0,random_state= 42)


rng = np.random.RandomState(42)
# os._exit()

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Don't cheat - fit only on training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # apply same transformation to test data
# # X_test = scaler.transform(X_test)


# X_train = X_train.reshape(1,-1)
# Y_train = Y_train.reshape(1,-1)
# X_test = X_test.reshape(1,-1)
# Y_test = Y_test.reshape(1,-1)

print("Input Traning data shape : ", X_train.shape)
print("Output Traning data shape : ",Y_train.shape)
# print("Input Test data shape : ", X_test.shape)
# print("Output Test data shape : ",Y_test.shape)




#====================================================================================================================
#Recurent Neural Network
#Transpose is required because the code checks the number of inputs fed to network and no of inputs
#in the training data

##https://pyrenn.readthedocs.io/en/latest/



X_train = X_train.transpose()
Y_train = Y_train.transpose()
# X_test = X_test.transpose()
# Y_test = Y_test.transpose()

import matplotlib.pyplot as plt
import pyrenn as prn

P = X_train
Y = Y_train
# Ptest = X_test
# Ytest = Y_test

# P0test=X_test.transpose()[0:3]
# Ptest=X_test.transpose()[3:4805]
# Y0test=Y_test.transpose()[0:3]
# Ytest=Y_test.transpose()[3:4805]
#
# Ptest=Ptest.transpose()
# Ytest= Ytest.transpose()
# P0test=P0test.transpose()
# Y0test= Y0test.transpose()

##
# #Create and train NN
#
# #create recurrent neural network with 1 input, 2 hidden layers with
# #2 neurons each and 1 output
# #the NN has a recurrent connection with delay of 1 timestep in the hidden
# # layers and a recurrent connection with delay of 1 and 2 timesteps from the output
# # to the first layer
net = prn.CreateNN([22,11,1],dIn=[0],dIntern=[],dOut=[1])

#
# #Train NN with training data P=input and Y=target
# #Set maximum number of iterations k_max to 100
# #Set termination condition for Error E_stop to 1e-3
# #The Training will stop after 100 iterations or when the Error <=E_stop
net = prn.train_LM(P,Y,net,verbose=True,k_max=100,E_stop= 1e-3)

prn.saveNN(net,"RNNmodel_st.mdl")
print("saved")
os._exit(0)
# print("loading")
# net = prn.loadNN("RNNmodel.mdl")
# print("loaded")
###
#Calculate outputs of the trained NN for train and test data
y = prn.NNOut(P,net)

# for i, o in zip(P, Y):
#     output = winner_net.activate(i)
#     print("input {!r}, expected output {!r}, got {!r}".format(i, o, output))
    # print("expected output {!r}, got {!r}".format(o, output))

# print(y)
# os._exit(0)
# ytest = prn.NNOut(Ptest,net)

###
#Plot results
fig = plt.figure(figsize=(11,7))
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212)
fs=18

#Train Data
ax0.set_title('Train Data',fontsize=fs)
ax0.plot(y,color='b',lw=2,label='NN Output')
ax0.plot(Y,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Train Data')
ax0.tick_params(labelsize=fs-2)
ax0.legend(fontsize=fs-2,loc='upper left')
ax0.grid()

#Test Data
# ax1.set_title('Test Data',fontsize=fs)
# ax1.plot(ytest,color='b',lw=2,label='NN Output')
# ax1.plot(Ytest,color='r',marker='None',linestyle=':',lw=3,markersize=8,label='Test Data')
# ax1.tick_params(labelsize=fs-2)
# ax1.legend(fontsize=fs-2,loc='upper left')
# ax1.grid()

fig.tight_layout()
plt.show()


##=============================================================
### Echo state model

# esn = ESN.ESN(n_inputs = 22,
#           n_outputs = 3,
#           n_reservoir = 100,
#           spectral_radius = 0.5,
#           sparsity = 0,
#           noise = 0.01,
#           out_activation=lambda x: x,
#           inverse_out_activation=lambda x: x,
#           random_state = rng,
#           silent = False)
#
#
#
# pred_train = esn.fit(X_train,Y_train,inspect=False)
#
#
# filename= 'ESNmodel.pkl'
# with open(filename, 'wb') as output:
#     pickle.dump(esn,output)
#
# print("test error:")
# pred_test = esn.predict(X_test)
# print(np.sqrt(np.mean((pred_test - Y_test)**2)))
#=====================================================================
