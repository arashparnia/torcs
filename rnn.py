import pandas as pd
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
data2 = pd.read_csv( mypath + datafile2, index_col=False)
data3 = pd.read_csv( mypath + datafile3, index_col=False)

data = pd.concat([data3,data2,data1])
# data = data1


data  = data.fillna(data.interpolate(),axis=0,inplace=False)
d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)
Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])



# X = X.values.tolist()
# Y = Y.values.tolist()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2,random_state= 42)


rng = np.random.RandomState(42)

print(ESN)
esn = ESN.ESN(n_inputs = 2,
          n_outputs = 1,
          n_reservoir = 200,
          spectral_radius = 0.25,
          sparsity = 0.95,
          noise = 0.001,
          input_shift = [0,0],
          input_scaling = [0.01, 3],
          teacher_scaling = 1.12,
          teacher_shift = -0.7,
          out_activation = np.tanh,
          inverse_out_activation = np.arctanh,
          random_state = rng,
          silent = False)



pred_train = esn.fit(X_train,Y_train)

print("test error:")
pred_test = esn.predict(X_test)
print(np.sqrt(np.mean((pred_test - Y_test)**2)))


