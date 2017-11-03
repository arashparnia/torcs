from pprint import pprint

from pandas import scatter_matrix
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


from sklearn.externals import joblib


datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
mypath = './train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
data2 = pd.read_csv( mypath + datafile2, index_col=False)
data3 = pd.read_csv( mypath + datafile3, index_col=False)

data = pd.concat([data1,data2,data3])
# data = data1



data.dropna(axis=0,inplace=True)
# print(list(data.head()))
Y = data[['STEERING']]
X = data[[ 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17']]



train_X, test_X, train_y, test_y = train_test_split(X,Y,test_size= 0.3,random_state= 42)
# print(list(train_y))


clf = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 22), random_state=42,verbose = False)
# clf = linear_model.LinearRegression()
clf.fit(train_X,train_y )


pred_y =  clf.predict(test_X)

pred_y = pd.DataFrame(pred_y)
# print(pred_y.head())
# print(test_y.head())

# print(accuracy_score(test_y,pred_y,normalize=False))
kfold = model_selection.KFold(n_splits=10,random_state=42)
scoring = "neg_mean_absolute_error"
results = model_selection.cross_val_score(clf,X,Y,cv=kfold,scoring=scoring)
print("-----------------------------")
print(results.mean(),results.std())
#

# # ----------------------------------------------------------------------------------
# # Create linear regression object
# regr = linear_model.LinearRegression()
#
# # Train the model using the training sets
# regr.fit(train_X, train_y)
#
# # Make predictions using the testing set
# pred = regr.predict(test_X)
#
# # for p,q in zip(pred,list(test_y)):
# #     print("predicted: " , p, " Actual ", q)
#
# # for p in test_y:
# #     print (p)
# # for p in pred:
# #     print(p)
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(test_y, pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(test_y, pred))
#
# # Plot outputs
# plt.scatter(test_X, test_y,  color='black')
# plt.plot(test_X, pred, color='blue', linewidth=3)
#
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

joblib.dump(clf, 'nnmodel.pkl')

# load with this
# clf = joblib.load('filename.pkl')
#   commands to car = clf.predict(input from car sensors)