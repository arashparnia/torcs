from pandas import scatter_matrix
from pandas.io.parsers import read_csv
from sklearn import preprocessing, model_selection
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


from sklearn.externals import joblib


datafile = 'aalborg.csv'
mypath = './train_data/'
data = pd.read_csv( mypath + datafile)
data.dropna(axis=0,inplace=True)
# print(list(data.head()))
Y = data[['ACCELERATION','BRAKE','STEERING']]
X = data[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17']]


train_X, test_X, train_y, test_y = train_test_split(X,Y,test_size= 0.3,random_state= 42)
print(list(train_y))
print(list(test_X))

clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 100), random_state=42,verbose = True)
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


joblib.dump(clf, 'model.pkl')

# load with this
# clf = joblib.load('filename.pkl')
#   commands to car = clf.predict(input from car sensors)