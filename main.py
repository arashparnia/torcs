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


# do this on simulator
# def drive(self, carstate: State) -> Command:
#     """
#     Produces driving command in response to newly received car state.
#
#     This is a dummy driving routine, very dumb and not really considering a
#     lot of inputs. But it will get the car (if not disturbed by other
#     drivers) successfully driven along the race track.
#     """
#     command = Command()
#     # self.steer(carstate, 0.0, command)
#
#     # ACC_LATERAL_MAX = 6400 * 5
#     # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
#     # v_x = 80
#     #
#     # self.accelerate(carstate, v_x, command)
#     #
#     # if self.data_logger:
#     #     self.data_logger.log(carstate, command)
#
#
#     test_X = [carstate.angle] + list(
#         carstate.distances_from_edge[0:18])
#
#     # test_X = [carstate.speed_x, carstate.distance_from_center, carstate.angle] + list(
#     #     carstate.distances_from_edge[0:18])
#
#     test_X = np.asarray(test_X)
#     # print(test_X)
#     test_X = test_X.reshape(1, -1)
#
#     linear = joblib.load('./linearmodel.pkl')
#     linearSTEERING = linear.predict(test_X)
#
#     nn = joblib.load('./nnmodel.pkl')
#     nnSTEERING = nn.predict(test_X)
#
#     # STEERING
#     # if (STEERING > 1):
#     #     STEERING = STEERING * 0.5
#     #     STEERING = min (STEERING , 1)
#     # if (STEERING < -1):
#     #     STEERING = STEERING * 0.5
#     #     STEERING = max(STEERING, -1)
#
#
#
#     # steering_error = 0 - carstate.distance_from_center
#     # STEERING_DEFAULT = self.steering_ctrl.control(
#     #     steering_error,
#     #     carstate.current_lap_time
#     # )
#
#     if (carstate.distances_from_edge[9] > 100):
#         command.steering = linearSTEERING
#     else:
#         command.steering = nnSTEERING
#     print(carstate.distances_from_edge[9], linearSTEERING, nnSTEERING)
#     # command.brake = BRAKE
#
#
#
#
#     # if (nnSTEERING < 0.9 or nnSTEERING > -0.9 ):
#     #     command.accelerator = 0.5
#     #     command.brake = 0
#     # else:
#     #     command.accelerator = 0
#     #     command.brake = 0.8
#
#
#     if carstate.rpm > 8000:
#         command.gear = carstate.gear + 1
#     if carstate.rpm < 2500:
#         command.gear = carstate.gear - 1
#
#     if not command.gear:
#         command.gear = carstate.gear or 1
#
#     if abs(carstate.distance_from_center >= 1):
#         command.accelerator = 0.3
#
#     print(carstate.speed_x)
#     if (carstate.speed_x > 10):
#         command.accelerator = 0
#         command.brake = 0
#     else:
#         command.accelerator = 0.5
#         command.brake = 0
#
#     return command
