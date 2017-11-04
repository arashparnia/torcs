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
Y = data[['ACCELERATION','BRAKE','STEERING']]
X = data[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
# Y = data[['STEERING']]
# X = data[['TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17']]



train_X, test_X, train_y, test_y = train_test_split(X,Y,test_size= 0.5,random_state= 42)
# print(list(train_y))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(train_X)
train_X = scaler.transform(train_X)
# apply same transformation to test data
test_X = scaler.transform(test_X)



clf = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 100), random_state=42,verbose = False,warm_start=False,learning_rate='adaptive',activation='logistic')

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
# print(pred_y)
# print(min(pred_y) ,pred_y, max(pred_y))
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


# do this on simulator paste this to the mydriver file all of this

# from pytocl.driver import Driver
# from pytocl.car import State, Command
# from sklearn.externals import joblib
# import pandas as pd
# import numpy as np
# import os
# from neupy import algorithms, layers
# # from neupy import algorithms, layers, environment
# # from neupy.datasets import reber
# # old_distance = None
# class MyDriver(Driver):
#     old_distance = None
#     # Override the `drive` method to create your own driver
#     ...
#
#
#     def drive(self, carstate: State) -> Command:
#         """
#         Produces driving command in response to newly received car state.
#
#         This is a dummy driving routine, very dumb and not really considering a
#         lot of inputs. But it will get the car (if not disturbed by other
#         drivers) successfully driven along the race track.
#         """
#         command = Command()
#         # self.steer(carstate, 0.0, command)
#
#         # ACC_LATERAL_MAX = 6400 * 5
#         # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
#         # v_x = 80
#         #
#         # self.accelerate(carstate, v_x, command)
#         #
#         # if self.data_logger:
#         #     self.data_logger.log(carstate, command)
#
#
#
#         test_X = [carstate.speed_x, carstate.distance_from_center, carstate.angle] + list(
#             carstate.distances_from_edge[0:19])
#
#         test_X = np.asarray(test_X)
#         # print(test_X)
#         test_X = test_X.reshape(1, -1)
#
#         # from sklearn.preprocessing import StandardScaler
#         # scaler = StandardScaler()
#         # scaler.fit(test_X)
#         # test_X = scaler.transform(test_X)
#
#         # linear = joblib.load('./linearmodel.pkl')
#         # linearpred = linear.predict(test_X)
#         # linearACCELERATION = linearpred[0, 0]
#         # linearBREAK = linearpred[0, 1]
#         # linearSTEERING = linearpred[0,2]
#
#
#         nn = joblib.load('./nnmodel.pkl')
#         nnpred = nn.predict(test_X)[0]
#         nnACCELERATION = nnpred[0]
#         nnBREAK = nnpred[1]
#         nnSTEERING = nnpred[2]
#
#
#         # linearSTEERING = linearSTEERING * 0.001
#         if (carstate.distances_from_edge[9] > 100):
#             command.steering = nnSTEERING * 0.1
#             command.accelerator = nnACCELERATION
#             command.brake = nnBREAK * 0.1
#         else:
#             command.steering = nnSTEERING
#             command.accelerator = nnACCELERATION
#             command.brake = nnBREAK
#
#         if ((command.accelerator -  command.brake ) < 0.1  and carstate.speed_x < 2):
#             command.brake = 0
#
#         # if nnSTEERING > 0.9 or nnSTEERING < -0.9:
#         #     command.steering = nnSTEERING * 0.1
#
#         # if linearSTEERING > 0.9 or linearSTEERING < -0.9:
#         #     linearSTEERING = linearSTEERING * 0.1
#
#
#
#         # if abs(nnACCELERATION) > 0.6:
#         #     nnBREAK = 0
#
#         # if (nnSTEERING > 1):
#         #     nnSTEERING = nnSTEERING * 0.1
#         #     nnSTEERING = min(nnSTEERING, 1)
#         # if (nnSTEERING < -1):
#         #     nnSTEERING = nnSTEERING * 0.1
#         #     nnSTEERING = max(nnSTEERING, -1)
#         #
#         # if (linearSTEERING > 1):
#         #     linearSTEERING = linearSTEERING * 0.1
#         #     linearSTEERING = min(linearSTEERING, 1)
#         # if (linearSTEERING < -1):
#         #     linearSTEERING = linearSTEERING * 0.1
#         #     linearSTEERING = max(linearSTEERING, -1)
#
#         # steering_error = 0 - carstate.distance_from_center
#         # STEERING_DEFAULT = self.steering_ctrl.control(
#         #     steering_error,
#         #     carstate.current_lap_time
#         # )
#
#         # if (carstate.distances_from_edge[9] > 70):
#         #     command.steering = linearSTEERING
#         # else:
#         #     command.steering = nnSTEERING
#
#         # command.steering = linearSTEERING
#         # command.brake = BRAKE
#
#         print(command.steering , command.brake, command.accelerator)
#
#
#         # if (nnSTEERING < 0.9 or nnSTEERING > -0.9 ):
#         #     command.accelerator = 0.5
#         #     command.brake = 0
#         # else:
#         #     command.accelerator = 0
#         #     command.brake = 0.8
#
#
#
#         if carstate.rpm > 9000:
#             command.gear = carstate.gear + 1
#         if carstate.rpm < 2500:
#             command.gear = carstate.gear - 1
#
#         if not command.gear:
#             command.gear = carstate.gear or 1
#
#         # if abs(carstate.distance_from_center >= 1):
#         #     command.accelerator = 0.3
#
#         # # if (carstate.speed_x > 15):
#         # if nnSTEERING > 1:
#         #     command.accelerator = 0
#         #     command.brake = 0.1
#         # else:
#         #     command.accelerator = 0.7
#         #     command.brake = 0
#
#
#
#         return command
#
#
#
#
#         # """Command to drive car during next control cycle.
#         #
#         #     Attributes:
#         #         accelerator: Accelerator, 0: no gas, 1: full gas, [0;1].
#         #         brake:  Brake pedal, [0;1].
#         #         gear: Next gear. -1: reverse, 0: neutral,
#         #             [1;6]: corresponding forward gear.
#         #         steering: Rotation of steering wheel, -1: full right, 0: straight,
#         #             1: full left, [-1;1]. Full turn results in an approximate wheel
#         #             rotation of 21 degrees.
#         #         focus: Direction of driver's focus, resulting in corresponding
#         #             ``State.focused_distances_from_edge``, [-90;90], deg.
#         #     """
#
#
#
#         # """State of car and environment, sent periodically by racing server.
#         #
#         # Update the state's dictionary and use properties to access the various
#         # sensor values. Value ``None`` means the sensor value is invalid or unset.
#         #
#         # Attributes:
#         #     sensor_dict: Dictionary of sensor key value pairs in string form.
#         #     angle: Angle between car direction and track axis, [-180;180], deg.
#         #     current_lap_time: Time spent in current lap, [0;inf[, s.
#         #     damage: Damage points, 0 means no damage, [0;inf[, points.
#         #     distance_from_start:
#         #         Distance of car from start line along track center, [0;inf[, m.
#         #     distance_raced:
#         #         Distance car traveled since beginning of race, [0;inf[, m.
#         #     fuel: Current fuel level, [0;inf[, l.
#         #     gear: Current gear. -1: reverse, 0: neutral,
#         #         [1;6]: corresponding forward gear.
#         #     last_lap_time: Time it took to complete last lap, [0;inf[, s.
#         #     opponents: Distances to nearest opponents in 10 deg slices in
#         #         [-180;180] deg. [0;200], m.
#         #     race_position: Position in race with respect to other cars, [1;N].
#         #     rpm: Engine's revolutions per minute, [0;inf[.
#         #     speed_x: Speed in X (forward) direction, ]-inf;inf[, m/s.
#         #     speed_y: Speed in Y (left) direction, ]-inf;inf[, m/s.
#         #     speed_z: Speed in Z (up) direction, ]-inf;inf[, m/s.
#         #     distances_from_edge: Distances to track edge along configured driver
#         #         range finders, [0;200], m.
#         #     focused_distances_from_edge: Distances to track edge, five values in
#         #         five degree range along driver focus, [0;200], m. Can be used only
#         #         once per second and while on track, otherwise values set to -1.
#         #         See ``focused_distances_from_edge_valid``.
#         #     distance_from_center: Normalized distance from track center,
#         #         -1: right edge, 0: center, 1: left edge, [0;1].
#         #     wheel_velocities: Four wheels' velocity, [0;inf[, deg/s.
#         #     z: Distance of car center of mass to track surface, ]-inf;inf[, m.
#         # """
#
#     # current_lap_time: 662.884
#     # fuel: 93.1477
#     # rpm: 942.478
#     # last_lap_time: 0.0
#     # z: 0.342919
#     # speed_z: -0.003822
#     # angle: -53.27458981951567
#     # distance_from_start: 3083.42
#     # opponents: (
#     # 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
#     # 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
#     # 200.0, 200.0, 200.0, 200.0)
#     # damage: 654
#     # speed_y: -0.045844444444444445
#     # distances_from_edge: (
#     # -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
#     # race_position: 1
#     # distance_raced: 3108.42
#     # speed_x: 0.05645527777777778
#     # distance_from_center: 1.07016
#     # wheel_velocities: (-89.14764925999505, 0.4694283322552494, 63.65331920785393, 293.37788237658185)
#     # focused_distances_from_edge: (-1.0, -1.0, -1.0, -1.0, -1.0)
#     # gear: 1
