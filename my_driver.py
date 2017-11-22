from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
import pyrenn as prn
import os
import neat
#from neupy import algorithms, layers
# from neupy import algorithms, layers, environment
# from neupy.datasets import reber
# old_distance = None
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# stuck = False
# MAX_UNSTUCK_SPEED = 5
# MIN_UNSTUCK_DIST = 3
# MAX_UNSTUCK_ANGLE = 10
class MyDriver(Driver):
    stuck = False
    MAX_UNSTUCK_ANGLE = 15.0 / 180.0 * np.pi  #/ *[radians] * /
    UNSTUCK_TIME_LIMIT = 2.0# / *[s] * /
    MAX_UNSTUCK_SPEED = 5.0# / *[m / s] * /
    MIN_UNSTUCK_DIST = 3.0# / *[m] * /
    MAX_UNSTUCK_COUNT = 100
    stuck = 0
    counter = 0
    # Override the `drive` method to create your own driver
    ...





    def drive(self, carstate: State) -> Command:
        """

        fitness calculation

        """
        # if (self.isStuck(carstate)):
        #     print("STUCKKKK")
        command = Command()

        data = [carstate.speed_x, carstate.distance_from_center, carstate.angle] + list(
            carstate.distances_from_edge[0:19])
               # + [self.old_acc,self.old_brk,self.old_str]

        data = pd.DataFrame(data)

        data = np.array(data)

        # net = prn.loadNN("RNNmodel_acc.mdl")
        # out = prn.NNOut(data, net)

        print("Unpickled network in driver")
        with open('network.p', 'rb') as input:
            g= pickle.load(input)

        nw_activate = neat.generate_network(g)
        print("generated network")
        data1=nw_activate(data)

        command.acceleration= data1[0]
        command.brake=data1[1]
        command.steering=data1[2]
        # import pickle
        # esn = pickle.load(open('ESNmodel.pkl', 'rb'))
        # out= esn.predict(data)
        # data = np.array(data)
        # data = data.reshape(1, -1)

        # nn_acc = joblib.load('nnmodel_acc.pkl')
        # nnpred_acc = nn_acc.predict(data)
        #
        # nn_brk = joblib.load('nnmodel_brk.pkl')
        # nnpred_brk = nn_brk.predict(data)


        # nnACCELERATION = out[0]
        # nnBREAK = out[1]
        # nnSTEERING = out[2]

        print(command.accelerator, command.brake,command.steering)

        # if self.isStuck(carstate) :
        #     command.steering = -carstate.angle # / carstate. ->_steerLock;
        #     command.gear = -1  # reverse gear
        #     command.accelerator = 0.5  # 30 % accelerator pedal
        #     command.brake = 0.0 # no brakes
        # else :
        #     if (carstate.distances_from_edge[9] > 70):
        #         command.steering = nnSTEERING * 0.01
        #         command.accelerator =  min (nnACCELERATION , 1)
        #         command.brake = min (nnBREAK * 0.01,1)
        #     else:
        #         command.steering = nnSTEERING * 0.1
        #         command.accelerator = min(nnACCELERATION ,1)
        #         command.brake = min(nnBREAK  ,1 )
        # command.steering = nnSTEERING
        # command.accelerator = min(nnACCELERATION, 1)
        # command.brake = min(nnBREAK, 1)



        # print(carstate.distance_from_center, carstate.angle)
        # if abs(carstate.distance_from_center) >= 0.6:
        #     if (carstate.angle > 0):
        #         command.steering = command.steering - 1
        #     else:
        #         command.steering = command.steering + 1


        print("distance from start : ", carstate.distance_from_start)

        # if(carstate.distances_from_edge>1 or carstate.distances_from_edge<-1):
        #     print("fitness loop")
        #     fitness= carstate.distance_from_start
        #     f = open('fitness.txt', 'w')
        #     f.write(str(fitness) + "\n")
        #     f.close()
        #     command.meta=1
        # if(self.counter > 50):
        #     print("restarting")
        #     command.meta=1

        print("Counter value is: ", self.counter)


                # make this array
        if carstate.rpm > 7000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        if self.data_logger:
            self.data_logger.log(carstate, command)
        return command



    def isStuck(self, carstate: State):

        if carstate.angle > self.MAX_UNSTUCK_ANGLE and carstate.speed_x < self.MAX_UNSTUCK_SPEED and abs(
                carstate.distances_from_edge[9]) > self.MIN_UNSTUCK_DIST:

            if (self.stuck > self.MAX_UNSTUCK_COUNT and carstate.distances_from_edge[9] * carstate.angle < 0.0):
                return True
            else:
                self.stuck = self.stuck + 1
                return False
        else:
            self.stuck = 0
        return False

















        # values = data.values
        # # integer encode direction
        # encoder = LabelEncoder()
        # # values[:, 4] = encoder.fit_transform(values[:, 4])
        # # ensure all data is float
        # values = values.astype('float32')
        # # normalize features
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled = scaler.fit_transform(values)
        # # frame as supervised learning
        # reframed = self.series_to_supervised(scaled, 1, 1)
        # reframed.drop(reframed.columns[[range(28, 50)]], axis=1, inplace=True)
        # print(reframed.head())
        #
        # # values = scaled.values
        # test_X = values
        # #test_X = test_X.reshape((0, 1,27))
        # # test_X = np.asarray(values)
        # print(test_X.shape)


        # test_X = test_X.reshape((1, 1, 28))
        # print(test_X.shape)
        # test_X = test_X.reshape(1, -1)








































































        # """Command to drive car during next control cycle.
        #
        #     Attributes:
        #         accelerator: Accelerator, 0: no gas, 1: full gas, [0;1].
        #         brake:  Brake pedal, [0;1].
        #         gear: Next gear. -1: reverse, 0: neutral,
        #             [1;6]: corresponding forward gear.
        #         steering: Rotation of steering wheel, -1: full right, 0: straight,
        #             1: full left, [-1;1]. Full turn results in an approximate wheel
        #             rotation of 21 degrees.
        #         focus: Direction of driver's focus, resulting in corresponding
        #             ``State.focused_distances_from_edge``, [-90;90], deg.
        #     """



        # """State of car and environment, sent periodically by racing server.
        #
        # Update the state's dictionary and use properties to access the various
        # sensor values. Value ``None`` means the sensor value is invalid or unset.
        #
        # Attributes:
        #     sensor_dict: Dictionary of sensor key value pairs in string form.
        #     angle: Angle between car direction and track axis, [-180;180], deg.
        #     current_lap_time: Time spent in current lap, [0;inf[, s.
        #     damage: Damage points, 0 means no damage, [0;inf[, points.
        #     distance_from_start:
        #         Distance of car from start line along track center, [0;inf[, m.
        #     distance_raced:
        #         Distance car traveled since beginning of race, [0;inf[, m.
        #     fuel: Current fuel level, [0;inf[, l.
        #     gear: Current gear. -1: reverse, 0: neutral,
        #         [1;6]: corresponding forward gear.
        #     last_lap_time: Time it took to complete last lap, [0;inf[, s.
        #     opponents: Distances to nearest opponents in 10 deg slices in
        #         [-180;180] deg. [0;200], m.
        #     race_position: Position in race with respect to other cars, [1;N].
        #     rpm: Engine's revolutions per minute, [0;inf[.
        #     speed_x: Speed in X (forward) direction, ]-inf;inf[, m/s.
        #     speed_y: Speed in Y (left) direction, ]-inf;inf[, m/s.
        #     speed_z: Speed in Z (up) direction, ]-inf;inf[, m/s.
        #     distances_from_edge: Distances to track edge along configured driver
        #         range finders, [0;200], m.
        #     focused_distances_from_edge: Distances to track edge, five values in
        #         five degree range along driver focus, [0;200], m. Can be used only
        #         once per second and while on track, otherwise values set to -1.
        #         See ``focused_distances_from_edge_valid``.
        #     distance_from_center: Normalized distance from track center,
        #         -1: right edge, 0: center, 1: left edge, [0;1].
        #     wheel_velocities: Four wheels' velocity, [0;inf[, deg/s.
        #     z: Distance of car center of mass to track surface, ]-inf;inf[, m.
        # """

    # current_lap_time: 662.884
    # fuel: 93.1477
    # rpm: 942.478
    # last_lap_time: 0.0
    # z: 0.342919
    # speed_z: -0.003822
    # angle: -53.27458981951567
    # distance_from_start: 3083.42
    # opponents: (
    # 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
    # 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0,
    # 200.0, 200.0, 200.0, 200.0)
    # damage: 654
    # speed_y: -0.045844444444444445
    # distances_from_edge: (
    # -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
    # race_position: 1
    # distance_raced: 3108.42
    # speed_x: 0.05645527777777778
    # distance_from_center: 1.07016
    # wheel_velocities: (-89.14764925999505, 0.4694283322552494, 63.65331920785393, 293.37788237658185)
    # focused_distances_from_edge: (-1.0, -1.0, -1.0, -1.0, -1.0)
    # gear: 1



        # self.steer(carstate, 0.0, command)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        # v_x = 80
        #
        # self.accelerate(carstate, v_x, command)
        #
        # if self.data_logger:
        #     self.data_logger.log(carstate, command)






        # --------------------------------------------------------------

        # linear = joblib.load('./linearmodel.pkl')
        # linearpred = linear.predict(test_X)
        # linearACCELERATION = linearpred[0, 0]
        # linearBREAK = linearpred[0, 1]
        # linearSTEERING = linearpred[0, 2]
        # --------------------------------------------------------------
        # import keras
        # import tensorflow as tf
        # import h5py
        # conv = keras.models.load_model('./convmodel.mdl')
        # convpred = conv.predict(test_X)
        # --------------------------------------------------------------


        # --------------------------------------------------------------

        # if abs(carstate.distance_from_center) > 1:
        # #
        #
        #
        #     if carstate.speed_x < 1 :
        #         print('recovery mode ', command.steering)
        #         self.offroad = True
        #         # command.gear = -1
        #         # command.focus = 0
        #         command.brake = 0
        #         command.accelerator = 0.4
        #         if carstate.angle > 10:
        #             command.steering = command.steering + 1
        #         if carstate.angle < 10:
        #             command.steering = command.steering - 1
        #         if carstate.angle == 0:
        #             command.steering = 0
        #     else:
        #
        #         # command.focus = 0
        #         command.brake = 0.3
        #         command.accelerator = 0
        # else:
        # from keras.models import load_model
        # print("CHECK !!!!!")
        # model = load_model('LSTMmodel.mdl')
        # lstmpred = model.predict(test_X)
        # print(lstmpred)
        # nn = joblib.load('./nnmodel.pkl')
        # nnpred = nn.predict(test_X)[0]
        # nnACCELERATION = nnpred[0]
        # nnBREAK = nnpred[1]
        # nnSTEERING = nnpred[2]
        # linearSTEERING = linearSTEERING * 0.001
        # print(nnACCELERATION,nnBREAK,nnSTEERING)


#     def isStuck(self, carstate: State) -> bool:
#         trackangle = RtTrackSideTgAngleL( & (car->_trkPos));
#         angle = trackangle - car->_yaw;
#         NORM_PI_PI(angle);
#
#         angle =
#         if (fabs(angle) > MAX_UNSTUCK_ANGLE & &
#             car->_speed_x < MAX_UNSTUCK_SPEED & &
#                             fabs(car->_trkPos.toMiddle) > MIN_UNSTUCK_DIST) {
#         if (stuck > MAX_UNSTUCK_COUNT & & car->_trkPos.toMiddle * angle < 0.0) {
#         return true;
#         } else {
#         stuck + +;
#         return false;
#
#     }
#     } else {
#         stuck = 0;
#     return false;
#
# }
# }
#         if (self.stuck < 100):
#             self.stuck = self.stuck +1
#             return False
#         else:
#             return True


  # if abs(carstate.distance_from_center) > 1:
        #     command.accelerator = 0.4
        # if (carstate.angle > 100 and command.steering < 0.5) or (carstate.angle < 100 and command.steering > 0.5):
        #     command.steering = 0 - command.steering

        # if self.offroad == True and abs(carstate.distance_from_center) < 0.5:
        #     print ('CEHCK')
        #     command.brake = 1
        #     command.gear = 1
        #     command.accelerator =0
        #     self.offroad = False


        # if ((command.accelerator -  command.brake ) < 0.1  and carstate.speed_x < 2):
        #     command.brake = 0

        # if nnSTEERING > 0.9 or nnSTEERING < -0.9:
        #     command.steering = nnSTEERING * 0.1

        # if linearSTEERING > 0.9 or linearSTEERING < -0.9:
        #     linearSTEERING = linearSTEERING * 0.1



        # if abs(nnACCELERATION) > 0.6:
        #     nnBREAK = 0

        # if (nnSTEERING > 1):
        #     nnSTEERING = nnSTEERING * 0.1
        #     nnSTEERING = min(nnSTEERING, 1)
        # if (nnSTEERING < -1):
        #     nnSTEERING = nnSTEERING * 0.1
        #     nnSTEERING = max(nnSTEERING, -1)
        #
        # if (linearSTEERING > 1):
        #     linearSTEERING = linearSTEERING * 0.1
        #     linearSTEERING = min(linearSTEERING, 1)
        # if (linearSTEERING < -1):
        #     linearSTEERING = linearSTEERING * 0.1
        #     linearSTEERING = max(linearSTEERING, -1)

        # steering_error = 0 - carstate.distance_from_center
        # STEERING_DEFAULT = self.steering_ctrl.control(
        #     steering_error,
        #     carstate.current_lap_time
        # )

        # if (carstate.distances_from_edge[9] > 70):
        #     command.steering = linearSTEERING
        # else:
        #     command.steering = nnSTEERING

        # command.steering = linearSTEERING
        # command.brake = BRAKE

        # print(command.steering , command.brake, command.accelerator)


        # if (nnSTEERING < 0.9 or nnSTEERING > -0.9 ):
        #     command.accelerator = 0.5
        #     command.brake = 0
        # else:
        #     command.accelerator = 0
        #     command.brake = 0.8
