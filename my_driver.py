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
    flag = False
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

        # print("Unpickled network in driver")
        with open('network.p', 'rb') as input:
            g= pickle.load(input)

        nw_activate = neat.generate_network(g)
        # print("generated network")
        data1=nw_activate(data)
        command.accelerator= data1[0]
        command.brake=data1[1]
        command.steering=data1[2]

    

        # print("distance from start : ", carstate.distance_from_start)
        self.counter = self.counter + 1
        # print(self.counter)
        # if (self.counter < 200 and (command.accelerator != 1 or command.brake != 0)):
        if (carstate.distance_from_start < 10):
            self.flag = True

        if (self.counter > 500 and self.flag == False ):
            # print("distancer from start: ", carstate.distance_from_start)
            # print("fitness " ,0 )
            fitness = 0
            f = open('fitness.txt', 'w')
            f.write(str(fitness) + "\n")
            f.close()
            command.meta = 1

        if((carstate.distance_from_center>1 or carstate.distance_from_center<-1 or self.counter >8000)  and self.flag == True):
            fitness= carstate.distance_from_start
            print(self.counter)
            # print("distancer from start: ", carstate.distance_from_start)
            # print("fitness loop ", fitness)
            f = open('fitness.txt', 'w')
            f.write(str(fitness) + "\n")
            f.close()
            command.meta=1

        if carstate.rpm > 9000:
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
