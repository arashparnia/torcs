from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
import pyrenn as prn
import os
import neatlite
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class MyDriver(Driver):

    # reward = 0
    # punishment = 0
    # Override the `drive` method to create your own driver
    ...

    def drive(self, carstate: State) -> Command:
        """

        fitness calculation

        """

        command = Command()
##########################################################


        distance_from_edge =  list(carstate.distances_from_edge[0:19])
        # opponents = list(carstate.opponents[0:36])
        # data = [carstate.speed_x,carstate.speed_y, carstate.distance_from_center, carstate.angle] + distance_from_edge + opponents
        # distance_from_edge_left = abs(carstate.distances_from_edge[9] - carstate.distances_from_edge[8])
        # distance_from_edge_right = abs(carstate.distances_from_edge[9] - carstate.distances_from_edge[10])

        # data = [carstate.speed_x, carstate.distance_from_center, carstate.angle] + distance_from_edge
        # data = pd.DataFrame(data)
        # data = np.array(data)
##########################################################


        # =======================================================================================
        # net_break = joblib.load('nnmodel_brk.pkl')
        # data_break = net_break.predict(data.transpose())
        # command.brake = data_break[0]

        # net_acc = joblib.load('nnmodel_acc.pkl')
        # data_acc = net_acc.predict(data.transpose())
        # command.accelerator = data_acc[0]
        # ===============================================================================
        command.accelerator =1
        command.brake = 0
        command.steering =0

        # =============================================================================
        # data = [carstate.speed_x, carstate.speed_y,carstate.distance_from_center, carstate.angle] + distance_from_edge + [
        #     distance_from_edge_left, distance_from_edge_right]

        # with open('neat_torcs_winner_steering1.pkl', 'rb') as input:
        #     net_steering = pickle.load(input)
        # steering_data = net_steering.activate(data)
        # command.steering = steering_data[0]
# ========================================================================



# ========================================================================
        angle = carstate.angle / np.math.pi
        speed_x = (max(0, min(carstate.speed_x, 200)) / 100) - 1
        distance_from_center = max(min(carstate.distance_from_center / 2, 1), -1)
        distance_from_edge = [((x + 1) / 100.5) - 1 for x in carstate.distances_from_edge]

        distance_from_edge_left = abs(distance_from_edge[9] - distance_from_edge[8])/2
        distance_from_edge_right = abs(distance_from_edge[9] - distance_from_edge[10])/2

        data = [speed_x, distance_from_center, angle] + distance_from_edge + [
            distance_from_edge_left, distance_from_edge_right]


        with open('neat_torcs_winner_steering_new.pkl', 'rb') as input:
            net_steering = pickle.load(input)
        steering_data = net_steering.activate(data)
        command.steering = steering_data[0]

        # with open('neat_torcs_winner_brake.pkl', 'rb') as input:
        #     net_brake = pickle.load(input)
        # brake_data = net_brake.activate(data)
        # brake_data[0] = brake_data[0] * 0.1
        # print(brake_data[0])
        # command.brake = brake_data[0]
        # command.accelerator = 1 - brake_data[0]
        # if (command.brake > 0.5):
        #     command.accelerator = 0

# ========================================================================
        with open('network.p', 'rb') as input:
            net = pickle.load(input)
            # nw_activate = neatlite.generate_network(net)
            # data_ = nw_activate(data)
        data_ = net.activate(data)
        # data = data_[0]


        # print(data_)
        command.brake = data_[0]
        if data_[0] > 0.5:
            command.accelerator = 0




        # if command.brake > 0:
        #     command.accelerator=0
        # if (data > 0.0):
        #     command.accelerator = 1
        #     command.brake =0
        # if (data < 0.0):
        #     command.accelerator = 0
        #     command.brake = abs(data)

        #
        # if command.brake > 0.3:
        #     command.accelerator = 0
        # else:
        #     command.accelerator = 1
        #

        # if (command.steering >0.4):
        #     command.accelerator = 0
        #         #
        # command.accelerator = 1
        # MAXSPEED = min(distance_from_edge_left , distance_from_edge_right)
        # if (carstate.distances_from_edge[9] < 200):
        #     if (carstate.speed_x > max(MAXSPEED, 35) ):
        #         command.brake=   0.7 # max(0.7,command.brake)
        #         command.accelerator =  0 #min(0.3,command.accelerator)
        # else:
        #     command.steering = command.steering * 0.1
        #     command.brake =  command.brake * 0.1

        #
        #     if (distance_from_edge_right < 25 or distance_from_edge_left < 25):
        #         if (carstate.speed_x > 15):
        #             command.brake = max(0.6, command.brake)
        #     if (distance_from_edge_right < 5 or distance_from_edge_left < 5):
        #         if (carstate.speed_x > 10):
        #             command.brake = max(0.6, command.brake)


        #
        # if carstate.distances_from_edge[9] > 70:
        #     command.steering = command.steering * 0.1

    ##########################################################
    # FITNESS

    #################################
        #
        MAX_DISTANCE = 4400
        MAX_COUNTER = 45
        TRACK_LENGTH = MAX_DISTANCE - 100
        # self.counter = self.counter + 1
        fitness = carstate.distance_from_start
        # fitness = 10000 - carstate.current_lap_time
        # fitness += self.reward

        if carstate.distance_from_start > TRACK_LENGTH:
            fitness =0

        # self.reward +=  ( 180  - (abs(carstate.angle))) * 0.001

        # fitness += self.reward
        counter = carstate.current_lap_time

        # if (carstate.distance_from_start > MAX_DISTANCE and carstate.distance_raced > 100):
        #     self.fitness_to_file(fitness)
        #     print("COUNTER DISTANCE    fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
        #     command.meta = 1
        if (counter > MAX_COUNTER):
            self.fitness_to_file(fitness)
            print("COUNTER MAX    fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1
        # if (abs(carstate.angle) > 5 and carstate.current_lap_time > 5):
        #     self.fitness_to_file(fitness)
        #     print("BAD TURN   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
        #     command.meta = 1

        if (counter > 5 and counter < 10 and carstate.speed_x < 4):
            # fitness = 0
            self.fitness_to_file(fitness)
            print("DID NOT MOVE   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if (counter > 20 and carstate.speed_x < 1):
            # fitness = 0
            self.fitness_to_file(fitness)
            print("STOPED MOVING   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if (carstate.distance_raced < -2 ):
            fitness = 0
            self.fitness_to_file(fitness)
            print("BACKWARD fitness: ", fitness , " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if ( abs(carstate.distance_from_center) > 0.9 ):
            # fitness = 0
            self.fitness_to_file(fitness)
            print("OUT OF TRACK   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        # if (carstate.damage > 5000):
        #     self.fitness_to_file(fitness)
        #     print("DAMAGED        fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
        #     command.meta = 1
        if (carstate.distance_from_start > TRACK_LENGTH - 100  and carstate.distance_from_start < TRACK_LENGTH):
            self.fitness_to_file(TRACK_LENGTH)
            print("MAX FITNESS   fitness: ", TRACK_LENGTH, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1#########################




        # prevent sliding when out of track
        # if carstate.distance_from_center > 1.1 : # out of track to left
        #     if carstate.speed_x > 5 :
        #         command.brake =1
        #     else:
        #         command.brake = 0
        #     command.accelerator = 0.1
        #     command.steering =  -0.5
        #
        # elif carstate.distance_from_center < -1.1: # to right
        #     if carstate.speed_x > 5 :
        #         command.brake =1
        #     else:
        #         command.brake = 0
        #     command.accelerator = 0.1
        #     command.steering =  0.5


        if carstate.rpm > 9000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 4500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        if self.data_logger:
            self.data_logger.log(carstate, command)

        #initial acceleration
        if (carstate.current_lap_time < 6):
            command.accelerator = 1
            command.brake = 0
            command.steering = 0
        # prevent stoping completly
        # if (carstate.speed_x < 20):
        #     command.accelerator = 1
        #     command.brake=0

        # if (carstate.distances_from_edge[9] > 200):
        #     command.accelerator=1
        #     command.brake=0
            # command.steering = command.steering * 0.1
        return command

    def fitness_to_file(self,fitness):
        f = open('fitness.txt', 'w')
        f.write(str(fitness) + "\n")
        f.close()

