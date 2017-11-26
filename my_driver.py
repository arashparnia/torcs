from pytocl.driver import Driver
from pytocl.car import State, Command
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
import pyrenn as prn
import os
import neat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class MyDriver(Driver):

    reward = 0
    punishment = 0
    # Override the `drive` method to create your own driver
    ...

    def drive(self, carstate: State) -> Command:
        """

        fitness calculation

        """

        command = Command()
##########################################################
        distance_from_edge = list(carstate.distances_from_edge[0:19])
        opponents = list(carstate.opponents[0:36])
        data = [carstate.speed_x,carstate.speed_y, carstate.distance_from_center, carstate.angle] + distance_from_edge + opponents
        # data = [carstate.speed_x, carstate.distance_from_center,
        #         carstate.angle] + distance_from_edge
        data = pd.DataFrame(data)
        data = np.array(data)
##########################################################
        #for old neat use below
        # print("Unpickled the population in driver")
        with open('network.p', 'rb') as input:
            net = pickle.load(input)
        data = net.activate(data)
        # command.accelerator = data[0]
        # command.brake = data[1]
        # command.steering = data[2]
        if (carstate.speed_x < 60):
            command.accelerator = 0.5
        command.brake = 0
        command.steering = data[0]
##########################################################

        self.reward +=  1 - abs(carstate.distance_from_center)

        MAX_COUNTER = 60

        # self.counter = self.counter + 1

        fitness = carstate.distance_from_start
        fitness += self.reward

        if fitness > 22000:
            fitness =0

        # self.reward +=  ( 180  - (abs(carstate.angle))) * 0.001

        # fitness += self.reward
        counter = carstate.current_lap_time

        if (counter > MAX_COUNTER):
            self.fitness_to_file(fitness)
            print("COUNTER MAX    fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if (counter > 5 and counter < 20 and carstate.speed_x < 4):
            fitness = 0
            self.fitness_to_file(fitness)
            print("DID NOT MOVE   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if (carstate.distance_raced < -2 ):
            fitness = 0
            self.fitness_to_file(fitness)
            print("BACKWARD fitness: ", fitness , " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if ( abs(carstate.distance_from_center) > 1.1 ):
            print(carstate.distance_raced)
            print(carstate.distance_from_start)
            self.fitness_to_file(fitness)
            print("OUT OF TRACK   fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1

        if (carstate.damage > 5000):
            self.fitness_to_file(fitness)
            print("DAMAGED        fitness: ", fitness, " counter: ", counter, " Distance: ",carstate.distance_from_start)
            command.meta = 1
##########################################################
        # print("Min opponent distance is", map(min,(zip(*carstate.opponents))))
        # print("list of tuple is: ", (carstate.opponents)[0])
        # print("Damage : ", carstate.damage)
        # print("opponents : ", carstate.opponents)

        if carstate.rpm > 9000:
            command.gear = carstate.gear + 1
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command

    def fitness_to_file(self,fitness):
        f = open('fitness.txt', 'w')
        f.write(str(fitness) + "\n")
        f.close()