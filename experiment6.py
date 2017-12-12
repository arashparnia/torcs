import neat
import logging
import pickle as pickle
import gym
import numpy as np
import pandas as pd
import copy
import sklearn.metrics
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
# from pureples.shared.gym_runner import run_es

from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork
import neat
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
# data.dropna(axis=0,inplace=True)
d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)
# Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
Y = pd.DataFrame(d1[['STEERING']])
# X = pd.DataFrame(d2[['TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1' , 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])
X = pd.DataFrame(d2[[ 'ANGLE_TO_TRACK_AXIS' ]])

X = (X - X.mean()) / (X.max() - X.min())
Y = (Y - Y.mean()) / (Y.max() - Y.min())
rng = np.random.RandomState(42)
inputs = np.array(X)
outputs = np.array(Y)

print(inputs.shape)




def eval_genomes(genomes, config):
    # start torcs serever
    # start torcs client
    # wait until finished
    # fitness = results
    for genome_id, genome in genomes:
        genome.fitness = 0
        # cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        # net = create_phenotype_network(cppn, sub, "sigmoid")

        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()
        # net = neat.nn.FeedForwardNetwork.create(genome, config)

        # make a file from net in torcs folder
        # run torcs
        # read fitness

        predictions = []
        for input, output_real in zip((inputs), (outputs)):
            output_pred = net.activate(input)
            # print(output_pred,output_real)
            predictions.append(output_pred)

        genome.fitness = 0 - sklearn.metrics.mean_squared_error(outputs,predictions)
        # print(genome.fitness)

# Network input and output coordinates.
# input_coordinates = [(-0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.), (0.33, -1.)]
# output_coordinates = [(-0.5, 1.),(-0.5, 1.),(-0.5, 1.)]
#
# sub = Substrate(input_coordinates, output_coordinates)

# Network input and output coordinates.
# input_coordinates = [ (-0.8, -1.), (-0.7, -1.), (-0.6, -1.), (-0.5, -1.), (-0.4, -1.), (-0.3, -1.), (-0.2, -1.), (-0.1, -1.), (-0.05, -1.), (0.01, -1.), (0.01, -1.), (0.05, -1.), (0.1, -1.), (0.2, -1.), (0.3, -1.), (0.4, -1.), (0.5, -1.), (0.6, -1.), (0.7, -1.), (0.8, -1.), (0.9, -1.)]
input_coordinates = [(-0.5, -1.) ]
output_coordinates = [(0.5,1.)]

# hidden_coordinates = [[(0.0, 0.0)]]

sub = Substrate(input_coordinates,output_coordinates)#,hidden_coordinates)
# activations = len(hidden_coordinates) + 2


# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0,
          "max_depth": 100,
          "variance_threshold": 0.9,
          "band_threshold": 0.3,
          "iteration_level": 10,
          "division_threshold": 0.5,
          "max_weight": 8.0,
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_neat')




# Setup logger and environment.
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# env = gym.make("MountainCar-v0")

def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


stats_one = neat.statistics.StatisticsReporter()
pop = ini_pop(None, stats_one, config, True)
winner = pop.run(eval_genomes, 1)

print("done")
# stats_ten = neat.statistics.StatisticsReporter()
# pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, True)

# winner = pop.run(eval_genomes, 100)
#

# print(winner)
# Save CPPN if wished reused and draw it + winner to file.
# cppn = neat.nn.FeedForwardNetwork.create(winner, config)
# net = create_phenotype_network(cppn, sub)
# draw_net(cppn, filename="hyperneat_torcs_cppn")
# draw_net(net, filename="hyperneat_torcs_winner")
# with open('hyperneat_torcs_cppn.pkl', 'wb') as output:
#     pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)


# Save CPPN if wished reused and draw it + winner to file.
# cppn = neat.nn.FeedForwardNetwork.create(winner, config)
# network = ESNetwork(sub, cppn, params)
# net = network.create_phenotype_network(filename="es_hyperneat_mountain_car_small_winner")
# draw_net(cppn, filename="es_hyperneat_mountain_car_small_cppn")
# with open('es_hyperneat_mountain_car_small_cppn.pkl', 'wb') as output:
#     pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

