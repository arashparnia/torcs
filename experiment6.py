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
from pureples.shared.gym_runner import run_es
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
Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])


rng = np.random.RandomState(42)
inputs = np.array(X)
outputs = np.array(Y)




def eval_genomes(genomes, config):
    # start torcs serever
    # start torcs client
    # wait until finished
    # fitness = results
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # make a file from net in torcs folder
        # run torcs
        # read fitness

        predictions = []
        for input, output_real in zip((inputs), (outputs)):
            output_pred = net.activate(input)
            predictions.append(output_pred)

        genome.fitness = 0 - sklearn.metrics.mean_squared_error(outputs,predictions)

# Network input and output coordinates.
input_coordinates = [(-0.33, -1.), (0.33, -1.)]
output_coordinates = [(-0.5, 1.), (0., 1.), (0.5, 1.)]

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0,
          "max_depth": 1,
          "variance_threshold": 0.03,
          "band_threshold": 0.3,
          "iteration_level": 1,
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

# Run!
# Create population and train the network. Return winner of network running 100 episodes.
stats_one = neat.statistics.StatisticsReporter()
pop = ini_pop(None, stats_one, config, False)
winner = pop.run(eval_genomes, 1)
print(winner)
# Save CPPN if wished reused and draw it + winner to file.
cppn = neat.nn.FeedForwardNetwork.create(winner, config)
network = ESNetwork(sub, cppn, params)
net = network.create_phenotype_network(filename="es_hyperneat_mountain_car_small_winner")
draw_net(cppn, filename="es_hyperneat_mountain_car_small_cppn")
with open('es_hyperneat_mountain_car_small_cppn.pkl', 'wb') as output:
    pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

