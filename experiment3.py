import neat
import logging
import pickle as pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.gym_runner import run_neat

import copy
import os

import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
import visualize
import sklearn.metrics
import neat



print("--------------------------------------------------------------------------------------------------------------------")

def eval_fitness(genomes, config):
    # start torcs serever
    # start torcs client
    # wait until finished
    # fitness = results
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.RecurrentNetwork.create(genome, config)

        # make a file from net in torcs folder
        # run torcs
        # read fitness

        predictions = []
        for input, output_real in zip((inputs), (outputs)):
            output_pred = net.activate(input)
            predictions.append(output_pred)
            # print(output_pred,output_real)
            # genome.fitness += (  abs(output_real[0] - output_pred[0])  )
        # print(predictions[10],outputs[10])

        # fitness = 0 - sklearn.metrics.mean_squared_error(outputs,predictions)
        genome.fitness = 0 - sklearn.metrics.mean_squared_error(outputs,predictions)
        # f = open('fitness.txt', 'w')
        # f.write(str(fitness) + "\n")
        # f.close()
        #
        # fitness_file = open('fitness.txt', 'r')
        # for f in fitness_file:
        #     genome.fitness  = float(f)
        # genome.fitness -= abs(sklearn.metrics.r2_score(outputs,predictions))
        # print(genome.fitness)


def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    trials = 1
    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials is 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run(gens, eval_fitness):
    winner, stats = run_neat(gens, eval_fitness, 200, config, max_trials=0)
    print("neat_torcs done")
    return winner, stats



datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
mypath = './train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
data2 = pd.read_csv( mypath + datafile2, index_col=False)
data3 = pd.read_csv( mypath + datafile3, index_col=False)

data = pd.concat([data1,data2,data3])

data  = data.fillna(0,axis=0,inplace=False)


# data.dropna(axis=0,inplace=True)

d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)


Y = d1[['ACCELERATION','BRAKE','STEERING']]
X = d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
# Y = data[['STEERING']]
# X = data[['SPEED','TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]



inputs = np.array(X)
outputs = np.array(Y)
# -----------------------------------------------------------

# Config for NEAT.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_torcs')


# Setup logger and environment.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Run!
winner = run(10, eval_fitness)[0]

# Save net if wished reused and draw it to file.
net = neat.nn.RecurrentNetwork.create(winner, config)
draw_net(net, filename="neat_torcs_winner")
with open('neat_torcs_winner.pkl', 'wb') as output:
    pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
