import neat
import logging
import pickle as pickle
# import gym
# from pureples.shared.visualize import draw_net
# from pureples.shared.gym_runner import run_neat

import copy
import os

import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
# import visualize
import sklearn.metrics
import neat
from pytocl.main import main
from my_driver import MyDriver




print("--------------------------------------------------------------------------------------------------------------------")


def eval_fitness(genomes, config):
    # start torcs serever
    # start torcs client
    # wait until finished
    # fitness = results
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        pickle.dump(net,open('network.p','wb'))
        print("Starting simulation...")

        # run_torcs()
        last_path = (os.getcwd())
        os.chdir("../")
        # print(os.getcwd())
        os.system('./torcs_tournament.py quickrace.yml') # forza
        os.chdir(last_path)

        # main(MyDriver(logdata=False))

        # genome.fitness =
        # print(genome.fitness)

        fitness_file = open('fitness.txt', 'r')
        for f in fitness_file:
            genome.fitness = float(f)
        print("Fitness: ",genome.fitness)


def ini_pop(state, stats, config, output):
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop




# -----------------------------------------------------------

# Config for NEAT.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_torcs1')


# Setup logger and environment.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Run!
# trials = 1
# Create population and train the network. Return winner of network running 100 episodes.

stats_one = neat.statistics.StatisticsReporter()
pop = ini_pop(None, stats_one, config, False)
pop.add_reporter(neat.Checkpointer(0))
winner = pop.run(eval_fitness, 100)

# pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-259')
# stats_ten = neat.statistics.StatisticsReporter()
# pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, False)
# pop.add_reporter(neat.Checkpointer(0))
# winner = pop.run(eval_fitness, 100)


# Save net if wished reused and draw it to file.
net = neat.nn.FeedForwardNetwork.create(winner, config)
# draw_net(net, filename="neat_torcs_winner")
with open('neat_torcs_winner.pkl', 'wb') as output:
    pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
