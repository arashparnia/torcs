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
        net = neat.nn.RecurrentNetwork.create(genome, config)

        pickle.dump(net,open('network.p','wb'))
        print("Starting simulation...")

        # run_torcs from yaml
        # last_path = (os.getcwd())
        # os.chdir("../")
        # # print(os.getcwd())
        # os.system('./torcs_tournament.py quickrace.yml') # forza
        # os.chdir(last_path)

        main(MyDriver(logdata=False))

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


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    trials = 1
    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()

    pop = neat.Checkpointer.restore_checkpoint('neat-checkpoint-11')
    # pop = ini_pop(None, stats_one, config, output)
    pop.add_reporter(neat.Checkpointer(1))
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


# -----------------------------------------------------------

# Config for NEAT.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_torcs')


# Setup logger and environment.
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Run!
winner = run(1000, eval_fitness)[0]

# Save net if wished reused and draw it to file.
net = neat.nn.RecurrentNetwork.create(winner, config)
# draw_net(net, filename="neat_torcs_winner")
with open('neat_torcs_winner.pkl', 'wb') as output:
    pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)
