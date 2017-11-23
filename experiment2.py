"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import copy
import os
import pickle

import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
# import visualize
import sklearn.metrics
import neat
from pytocl.main import main
from my_driver import MyDriver


print("--------------------------------------------------------------------------------------------------------------------")

def eval_genomes(genomes, config):
    # start torcs serever
    # start torcs client
    # wait until finished
    # fitness = results
    for genome_id, genome in genomes:
        genome.fitness = 0

        net = neat.nn.RecurrentNetwork.create(genome, config)

        with open('network.p', 'wb') as handle:
            pickle.dump(net, handle, protocol=pickle.HIGHEST_PROTOCOL)

        main(MyDriver(logdata=False))

        # predictions = []
        # for input, output_real in zip((inputs), (outputs)):
        #     output_pred = te.activate(input)
        #     predictions.append(output_pred)
        #     # genome.fitness += (  abs(output_real[0] - output_pred[0])  )


        # fitness = 0 - sklearn.metrics.mean_squared_error(outputs,predictions)
        # f = open('fitness.txt', 'w')
        # f.write(str(fitness) + "\n")
        # f.close()

        fitness_file = open('fitness.txt', 'r')
        for f in fitness_file:
            fitness = f

        genome.fitness  = float(f)

        # genome.fitness -= abs(sklearn.metrics.r2_score(outputs,predictions))
        # print(genome.fitness)



def run(config_file):
    # Load configuration.

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

    # Create the population, which is the top-level object for a NEAT run.
    # p = neat.Population(config)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-63')

    #
    # thenet = neat.nn.FeedForwardNetwork.create(p.best_genome, config)
    # output = thenet.activate(i)
    #
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(0))

    # Run for up to 300 generations.
    print("before fitness")
    winner = p.run(eval_genomes, 10)

    import gzip

    try:
        import pickle as pickle  # pylint: disable=import-error
    except ImportError:
        import pickle  # pylint: disable=import-error

    def save_object(obj, filename):
        with gzip.open(filename, 'w', compresslevel=5) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_object(filename):
        with gzip.open(filename) as f:
            obj = pickle.load(f)
            return obj

    save_object(winner,"neat.pkl")
    # winner = load_object("neat.pkl")

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    # winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    # for i, o in zip(inputs, outputs):
    #     output = winner_net.activate(i)
    #     # print("input {!r}, expected output {!r}, got {!r}".format(i, o, output))
    #     print("expected output {!r}, got {!r}".format( o, output))

    # node_names = {0:'Acc', 1: 'Brk', 2:'Str'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=True, view=True)
    # visualize.plot_species(stats, view=True)


    # p.run(eval_genomes, 1)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)
