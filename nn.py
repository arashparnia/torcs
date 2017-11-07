"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import copy
import os
import neat
import pandas as pd
from sklearn.model_selection import train_test_split

import visualize

# 2-input XOR inputs and expected outputs.

# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
mypath = './train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
data2 = pd.read_csv( mypath + datafile2, index_col=False)
data3 = pd.read_csv( mypath + datafile3, index_col=False)

data = pd.concat([data1,data2,data3])

data  = data.fillna(0,axis=0,inplace=False)


data.dropna(axis=0,inplace=True)

d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)


Y = d1[['ACCELERATION','BRAKE','STEERING']]
X = d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
# Y = data[['ACCELERATION']]
# X = data[['SPEED','TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]



X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.8,random_state= 42)


# -----------------------------------------------------------

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Don't cheat - fit only on training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # apply same transformation to test data
# X_test = scaler.transform(X_test)


# -----------------------------------------------------------


inputs = X_train.values.tolist()
outputs = Y_train.values.tolist()
# -----------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------------")
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = -1000
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for i, o in zip((inputs), (outputs)):
            output = net.activate(i)
            genome.fitness -= (  abs(o[0] - output[0])   + abs(o[1] - output[1]) + abs(o[2] - output[2]) )


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-115')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(0))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10000000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i, o in zip(inputs, outputs):
        output = winner_net.activate(i)
        print("input {!r}, expected output {!r}, got {!r}".format(i, o, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=True, view=True)
    visualize.plot_species(stats, view=True)


    # p.run(eval_genomes, 1)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    run(config_path)