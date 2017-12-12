"""
2-input XOR example -- this is most likely the simplest possible example.
"""
import pickle as pickle
# from __future__ import print_function

import copy
import os

import pandas as pd
from pureples.shared.visualize import draw_net

from sklearn.model_selection import train_test_split
import numpy as np
import visualize
import sklearn.metrics
import neat
# 2-input XOR inputs and expected outputs.

# xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
# xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]


datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
mypath = './train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
data2 = pd.read_csv( mypath + datafile2, index_col=False)
# data3 = pd.read_csv( mypath + datafile3, index_col=False)

data = pd.concat([data1,data2])
# data = data1


# data = data1
# data  = data.fillna(data.,axis=0,inplace=False)

# print(data.shape)
data.dropna(axis=0,inplace=True)
# print(data.shape)

d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)


# Y = d1[['ACCELERATION','BRAKE','STEERING']]
# X = d2[[ 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
# Y = data[['SPEED']]
# X = data[['SPEED','TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']]
Y = pd.DataFrame(d1[['ACCELERATION','BRAKE']])
X = pd.DataFrame(d2[['SPEED','TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1' , 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])


X['TRACK_EDGE_9_8'] = abs(d2.TRACK_EDGE_9 - d2.TRACK_EDGE_8)
X['TRACK_EDGE_9_10'] = abs(d2.TRACK_EDGE_9 -  d2.TRACK_EDGE_10)


# print(X)
# X[['TRACK_EDGE_9_8']] = abs(d2[['TRACK_EDGE_9']] - d2[['TRACK_EDGE_8']])
# X[['TRACK_EDGE_9_10'] = abs(d2[['TRACK_EDGE_9']] - d2[['TRACK_EDGE_10']])
from sklearn import preprocessing


X = (X - X.mean()) / (X.max() - X.min())

# Y = (Y - Y.mean()) / (Y.max() - Y.min())
# x = X.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# # x_scaled = min_max_scaler.fit_transform(x)
# # X = pd.DataFrame(x_scaled)
#
# x1 = pd.DataFrame(min_max_scaler.fit_transform(X.T), columns=X.columns, index=X.index)
# X = x1.transpose
# # X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.0,random_state= 42)


# -----------------------------------------------------------

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# # Don't cheat - fit only on training data
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# # apply same transformation to test data
# X_test = scaler.transform(X_test)


# -----------------------------------------------------------

# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)

inputs = np.array(X)
outputs = np.array(Y)
# -----------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------------")

def eval_genomes(genomes, config):
    s=0
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

            # s+=1
            # if s % 10000 == 0:
            #     print("Predicted: " , output_pred,"      Real: ",output_real)
            # genome.fitness += (  abs(output_real[0] - output_pred[0])  )


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



def run(config_file):
    # Load configuration.

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    #
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-100')
    #
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
    winner = p.run(eval_genomes, 150)

    # import gzip
    #
    # try:
    #     import pickle as pickle  # pylint: disable=import-error
    # except ImportError:
    #     import pickle  # pylint: disable=import-error
    #
    # def save_object(obj, filename):
    #     with gzip.open(filename, 'w', compresslevel=5) as f:
    #         pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # def load_object(filename):
    #     with gzip.open(filename) as f:
    #         obj = pickle.load(f)
    #         return obj

    # save_object(winner,"neat.pkl")
    # winner = load_object("neat.pkl")

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    draw_net(winner_net, filename="neat_torcs_winner_steering")
    with open('neat_torcs_steering.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)


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
    config_path = os.path.join(local_dir, 'config2')
    run(config_path)