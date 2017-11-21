# ‘r’ – Read mode which is used when the file is only being read
# ‘w’ – Write mode which is used to edit and write new information to the file (any existing files with the same name will be erased when this mode is activated)
# ‘a’ – Appending mode, which is used to add new data to the end of the file; that is new information is automatically amended to the end
# ‘r+’ – Special read and write mode, which is used to handle both actions when working with a file


# from standard_neat import start_neuroevolution

import numpy as np
import pandas as pd
import copy

import sklearn
from sklearn.model_selection import train_test_split


datafile1 = 'aalborg.csv'
datafile2 = 'alpine-1.csv'
datafile3 = 'f-speedway.csv'
import os
mypath = os.getcwd()
mypath += '/train_data/'

data1 = pd.read_csv( mypath + datafile1 , index_col=False)
# data2 = pd.read_csv( mypath + datafile2, index_col=False)
# data3 = pd.read_csv( mypath + datafile3, index_col=False)

# data = pd.concat([data3,data2,data1])
data = data1

data  = data.fillna(data.interpolate(),axis=0,inplace=False)
data.dropna(axis=0,inplace=True)
d1 = copy.deepcopy(data)
d2 = copy.deepcopy(data)
Y = pd.DataFrame(d1[['ACCELERATION','BRAKE','STEERING']])
# Y = pd.DataFrame(d1[['STEERING']])
# Y = pd.DataFrame(d1[['STEERING']])
# X = pd.DataFrame(d2[['TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS']])
X = pd.DataFrame(d2[['SPEED', 'TRACK_POSITION', 'ANGLE_TO_TRACK_AXIS', 'TRACK_EDGE_0', 'TRACK_EDGE_1', 'TRACK_EDGE_2', 'TRACK_EDGE_3', 'TRACK_EDGE_4', 'TRACK_EDGE_5', 'TRACK_EDGE_6', 'TRACK_EDGE_7', 'TRACK_EDGE_8', 'TRACK_EDGE_9', 'TRACK_EDGE_10', 'TRACK_EDGE_11', 'TRACK_EDGE_12', 'TRACK_EDGE_13', 'TRACK_EDGE_14', 'TRACK_EDGE_15', 'TRACK_EDGE_16', 'TRACK_EDGE_17', 'TRACK_EDGE_18']])

# X = X.values.tolist()
# Y = Y.values.tolist()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2,random_state= 42)


rng = np.random.RandomState(42)
# os._exit()

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
inputs = X_train
outputs = Y_train
import neat


def fitness(pop):

    """
    Recieves a list of pop. Modify ONLY their
    fitness values
    """
    for g in pop:
        g['fitness'] = 0
        nw_activate = neat.generate_network(g)
        # for xi, xo in zip(xor_inputs, xor_outputs):
        #     output = nw_activate(xi)
        #     g['fitness'] -= (output[0] - xo[0]) ** 2

        predictions = []
        for input in inputs:
            output_pred = nw_activate(input)
            predictions.append(output_pred)
            # genome.fitness += (  abs(output_real[0] - output_pred[0])  )
        fitness = 0 - sklearn.metrics.mean_squared_error(outputs, predictions)
        f = open('fitness.txt', 'w')
        f.write(str(fitness) + "\n")
        f.close()

        fitness_file = open('fitness.txt', 'r')
        for f in fitness_file:
            g['fitness'] = float(f)


nn = neat.main(fitness=fitness, gen_size=100, pop_size=20, verbose=True, fitness_thresh=0,save=True)
fit = None

while True:
    try:
        fit = next(nn)
    except:
        break


