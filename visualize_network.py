
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


# winner_net = neat.nn.RecurrentNetwork.create(winner, config)
filename = 'network_brake'
with open(filename+'.pkl', 'rb') as input:
    winner_net = pickle.load(input)
draw_net(winner_net, filename=filename)