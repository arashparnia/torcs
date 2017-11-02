from pandas import scatter_matrix
from pandas.io.parsers import read_csv
from sklearn import preprocessing, model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd

datafile = 'aalborg.csv'
mypath = '../training/'
data = pd.read_csv( mypath + 'all.csv')

