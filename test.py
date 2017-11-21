import pandas as pd
import matplotlib.pyplot as plt
import pyrenn as prn
import numpy as np
df = pd.ExcelFile('example_data.xlsx').parse('compressed_air')
P = np.array([df['P1'].values,df['P2'].values,df['P3'].values])
Y = np.array([df['Y1'].values,df['Y2']])
Ptest = np.array([df['P1test'].values,df['P2test'].values,df['P3test'].values])
Ytest = np.array([df['Y1test'].values,df['Y2test']])