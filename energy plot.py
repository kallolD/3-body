import numpy as np
import pandas as pd
import xlrd as xl
import matplotlib.pyplot as plt

data_file = pd.read_csv('eng.csv')
time = data_file['time']
energy = data_file['frac']

plt.plot(time, energy, linewidth=0.7)
axes = plt.gca()
axes.set_ybound([-0.00001,0.00001])
plt.xlabel('t')
plt.ylabel('Fractional change in Energy')
plt.title('Plot showing fractional change in energy')
plt.show()

