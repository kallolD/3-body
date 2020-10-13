import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def solvr(Y, t):

    return [Y[1], ((-0.8*Y[1] - 8*(Y[0]-0.5) + 40*(Y[2]-Y[0]-1))/1), Y[3], ((-0.5*Y[3] - 40*(Y[2]-Y[0]-1))/1.5)]


a_t = np.linspace(0,10, 250)
asol = odeint(solvr, [0.5, 0.0, 2.25, 0.0], a_t)

#plt.plot(a_t, asol[:,0], 'b')
#plt.plot(a_t, asol[:,2], 'g')
#plt.xlabel('t')
#plt.ylabel('Displacement')
plt.plot(asol[:,0], asol[:,2])
plt.show()
