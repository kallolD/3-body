import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def solve(Y, t):
    a=0.012277471
    b=1-a
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]


a=0.012277471
b=1-a
t = np.linspace(0,10,1000000)
asol = odeint(solve, [1.2, 0, 0, -1.0439 ], t)


def K():
    a = 0.012277471
    b = 1 - a
    D3 = ((asol[t, 0] + a) ** 2 + (asol[t, 2]) ** 2) ** (1 / 2)
    D4 = ((asol[t, 0] - b) ** 2 + (asol[t, 2]) ** 2) ** (1 / 2)
    return [0.5*(asol[t, 2]**2 + asol[t, 3]**2 - asol[t, 0]**2 - asol[t, 1]**2) - a/D4 - b/D3 ]



astack = np.c_[t, asol[:,0], asol[:, 1], asol[:,2], asol[:,3]]
#np.savetxt('approx.csv', astack, delimiter=',', header='t, x, xd, y, yd', comments='')
#plt.plot(t, K)

#bstack = np.c_[t, K(t)]
#np.savetxt('new.csv', bstack, delimiter=',', header='t, K', comments='')
plt.plot(asol[:,0],asol[:,2], 'r', linewidth=0.7)
#plt.scatter(0,0, linewidths=1.0)
#plt.scatter(-0.012277471,0, linewidths=1.0)
#plt.scatter(b,0)
#plt.axhline(y=0, color='black', linewidth=1.0)
#plt.axvline(x=0, color='black', linewidth=1.0)
#plt.xlabel('x/R')
#plt.ylabel('y/R')
#plt.title('Trajectory of the 3rd mass in Earth Moon system\n '
 #         '(Initial y velocity = -1.0494)')
plt.show()

