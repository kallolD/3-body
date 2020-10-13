import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy import optimize


x0=0.994
y0=0
xd0=0
T=11.1243403372660851350
a=0.012277471
b=1-a


def f(Y, t):
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]


i=-2.032
while i<3:
    yd0=i
    print("i=", i)
    a_t=np.linspace(0,T,1000)
    r=odeint(f, [x0, xd0, y0, yd0], a_t)
    astack=np.c_[a_t, r[:,0], r[:,1], r[:,2], r[:,3]]
    s=np.array(astack)
    xT="%.4f" % s[999,1]
    g="%.4f" % s[0,1]
    yT="%.4f" % s[999,3]
    h="%.4f" % s[0,3]
    if (xT==g and yT==h):
           print(yd0)
           plt.plot(s[:,1], s[:,3])
           plt.show()
           break
    i = i + 1e-6
