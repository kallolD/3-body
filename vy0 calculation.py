pimport numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import odeint



x0=0.994
y0=0
xd0=0
T=11.1243403372660851350
a=0.012277471
b=1-a

#This is where I defined the set of differential equations
def f(Y, t):
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]



#The solution is carried out for various possible values of vy0 in the following loop.

#For a particular vy0 the equations are solved and the values of the initial points (x0,y0) and
#final points (xT,yT) are compared. 

#The vy0 values for which the above condition is met are acceptable values of vy0.


i=-2.032                                            # a trial vy0 
while i<3:
    yd0=i
    print("i=", i)
    a_t=np.linspace(0,T,1000)
    r=odeint(f, [x0, xd0, y0, yd0], a_t)
    astack=np.c_[a_t, r[:,0], r[:,1], r[:,2], r[:,3]]
    s=np.array(astack)
    g = "%.4f" % s[0, 1]                            # this is where I set the accuracy limit of the comparision
    h = "%.4f" % s[0, 3]
    xT="%.4f" % s[999, 1]
    yT="%.4f" % s[999, 3]
    if (xT == g and yT == h):                       # The condition which determines whether the value of vy0(i)
        print(yd0)                                  # is acceptable or not.
        plt.plot(s[:, 1], s[:, 3])
        plt.show()
        break
    i = i + 1e-6
