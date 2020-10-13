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

#defined the differential equations here
def f(Y, t):
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]

#define the callable function for minimization. i.e, the function f
def objective(x):
    return (x[0]-x0)**2 + (x[1]-xd0)**2 + (x[2]-y0)**2 + (x[3]-x[4])**2

#defined the constraint, i.e, the acceptable values of xT, xdT, yT, ydT. These are those values that are solutions of
#the set of differential equations
def constraint1(x):
    t = np.linspace(0, T, 1000)
    r = odeint(f, [x0, xd0, y0, x[4]], t)
    return x[0] +  x[1] + x[2] + x[3] - r[999, 0] - r[999, 1] - r[999, 2] - r[999, 3]

#the minimisation is carried out here
a1=(-10,10)
b1=(-3,0)
bnds=(a1,a1,a1,b1, b1)
con1={'type': 'eq', 'fun':constraint1}
optimize.minimize(objective, [1,0,0,3,2],bounds=bnds, constraints=con1 )
