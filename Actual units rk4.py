import numpy as np
import matplotlib.pyplot as plt

#set mass ratios
a=0.012277471
b=1-a

#define the equations as functions
def e(xd):
    return xd
def f(x, y, yd):
    D1 = ((x + a) ** 2 + y ** 2) ** (3 / 2)
    D2 = ((x - b) ** 2 + y ** 2) ** (3 / 2)
    return x + 2*yd - b*(x+a)/D1 - a*(x-b)/D2
def g(yd):
    return yd
def h(x, xd, y):
    D1 = ((x + a) ** 2 + y ** 2) ** (3 / 2)
    D2 = ((x - b) ** 2 + y ** 2) ** (3 / 2)
    return y - 2*xd - b*(y)/D1 - a*(y)/D2

#set initial conditions and time bounds
t0=0
t1=5.4634
n=1000
w=(t1-t0)/n
t=0
x=0.994
xd=0
y=0
yd=-2.113898
s = np.empty((n,5))
E = np.empty((n,2))
i=0
while (i<n):
    k0=w*e(xd)
    l0=w*f(x, y, yd)
    m0=w*g(yd)
    n0=w*h(x, xd, y)
    k1=w*e(xd+0.5*l0)
    l1=w*f(x+0.5*k0, y+0.5*m0, yd+0.5*n0)
    m1=w*g(yd+0.5*n0)
    n1=w*h(x+0.5*k0, xd+0.5*l0, y+0.5*m0)
    k2=w*e(xd+0.5*l1)
    l2=w*f(x+0.5*k1, y+0.5*m1, yd+0.5*n1)
    m2=w*g(yd+0.5*n1)
    n2=w*h(x+0.5*k1, xd+0.5*l1, y+0.5*m1)
    k3=w*e(xd+l2)
    l3=w*f(x+k2, y+m2, yd+n2)
    m3=w*g(yd+n2)
    n3=w*h(x+k2, xd+l2, y+m2)
    s[i,0]= t = t0 + (i*w)
    s[i,1]=x=x + (k0 + 2*k1 + 2*k2 + k3)/6
    s[i,2]=xd=xd + (l0 + 2*l1 + 2*l2 + l3)/6
    s[i,3]=y=y + (m0 + 2*m1 + 2*m2 + m3)/6
    s[i,4]=yd=yd + (n0 + 2*n1 + 2*n2 + n3)/6
    #print(s[i,0], s[i,1],s[i,2],s[i,3],s[i,4])
    #K = 0.5*(xd**2 + yd**2 - x**2 - y**2) - a*((x - b)**2 + y**2)**(-0.5) - b*((x + a) ** 2 + y ** 2)**(-0.5)
    #D1 = ((x + a) ** 2 + y ** 2) ** (1 / 2)
    #D2 = ((x - b) ** 2 + y ** 2) ** (1 / 2)
    #C = ((2 * 3.14 / 5.4634) ** 2) * (x ** 2 + y ** 2) - (xd ** 2 + yd ** 2) + 2 * (b / D1 + a / D2)

    E[i, 0] = 384400*s[i,1]
    #E[i, 1] = (K + 1.04773934766827)/(-1.04773934766827)
    #E[i,1] =   (K+1.04202736094963)/(-1.04202736094963)
    E[i, 1] = 384400*s[i,3]
    #print(E[i,1])
    #E[i,1]=K
    print(E[i,1])

    i += 1

#astack = np.c_[s[:,0], s[:, 1], s[:,2], s[:,3], s[:,4]]
#np.savetxt('rk4.csv', astack, delimiter=',', header='t, x, xd, y, yd', comments='')
#sol = np.array(astack)

bstack = np.c_[E[:,0], E[:, 1]]
#np.savetxt('rk4 eng jacobi.csv', bstack, delimiter=',', header='t, eng', comments='')
#sol = np.array(astack)

plt.plot(E[:,0], E[:,1])
#axes = plt.gca()
#axes.set_ybound([-0.0000001,0.0000001])
plt.xlabel('time')
plt.ylabel('Fractional change in energy')
plt.title('Plot of fractional change in energy rk4')
plt.show()
