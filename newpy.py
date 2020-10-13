import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

Y0, t0 = [0.994, 0, 0, -2.0317326295573368356], 0

a=0.012277471
b=1-a


def f(t, Y):
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]


r = ode(f).set_integrator('dopri5')
r.set_initial_value(Y0, t0)
t1=10.124

N=1000
t = np.linspace(t0, t1, N)
sol = np.empty((N, 4))
sol[0] = Y0
s = np.empty((N, 5))


k = 1
while r.successful() and r.t < t1:
    r.integrate(t[k])
    sol[k] = r.y
    k += 1


#astack = np.c_[t, sol[:,0], sol[:, 1], sol[:,2], sol[:,3]]
#np.savetxt('dopri5.csv', astack, delimiter=',', header='t, x, xd, y, yd', comments='')
#s = np.array(astack)

#E=np.empty((N, 2))
#p=1
#while s[p,0]<t1:
#    x=s[p,1]
#    xd=s[p,2]
#    y=s[p,3]
#    yd=s[p,4]
#    K = 0.5*(xd**2 + yd**2 - x**2 - y**2) - a*((x-b)**2 + y**2)**(-0.5) - b*((x+a)**2 + y**2)**(-0.5)

#    E[p,0]=s[p,0]
#    E[p,1]= (K+1.04773934766827)/(-1.04773934766827)
    #E[p,1]=K
#    print(E[p,1])

#    p += 1

#bstack=np.c_[E[:,0], E[:,1]]
#np.savetxt('dopri5 energy.csv', bstack, delimiter=',', header='t, E', comments=',')

plt.plot(sol[:,0], sol[:,2])
#axes = plt.gca()
#axes.set_ybound([-0.0000000001,0.0000000001])
plt.xlabel('t')
plt.ylabel('Fractional change in energy')
#plt.scatter(-a,0)
#plt.scatter(b,0)
plt.title('fractional change in energy\n by dopri5')
plt.show()




