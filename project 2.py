import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

Y0, t0 = [0.994, 0, 0, -2.11389879669], 0

a=0.012277471
b=1-a


def f(t, Y):
    D1=((Y[0] + a)**2 + (Y[2])**2)**(3/2)
    D2=((Y[0] - b)**2 + (Y[2])**2)**(3/2)
    return [Y[1], (Y[0] + 2*Y[3] - b*(Y[0]+a)/D1 - a*(Y[0]-b)/D2), Y[3], (Y[2] - 2*Y[1] - b*(Y[2])/D1 - a*(Y[2])/D2)]


r = ode(f).set_integrator('dop853')
r.set_initial_value(Y0, t0)
t1=5.4367

N=1001
t = np.linspace(t0, t1, N)
sol = np.empty((N, 4))
sol[0] = Y0
s = np.empty((N, 5))


k = 1
while r.successful() and r.t < t1:
    r.integrate(t[k])
    sol[k] = r.y
    k += 1


astack = np.c_[t, sol[:,0], sol[:, 1], sol[:,2], sol[:,3]]
np.savetxt('dopri5.csv', astack, delimiter=',', header='t, x, xd, y, yd', comments='')
s = np.array(astack)

E=np.empty((N, 2))
p=0
while s[p,0]<t1:
    x=s[p,1]
    xd=s[p,2]
    y=s[p,3]
    yd=s[p,4]
    K = 0.5*(xd**2 + yd**2 - x**2 - y**2) - a*((x-b)**2 + y**2)**(-0.5) - b*((x+a)**2 + y**2)**(-0.5)
    D1=((x+a)**2 + y**2)**(1/2)
    D2=((x-b)**2 + y**2)**(1/2)
    C=((2*3.14/5.4634)**2)*(x**2 + y**2) -(xd**2 + yd**2) + 2*(b/D1 + a/D2)
    E[p,0]=2.04E06*s[p,0]
    #E[p,1]=((K + 1.1970936678205)/(-1.1970936678205))*100
    E[p,1]=1.05E06*C
    print(E[p,1])

    p += 1

bstack=np.c_[E[:,0], E[:,1]]
np.savetxt('dopri5 energy.csv', bstack, delimiter=',', header='t, E', comments=',')

plt.plot(E[1:1000,0], E[1:1000,1])
axes = plt.gca()
axes.set_ybound([-1000000,1E07])
plt.ylabel('Jacobi Constant C')
plt.xlabel('time')
#plt.scatter(-a,0)
#plt.scatter(b,0)
plt.title('Jacobi integral\n')
plt.show()




