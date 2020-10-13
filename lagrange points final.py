import matplotlib.pyplot as plt
import numpy as np
import mpl_toolkits as mpl
import math

fig = plt.figure()
ax = fig.add_subplot(111)
u = np.linspace(-15,15,50)
x, y = np.meshgrid(u,u)


z = - (10/(np.sqrt((x + 0.91)**2 + y**2))) - 1/(np.sqrt((x - 9.09)**2 + y**2)) - 0.0055*((x**2) + (y**2))

ax.contour(x,y,z,1000)

plt.show()



