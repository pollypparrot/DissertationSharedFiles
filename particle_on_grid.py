import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import randint

mean = 0
comp = 50   # number of computations
dt = 0.01   # time step size
diffusivity = 1e-5

# calculate standard deviation at given time
def stand_dev(time):
    return np.sqrt(2 * diffusivity * time)

# intiailize arrays to store position information
x_array = np.random.normal(loc=mean, scale=stand_dev(dt), size=(512,1))
y_array = np.random.normal(loc=mean, scale=stand_dev(dt), size=(512,1))
# set starting postion at origin
x_array[0] = 256
y_array[0] = 256

grid = 101 # n x n grid size

# plots initial position
Z = np.zeros((grid,grid))   # grid data
x = 50
y = 50
Z[x][y] = 1
s = 3 # splodge size
for i in range(grid-1): # colours nearby pixels
    for j in range(grid-1):
        Z[i][j] = 255*np.exp(-((i-x)**2+(j-y)**2) / (2*s**2))
plt.imshow(Z, cm.get_cmap("gray"),vmax=15, vmin=0,interpolation='nearest')

# plots moving particle
for computations in range(comp):
    if x or y < grid:
        Z[x][y] = 0
        x += randint(-3, 3)
        y += randint(-3, 3)
        Z[x][y] = 1
        for i in range(grid-1): # colours nearby pixels
            for j in range(grid-1):
                Z[i][j] = 255*np.exp(-((i-x)**2+(j-y)**2) / (2*s**2))
    else:
        break
    plt.imshow(Z, cm.get_cmap("gray"), interpolation='nearest')
    plt.show()
    plt.pause(0.01)
# plots moving particle


