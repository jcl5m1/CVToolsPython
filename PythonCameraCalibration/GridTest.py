import scipy.cluster.vq
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from matplotlib import animation
import matplotlib
from matplotlib import collections  as mc
from sklearn.covariance import MinCovDet

import matplotlib.animation as animation


def angular_MCD(data):
    unit_circle = np.zeros((len(data), 2))
    unit_circle[:,0] = [math.cos(x) for x in data]
    unit_circle[:,1] = [math.sin(x) for x in data]
    S = MinCovDet().fit(unit_circle)
    theta = math.atan2(S.location_[1], S.location_[0])
    det = np.linalg.det(S.covariance_)
    return theta, det


def closest_point(points, index):
    closest_dist = sys.float_info.max
    closest_id = -1
    for i in range(len(points)):
        if i == index:
            continue
        dist = np.linalg.norm(points[i] - points[index])
        if dist < closest_dist:
            closest_dist = dist
            closest_id = i
    return closest_id, closest_dist


def create_grid(n, noise, outliers):
    points = np.zeros((n*n + outliers, 2))
    index = 0
    for r in range(-n/2, n/2):
        for c in range(-n/2, n/2):
            points[index, 0] = c+noise*(random.random()-0.5)
            points[index, 1] = r+noise*(random.random()-0.5)
            index += 1

    for i in range(outliers):
        points[index, 0] = 2*n*(random.random()-0.5)
        points[index, 1] = 2*n*(random.random()-0.5)
        index += 1

    return points


def rotate(points, angle):
    r = np.array([math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle)]).reshape(2, 2)
    return points.dot(r)



class Grid:
    def __init__(self, dim=10, noise=0.1, outliers=0):
        self.points = create_grid(dim, noise, outliers)
        self.angles = np.zeros(len(self.points))
        self.lengths = np.zeros(len(self.points))
        self.angle = []
        self.length = []
        self.linedata = np.zeros((3*len(self.points), 2))

    def step(self, rotation=0):
        self.points = rotate(self.points, rotation)

    def analyze(self):
        for id1 in range(len(self.points)):
            id2 = closest_point(self.points, id1)[0]
            self.linedata[3*id1] = self.points[id1]
            self.linedata[3*id1+1] = self.points[id2]
            self.linedata[3*id1+2] = [None, None]
            a = math.atan2((self.points[id1, 1] - self.points[id2, 1]),(self.points[id1,0] - self.points[id2,0]))
            self.angles[id1] = 4*a # because we are repeating every pi/2, compress the angle space by 4x
            self.lengths[id1] = np.linalg.norm(self.points[id1] - self.points[id2])
        self.angle = angular_MCD(self.angles)
        S = MinCovDet().fit(self.lengths.reshape((len(self.lengths),1)))
        self.length = S.location_, S.covariance_


grid_dim = 10
grid_noise = 0.1
grid_outliers = 10
grid = Grid(grid_dim, grid_noise, grid_outliers)


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()

scale = 20
gridplot = fig.add_subplot(141, aspect='equal', autoscale_on=False, xlim=(-scale, scale), ylim=(-scale, scale))
angleplot = fig.add_subplot(142, aspect='equal', autoscale_on=False, xlim=(-1.2,1.2), ylim=(-1.2,1.2))
lengthplot = fig.add_subplot(143, aspect='equal', autoscale_on=False, xlim=(-.5,.5), ylim=(0,3))
gridplot2 = fig.add_subplot(144, aspect='equal', autoscale_on=False, xlim=(-scale, scale), ylim=(-scale, scale))
#angleplot = fig.add_subplot(122, autoscale_on=False, xlim=(-.5, 0.5), ylim=(-14.2, 14.2) )


line, = gridplot.plot([])
data, = gridplot.plot([], [], 'ro')

angle_data, = angleplot.plot([],[], 'ro')
mean_angle_data = plt.Circle((0, 0), 0.25, fc='c')
angleplot.add_patch(mean_angle_data)

length_data, = lengthplot.plot([],[], 'ro')
mean_length_data = plt.Circle((0, 0), 0.25, fc='c')
lengthplot.add_patch(mean_length_data)

data2, = gridplot2.plot([], [], 'ro')

# initialization function: plot the background of each frame
def init():
    data.set_data([], [])
    return

# animation function.  This is called sequentially
def animate(i):
    global grid, gridplot
    grid.points = create_grid(grid_dim, grid_noise, grid_outliers)
    grid.step(i/30.0)
    grid.analyze()

    data.set_data(grid.points[:, 0], grid.points[:, 1])
#    line.set_data(grid.linedata[:, 0], grid.linedata[:, 1])

    mean_angle_data.center = (math.cos(grid.angle[0]), math.sin(grid.angle[0]))
    mean_angle_data.radius = grid.angle[1]+.1
    angle_data.set_data([math.cos(x) for x in grid.angles], [math.sin(x) for x in grid.angles])

    length_data.set_data([0], grid.lengths)
    mean_length_data.center = (0,grid.length[0])
    mean_length_data.radius = math.sqrt(grid.length[1])+.1

    rotated_points = rotate(grid.points, grid.angle[0]/4)
    data2.set_data(rotated_points[:,0], rotated_points[:,1])

    return


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20000, interval=20, blit=False)

plt.show()