import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from matplotlib import animation
from sklearn.covariance import MinCovDet
import matplotlib.gridspec as gridspec
from matplotlib.patches import *

import matplotlib.animation as animation

def closest_point(points, point, index=-1):
    closest_dist = sys.float_info.max
    closest_id = -1
    for i in range(0, len(points)):
        if i == index:
            continue
        dist = np.linalg.norm(points[i] - point)
        if dist < closest_dist:
            closest_dist = dist
            closest_id = i
    return closest_id, closest_dist


def create_grid(n, noise, outliers):
    points = np.zeros((n*n + outliers, 2))
    index = 0
    x_offset = 3*(random.random()-0.5)
    y_offset = 3*(random.random()-0.5)
    for r in range(-n/2, n/2):
        for c in range(-n/2, n/2):
            points[index, 0] = (c+noise*(random.random()-0.5))+x_offset
            points[index, 1] = (r+noise*(random.random()-0.5))+y_offset
            index += 1

    for i in range(outliers):
        points[index, 0] = 2*n*(random.random()-0.5)
        points[index, 1] = 2*n*(random.random()-0.5)
        index += 1

    return points

def cov_ellipse(ellipse, cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellipse.center = pos
    ellipse.width = width
    ellipse.height = height
    ellipse.angle = theta


def rotate(points, angle):
    r = np.array([math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle)]).reshape(2, 2)
    return points.dot(r)

class Grid:
    def __init__(self, dim=10, noise=0.1, outliers=0):
        self.points = create_grid(dim, noise, outliers)
        self.polardata = np.zeros((len(self.points), 2))
        self.polar_cov = 0
        self.inlier_points = np.zeros((len(self.points), 2))
        self.inlier_indicies = np.zeros((len(self.points), 1))
        self.normalized_points = np.zeros((len(self.points), 2))
        self.theta = 0
        self.step_size = 1
        self.linedata = np.zeros((3*len(self.points), 2))
        self.normalized_point_ids = []
        self.bounds = [0, 0, 0, 0]

    def step(self, rotation=0):
        self.points = rotate(self.points, rotation)

    def analyze(self, mahalanobis_tolerance=2):
        self.inlier_points = np.zeros((len(self.points), 2))
        for id1 in range(len(self.points)):
            id2 = closest_point(self.points, self.points[id1], id1)[0]

            #keep lines fro plotting purposes
            self.linedata[3*id1] = self.points[id1]
            self.linedata[3*id1+1] = self.points[id2]
            self.linedata[3*id1+2] = [None, None]

            # we are repeating every pi/2, so we compress the angle space by 4x
            a = 4*math.atan2((self.points[id1, 1] - self.points[id2, 1]), (self.points[id1, 0] - self.points[id2, 0]))
            r = np.linalg.norm(self.points[id1] - self.points[id2])
            self.polardata[id1] = [r*math.cos(a), r*math.sin(a)]

        #find the minimal covariance inlier cluster
        self.polar_cov = MinCovDet().fit(self.polardata)

        # extract the grid angle and size.  angle is divided by 4 because
        # we previously scaled it up to repeat every 90 deg
        self.theta = math.atan2(-self.polar_cov.location_[1], self.polar_cov.location_[0])/4
        self.step_size = np.linalg.norm(self.polar_cov.location_)

        # extract inlier points
        polar_mahal = self.polar_cov.mahalanobis(self.polardata)**(0.33)
        inlier_count = 0
        for i in range(len(polar_mahal)):
            if polar_mahal[i] < mahalanobis_tolerance: # stdev tolerance to outliers
                self.inlier_points[inlier_count] = self.points[i]
                self.inlier_indicies[inlier_count] = i
                inlier_count += 1

        self.normalized_points = rotate(self.inlier_points[:inlier_count], -self.theta)/self.step_size

        #enumerate grid IDs
        origin_id = closest_point(self.normalized_points, np.mean(self.normalized_points))[0]
        self.normalized_points = self.normalized_points - self.normalized_points[origin_id]
        inlier_count = 0

        self.bounds = [sys.maxint, sys.maxint, -sys.maxint, -sys.maxint]
        for p in self.normalized_points:
            x = round(p[0])
            y = round(p[1])
            d = np.linalg.norm(p-[x, y])
            if d < 0.4: #tolerance from unit position
                self.normalized_points[inlier_count] = [x, y]
                if (x < self.bounds[0]):
                    self.bounds[0] = x
                if (x > self.bounds[2]):
                    self.bounds[2] = x
                if (y < self.bounds[1]):
                    self.bounds[1] = y
                if (y > self.bounds[3]):
                    self.bounds[3] = y
                inlier_count += 1

        self.normalized_points = self.normalized_points[:inlier_count]


grid_dim = 10
grid_noise = 0.2
grid_outliers = 20
grid = Grid(grid_dim, grid_noise, grid_outliers)

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(14, 4))

plot_range = 30
grid_random_scale = 4
gridplot = fig.add_subplot(131, aspect='equal', autoscale_on=False, xlim=(-plot_range, plot_range), ylim=(-plot_range, plot_range))
polar_plot = fig.add_subplot(132, aspect='equal', autoscale_on=False, xlim=(-grid_random_scale*2, grid_random_scale*2), ylim=(-grid_random_scale*2, grid_random_scale*2))
gridplot2 = fig.add_subplot(133, aspect='equal', autoscale_on=False, xlim=(-1*grid_dim, 1*grid_dim), ylim=(-1*grid_dim, 1*grid_dim))


point_edges, = gridplot.plot([])
point_data, = gridplot.plot([], [], 'ro')

polar_point_data, = polar_plot.plot([],[], 'ro')
polar_data_ellipse = Ellipse(xy=(0,0), width=1, height=1, angle=0, fc='w')
polar_plot.add_patch(polar_data_ellipse)

data2, = gridplot2.plot([], [], 'ro')
data2_ellipse = Ellipse(xy=(0,0), width=1, height=1, angle=0, fc='g')
data2_rect = Rectangle(xy=(0,0), width=1, height=1, fc='w')
gridplot2.add_patch(data2_rect)
gridplot2.add_patch(data2_ellipse)

# initialization function: plot the background of each frame
def init():
#    data.set_data([], [])
    return

# animation function.  This is called sequentially
def animate(i):
    global grid, gridplot, polar_data_ellipse
    grid.points = create_grid(grid_dim, grid_noise, grid_outliers) * (grid_random_scale*random.random() + 2)

    grid.step(math.pi*2*random.random())
    mahalanobis_tolerance = 2
    grid.analyze(mahalanobis_tolerance)

    point_data.set_data(grid.points[:, 0], grid.points[:, 1])
    point_edges.set_data(grid.linedata[:, 0], grid.linedata[:, 1])

    cov_ellipse(polar_data_ellipse, grid.polar_cov.covariance_, grid.polar_cov.location_, mahalanobis_tolerance)

    polar_point_data.set_data(grid.polardata[:, 0], grid.polardata[:, 1])

    data2.set_data(grid.normalized_points[:, 0], grid.normalized_points[:, 1])

    print grid.bounds
    data2_rect._x = grid.bounds[0]
    data2_rect._y = grid.bounds[1]
    data2_rect._width = grid.bounds[2] - grid.bounds[0]
    data2_rect._height = grid.bounds[3] - grid.bounds[1]
#    data2_ellipse.center = grid.origin_point

    return polar_data_ellipse


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20000, interval=20, blit=False)

plt.show()