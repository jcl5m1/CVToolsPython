import scipy.cluster.vq
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys
from matplotlib import animation

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


def create_grid(n, noise):
    points = np.zeros((n*n, 2))
    index = 0
    for r in range(-n/2, n/2):
        for c in range(-n/2, n/2):
            points[index, 0] = c+noise*(random.random()-0.5)
            points[index, 1] = r+noise*(random.random()-0.5)
            index += 1
    return points


def rotate(points, angle):
    r = np.array([math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle)]).reshape(2, 2)
    return points.dot(r)

"""


#create a noisy grid of points
fig_height = 7
fig = plt.figure(num=None, figsize=(2*fig_height, fig_height), dpi=80, facecolor='w', edgecolor='k')
leftfig = fig.add_subplot(1, 2, 1)
pts = leftfig.plot([], [], 'r')



pts = create_grid(15, .25)

angles = np.zeros((len(pts)))
centroids = scipy.cluster.vq.kmeans(angles, 2)[0]
idx = scipy.cluster.vq.vq(angles, centroids)[0]
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1)


def init():
    return


def animate(i):
    global centroids, idx, pts
    pts = rotate(pts, (math.pi * i) / 60)
    return

    plt.scatter(pts[:, 0], pts[:, 1])
    b = np.mat('1;1')
    for id1 in range(len(pts)):
        id2 = closest_point(pts, id1)[0]
        plt.subplot(1, 2, 1)
        plt.plot([pts[id1, 0], pts[id2, 0]], [pts[id1, 1], pts[id2, 1]], 'r')

        A = np.mat(np.concatenate((pts[id1], pts[id2])).reshape(2, 2))
        x = np.linalg.solve(A, b)

        a = math.atan(-x[0]/x[1])
        angles[id1] = a
    centroids = scipy.cluster.vq.kmeans(angles, 2)[0]
    idx = scipy.cluster.vq.vq(angles, centroids)[0]
    plt.subplot(1, 2, 2)
    plt.plot(angles[idx == 0], 'ob', angles[idx == 1], 'or')
    plt.plot(centroids[:], centroids[:], 'sg', markersize=8)

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)




# First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
#line, = ax.plot([], [], lw=2)


# initialization function: plot the background of each frame
#def init():
#    line.set_data([], [])
#    return line,


# animation function.  This is called sequentially
#def animate(i):
#    x = np.linspace(0, 2, 1000)
#    y = np.sin(2 * np.pi * (x - 0.01 * i))
#    line.set_data(x, y)
#    return line,


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

#plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Grid:
    def __init__(self, dim=10, noise=0.05):
        self.points = create_grid(dim, noise)

    def step(self, rotation=0):
        self.points = rotate(self.points, rotation)

    def analyze(self):
        for id1 in range(len(self.points)):
            id2 = closest_point(self.points, id1)[0]
#            plt.subplot(1, 2, 1)
#            plt.plot([pts[id1, 0], pts[id2, 0]], [pts[id1, 1], pts[id2, 1]], 'r')


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()

scale = 10
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-scale, scale), ylim=(-scale, scale))
data, = ax.plot([], [], 'ro')
line, = ax.plot([], [], 'bo-')
grid = Grid(10, 0.3)

# initialization function: plot the background of each frame
def init():
    data.set_data([], [])
    return data, line

# animation function.  This is called sequentially
def animate(i):
    global grid, ax
    grid.points = create_grid(10, .3)
    grid.step(i/30.0)
  #  ax.plot([0, 5], [0, 5], 'o-')
    data.set_data(grid.points[:, 0], grid.points[:, 1])
#    line.set_data([0, 5], [0, 5])
    line.set_data([0, 5], [0, 5+math.sin(i/10.0)])

    return data, line

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=20000, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()