__author__ = 'johnnylee'

import geometery as geo
import random
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

points = np.zeros((4, 2))
fig = plt.figure()
ax = fig.add_subplot(111,  aspect='equal', xlim=(0, 1), ylim=(0, 1))
data_lines, = ax.plot([])
data_pts, = ax.plot([], [], 'ro')
start_pt, = ax.plot([], [], 'go')
plt_text = ax.text(0.05, 0.05, 'text',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)


def generatePoints(pts):
    for i in range(len(pts)):
        pts[i, 0] = random.random()
        pts[i, 1] = random.random()

def runTest():
    global points, data_pts, fig, data_lines
    generatePoints(points)
    quad_ids = [0, 1, 2, 3]
    quad_ids = geo.ClockwiseSortQuad(quad_ids, points)

#    print geo.QuadIsConvex(quad_ids, points)

    sorted_data = np.zeros((len(points), 2))
    for i in range(len(points)):
        sorted_data[i] = points[quad_ids[i]]

#    print geo.TriangleOrientation(sorted_data[0], sorted_data[1], sorted_data[2]), \
#        geo.TriangleOrientation(sorted_data[1], sorted_data[2], sorted_data[3]), \
#        geo.TriangleOrientation(sorted_data[2], sorted_data[3], sorted_data[0]), \
#        geo.TriangleOrientation(sorted_data[3], sorted_data[0], sorted_data[1])

    start_pt.set_data(sorted_data[0, 0], sorted_data[0, 1])
    data_pts.set_data(sorted_data[:, 0], sorted_data[:, 1])
    data_lines.set_data(sorted_data[:, 0], sorted_data[:, 1])
    plt_text.set_text(geo.QuadIsConvex(quad_ids, points))
    fig.canvas.draw()

def press(event):
    runTest()

runTest()
cid = fig.canvas.mpl_connect('key_press_event', press)
plt.show()
