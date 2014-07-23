__author__ = 'jcl5m'
import numpy as np
import random
import math
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial import Delaunay

import sys
sys.path.append('../common/')
import transformations as xform
from homography import *
from pylab import get_current_fig_manager


plotsize = 5
neighbor_count = 3
point_count = 16
points = np.zeros((point_count, 2))
fig = plt.figure(figsize=(14, 7))
dotplot = fig.add_subplot(121, aspect='equal', autoscale_on=False, xlim=(-plotsize, plotsize), ylim=(-plotsize, plotsize))
neighborplot = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(-plotsize, plotsize), ylim=(-plotsize, plotsize))

#Put figure window on top of all other windows
#fig.canvas.manager.window.attributes('-topmost', 1)
#After placing figure window on top, allow other windows to be on top of it later
#fig.canvas.manager.window.attributes('-topmost', 0)

point_edges, = dotplot.plot([])
point_data, = dotplot.plot([], [], 'ro')
mouse_point_data, = dotplot.plot([], [], 'go')

neighbor_edges, = neighborplot.plot([])
neighbor_data, = neighborplot.plot([], [], 'ro')
neighbor_mouse_point, = neighborplot.plot([], [], 'go')

mouse_pos = [0, 0]

point_text = dotplot.text(0.05, 0.05, 'text',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=dotplot.transAxes)

def generateRandomPoints(points, scale):
    for p in points:
        r = scale*(random.random() - 0.5)
        theta = math.pi*2*(random.random() - 0.5)
        p[0] = r*math.cos(theta)
        p[1] = r*math.sin(theta)
    return points

def generateRandomGridPoints(points, scale):
    for p in points:
        unique = False
        while not unique:
            x = round(scale*(random.random() - 0.5))
            y = round(scale*(random.random() - 0.5))
            unique = True
            for q in points:
                if (q[0] == x) and (q[1] == y):
                    unique = False
                    #print "not unique"
                    break
        p[0] = x
        p[1] = y
    return points


def perspectiveMatrix(cx=0,cy=0, fov=1):
    projMatrix = np.eye(4)
    projMatrix[2,0] = -cx
    projMatrix[2,1] = -cy
    projMatrix[2,3] = fov
    projMatrix[3,3] = 0
    return projMatrix

def randomizedTransform(points, t, fov):
    points3D = np.zeros((len(points),4))
    result_points = np.zeros((len(points),2))
    worldMatrix = np.eye(4)
    projectionMatrix = np.eye(4)

    #homogenous coordinates
    for i in range(len(points)):
        points3D[i] = [points[i][0], points[i][1], 0, 1]

    theta = t/57.0
    phi = t/51.0
    rho = t/59.0
    rotMatrix = xform.euler_matrix(theta, phi, rho)

    worldMatrix = np.dot(worldMatrix, rotMatrix)
    worldMatrix[3,0] = 1*math.sin(t/61.0) #move it away form the camera after rotation
    worldMatrix[3,1] = 1*math.sin(t/81.0) #move it away form the camera after rotation
    worldMatrix[3,2] = 12 - 5*math.sin(t/31.0) #move it away form the camera after rotation
    points3D = np.dot(points3D, worldMatrix)

    pM = perspectiveMatrix(0, 0, fov)
    points3D = np.dot(points3D, pM)

    for i in range(len(points)):
        result_points[i] = [points3D[i][0]/points3D[i][3], points3D[i][1]/points3D[i][3]]

    #randomize order
    if False:
        for s in range(len(result_points)):
            d = int(round((len(result_points)-1)*random.random()))
            temp = result_points[s].copy()
            result_points[s] = result_points[d]
            result_points[d] = temp

    return result_points


def onclick(event):
    global mouse_pos
    if event.button == 1:
        mouse_pos = [event.xdata, event.ydata]

def generateDelaunyEdges(points):
    tri = Delaunay(points)

    #collect array of each edge
    edges = []
    for k in range(len(points)):
        indices, indptr = tri.vertex_neighbor_vertices
        k_edges = indptr[indices[k]:indices[k+1]]
        for e in k_edges:
            edges.append([k, e])
    edges = np.reshape(edges, (len(edges), 2))

    #transport edge ids to line drawing list
    linedata = np.zeros((len(edges)*3, 2))
    for i in range(len(edges)):
        id1 = edges[i][0]
        id2 = edges[i][1]
        linedata[3*i] = points[id1]
        linedata[3*i+1] = points[id2]
        linedata[3*i+2] = [None, None]
    return linedata, tri


def baycentricPoints(points, transform):
    result_points = np.zeros((len(points), 2))

    for i in range(len(points)):
        result_points[i] = transform[:2, :].dot(points[i]-transform[2, :])
    return result_points


def closest_point(points, point, index=-1, count=1):
    closest = np.zeros((count,2))
    for c in closest:
        c[0] = -1
        c[1] = sys.float_info.max

    for i in range(0, len(points)):
        if i == index:
            continue
        dist = np.linalg.norm(points[i] - point)
        for ci in range(len(closest)):
            if(dist < closest[ci][1]): #shift down and insert
                for ci2 in range(len(closest)-1, ci, -1):
                    closest[ci2] = closest[ci2-1]
                closest[ci][0] = i
                closest[ci][1] = dist
                break
    return closest


def generateClosestEdges(points, count):
    if count > (len(points)-1):
        count = (len(points)-1)

    linedata = np.zeros((len(points)*3*count, 2))
    closest_ids = np.zeros((len(points)*count, 3))
    index = 0
    for i in range(len(points)):
        closest = closest_point(points, points[i], i, count)
        for c in closest:
            linedata[3*index] = points[i]
            linedata[3*index+1] = points[c[0]]
            linedata[3*index+2] = [None, None]
            closest_ids[index] = [i, c[0], c[1]]
            index += 1
    return linedata, closest_ids

def triangleArea(a, b, c):
    return 0.5*math.fabs(a[0]*(b[1] - c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))


cid = fig.canvas.mpl_connect('button_press_event', onclick)

generateRandomGridPoints(points, 10)

# animation function.  This is called sequentially
def animate(t):

    points[0] = [0, 0]
    points[1] = [3, 0]
    points[2] = [0, 3]
    points[3] = [3, 3]
    transformed_points = randomizedTransform(points, t, .2)
    linedata, tri = generateDelaunyEdges(transformed_points)


    pa = [transformed_points[0], transformed_points[1], transformed_points[2], transformed_points[3]]
    pa = np.reshape(pa, (4, 2))
    pb = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    h = find_homography(pa, pb)
    b_points = apply_homography(h, transformed_points)


#    print tri.simplices
#    for i in range(len(tri.simplices)):
#        print tri.neighbors[i]
#    tri_transform = tri.transform[0]

 #   b_points = baycentricPoints(transformed_points, tri_transform)
  #  mb_point = baycentricPoints([mouse_pos], tri_transform)


#    linedata, closest_ids = generateClosestEdges(transformed_points, neighbor_count)

#    dotplot.add_patch(plt.Line2D(linedata[:, 0], linedata[:, 1], color=someColors[t % 5]))
#    dotplot.add_patch(plt.Circle(xy=[5,2], radius=1, color=someColors[t % 5]))
#        dotplot.add_line(plt.Line2D([l[0][0],l[1][0]],[l[0][1],l[1][1]], color='k'))
#    dotplot.add_line(plt.Line2D(linedata[:, 0], linedata[:, 1], color=someColors[t % 5]))

#    text_data = ""
#    n_edges = np.zeros((neighbor_count*3, 2))

#    count = 0
#    nearest_set = []
#    for ids in closest_ids:
#        if ids[0] == 0:
#            nearest_set.append([ids[0], ids[1]])
#            area1 = triangleArea(points[ids[0]], points[ids[1]])
#            text_data += str(ids[0]) + " " + str(ids[1]) + " " + str(ids[2]) + "\n"
#            n_edges[3*count] = [points[ids[0]][0], points[ids[0]][1]]
#            n_edges[3*count+1] = [points[ids[1]][0], points[ids[1]][1]]
#            n_edges[3*count+2] = [None, None]
#            count += 1

#    area1 = triangleArea(transformed_points[0], transformed_points[1], transformed_points[2])
#    area2 = triangleArea(transformed_points[1], transformed_points[2], transformed_points[3])
#    text_data += str(area1/area2)

#    area1 = triangleArea(points[nearest_set[0, 0]],points[nearest_set[0, 1]],points[nearest_set[1, 1]])
#    point_text.set_text(text_data)

#    mouse_point_data.set_data(mouse_pos[0], mouse_pos[1])
#    neighbor_mouse_point.set_data(mb_point[0][0], mb_point[0][1])

 #   point_edges.set_data(linedata[:, 0], linedata[:, 1])
    point_data.set_data(transformed_points[:, 0], transformed_points[:, 1])

    neighbor_data.set_data(b_points[:, 0], b_points[:, 1])
#    neighbor_edges.set_data(n_edges[:, 0], n_edges[:, 1])

    return point_data

anim = animation.FuncAnimation(fig,  animate, frames=20000, interval=20, blit=False)

plt.show()

