__author__ = 'johnnylee'

import math
from scipy.spatial import Delaunay
import numpy as np

def triangleArea(a, b, c):
    return 0.5*math.fabs(a[0]*(b[1] - c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))

def CrossProductZ(a, b):
    return a[0] * b[1] - a[1] * b[0];

def TriangleOrientation(a, b, c):
    v = CrossProductZ(a, b) + CrossProductZ(b, c) + CrossProductZ(c, a)
    return v



def ClockwiseSortQuad(quad_ids, points):
    a = points[quad_ids[0]]
    b = points[quad_ids[1]]
    c = points[quad_ids[2]]
    d = points[quad_ids[3]]

    if (TriangleOrientation(a, b, c) < 0.0):
        if TriangleOrientation(a, c, d) < 0.0:
            return quad_ids
        elif TriangleOrientation(a, b, d) < 0.0:
            return [quad_ids[0], quad_ids[1], quad_ids[3], quad_ids[2]]
        else:
            return [quad_ids[3], quad_ids[1], quad_ids[2], quad_ids[0]]

    elif TriangleOrientation(a, c, d) < 0.0:
        if TriangleOrientation(a, b, d) < 0.0:
            return [quad_ids[0], quad_ids[2], quad_ids[1], quad_ids[3]]
        else:
            return [quad_ids[1], quad_ids[0], quad_ids[2], quad_ids[3]]
    else:
        return [quad_ids[2], quad_ids[1], quad_ids[0], quad_ids[3]]

def QuadIsConvex(quad_ids, points):
    a = points[quad_ids[0]]
    b = points[quad_ids[1]]
    c = points[quad_ids[2]]
    d = points[quad_ids[3]]

    if (TriangleOrientation(a, b, c) >= 0.0):
        return False
    if (TriangleOrientation(b, c, d) >= 0.0):
        return False
    if (TriangleOrientation(c, d, a) >= 0.0):
        return False
    if (TriangleOrientation(d, a, b) >= 0.0):
        return False
    return True


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
