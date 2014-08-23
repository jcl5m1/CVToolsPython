__author__ = 'jcl5m'
import numpy as np
import random
import math
import matplotlib
matplotlib.use('Qt4Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
import math
import sys
sys.path.append('../common/')
import transformations as xform
import geometery as geo
from homography import *
from pylab import get_current_fig_manager

plotsize = 5
plotsize2 = 50
neighbor_count = 3
point_count = 50
points = np.zeros((point_count, 2))
fig = plt.figure(figsize=(14, 7))
dotplot = fig.add_subplot(121, aspect='equal', autoscale_on=False, xlim=(-plotsize, plotsize), ylim=(-plotsize, plotsize))
warpedplot = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(0, plotsize2), ylim=(0, plotsize2))
#warpedplot = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(-10, 10), ylim=(-10, 10))

point_edges, = dotplot.plot([])
point_data, = dotplot.plot([], [], 'bo')
feature_point_data, = dotplot.plot([], [], 'go')
selection_point_data, = dotplot.plot([], [], 'ro')

warped_edges, = warpedplot.plot([])
warped_data, = warpedplot.plot([], [], 'ro')

mouse_pos = [0, 0]

plt_text = dotplot.text(0.05, 0.05, 'text',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=dotplot.transAxes)

def generateRandomPoints(points, scale):
    for p in points:
        p[0] = scale*(random.random() - 0.5)
        p[1] = scale*(random.random() - 0.5)
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




def randomizedTransform(points, t, fov):
    points3D = np.zeros((len(points),4))
    result_points = np.zeros((len(points),2))
    worldMatrix = np.eye(4)
    projectionMatrix = np.eye(4)

    #homogenous coordinates
    for i in range(len(points)):
        points3D[i] = [points[i][0], points[i][1], 0, 1]
    t = t/10.0
    theta = t/57.0
    phi = t/51.0
    rho = t/59.0
    rotMatrix = xform.euler_matrix(theta, phi, rho)

    worldMatrix = np.dot(worldMatrix, rotMatrix)
    worldMatrix[3,0] = 1*math.sin(t/61.0) #move it away form the camera after rotation
    worldMatrix[3,1] = 1*math.sin(t/81.0) #move it away form the camera after rotation
    worldMatrix[3,2] = 12 - 5*math.sin(t/31.0) #move it away form the camera after rotation
    points3D = np.dot(points3D, worldMatrix)

    pM = xform.perspectiveMatrix(0, 0, fov)
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

def press(event):
    global h, pause, step, selected_quad_id
#    print('press', event.key)
#    sys.stdout.flush()
    if event.key==' ':
        pause = not pause
    if event.key=='.':
        step = True
    if event.key=='n':
        selected_quad_id += 1


cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', press)

#generateRandomPoints(points, 10)
#np.save("points", points)

points = np.load('points.npy')

h = []
pause = False
step = False
time = 0
selected_quad_id = 0
curr_triangle = 0


def GetFeaturePoints(tri, quad):
    feature_point_ids = []
    indices, indptr = tri.vertex_neighbor_vertices
    for p in quad:
        for p2 in indptr[indices[p]:indices[p+1]]:
            if (p2 not in quad) and (p2 not in feature_point_ids): #add this point to the feature set
                feature_point_ids.append(p2)
    return  feature_point_ids

def GetDewarpedFeaturePoints(quad, feature_ids, points):
    pa = [points[quad[0]], points[quad[1]], points[quad[2]], points[quad[3]]]
    pa = np.reshape(pa, (4, 2))

    min_cube = -.5
    max_cube = .5
    pb = np.array([[min_cube, min_cube], [min_cube, max_cube], [max_cube, max_cube], [max_cube, min_cube]])

    feature_points = []
    for fp_id in feature_ids:
        feature_points.append(points[fp_id])
    h = find_homography(pa, pb)
    b_points = apply_homography(h, feature_points)
    for p in b_points:
        theta = math.atan2(p[1], p[0])
        m = math.sqrt(p[1]*p[1] + p[0]*p[0])
        p[0] = m
        p[1] = theta

    return b_points

def ExtractValidQuads(points, tri):
    simplex_pairs = []
    quads = []
    for s1 in range(len(tri.simplices)):
        for s2 in tri.neighbors[s1]:
            quad = tri.simplices[s1]
            if s2 == -1:  # no neighbor in this direction, skip
                continue
            for p in tri.simplices[s2]: #for each point in neighbor triangle
                if (p not in quad) and (len(quad) == 3): #if the point is new, add it
                    quad = np.append(quad, p)
            newpair = True
            for sp in simplex_pairs:
                if (sp[0] == s2) and (sp[1] == s1):
                    newpair = False
            if newpair == False:
                continue
            #check if quad is convex
            quad = geo.ClockwiseSortQuad(quad, points)
            if not geo.QuadIsConvex(quad, points):
                continue
            simplex_pairs.append([s1, s2])
            quads.append(quad)
    return quads, simplex_pairs


def ScaleToImageCoords(pt, size):
    r = int((pt[0]/20.0)*size)
    c = int((pt[1]/(math.pi*2) + 0.5)*size)
    if (r < 0) or (r >= size):
        return -1
    if (c < 0) or (c >= size):
        return -1
    return [r, c]

def GenerateFeatureImage(tri, quads, pts):
    global imgplot

    img_data = np.zeros((plotsize2, plotsize2))

    for quad in quads:
        feature_point_ids = GetFeaturePoints(tri, quad)
        dewarped_feature_points = GetDewarpedFeaturePoints(quad, feature_point_ids, pts)
        for fpt in dewarped_feature_points:
#            coord = ScaleToImageCoords(fpt, img_data.shape[0])
#            if coord == -1:
#                continue
#            img_data[coord] += 1

            r = int((fpt[0]/20)*img_data.shape[1])
            c = int((fpt[1]/(math.pi*2) + 0.5)*img_data.shape[0])
            if (r < 0) or (r >= plotsize2):
                continue
            if (c < 0) or (c >= plotsize2):
                continue
            img_data[r, c] += 1
    imgplot = plt.imshow(img_data)

# animation function.  This is called sequentially
def animate(t):
    global h, time, pause, step, selected_quad_id

    if pause and (step == False):
        return

    step = False
    time = time + 5
#    time = 0

    transformed_points = randomizedTransform(points, time, .2)
    linedata, tri = geo.generateDelaunyEdges(transformed_points)

    quads, simplex_pairs = ExtractValidQuads(transformed_points, tri)

    selected_quad_id = selected_quad_id % len(quads)

    GenerateFeatureImage(tri, quads, transformed_points)

    selected_quad = quads[selected_quad_id]
    selected_pts = []
    feature_pts = []
    warped_feature_pts = []
    plt_text.set_text(str(selected_quad_id) + '/' + str(len(quads)) + '\n' + str(selected_quad))

    for id in selected_quad:
        selected_pts.append(transformed_points[id])
    fids = GetFeaturePoints(tri, selected_quad)
    for fid in fids:
        feature_pts.append(transformed_points[fid])

    for wfp in GetDewarpedFeaturePoints(selected_quad, fids, transformed_points):

        r = ((wfp[0]/20)*plotsize2)
        c = ((wfp[1]/(math.pi*2) + 0.5)*plotsize2)
        warped_feature_pts.append([c, r])

    selected_pts = np.array(selected_pts)
    selection_point_data.set_data(selected_pts[:, 0], selected_pts[:, 1])
    fpts = np.array(feature_pts)
    feature_point_data.set_data(fpts[:, 0], fpts[:, 1])
    point_edges.set_data(linedata[:, 0], linedata[:, 1])

    warped_feature_pts = np.array(warped_feature_pts)
    warped_data.set_data(warped_feature_pts[:, 0], warped_feature_pts[:, 1])

#    point_data.set_data(transformed_points[:, 0], transformed_points[:, 1])


    return point_data


linedata, tri = geo.generateDelaunyEdges(points)
quads, simplex_pairs = ExtractValidQuads(points, tri)

anim = animation.FuncAnimation(fig,  animate, frames=20000, interval=20, blit=False)


plt.show()

cfm2=get_current_fig_manager().window
cfm2.activateWindow()
cfm2.raise_()