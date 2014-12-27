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

plotsize = 6
plotsize2 = 128
neighbor_count = 3
point_count = 50
points = np.zeros((point_count, 2))
fig = plt.figure(figsize=(14, 7))
dotplot = fig.add_subplot(121, aspect='equal', autoscale_on=False, xlim=(-plotsize, plotsize), ylim=(-plotsize, plotsize))
warpedplot = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(0, plotsize2), ylim=(0, plotsize2))
#warpedplot = fig.add_subplot(122, aspect='equal', autoscale_on=False, xlim=(-6, 6), ylim=(-6, 6))

currHistogramData = np.zeros((plotsize2, plotsize2))
maxHistogramData = np.zeros((plotsize2, plotsize2))

point_edges, = dotplot.plot([])
point_data, = dotplot.plot([], [], 'bo')
feature_point_data, = dotplot.plot([], [], 'go')
selection_point_data, = dotplot.plot([], [], 'ro')

warped_edges, = warpedplot.plot([])
warped_data, = warpedplot.plot([], [], 'go')

mouse_pos = [0, 0]

plt_text = dotplot.text(0.05, 0.05, 'text',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=dotplot.transAxes)

def generateRandomPoints(points, scale):

    #reset
    for p in points:
        p[0] = 0
        p[1] = 0

    #create candidate, add if separated from existing point by > 10% of scale
    for p in points:
        separated = False
        while not separated:
#            m = scale*random.random()
            m = 0.5*scale*random.gauss(0, 1)
            theta = math.pi*2*random.random()
            x = m*math.cos(theta)
            y = m*math.sin(theta)
            minDist = scale
            #searching through points, and get min distance
            for q in points:
                dx = q[0] - x
                dy = q[1] - y
                dist = math.sqrt(dx*dx+dy*dy)
                if(dist < minDist):
                    minDist = dist
            separated = (minDist > (scale/20))
        #sufficiently far from existing points, so add
        p[0] = x
        p[1] = y
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

    angleRange = math.pi/4
    for i in range(len(points)):
        points3D[i] = [points[i][0], points[i][1], 0, 1]
    theta = angleRange*random.uniform(-1.0,1.0)
    phi = angleRange*random.uniform(-1.0,1.0)
    rho = math.pi*random.uniform(-1.0,1.0)
    rotMatrix = xform.euler_matrix(theta, phi, rho)

    worldMatrix = np.dot(worldMatrix, rotMatrix)
    worldMatrix[3, 0] = 0#5*math.sin(t/61.0) #move it away form the camera after rotation
    worldMatrix[3, 1] = 0#5*math.sin(t/81.0) #move it away form the camera after rotation
    worldMatrix[3, 2] = 10#12 - 5*math.sin(t/31.0) #move it away form the camera after rotation
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
    global h, pause, step, selected_quad_id, points, maxHistogramData
#    print('press', event.key)
#    sys.stdout.flush()
    if event.key==' ':
        pause = not pause
    if event.key=='.':
        step = True
    if event.key=='n':
        selected_quad_id += 1
        step = True

    if event.key == 'r':
        print("randomizing points")
        generateRandomPoints(points, 7)
        maxHistogramData = np.zeros((plotsize2, plotsize2))

    if event.key == 'z':
        print("saving points")
        np.save("points", points)


cid = fig.canvas.mpl_connect('button_press_event', onclick)
cid = fig.canvas.mpl_connect('key_press_event', press)


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
    return feature_point_ids

def GetDewarpedFeaturePoints(quad, feature_ids, points):
    pa = [points[quad[0]], points[quad[1]], points[quad[2]], points[quad[3]]]
    pa = np.reshape(pa, (4, 2))

    min_square = -1
    max_square = 1
    pb = np.array([[min_square, min_square], [min_square, max_square], [max_square, max_square], [max_square, min_square]])

    feature_points = []
    for fp_id in feature_ids:
        feature_points.append(points[fp_id])
    h = find_homography(pa, pb)
    b_points = apply_homography(h, feature_points)

    for p in b_points:
        #mod by pi/2 provides 90 degree symmetry since we don't know which way is up
        theta = (math.atan2(p[1], p[0]) + math.pi)%(math.pi/2)
        m = math.log(math.sqrt(p[1]*p[1] + p[0]*p[0]))
        p[0] = m*(plotsize2/5)
        p[1] = theta*plotsize2/(math.pi/2)

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
    global imgplot, currHistogramData, maxHistogramData

    #faster way to do this?
    currHistogramData = np.zeros((plotsize2, plotsize2))
    updates = 0
    for quad in quads:
        feature_point_ids = GetFeaturePoints(tri, quad)
        dewarped_feature_points = GetDewarpedFeaturePoints(quad, feature_point_ids, pts)
        for fpt in dewarped_feature_points:
#            coord = ScaleToImageCoords(fpt, img_data.shape[0])
#            if coord == -1:
#                continue
#            img_data[coord] += 1

            r = fpt[0] #int((fpt[0]/20)*img_data.shape[1])
            c = fpt[1] #int((fpt[1]/(math.pi*2) + 0.5)*img_data.shape[0])
            if (r < 0) or (r >= plotsize2):
                continue
            if (c < 0) or (c >= plotsize2):
                continue
            currHistogramData[r, c] += 1
            if currHistogramData[r, c] > maxHistogramData[r, c]:
                maxHistogramData[r, c] = currHistogramData[r, c]
                updates += 1

    imgplot = plt.imshow(maxHistogramData)
    return updates

# animation function.  This is called sequentially
def animate(t):
    global h, time, pause, step, selected_quad_id

    if pause and (step == False):
        return

    step = False
    time = time + 1

    #do triangulation and extract quads
    transformed_points = randomizedTransform(points, time, .2)
    linedata, tri = geo.generateDelaunyEdges(transformed_points)
    quads, simplex_pairs = ExtractValidQuads(transformed_points, tri)

    #generate a finger print image for this triangulation
    print("updates:", GenerateFeatureImage(tri, quads, transformed_points))

    #identify selected quad -----
#    selected_quad_id = selected_quad_id % len(quads)
#    selected_quad = quads[selected_quad_id]
#    selected_pts = []
#    feature_pts = []
#    warped_feature_pts = []
#    plt_text.set_text(str(selected_quad_id) + '/' + str(len(quads)) + '\n' + str(selected_quad))
#    for id in selected_quad:
#        selected_pts.append(transformed_points[id])
#    fids = GetFeaturePoints(tri, selected_quad)
#    for fid in fids:
#        feature_pts.append(transformed_points[fid])
#    for wfp in GetDewarpedFeaturePoints(selected_quad, fids, transformed_points):
#        warped_feature_pts.append([wfp[1], wfp[0]])
#    selected_pts = np.array(selected_pts)
#    fpts = np.array(feature_pts)
#    selection_point_data.set_data(selected_pts[:, 0], selected_pts[:, 1])
#    feature_point_data.set_data(fpts[:, 0], fpts[:, 1])
#    warped_feature_pts = np.array(warped_feature_pts)
#    warped_data.set_data(warped_feature_pts[:, 0], warped_feature_pts[:, 1])
    #end selected quad ui -------------

    #draw points and delauny edges
#    point_edges.set_data(linedata[:, 0], linedata[:, 1])
    point_data.set_data(transformed_points[:, 0], transformed_points[:, 1])
    return point_data


points = np.load('points.npy')
anim = animation.FuncAnimation(fig,  animate, frames=20000, interval=20, blit=False)

plt.show()
#cfm2 = get_current_fig_manager().window
#cfm2.activateWindow()
#cfm2.raise_()