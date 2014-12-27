#Copyright [2014] [Google]
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

__author__ = 'johnnylee'
__version__ = '2014.10.29'
import struct
import numpy as np
import sys
import os
import re
import math
import random
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import argparse

viewRotation = 0
subsample = 10000
viewTopDown = False
resultOnlyOutput = False

class ply:
    version = 1.0
    format = "binary_little_endian"
    rendervertices = np.array(0)
    vertices = np.array(0)
    rendercount = subsample
    isCorner = False

    mean = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    cornerIntersection = np.array([0, 0, 1])
    totalBelow1percent = 0
    leftMean = np.array([0, 0, 0])
    leftNormal = np.array([0, 0, 1])
    leftBelow1Percent = 0
    leftBelow4Percent = 0
    rightMean = np.array([0, 0, 0])
    rightNormal = np.array([0, 0, 1])
    rightBelow1Percent = 0
    rightBelow4Percent = 0
    cornerAngle = 0

    propertyBytes = {'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2, 'int': 4, 'uint': 4, 'float': 4, 'double': 8}
    propertyFormatKey = {'char': 'b', 'uchar': 'B', 'short': 'h', 'ushort': 'H', 'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd'}

    elements = {}

    def __init__(self):
        return

    def cleanLine(self, line):
        return re.sub(' +', ' ', line.rstrip()) #removes redundant white space

    def parseElement(self, plyfile, element):
        properties = element[1]
        for i in range(10):
            line = plyfile.readline()
            parts = self.cleanLine(line).split(' ')

            if parts[0] == "element":
                plyfile.seek(-len(line),1) #rewind to the beginning of the line
                return 0
            if parts[0] == "end_header":
                plyfile.seek(-len(line),1) #rewind to the beginning of the line
                return 0

            if parts[0] == "property":
                if parts[1] == 'list':
                    if not resultOnlyOutput:
                        print "List Properties unsupported"
                else:
                    properties.append((self.propertyFormatKey[parts[1]], self.propertyBytes[parts[1]], parts[2]))

    def parseHeader(self, plyfile):
        line = self.cleanLine(plyfile.readline())
        if line != "ply":
            if not resultOnlyOutput:
                print "Not a valid ply file"
            return -1

        while True:
            line = plyfile.readline()
            parts = self.cleanLine(line).split(' ')
            if parts[0] == "comment":
                continue
            if parts[0] == "format":
                if parts[1] == "binary_little_endian":
                    self.format = parts[1]
                    self.version = parts[2]
                    continue
                if parts[1] == "binary_big_endian":
                    print "ERROR: only supports binary little endian format currently"
                    return -1
                if parts[1] == "ascii":
                    self.format = parts[1]
                    self.version = parts[2]
                    continue
                print "ERROR: unknown format"
                return -1
            if parts[0] == "element":
                self.elements[parts[1]] = (int(parts[2]), [])
                if self.parseElement(plyfile, self.elements[parts[1]]) != 0:
                    break
            if parts[0] == "end_header":
                break
        return 0

    def load(self, plyfile, xOffsetMM, minHorzAngle, maxHorzAngle, minVertAngle, maxVertAngle, rotz, scale):
        global subsample
        if self.parseHeader(plyfile) != 0:
            print "load error"
            return

        #load vertices
        vertexCount = self.elements['vertex'][0]
        properties = self.elements['vertex'][1]
        fmt = ""
        tot = 0
        for p in properties:
            fmt += p[0]
            tot += p[1]

        self.rendervertices = np.zeros((self.rendercount, 3))
        self.vertices = np.zeros((vertexCount, 3))
        if not resultOnlyOutput:
            print "Vertices: ", vertexCount

        outIndex = 0
        for i in range(vertexCount):
            if ((i+1) % 100000) == 0:
                print("    loading: {0:.2f}%".format(100*float(i)/vertexCount))
            if self.format == "binary_little_endian":
                v = struct.unpack(fmt, plyfile.read(tot))
            if self.format == "ascii":
                parts = plyfile.readline().split(' ')
                v = np.array((float(parts[0]), float(parts[1]), float(parts[2])))

            if rotz:
                temp = v[1]
                v[1] = -v[0]
                v[0] = temp

            v *= scale

            #gets rid of things at zero distance or on z=0 plane
            if math.fabs(v[2]) < sys.float_info.epsilon:
                continue
            mag = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            if mag < sys.float_info.epsilon:
                continue

            #angle crop
            horzAngle = 180*math.atan(v[0]/-v[2])/math.pi
            vertAngle = 180*math.atan(v[1]/-v[2])/math.pi
            if(horzAngle < minHorzAngle) or (horzAngle > maxHorzAngle):
                continue
            if(vertAngle < minVertAngle) or (vertAngle > maxVertAngle):
                continue

            #include vertex and apply offset
            self.vertices[outIndex, :3] = v[:3]
            self.vertices[outIndex, 0] += xOffsetMM
            outIndex += 1

        #truncate
        self.vertices = self.vertices[:outIndex]

        #subsample for rendering
        for i in range(self.rendercount):
            self.rendervertices[i] = self.vertices[random.randrange(0, outIndex-1)]

        plyfile.close()
        if not resultOnlyOutput:
            print "Horz Offset: ", xOffsetMM
            print "CroppingAngles: ", (minHorzAngle, maxHorzAngle), (minVertAngle, maxVertAngle)
            print "Done Loading. Uncropped Vertices: ", outIndex

    def computeAnalysis(self, isCorner, bestfit, dist):
        global flatWallAngleThreshold
        self.isCorner = isCorner

        self.mean = np.mean(self.vertices, 0)
        if self.mean[2] > 0:
            for v in self.vertices:
                v[2] *= -1
            for v in self.rendervertices:
                v[2] *= -1
        if not isCorner:
            if bestfit:
                self.mean, self.normal = self.planeFit(self.vertices.transpose())
            else:
                self.mean = np.array((0, 0, dist))
                self.normal = np.array((0, 0, -1))

            self.totalBelow1percent = self.computePercentageAboveError(self.vertices, self.mean, self.normal, 0.01)
            self.totalBelow4percent = self.computePercentageAboveError(self.vertices, self.mean, self.normal, 0.04)
        else:
            #does not do a best fit corner, just splits along x = 0, does not reject outliers
            #left plane
            leftPoints = np.zeros(self.vertices.shape)
            index = 0
            for p in self.vertices:
                if p[0] < 0:
                    leftPoints[index] = p
                    index += 1
            leftPoints = leftPoints[:index]

            if bestfit:
                self.leftMean, self.leftNormal = self.planeFit(leftPoints.transpose())
            else:
                self.leftMean = np.array((-500, 0, dist+500))
                self.leftNormal = np.array((math.sqrt(2)/2, 0, math.sqrt(2)/2))

            self.leftBelow1Percent = self.computePercentageAboveError(leftPoints, self.leftMean, self.leftNormal, 0.01)
            self.leftBelow4Percent = self.computePercentageAboveError(leftPoints, self.leftMean, self.leftNormal, 0.04)

            #right plane
            rightPoints = np.zeros(self.vertices.shape)
            index = 0
            for p in self.vertices:
                if p[0] > 0:
                    rightPoints[index] = p
                    index += 1
            rightPoints = rightPoints[:index]

            if bestfit:
                self.rightMean, self.rightNormal = self.planeFit(rightPoints.transpose())
            else:
                self.rightMean = np.array((500, 0, dist+500))
                self.rightNormal = np.array((-math.sqrt(2)/2, 0, math.sqrt(2)/2))

            self.rightBelow1Percent = self.computePercentageAboveError(rightPoints, self.rightMean, self.rightNormal, 0.01)
            self.rightBelow4Percent = self.computePercentageAboveError(rightPoints, self.rightMean, self.rightNormal, 0.04)

            #compute corner line intersection with y=0 plane
            lpDir = np.cross(np.array((0, 1, 0)), self.leftNormal)
            rpDir = np.cross(np.array((0, 1, 0)), self.rightNormal)

            #project the means to the y=0 plane
            lp = self.leftMean
            lp[1] = 0
            rp = self.rightMean
            rp[1] = 0

            #compute the intersection point of two lines in the y=0 plane
            z = rp - lp
            a = np.linalg.norm(np.cross(z, rpDir))
            b = np.linalg.norm(np.cross(lpDir, rpDir))
            if b > sys.float_info.epsilon:
                if lpDir.dot(z) > 0:
                    self.cornerIntersection = lp + (a/b)*lpDir
                else:
                    self.cornerIntersection = lp - (a/b)*lpDir
            self.cornerAngle = 180 - 180*math.acos(self.leftNormal.dot(self.rightNormal))/math.pi

    def computePercentageAboveError(self, points, mean, normal, errorPercentThreshold):
        totalBelowThresh = 0
        totalCount = 0
        for p in points:
            #divides by distance to camera
            div = math.sqrt(p.dot(p))
            if math.fabs(div) > sys.float_info.epsilon:
                err = math.fabs((p-mean).dot(normal)/div)
                if err < errorPercentThreshold:
                    totalBelowThresh += 1
                totalCount += 1
        return float(100.0)*totalBelowThresh/float(totalCount)

    def planeFit(self, points):
        from numpy.linalg import svd
        points = np.reshape(points, (points.shape[0], -1))
        assert points.shape[0] < points.shape[1]
        ctr = points.mean(axis=1)
        x = points - ctr[:, None]
        M = np.dot(x, x.T)
        n = svd(M)[0][:, -1]
        #flip the normal if needed so it is facing toward the origin
        if ctr.dot(n) > 0:
            n *= -1
        return ctr, n

    def printAnalyis(self, filename, result_only):

        if result_only:
            if(self.isCorner):
                print filename, self.vertices.shape[0], self.cornerIntersection[2], (self.leftBelow1Percent + self.rightBelow1Percent)/2.0, (self.leftBelow4Percent + self.rightBelow4Percent)/2.0, self.cornerAngle
            else:
                print filename,  self.vertices.shape[0], self.mean[2], self.totalBelow1percent, self.totalBelow4percent
            return
        if self.isCorner:
            print "Test Corner Intersection (mm): ", self.cornerIntersection
            print "Corner Angle(deg): ", self.cornerAngle
            print "Number of points: ", self.vertices.shape[0]
            print "Percent of points below 1% err: ", (self.leftBelow1Percent + self.rightBelow1Percent)/2.0
            print "Percent of points below 4% err: ", (self.leftBelow4Percent + self.rightBelow4Percent)/2.0
        else:
            print "Normal Vector: ", self.normal
            print "Test Plane Position (mm): ", self.mean
            print "Number of points: ", self.vertices.shape[0]
            print "Percent of points below 1% err: ", self.totalBelow1percent
            print "Percent of points below 4% err: ", self.totalBelow4percent

def generatePlyFile(isCorner):
    gridsize = 100
    dist = 4000
    scalediv = 2
    step = (2*dist/scalediv)/gridsize
    noise = 0.012

    genPly = "ply\n"
    genPly += "format binary_little_endian 1.0\n"
    genPly += "element vertex " + str(gridsize*gridsize) + "\n"
    genPly += "property float x\n"
    genPly += "property float y\n"
    genPly += "property float z\n"
    genPly += "end_header\n"
    counter = 0
    if isCorner:
        for x in range(-dist/scalediv, dist/scalediv, step):
            for y in range(-dist/scalediv, dist/scalediv, step):
                counter += 1
                v = np.array((float(x), float(y), float(-dist)))
                if x < 0:
                    v[2] = -dist - v[0]
                else:
                    v[2] = -dist + v[0]
                mag = math.sqrt(v.dot(v))
                v /= mag
                mag += random.uniform(-noise*mag, noise*mag) #average will be half
                genPly += struct.pack("fff", v[0]*mag, v[1]*mag, v[2]*mag)
    else: #flat wall
        for x in range(-dist/scalediv, dist/scalediv, step):
            for y in range(-dist/scalediv, dist/scalediv, step):
                counter += 1
                v = np.array((float(x), float(y), float(-dist)))
                mag = math.sqrt(v.dot(v))
                v /= mag
                mag += random.uniform(-noise*mag, noise*mag) #average will be half
                genPly += struct.pack("fff", v[0]*mag, v[1]*mag, v[2]*mag)
    file = open("testPly.ply", 'wb')
    file.write(genPly)
    file.close()
    file = open("testPly.ply", 'rb')
    return file

def renderPlane(point, normal, size):
    forward = np.array([0, 0, 1])
    normal = normal/np.linalg.norm(normal)
    rotVect = np.cross(forward, normal)

    if 1 + forward.dot(normal) < sys.float_info.epsilon:
        angle = 0
        rotVect = normal
    else:
        angle = 180*math.acos(forward.dot(normal))/math.pi
        rotVect = rotVect/np.linalg.norm(rotVect)

    glColor3f(1.0, 1.0, 1.0)
    glPushMatrix()
    glTranslatef(point[0], point[1], point[2])
    glRotatef(angle, rotVect[0], rotVect[1], rotVect[2])
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, size/2)
    glVertex3f(-size, -size, 0)
    glVertex3f(+size, +size, 0)
    glVertex3f(+size, -size, 0)
    glVertex3f(-size, +size, 0)
    glVertex3f(-size, -size, 0)
    glVertex3f(+size, -size, 0)
    glVertex3f(-size, +size, 0)
    glVertex3f(+size, +size, 0)
    glVertex3f(-size, -size, 0)
    glVertex3f(-size, +size, 0)
    glVertex3f(+size, -size, 0)
    glVertex3f(+size, +size, 0)
    glEnd()
    glPopMatrix()

def initPersectiveView():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(70, 1, 0.1, 10000)
    gluLookAt(0, 1000, 4000, 0, 0, 0, 0, 1, 0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def initTopDownView():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-3000, 3000, -3000, 3000, -10000, 10000)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(90, 1, 0, 0)

def displayFunc():
    global viewRotation, myPly, viewTopDown
    glClear(GL_COLOR_BUFFER_BIT)

    if viewTopDown:
        initTopDownView()
        viewRotation = 0
    else:
        initPersectiveView()
        viewRotation += 1.0
    glRotatef(viewRotation, 0, 1, 0)
    if myPly.mean[2] > 0:
        glTranslatef(myPly.mean[0], myPly.mean[1], myPly.mean[2])
    else:
        glTranslatef(-myPly.mean[0], -myPly.mean[1], -myPly.mean[2])

    #render points
    glPointSize(2.0)
    glColor3f(1.0, 0.0, 0.0)
    glPushMatrix()

    glBegin(GL_POINTS)
    for v in myPly.rendervertices:
        if(myPly.isCorner):
            if(v[0] < 0):
                glColor3f(1.0, 0.0, 0.0)
            else:
                glColor3f(0.0, 1.0, 1.0)
        glVertex3f(v[0], v[1], v[2])
    glEnd()

    #render best fit planes
    if myPly.isCorner:
        renderPlane(myPly.leftMean, myPly.leftNormal, 500)
        renderPlane(myPly.rightMean, myPly.rightNormal, 500)
    else:
        renderPlane(myPly.mean, myPly.normal, 500)
    glPopMatrix()

    #render frustum lines
    glColor3f(0.4, 0.4, 0.4)
    frustumDist = -4000
    frustumWidthRatio = 0.5
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, frustumDist)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(-frustumWidthRatio*frustumDist, 0.0, frustumDist)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(frustumWidthRatio*frustumDist, 0.0, frustumDist)

    for i in range(-1000, frustumDist-1, -1000):
        glVertex3f(frustumWidthRatio*i, 0.0, i)
        glVertex3f(-frustumWidthRatio*i, 0.0, i)
    glEnd()

    #render Z- label
    glPushMatrix()
    labelSize = 100
    glTranslatef(0, 0, frustumDist - 2*labelSize)
    glBegin(GL_LINES)
    glVertex3f(labelSize, 0.0, labelSize)
    glVertex3f(-labelSize, 0.0, labelSize)
    glVertex3f(-labelSize, 0.0, labelSize)
    glVertex3f(labelSize, 0.0, -labelSize)
    glVertex3f(-labelSize, 0.0, -labelSize)
    glVertex3f(labelSize, 0.0, -labelSize)

    glVertex3f(-2*labelSize, 0.0, 0.0)
    glVertex3f(- labelSize, 0.0, 0.0)

    glEnd()
    glPopMatrix()

    glutSwapBuffers()
    glutPostRedisplay()

def keyboardFunc(key, x, y):
    global viewTopDown
    if(viewTopDown):
        viewTopDown = False
    else:
        viewTopDown = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='filename', help='PLY file (binary little endian or ascii)')
    parser.add_argument('-x', dest='xOffset', help='Horz offset for corner split', default=0)
    parser.add_argument('-c', nargs=4, dest='angle', help='Crop angles in degrees (minH, maxH, minV, maxV)')
    parser.add_argument('-d', dest='distance', help='Known distance to wall or corner point(mm)', default=-4000)
    parser.add_argument('-s', dest='scale', help='Adjust scale of the data', default=1)
    parser.add_argument('-rotz', action="store_true", default=False, help='Rot 90 deg on Z axis')
    parser.add_argument('-corner', action="store_true", default=False, help='Corner Dataset')
    parser.add_argument('-best_fit', action="store_true", default=False, help='Use Best Fit Plane')
    parser.add_argument('-no_vis', action="store_true", default=False, help='No visualization')
    parser.add_argument('-result_only', action="store_true", default=False, help='Result text only')
    results = parser.parse_args()

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    resultOnlyOutput = results.result_only

    xOffsetMM = 0
    cropAngles = [-90, 90, -90, 90]
    if results.angle is not None:
        cropAngles[0] = float(results.angle[0])
        cropAngles[1] = float(results.angle[1])
        cropAngles[2] = float(results.angle[2])
        cropAngles[3] = float(results.angle[3])
    if results.xOffset is not None:
        xOffsetMM = float(results.xOffset)

    myPly = ply()
    if results.filename is None:
        if not resultOnlyOutput:
            print "Generating Synthetic Data"
        file = generatePlyFile(results.corner)
    else:
        filename = results.filename
        if not resultOnlyOutput:
            print "Loading ", filename
        file = open(filename, 'rb')

    myPly.load(file, xOffsetMM, cropAngles[0], cropAngles[1], cropAngles[2], cropAngles[3], results.rotz, int(results.scale))
    myPly.computeAnalysis(results.corner, results.best_fit, int(results.distance))
    myPly.printAnalyis(results.filename, results.result_only)

    if not results.no_vis and not resultOnlyOutput:
        print " - press any key to toggle top down view"
        print " - use host OS keyboard command to quit"
        glutInit()
        glutInitWindowSize(800, 800)
        glutCreateWindow("Ply Data")
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutDisplayFunc(displayFunc)
        glutKeyboardFunc(keyboardFunc)
        initPersectiveView()
        glutMainLoop()