__author__ = 'johnnylee'
import struct
import numpy as np
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
flatWallAngleThreshold = 150

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
    totalRMSError = 0
    leftMean = np.array([0, 0, 0])
    leftNormal = np.array([0, 0, 1])
    leftRMSError = 0
    rightMean = np.array([0, 0, 0])
    rightNormal = np.array([0, 0, 1])
    rightRMSError = 0

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
                    print "List Properties unsupported"
                else:
                    properties.append((self.propertyFormatKey[parts[1]], self.propertyBytes[parts[1]], parts[2]))

    def parseHeader(self, plyfile):
        line = self.cleanLine(plyfile.readline())
        if line != "ply":
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
                    print "ERROR: only supports binary little endian format currently"
                    return -1
                print "ERROR: unknown format"
                return -1
            if parts[0] == "element":
                self.elements[parts[1]] = (int(parts[2]), [])
                if self.parseElement(plyfile, self.elements[parts[1]]) != 0:
                    break
            if parts[0] == "end_header":
                break
        return 0

    def load(self, plyfile, xOffsetMM, minHorzAngle, maxHorzAngle, minVertAngle, maxVertAngle):
        global subsample
        if self.parseHeader(plyfile) != 0:
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
        print "Vertices: ", vertexCount

        outIndex = 0
        for i in range(vertexCount):
            if ((i+1) % 100000) == 0:
                print("    loading: {0:.2f}%".format(100*float(i)/vertexCount))
            v = struct.unpack(fmt, plyfile.read(tot))

            #gets or of things at zero distance or on z plane
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
        print "Horz Offset: ", xOffsetMM
        print "CroppingAngles: ", (minHorzAngle, maxHorzAngle), (minVertAngle, maxVertAngle)
        print "Done Loading. Uncropped Vertices: ", outIndex

    def computeAnalysis(self):
        global flatWallAngleThreshold
        self.isCorner = True
        self.mean, self.normal = self.planeFit(self.vertices.transpose())

        self.totalRMSError = self.computeRMSPercentageError(self.vertices, self.mean, self.normal)

        #does not do a best fit corner, just splits along x = 0, does not reject outliers
        #left plane
        leftPoints = np.zeros(self.vertices.shape)
        index = 0
        for p in self.vertices:
            if p[0] < 0:
                leftPoints[index] = p
                index += 1
        leftPoints = leftPoints[:index]
        self.leftMean, self.leftNormal = self.planeFit(leftPoints.transpose())
        self.leftRMSError = self.computeRMSPercentageError(leftPoints, self.leftMean, self.leftNormal)

        #right plane
        rightPoints = np.zeros(self.vertices.shape)
        index = 0
        for p in self.vertices:
            if p[0] > 0:
                rightPoints[index] = p
                index += 1
        rightPoints = rightPoints[:index]
        self.rightMean, self.rightNormal = self.planeFit(rightPoints.transpose())
        self.rightRMSError = self.computeRMSPercentageError(rightPoints, self.rightMean, self.rightNormal)

        #compute corner line intersection with y=0 plane
        lpDir = np.cross(np.array((0,1,0)), self.leftNormal)
        rpDir = np.cross(np.array((0,1,0)), self.rightNormal)

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

        #check if it looks like a corner
        angle = 180 - 180*math.acos(self.leftNormal.dot(self.rightNormal))/math.pi
        if angle > flatWallAngleThreshold:
            print "Best Fit planes are >", flatWallAngleThreshold, " degrees. Using flat wall analysis..."
            self.isCorner = False

    def computeRMSPercentageError(self, points, mean, normal):
        totalError = 0
        count = 0
        for p in points:
            #divides by distance to camera
            div = math.sqrt(p.dot(p))
            if math.fabs(div) > sys.float_info.epsilon:
                err = (p-mean).dot(normal)/div
                totalError += err*err
                count += 1
        return math.sqrt(totalError/count)

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

    def printAnalyis(self):
        print "Rendered points: ", self.rendercount
        if self.isCorner:
            print "Corner Intersection (mm): ", self.cornerIntersection
            print "Corner Angle(deg): ", 180 - 180*math.acos(self.leftNormal.dot(self.rightNormal))/math.pi
            print "Total RMS % Error (err/dist): ", 100*(self.leftRMSError + self.rightRMSError)/2.0
        else:
            print "Average Position (mm): ", self.mean
            print "Normal Vector: ", self.normal
            print "Total RMS % Error (err/dist): ", 100*self.totalRMSError

def generatePlyFile(isCorner):
    gridsize = 100
    dist = 4000
    scalediv = 2
    step = (2*dist/scalediv)/gridsize
    noise = 0.01

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
    print "Generated Points", counter
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

    if (myPly.mean[2] > 0):
        glScale(1.0, 1.0, -1.0)

    glBegin(GL_POINTS)
    glColor3f(1.0, 0.0, 0.0)
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
    glTranslatef(0,0,frustumDist - 2*labelSize)
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
    parser.add_argument('-f', action='store', dest='filename', help='PLY file (binary little endian)')
    parser.add_argument('-x', dest='xOffset', help='Horz offset for corner split')
    parser.add_argument('-c', nargs=4, dest='angle', help='Crop angles in degrees (minH, maxH, minV, maxV)')
    parser.add_argument('-test_wall', action="store_true", default=False, help='Synthetic Test Wall')
    parser.add_argument('-no_vis', action="store_true", default=False, help='No visualization')
    results = parser.parse_args()

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

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
        if results.test_wall:
            print "Synthetic wall test"
            myPly.load(generatePlyFile(False), xOffsetMM, cropAngles[0], cropAngles[1], cropAngles[2], cropAngles[3])
        else:
            #generate corner test is default case
            print "Synthetic corner test"
            myPly.load(generatePlyFile(True), xOffsetMM, cropAngles[0], cropAngles[1], cropAngles[2], cropAngles[3])
    else:
        filename = results.filename
        print "Loading ", filename
        myPly.load(open(filename, 'rb'), xOffsetMM, cropAngles[0], cropAngles[1], cropAngles[2], cropAngles[3])

    print "Computing statistics using", myPly.vertices.shape[0], "points"
    myPly.computeAnalysis()
    myPly.printAnalyis()

    if not results.no_vis:
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