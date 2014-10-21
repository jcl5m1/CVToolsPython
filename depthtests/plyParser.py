__author__ = 'johnnylee'
import struct
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D
import random

class ply:

    vertices = 0
    format = "binary_little_endian"
    points = np.array(0)

    elementBytes = {'char': 1, 'uchar': 1, 'short': 2, 'ushort': 2, 'int': 4, 'uint': 4, 'float': 4, 'double': 8}
    elementFormatKey = {'char': 'b', 'uchar': 'B', 'short': 'h', 'ushort': 'H', 'int': 'i', 'uint': 'I', 'float': 'f', 'double': 'd'}

    elements = []

    def __init__(self):
        return

    def parseHeaderLine(self, line):
        line = line.rstrip()
        line = re.sub(' +', ' ', line) #removes redundant white space
        parts = line.split(' ')
        if parts[0] == "comment":
            return 0
        if parts[0] == "format":
            if parts[1] == "binary_little_endian":
                self.format = parts[1]
                return 0
            if parts[1] == "binary_big_endian":
                print "only supports binary little endian format currently"
                return -1
            if parts[1] == "ascii":
                print "only supports binary little endian format currently"
                return -1
            print "unknown format"
            return -1
        if parts[0] == "element":
            if parts[1] == "vertex":
                self.vertices = int(parts[2])
                print "vertices: " + str(self.vertices)
            else:
                print "ERROR: this script only supports vertex elements"
            return 0
        if parts[0] == "property":
            self.elements.append((self.elementFormatKey[parts[1]], self.elementBytes[parts[1]], parts[2]))
#            if parts[1] == "float":
#                print "float: " + parts[2],
 #           else:
  #              print "ERROR: this script only support floats currently"
            return 0
        if parts[0] == "end_header":
            return 1

    def parseBinaryLittleEndianFloat(self, data, index):
        return
        print file.readline()
#        print struct.unpack('f',file.read(4))

    def read(self, file):

        line = file.readline().rstrip()
        if line != "ply":
            print "Not a valid ply file"
            return -1

        while True:
            line = file.readline()
            print line
            if self.parseHeaderLine(line) != 0:
                break

        print self.elements
        print len(self.elements)

        self.points = np.zeros((self.vertices, len(self.elements)))
        for i in range(self.vertices):
            for e in self.elements:
                self.points[i, 0] = struct.unpack('f'*3, file.read(4*3))
        file.close()

        return


    def plot(self):

        fig = plt.figure()
        randIndex = []
        for i in range(2000):
            randIndex.append(random.randrange(self.points.size/3))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.points[randIndex, 0], self.points[randIndex, 1], self.points[randIndex, 2])


#        ax = fig.add_subplot(131)
#        ax2 = fig.add_subplot(132)
#        ax2 = fig.add_subplot(133)
#        ax.plot(self.points[:, 0], self.points[:, 1])
 #       ax2.plot(self.points[:, 0], self.points[:, 2])
  #      ax2.plot(self.points[:, 1], self.points[:, 2])
        plt.show()


#filename = "pmd_wall_recheck_1m.ply"
#filename = "pmd_corner_1m_5200 (1).ply"
#filename = "pmd_corner_1m_wall_5200_ff (1).ply"
filename = "inuitive_objects_manual.ply"

myPly = ply()
myPly.read(open(filename, 'rb'))
myPly.plot()