__author__ = 'jcl5m1'

import numpy as np
import cv2
from itertools import *
from random import *
from math import *
import collections
from ImageUtilities import *

class TrainingRegion:
    p1 = (0,0)
    p2 = (0,0)
    center = (0,0)

    def __init__(self):
        self.set((0, 0), (0, 0))

    def set_p2(self, p2):
        self.set(self.p1, p2)

    def set(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.center = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)

    def unflip_corners(self):
        x1 = self.p1[0]
        x2 = self.p2[0]
        y1 = self.p1[1]
        y2 = self.p2[1]
        if self.p1[0] > self.p2[0]:
            x2 = self.p1[0]
            x1 = self.p2[0]
        if self.p1[1] > self.p2[1]:
            y2 = self.p1[1]
            y1 = self.p2[1]
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)
        self.center = ((self.p1[0] + self.p2[0])/2, (self.p1[1] + self.p2[1])/2)


    def set_using_radius(self, x, y, r):
        self.center = (x, y)
        self.p1 = (x-r, y-r)
        self.p2 = (x+r, y+r)

    def copy(self):
        region = TrainingRegion()
        region.p1 = self.p1
        region.p2 = self.p2
        region.center = self.center
        return region

    def recenter(self, x, y):
        dx = x-self.center[0]
        dy = y-self.center[1]
        self.center = (self.center[0] + dx, self.center[1] + dy)
        self.p1 = (self.p1[0] + dx, self.p1[1] + dy)
        self.p2 = (self.p2[0] + dx, self.p2[1] + dy)

    def is_equal(self, region):
        if cmp(self.p1, region.p1) != 0:
            return False
        if cmp(self.p2, region.p2) != 0:
            return False
        return True

    def offset(self, x, y):
        self.set((self.p1[0] + x, self.p1[1] + y),(self.p2[0] + x, self.p2[1] + y))

    def width(self):
        return int(fabs(self.p1[0]-self.p2[0]))

    def height(self):
        return int(fabs(self.p1[1]-self.p2[1]))

    def draw(self, img, color=(0, 0, 255)):
        cross_size = 5
        thickness = 1
        cv2.rectangle(img, self.p1, self.p2, color, thickness)
        cross_hair(img, self.center,cross_size, color, thickness)
