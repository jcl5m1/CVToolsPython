__author__ = 'johnny'

import numpy as np
import cv2
from itertools import *
from random import *
from math import *
import collections

def generate_random_pixel_order(img, margin):
    pixel_order = {}

    if margin < 0:
        raise ValueError('Error: negative margin')

    for c in range(margin,img.shape[0]-margin):
        for r in range(margin,img.shape[1]-margin):
            pixel_order[uniform(0.0,1.0)] = [c,r]
    od = collections.OrderedDict(sorted(pixel_order.items()))
    pixel_array = []
    for k, v in od.iteritems():
        pixel_array.append(v)
    return pixel_array

def cross_hair(img, p, size, color, thickness = 1):
    p1 = (p[0], p[1] + size)
    p2 = (p[0], p[1] - size)
    cv2.line(img, p1,p2, color, thickness)
    p1 = (p[0] + size, p[1])
    p2 = (p[0] - size, p[1])
    cv2.line(img, p1,p2, color, thickness)

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
        self.center = ((p1[0] +p2[0])/2, (p1[1] +p2[1])/2)

    def set_using_radius(self, x, y, r):
        self.center = (x,y)
        self.p1 = (x-r, y-r)
        self.p2 = (x+r, y+r)

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

class PixelSampleTest:
    x,y,v = 0, 0, 0
    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = 0

    def randomize(self, img, region):
        self.x = randint(-region.width()/2, region.width()/2)
        self.y = randint(-region.height()/2, region.height()/2)
        self.v = img[region.center[1] + self.y, region.center[0] + self.x ]

    def evaluate(self, img, center):
        return (img[center[1] + self.y,center[0] + self.x] == self.v).all()
#        return (img[center[0] + self.x, center[1] + self.y] == self.v).all()

class VisualDecisionTree:
    left, right, test, best_test, data = None, None, None, None, 0
    def __init__(self):
        self.left = None
        self.right = None
        self.test = None
        self.data = 0

    def train(self, img, region, pixel_order, depth = 5):
        debug_img = img.copy()
        #select a few negative random regions from the image as training set

        #randomize pixel test
        self.test = PixelSampleTest()
        self.test.randomize(img, region)

        training_size = 100
        training_set = [(region, 1)]

        index = 0
        while len(training_set) < training_size:
            if index >= len(pixel_order):
                break
            x, y = pixel_order[index]
            index += 1

            #only add training examples that are passing the test
            if self.test.evaluate(img, (x, y)):
                test_region = TrainingRegion()
                test_region.set_using_radius(x,y,region.width()/2)
#                test_region.set((y-region.height()/2,x-region.width()/2), ( y + region.height()/2,x + region.width()/2))
                if not region.is_equal(test_region):
                    training_set.append((test_region, 0))

        for t in training_set:
            if t[1] == 1:
                t[0].draw(debug_img, (0, 255, 0))
            if t[1] == 0:
                t[0].draw(debug_img, (0, 0, 255))

        #compute information gain
        #iterate a for a while for best test
        return debug_img

    def classifyImg(self, src_img, dst_img, margin):
        for c in range(margin,src_img.shape[0]-margin):
            for r in range(margin,src_img.shape[1] - margin):
                dst_img[c, r] = self.classify(src_img, (c, r))

    def classifyImgRandomSubsample(self, src_img, dst_img, pixel_order, start, count):
        stop = start+count
        if stop >= len(pixel_order):
            stop = len(pixel_order)-1
        for n in range(start, stop):
            c, r = pixel_order[n]
            dst_img[r, c] = self.classify(src_img, (c, r))

    def classify(self, img, p):
        if self.test is None:
            return randint(0,255)
        else:
            if self.test.evaluate(img, p):
                return 255
            else:
                return 0