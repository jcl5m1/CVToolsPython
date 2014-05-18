__author__ = 'johnny'

import numpy as np
import cv2
from itertools import *
from random import *
from math import *
import collections

from TrainingRegion import *



class PixelSampleTest:
    x,y,v = 0, 0, 0
    def __init__(self):
        self.x = 0
        self.y = 0
        self.v = 0

    def randomize(self, img, region):
        self.x = randint(-region.width()/2, region.width()/2-1)
        self.y = randint(-region.height()/2, region.height()/2-1)
        self.v = img[region.center[1] + self.y, region.center[0] + self.x ]

    def evaluate(self, img, sample_pos):
        return img[sample_pos[1] + self.y, sample_pos[0] + self.x][0] == self.v[0]
#        return (img[sample_pos[1] + self.y, sample_pos[0] + self.x] == self.v).all()

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

        training_size = 20
        training_set = [(region, 1)]

        index = 0
        while len(training_set) < training_size:
            if index >= len(pixel_order):
                break
            x, y = pixel_order[index]
            index += 1

            test_region = region.copy()
            test_region.recenter(x,y)

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
        for c in range(margin,src_img.shape[1]-margin):
            for r in range(margin,src_img.shape[0] - margin):
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