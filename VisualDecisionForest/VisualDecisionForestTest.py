__author__ = 'johnny'

import numpy as np
import cv2
from itertools import *
from random import *
from VisualDecisionForest import *
import time

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

tr = TrainingRegion()

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    global tr,ix,iy,drawing,mode, src_img, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        tr.set((x, y),(x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            tr.set_p2((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("image", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("debug", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("class", cv2.CV_WINDOW_AUTOSIZE)

src_img = cv2.imread('..//data//2048-screenshot.png')
tr.set_using_radius(80,200,50)
randomized_pixel_order = generate_random_pixel_order(src_img, tr.width()/2)
img = src_img.copy()

class_img = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
tree = VisualDecisionTree()
debug_img = tree.train(src_img, tr, randomized_pixel_order)

cv2.moveWindow("image", 0, 0)
cv2.moveWindow("debug", src_img.shape[0], 0)
cv2.moveWindow("class", src_img.shape[0]*2, 0)

cv2.setMouseCallback('image', mouse_callback)

pixel_index = 1
pixels_per_loop = 5000

while True:
    if drawing:
        img = src_img.copy()
        tr.draw(img)
    if pixel_index < len(randomized_pixel_order):
        tree.classifyImgRandomSubsample(src_img, class_img, randomized_pixel_order, pixel_index, pixels_per_loop)
        pixel_index += pixels_per_loop
    cv2.imshow('image', img)
    cv2.imshow('class', class_img)
    cv2.imshow("debug", debug_img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        debug_img = tree.train(src_img, tr, randomized_pixel_order)
        class_img.fill(0)
        pixel_index = 0
    elif k == 27:
        break

cv2.destroyAllWindows()