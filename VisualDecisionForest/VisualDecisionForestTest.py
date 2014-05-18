__author__ = 'johnny'

import numpy as np
import cv, cv2
from itertools import *
from random import *
import datetime
import time
from VisualDecisionForest import *
import time

dragging = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

curr_training_region = TrainingRegion()
template = []

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    global curr_training_region,ix,iy,dragging,mode, src_img, img, randomized_pixel_order

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        tr.set((x, y),(x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            tr.set_p2((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        tr.set_p2((x, y))
        print tr.p1, tr.p2
        randomized_pixel_order = generate_random_pixel_order(src_img, tr)
        reset_classifier()
        template_match()


def template_match():
    global  curr_training_region, src_img, template
    template = src_img[tr.p1[1]:tr.p2[1],tr.p1[0]:tr.p2[0]]
    cv2.imshow("template", template)
    result = cv2.matchTemplate(src_img, template,cv2.TM_CCOEFF_NORMED)
    print tr.center[0], tr.center[1]
    print result.shape
    print result[tr.center[1], tr.center[0]]
    result2 = cv2.threshold(result, 0.7, 1.0, cv2.THRESH_TOZERO)[1]
    cv2.imshow("result", result)
    cv2.imshow("result2", result2)
    for r in range(0, result2.shape[0]):
        for c in range(0, result2.shape[1]):
            v = result2[r,c]
            if v > 0.5:
                print v, c, r


def reset_classifier():
    global  debug_img, class_img, pixel_index
    debug_img = tree.train(src_img, curr_training_region, randomized_pixel_order)
    class_img.fill(0)
    pixel_index = 0

cv2.namedWindow("image", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("debug", cv2.CV_WINDOW_AUTOSIZE)
cv2.namedWindow("class", cv2.CV_WINDOW_AUTOSIZE)

#directory_prefix = "..//data//2048//"
directory_prefix = "..//data//ClashOfClans//"
downsize = 4

src_img = cv2.imread(directory_prefix + 'screenshots//screenshot.png')
src_img = cv2.resize(src_img, (0,0), fx=1.0/downsize, fy=1.0/downsize)

img = src_img.copy()



cv.MoveWindow("image", 0, 0)
cv.MoveWindow("debug", src_img.shape[1], 0)
cv.MoveWindow("class", src_img.shape[1]*2, 0)

cv2.setMouseCallback('image', mouse_callback)

pixel_index = 1
pixels_per_loop = 5000


curr_training_region.set((107, 263),(123, 281))
print curr_training_region.center
template_match()

randomized_pixel_order = generate_random_pixel_order(src_img, curr_training_region)
class_img = np.zeros((img.shape[0],img.shape[1],1), np.uint8)
tree = VisualDecisionTree()
debug_img = tree.train(src_img, curr_training_region, randomized_pixel_order)


while True:
    if dragging:
        img = src_img.copy()
        curr_training_region.draw(img)
    if pixel_index < len(randomized_pixel_order):
        tree.classifyImgRandomSubsample(src_img, class_img, randomized_pixel_order, pixel_index, pixels_per_loop)
        pixel_index += pixels_per_loop
    cv2.imshow('image', img)
    cv2.imshow('class', class_img)
    cv2.imshow("debug", debug_img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):
        reset_classifier()
    if k == ord('s'):
        name = directory_prefix + datetime.datetime.fromtimestamp(time.time()).strftime('templates//template_%d%h%Y_%H%M%S.png')
        print name
        cv2.imwrite(name, template)
    if k == 27:
        break

cv2.destroyAllWindows()