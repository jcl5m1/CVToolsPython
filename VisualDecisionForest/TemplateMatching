__author__ = 'johnny'

import numpy as np
import cv, cv2
import datetime
import time
import glob
import os
import subprocess
import sys
print sys.path

from TrainingRegion import *
from ImageUtilities import *


dragging = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

curr_training_region = TrainingRegion()
match_tolerance = 0.80
curr_template = []

pause = False

# mouse callback function
def mouse_callback(event,x,y,flags,param):
    global curr_training_region,ix,iy,dragging,mode, src_img, img, randomized_pixel_order, match_tolerance, curr_template

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        curr_training_region.set((x, y),(x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            curr_training_region.set_p2((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        curr_training_region.set_p2((x, y))
        curr_training_region.unflip_corners()
        if((curr_training_region.width() > 0) and (curr_training_region.height() > 0)):
            print curr_training_region.p1, curr_training_region.p2
            curr_template = create_template_from_region(curr_training_region, src_img)
            cv2.imshow("template", curr_template[0])
            print "template score: " + str(curr_template[1])
            template_match(curr_template, src_img, match_tolerance, img)

def create_template_from_region(tr, img):
    template = img[tr.p1[1]:tr.p2[1],tr.p1[0]:tr.p2[0]]
    result = cv2.matchTemplate(src_img, template,cv2.TM_CCOEFF)
    score = result[tr.p1[1], tr.p1[0]]
    return [template, score]


def template_match(template, img,tolerance, debug_img):
    result = cv2.matchTemplate(img, template[0],cv2.TM_CCOEFF)
    thresh = tolerance*template[1]
    thresh_result = cv2.threshold(result, thresh, 1.0, cv2.THRESH_TOZERO)[1]
#    cv2.imshow("threshold match", thresh_result)
    matches = []
    for r in range(0, thresh_result.shape[0]):
        for c in range(0, thresh_result.shape[1]):
            v = thresh_result[r,c]
            if v > 0.5:
                x = c+ template[0].shape[1]/2
                y = r+template[0].shape[0]/2
                matches.append([x,y])
                cross_hair(debug_img,(x, y),20,(0,0,255),2)
                cv2.circle(debug_img, (x,y),15,(0,0,255),2)
    return matches


def save_template(template):
    img_fname = directory_prefix + datetime.datetime.fromtimestamp(time.time()).strftime('templates/img_%d%h%Y_%H%M%S.png')
    print "saving template: " + img_fname
    cv2.imwrite(img_fname, template[0])
    score_fname = img_fname.replace("img_", "score_").replace(".png", ".txt")
    f = open(score_fname, 'w')
    f.write(img_fname + "\n" + str(template[1]) + "\n")
    f.close()

def load_templates(directory):
    templates = []

    for file in glob.glob(directory + "templates/*.txt"):
        f = open(file, 'r')
        img_fname = f.readline()[:-1]
        img = cv2.imread(img_fname)
        score = float(f.readline())
        templates.append([img, score])
    return templates

def grab_screenshot_from_device():
    print "Capturing Screenshot..."
#    subprocess.Popen('./../AndroidScripts/grab_screenshot.sh', shell=True)
    os.system('./../AndroidScripts/grab_screenshot.sh')
    print "done"
    return cv2.imread('screenshot.png')

def tap_device(x,y):
    cmd = 'adb shell input touchscreen tap ' + str(x) + ' ' + str(y)
    print cmd
    os.system(cmd)

cv2.namedWindow("image", cv2.CV_WINDOW_AUTOSIZE)

#directory_prefix = "..//data//2048//"
directory_prefix = "../data/ClashOfClans/"
downsize = 2

cv.MoveWindow("image", 0, 0)
cv2.setMouseCallback('image', mouse_callback)
targets = load_templates(directory_prefix)

counter = 0

#training--------------------------
src_img = cv2.imread(directory_prefix + 'screenshots/screenshot.png')
src_img = cv2.resize(src_img, (0,0), fx=1.0/downsize, fy=1.0/downsize)
img = src_img.copy()

while True:
    #live image ----------------
    do_live = True
    if pause == False:
        if do_live:
            src_img = grab_screenshot_from_device()
        src_img = cv2.resize(src_img, (0,0), fx=1.0/downsize, fy=1.0/downsize)
        img = src_img.copy()

        #check all template types
        for target_id in range(0,len(targets)):
            matches = template_match(targets[target_id], src_img,match_tolerance, img)
            for m in matches:
                tap_device(m[1]*downsize, 1080-m[0]*downsize) #clans puts 0,0, in upper right with x downwards
                cv2.circle(img,(m[0], m[1]),20,(0,255,255),5)
                break #only do one per type

    counter = (counter + 1) % 500

    if dragging:
        img = src_img.copy()
        curr_training_region.draw(img)
    cv2.imshow('image', img)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('s'):
        save_template(curr_template)
    if k == ord(' '):
        if pause:
            pause = False
        else:
            pause = True
        print 'Pause: '+ str(pause)
    if k == 27:
        break

cv2.destroyAllWindows()