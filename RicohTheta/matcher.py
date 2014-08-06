__author__ = 'jcl5m'
import cv2
import numpy as np
import math
import os
import argparse


def pinhole_from_cylindrical_dewarp(src_img, dst_img, src_center, dst_fov):
    src_h, src_w, src_depth = src_img.shape
    dst_h, dst_w, dst_depth = dst_img.shape

    src_f = float(src_h)/math.pi

    print math.tan(dst_fov/2.0)

    q = math.tan(dst_fov/2.0)*2.0*(0-dst_w/2.0)/float(dst_w)
    q2 = math.tan(dst_fov/2.0)*2.0*(200000-dst_w/2.0)/float(dst_w)
    print "FOV", q, math.atan(q)
    print "FOV", q2, math.atan(q2)

    for r in range(dst_h):
        for c in range(dst_w):
            ax = math.atan(math.tan(dst_fov/2.0)*2.0*(c-dst_w/2.0)/float(dst_w))
            ay = math.atan(math.tan(dst_fov/2.0)*2.0*(r-dst_h/2.0)/float(dst_w))

            src_x = src_f*ax+src_center[0]
            src_y = src_f*ay+src_center[1]

            if src_x < 0:
                continue
            if src_y < 0:
                continue
            if src_x >= src_w:
                continue
            if src_y >= src_h:
                continue
            dst_img[r, c, :] = src_img[src_y, src_x, :]
    return (src_h, src_w)


def drawMatches(img1, kp1, img2, kp2, matches, mask):
    color = (0,255,0)
    result_img = np.zeros((img1.shape[1], img1.shape[0] + img2.shape[0], 3), np.uint8)

    result_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    result_img[img1.shape[0]:(img1.shape[0]+img2.shape[0]), 0:img2.shape[1]] = img2

    for m, t in zip(matches, mask):
        if t == 0:
            continue
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        cv2.line(result_img, (int(p1[0]), int(p1[1])), (int(p2[0]), img1.shape[0]+int(p2[1])), color)

        cv2.circle(result_img, (int(p1[0]), int(p1[1])), 3, color)
        cv2.circle(result_img, (int(p2[0]), img1.shape[0]+int(p2[1])), 3, color)
    return result_img


file1 = "./data/R0010001.JPG"
file2 = "./data/R0010002.JPG"

print file1
print file2

img1 = cv2.imread(file1)
img2 = cv2.imread(file2)

downsample = 4

img1_size = img1.shape
img2_size = img1.shape
img1_small = cv2.resize(img1, (img1_size[1]/downsample, img1_size[0]/downsample))
img2_small = cv2.resize(img2, (img2_size[1]/downsample, img2_size[0]/downsample))

orb = cv2.ORB(2000, 1.2, 8, 31)

kp1, des1 = orb.detectAndCompute(img1_small, None)
kp2, des2 = orb.detectAndCompute(img2_small, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
matchesMask = mask.ravel().tolist()

img3 = drawMatches(img1_small, kp1, img2_small, kp2, matches, mask)

dewarp_img = np.zeros((500, 500, 3), np.uint8)

print img2_small.shape[1]/2, img2_small.shape[0]/2
pinhole_from_cylindrical_dewarp(img1_small, dewarp_img, (img1_small.shape[1]/2, img1_small.shape[0]/2), 2.0)

cv2.imshow("img3", img3)
cv2.moveWindow("img3", 0, 0)

cv2.imshow("dewarp", dewarp_img)


cv2.waitKey()
