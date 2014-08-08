__author__ = 'jcl5m'
import cv2
import numpy as np
import math
import os
import argparse
import sys
sys.path.append('../common/')
import transformations


def pinhole_from_cylindrical_dewarp(src_img, dst_img, src_center, dst_fov):
    src_h, src_w, src_depth = src_img.shape
    dst_h, dst_w, dst_depth = dst_img.shape

    src_f = float(src_h)/math.pi

    rot = transformations.euler_matrix(src_center[1],src_center[0],0)

    pixel_scaler = math.tan(dst_fov/2.0)*2.0/float(dst_w)

    for r in range(dst_h):
        for c in range(dst_w):

            dx = pixel_scaler*(c-dst_w/2.0)
            dy = pixel_scaler*(r-dst_h/2.0)

            azimuth = math.atan(dx)
            altitude = math.atan(dy/math.sqrt(dx*dx+1))

            y = math.sin(altitude)
            d = math.sqrt(1-y*y)
            x = math.sin(azimuth)*d
            z = math.cos(azimuth)*d
            p = [x, y, z, 1]
            p2 = rot.dot(p)


            altitude = math.asin(p2[1])
            azimuth = math.atan2(p2[0],p2[2])

            src_x = src_f*azimuth + src_img.shape[1]/2
            src_y = src_f*altitude + src_img.shape[0]/2

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


def cartesian_to_cylindrical(src_img,x,y,z):

    src_f = float(src_img.shape[0])/math.pi
    d = math.sqrt(z*z + x*x)
    if d == 0:
        altitude = math.pi/2
    else:
        altitude = math.atan(y/d)
    azimuth = math.atan2(x,z)
    cx = src_f*azimuth + src_img.shape[1]/2
    cy = -src_f*altitude + src_img.shape[0]/2

    return cx,cy

def drawMatches(img1, kp1, img2, kp2, matches, mask):
    color = (0,255,0)
    result_img = np.zeros((img1.shape[0]+img2.shape[0], img1.shape[1], 3), np.uint8)
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


def ExtractAndMatch(img1,img2):
    orb = cv2.ORB(2000, 1.2, 8, 31)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    return drawMatches(img1, kp1, img2, kp2, matches, mask)



file1 = "./data/R0010001.JPG"
file2 = "./data/R0010002.JPG"

print file1
print file2

img1_big = cv2.imread(file1)
img2_big = cv2.imread(file2)

downsample = 4

img1_size = img1_big.shape
img2_size = img1_big.shape
img1_small = cv2.resize(img1_big, (img1_size[1]/downsample, img1_size[0]/downsample))
img2_small = cv2.resize(img2_big, (img2_size[1]/downsample, img2_size[0]/downsample))

if True:
    dewarp_size = 400
    dewarp_img1 = np.zeros((dewarp_size, dewarp_size, 3), np.uint8)
    pinhole_from_cylindrical_dewarp(img1_small, dewarp_img1, (0,0), 2.0)
    dewarp_img2 = np.zeros((dewarp_size, dewarp_size, 3), np.uint8)
    pinhole_from_cylindrical_dewarp(img2_small, dewarp_img2, (-.20,0), 2.0)
    img3 = ExtractAndMatch(dewarp_img1,dewarp_img2)
else:
    img3 = ExtractAndMatch(img1_small, img2_small)

x_min = -1
x_max = 20
z_min = 5
z_max = 25
for pz in range(z_min,z_max):
    for px in range(x_min,x_max):
        x1,y1 = cartesian_to_cylindrical(img1_small,px,5,pz)
        x2,y2 = cartesian_to_cylindrical(img1_small,px+1,5,pz)
        cv2.line(img3,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255))
for px in range(x_min,x_max):
    for pz in range(z_min,z_max):
        x1,y1 = cartesian_to_cylindrical(img1_small,px,5,pz)
        x2,y2 = cartesian_to_cylindrical(img1_small,px,5,pz+1)
        cv2.line(img3,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255))

cv2.imshow("img3", img3)
cv2.moveWindow("img3", 0, 0)
#cv2.imshow("dewarp", dewarp_img1)


cv2.waitKey()
