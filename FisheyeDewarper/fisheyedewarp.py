__author__ = 'jcl5m'

import cv2
import numpy as np
import math
import os
import argparse

def pinhole_dewarp_LUT(src_img, dst_img, src_center, dst_fov, src_f):
    src_h, src_w, src_depth = src_img.shape
    dst_h, dst_w, dst_depth = dst_img.shape

    lut = []

    for r in range(dst_h):
        for c in range(dst_w):
            dst_x = (c-dst_w/2.0)*(math.tan(dst_fov/2.0)/float(dst_w))
            dst_y = (r-dst_h/2.0)*(math.tan(dst_fov/2.0)/float(dst_w))

            theta = math.atan(math.sqrt(dst_x*dst_x + dst_y*dst_y))
            dir_angle = math.atan2(dst_y, dst_x)

            src_x = theta*src_f*math.cos(dir_angle)+src_center[0]
            src_y = theta*src_f*math.sin(dir_angle)+src_center[1]

            if src_x < 0:
                lut.append([0, 0])
                continue
            if src_y < 0:
                lut.append([0, 0])
                continue
            if src_x >= src_w:
                lut.append([0, 0])
                continue
            if src_y >= src_h:
                lut.append([0, 0])
                continue
            lut.append([src_y, src_x])
            dst_img[r, c, :] = src_img[src_y, src_x, :]
    return lut, (src_h, src_w)


def cylindrical_dewarp(src_img, dst_img, src_center, dst_fov, src_f):
    src_h, src_w, src_depth = src_img.shape
    dst_h, dst_w, dst_depth = dst_img.shape
    lut = []

    for r in range(dst_h):
        for c in range(dst_w):
            dst_x = (c-dst_w/2.0)*(dst_fov/float(dst_w))
            dst_y = (r-dst_h/2.0)*(dst_fov/float(dst_w))
#            dst_y = (r-dst_h/2.0)*(math.pi/float(dst_h)) #force pi vertical
            px = math.tan(dst_x)
            py = math.tan(dst_y)*math.sqrt(px*px + 1)

            theta = math.atan(math.sqrt(px*px + py*py))
            dir_angle = math.atan2(py, px)

            src_x = theta*src_f*math.cos(dir_angle)+src_center[0]
            src_y = theta*src_f*math.sin(dir_angle)+src_center[1]
            if src_x < 0:
                lut.append([0, 0])
                continue
            if src_y < 0:
                lut.append([0, 0])
                continue
            if src_x >= src_w:
                lut.append([0, 0])
                continue
            if src_y >= src_h:
                lut.append([0, 0])
                continue
            lut.append([src_y, src_x])
            dst_img[r, c, :] = src_img[src_y, src_x, :]
    return lut, (src_h, src_w)


def warpUsingLUT(src_img, dst_img, lut):
    src_h, src_w, src_depth = src_img.shape
    dst_h, dst_w, dst_depth = dst_img.shape

    index = 0
    for r in range(dst_h):
        for c in range(dst_w):
            address = lut[index]
            index += 1
            dst_img[r, c, :] = src_img[address[0], address[1], :]

parser = argparse.ArgumentParser(description='Dewarp Circular Fisheye Image')
parser.add_argument('directory', help='directory containing circular fisheye images')
parser.add_argument('-postfix', help='post-fix to add to dewarped image filenames')
parser.add_argument('-dns', help='do not save', action="store_true")
parser.add_argument('-dewarp', help='type of dewarp: pinhole (default) or cylindrical')
parser.add_argument('-dst_w', help='width of the destination image')
parser.add_argument('-dst_h', help='height of the destination image')
parser.add_argument('-dst_fov', help='horizontal FOV of the destination image')
parser.add_argument('-src_f', help='source image focal number (default:970)')
parser.add_argument('-src_cx', help='source image center x in percent (default:0.5)')
parser.add_argument('-src_cy', help='source image center y in percent (default:0.5)')

args = parser.parse_args()

image_path = args.directory

#default values
src_f = 970 #hand tuned for sigma circular fisheye
dst_width = 1024
dst_height = 1024
dst_horz_fov = 2.6
src_center = [0.5, 0.5] #hand selected to be image center, needs to be calibrated
save_ouput = True
postfix = "_dewarp"
dewarp_type = 'pinhole'

if args.dst_w != None:
    dst_width = int(args.dst_w)
    dst_height = dst_width
if args.dst_h != None:
    dst_height = int(args.dst_h)
if args.dst_fov != None:
    dst_horz_fov = float(args.dst_fov)
if args.src_f != None:
    src_f = float(args.src_f)
if args.src_cx != None:
    src_center[0] = float(args.src_cx)
if args.src_cy != None:
    src_center[1] = float(args.src_cy)
if args.postfix != None:
    postfix = args.postfix
if args.dewarp != None:
    dewarp_type = args.dewarp
if args.dns:
    save_ouput = False


print "Loading images from:" + image_path
print  "output image:" + str(dst_width) + ", " + str(dst_height)
print  "output horz fov: " + str(dst_horz_fov)
print  "source center: " + str(src_center)
print  "source f: " + str(src_f)
print  "save output: " + str(save_ouput)
print  "postfix: " + postfix
print  "dewarp: " + dewarp_type

lut = 0
lut_dim = 0
dst_img = np.zeros((dst_height, dst_width, 3), np.uint8)

#for dirname, dirnames, filenames in os.walk(image_path):
for dirname, dirnames, filenames in os.walk(image_path):
    # print path to all filenames.
    for filename in filenames:
        file = os.path.join(dirname, filename)
        print file

        img = cv2.imread(file)
        center = [img.shape[1]*src_center[1], img.shape[0]*src_center[1]]

        dst_img.fill(0)

        if (img.shape[:2] != lut_dim):
            if(dewarp_type == 'pinhole'):
                lut, lut_dim = pinhole_dewarp_LUT(img, dst_img, center, dst_horz_fov, src_f)
            if(dewarp_type == 'cylindrical'):
                lut, lut_dim = cylindrical_dewarp(img, dst_img, center, dst_horz_fov, src_f)
        else:
            warpUsingLUT(img, dst_img, lut)

        #cv2.imshow('image', img)
        cv2.imshow('dest', dst_img)

        if save_ouput:
            output_file = file.replace(".JPG", postfix+".JPG")
            print "Saving: " + output_file
            cv2.imwrite(output_file, dst_img)
        key = cv2.waitKey(30)
        if key == 27:
            break

cv2.destroyAllWindows()