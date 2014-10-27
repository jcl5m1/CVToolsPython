import cv
import cv2
import numpy as np
from numpy import *
import math
from matplotlib import pyplot as plt
import sys
from scipy.optimize import leastsq
from transformations import pose_matrix

pause = False
        

#generate camera params and grid points ----------------------------------------
params = [320,240,130]
points = []
for c in range(11):
  for r in range(4):
    points.append([c,r*2+c%2])
  
def projectPoint(pt,params):
  x = params[0] + pt[0]*params[2]
  y = params[1] + pt[1]*params[2]
  return [x,y]

def unprojectPoint(pt,params):
  x = pt[0] - params[0]
  y = pt[1] - params[1]
  r = math.sqrt(x*x + y*y)
  if r > sys.float_info.epsilon:
    d = math.tan(r/params[2])/r
    x *= d
    y *= d
  return [x,y]

def projectPoints(pts,params):
  result = []
  for pt in pts:
    result.append(projectPoint(pt,params))
  return result

def unprojectPoints(pts,params):
  result = []
  for pt in pts:
    result.append(unprojectPoint(pt,params))
  return result

def plotUnprojectedPoints(pts, scale):
  result = []
  for pt in pts:
    result.append([pt[0]*scale + 320,pt[1]*scale+ 240])
  return result
  

pose = pose_matrix([1,2,3,0.4,0.5,0.8])
tpt = np.mat([1,0,0,1]).reshape((4,1))
print pose*tpt


#reprojection error ----------------------------------------
def residuals(p, y, x):
  err = np.linalg.norm(np.subtract(y,projectPoints(x,p)),axis=1)
  return err

#video playback ----------------------------------------
cap = cv2.VideoCapture( 'MVI_5743.MOV' )
waitPerFrameInMillisec = 16

nFrames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

def drawCrossHair(img, x, y, size):
  cv2.line(img,(x+size, y), (x-size, y), (0,0,255))
  cv2.line(img,(x, y+size), (x, y-size), (0,0,255))

while True:

  k = cv2.waitKey(5)
  if k == 32:
    pause = ~pause
  if k == 27:
    break

  if pause:
    continue

  ret ,img = cap.load()
  if ret == False:
      break
  
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  ret,thresh = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  
  observed = []
  for cnt in contours:
    area = cv2.contourArea(cnt)
    M = cv2.moments(cnt)
    if area < 20:
      continue
    if M['m00'] < 0.01:
      continue
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    observed.append((cx, cy))
        
  for o in observed:
    drawCrossHair(img, o[0], o[1], 4)

  if False: #do least squares
    params_lsq = leastsq(residuals, params, args=(observed, points[:len(observed)]))
    pts = projectPoints(points, params_lsq[0])
  
  pts = plotUnprojectedPoints(unprojectPoints(observed, params),50)
#  else:
#    pts = projectPoints(points, params)
  
  img *= 0
  for pt in pts:
    cv2.circle(img,(int(pt[0]), int(pt[1])),3,(0,255,0))
    
  cv2.imshow('image',img)

  if cap.get(cv.CV_CAP_PROP_POS_FRAMES) > (nFrames-2):
    cap.set(cv.CV_CAP_PROP_POS_FRAMES,0)
