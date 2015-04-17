__author__ = 'johnnylee'


import numpy as np
import cv2

pause = False
width = 640
height = 480
img = np.zeros((width,height,1), np.uint8)
frame = np.zeros((width,height,3), np.uint8)

start_x = 0
start_y = 0

def draw_line(event,x,y,flags,param):
    global pause, img, frame, start_x, start_y

    if event == cv2.EVENT_LBUTTONDOWN:
        pause = True
        start_x = x
        start_y = y
    if event == cv2.EVENT_LBUTTONUP:
        pause = False

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.line(img,(start_x, start_y), (x,y),(255,255,255),2)

    a = np.array([start_x, start_y])
    b = np.array([x, y])
    dist = np.linalg.norm(a-b)*180/640
    cv2.putText(img, str(round(dist,2)) +"deg", (start_x+5, start_y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255))

cap = cv2.VideoCapture(1)

cap.set(3,width)
cap.set(4,height)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_line)

while(True):

    if(not pause):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()