import cv, cv2
import random

def generate_random_pixel_order(img, region):
    pixel_array = []
    for r in range(region.height()/2,img.shape[0]-region.height()/2):
        for c in range(region.width()/2,img.shape[1]-region.width()/2):
            pixel_array.append([c,r])

    for i in range(0,len(pixel_array)):
        idx = random.randint(0,len(pixel_array)-1)
        temp = pixel_array[i]
        pixel_array[i] = pixel_array[idx]
        pixel_array[idx] = temp
    return pixel_array


def cross_hair(img, p, size, color, thickness = 1):
    p1 = (p[0], p[1] + size)
    p2 = (p[0], p[1] - size)
    cv2.line(img, p1,p2, color, thickness)
    p1 = (p[0] + size, p[1])
    p2 = (p[0] - size, p[1])
    cv2.line(img, p1,p2, color, thickness)
