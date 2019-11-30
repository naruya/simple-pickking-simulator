import cv2
import numpy as np


def layerize(size, pos, src):
    x, y = pos
    h, w = src.shape[:2]
    layer = np.zeros(list(size)+[4]).astype(np.uint8)
    x1, y1 = x - int(h/2), y - int(w/2)
    x2, y2 = x1+h, y1+w
    layer[x1:x2, y1:y2] = src
    return layer

def overlay(dst, src):
    assert src.shape == dst.shape
    assert set(np.unique(src[:,:,3])) <= set([0,255])
    
    mask = cv2.cvtColor(src[:,:,3], cv2.COLOR_GRAY2BGR) / 255.0
    mask = mask.astype(np.uint8)
    dst[:,:,:3] *=  1 - mask
    dst[:,:,:3] += src[:,:,:3] * mask
    dst[:,:,3] = 255
    return dst
