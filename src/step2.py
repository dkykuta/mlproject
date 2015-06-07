# -*- coding: utf-8 -*-

import utils
import imgfun
import cv2

def to_binary(former_img):
    img = cv2.adaptiveThreshold(former_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,101,15)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
#    return imgfun.autothreshold(former_img, invert=True)
