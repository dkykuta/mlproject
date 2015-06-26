# -*- coding: utf-8 -*-

import cv2

def to_binary(former_img, inverted=True):
    img = cv2.adaptiveThreshold(former_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY,101,15)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
