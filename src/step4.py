# -*- coding: utf-8 -*-

import cv2
import os
import pickle
from sklearn import svm
import numpy as np

digits_trained_classifier = '../learned/digits-train'

def make_features_array(img):
    f = [0] * 55
    mini = cv2.resize(img, (5,5)).reshape((25))
    _, med = cv2.threshold(cv2.resize(img, (15, 15)), 127, 255, cv2.THRESH_BINARY)
    f[0:25] = mini
    for i in xrange(15):
        val = len([x for x in med[i,:] if x != 0])
        f[25+i] = val
    for i in xrange(15):
        val = len([x for x in med[:,i] if x != 0])
        f[40+i] = val
    return f

def recognize_digit(former_img):
    f = make_features_array(former_img)
    if not os.path.exists(digits_trained_classifier):
        return "<nao foi treinado ainda>"
    clf = pickle.load(open(digits_trained_classifier))
    return clf.predict(f)[0]
