# -*- coding: utf-8 -*-

import numpy as np
import utils
import matplotlib.pyplot
import os

def save_debug_image(img, name):
    debugdir = '../output/debug/'
    if not os.path.exists(debugdir):
        os.makedirs(debugdir)
    a = utils.np_as_image(img)
    a.save('../output/debug/%s.png' % name)

def histogram(imgin):
    hist = [0] * 256
    if len(imgin.shape) == 3:
            _, h, w = imgin.shape
            img = imgin[0]
    else:
        img = imgin
        h, w = img.shape
    for i in xrange(h):
        for j in xrange(w):
            hist[img[i,j]] += 1
    return hist

def choose_limiar(img):
    def mediaH(hist, inf, sup):
        S = [x * hist[x] for x in xrange(inf, sup)]
        d = sum([hist[x] for x in xrange(inf, sup)])
        return float(sum(S)/max(d, 1))

    hist = histogram(img)
    T = 128
    notstable = True
    while notstable:
        m1 = mediaH(hist, 0, T)
        m2 = mediaH(hist, T, 256)

        T2 = int((m1 + m2)/2)
        if T2 == T:
            notstable = False
        T = T2
    return T

def autothreshold(img):
    return threshold(img, choose_limiar(img))

def threshold(img, limiar):
    if len(img.shape) == 3:
            _, w, h = img.shape
    else:
        w, h = img.shape
    ret = np.zeros((w, h))
    for i in xrange(w):
        for j in xrange(h):
            if img[0,i,j] > limiar:
                ret[i,j] = np.uint8(255)
            else:
                ret[i,j] = np.uint8(0)
    return ret