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

def save_extracted_digit(img, outdir='../output/extract'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    a = utils.np_as_image(img)
    exists = True
    i = 1
    while exists:
        path = '%s/exd_%03d.png' % (outdir, i)
        exists = os.path.exists(path)
        i += 1
    a.save(path)

def histogram(imgin):
    """ supoe imgin sendo uma imagem em escala de cinza """
    hist = [0] * 256
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

def autothreshold(img, invert=False):
    return threshold(img, choose_limiar(img), invert)

def threshold(img, limiar, invert=False):
    above = np.uint8(0) if invert else np.uint8(255)
    below = np.uint8(255) if invert else np.uint8(0)

    w, h = img.shape
    ret = np.zeros((w, h))
    for i in xrange(w):
        for j in xrange(h):
            if img[i,j] > limiar:
                ret[i,j] = above
            else:
                ret[i,j] = below
    return ret
