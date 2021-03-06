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
