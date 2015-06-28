# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot
import os
import cv2

def save_extracted_digit(img, outdir='../output/extract'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    exists = True
    i = 1
    while exists:
        path = '%s/exd_%03d.png' % (outdir, i)
        exists = os.path.exists(path)
        i += 1
    #a.save(path)
    cv2.imwrite(path, img)
