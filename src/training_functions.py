#! \usr\bin\python
# -*- coding: utf-8 -*-

import os
import cv2
import math
import pickle
import shutil
import numpy as np
from sklearn import svm

from step1 import extract_plate
from step2 import to_binary
from step3 import extract_last_digit
from step4 import make_features_array, recognize_digit
from exampleset import ExampleSet

def make_features_plate_training(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(17,17),0)
    img = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
    img = np.absolute(img)
    img = np.uint8(img)
    img = cv2.resize(img, (300, 100), interpolation = cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
        #show_cv2(img,'')
    img = np.array(img, dtype='double')
    return img.flatten()

def train_plates():
    plates_dir = '../training'
    outdir = '../learned'
    outfile = '%s/plates-train' % outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if os.path.exists(outfile):
        os.remove(outfile)

    fullset = ExampleSet()

    for dirname, y in [('plate', 1), ('not_plate', -1)]:
        currdir = os.path.join(plates_dir, dirname)
        files = os.listdir(currdir)
        for f in files:
            img = cv2.imread(os.path.join(currdir, f))
            fullset.add_info(make_features_plate_training(img), y)

    # essa porcaria ta muito concentrado
    fullset.shuffle()
    # crossvalidation
    error, gamma, C, kernel = fullset.crossvalidation(10)
    print "Parametros escolhidos: (kernel = %s, gamma = %s, C = %s)" % (kernel, gamma, C)

    # salva o classificador escolhido como melhor
    clf = svm.SVC(kernel = kernel, gamma = gamma, C=C)
    clf.fit(fullset.data, fullset.label)
    clf_file = open(outfile, "w")
    pickle.dump(clf, clf_file)
    clf_file.close()
    print clf.n_support_

def train_digits():
    digitsdir = '../training/digits'
    outdir = '../learned'
    outfile = '%s/digits-train' % outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if os.path.exists(outfile):
        os.remove(outfile)

    fullset = ExampleSet()

    for d in xrange(0,10):
        currdir = os.path.join(digitsdir, "%d" % d)
        files = os.listdir(currdir)
        for f in files:
            # img = utils.open_gray_image_as_np(os.path.join(currdir, f))
            img = np.uint8(cv2.imread(os.path.join(currdir, f), cv2.IMREAD_GRAYSCALE))
            fullset.add_info(make_features_array(img), "%d" % d)

    # essa porcaria ta muito concentrado
    fullset.shuffle()
    # crossvalidation
    error, gamma, C, kernel = fullset.crossvalidation(10)
    print "Parametros escolhidos: (kernel = %s, gamma = %s, C = %s)" % (kernel, gamma, C)

    # salva o classificador escolhido como melhor
    clf = svm.SVC(kernel = kernel, gamma = gamma, C=C)
    clf.fit(fullset.data, fullset.label)
    clf_file = open(outfile, "w")
    pickle.dump(clf, clf_file)
    clf_file.close()
    print clf.n_support_
