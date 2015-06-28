#! \usr\bin\python
# -*- coding: utf-8 -*-

import os
import cv2
import math
import pickle
import shutil
import numpy as np
from sklearn import svm

from step1 import extract_plate, plates_trained_classifier
from step2 import to_binary
from step3 import extract_last_digit
from step4 import make_features_array, recognize_digit, digits_trained_classifier
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

def test_plates():
    plates_dir = '../testing'
    clf = pickle.load(open(plates_trained_classifier))

    correct = 0
    total = 0

    for dirname, y in [('plate', 1), ('not_plate', -1)]:
        currdir = os.path.join(plates_dir, dirname)
        files = os.listdir(currdir)
        for f in files:
            img = cv2.imread(os.path.join(currdir, f))
            #TRAINING: fullset.add_info(make_features_plate_training(img), y)
            resp = clf.predict(make_features_array(img))
            total += 1
            correct += (1 if (d == int(resp[0])) else 0)

    print clf.n_support_
    print "%d/%d" % (correct, total)


def test_digits():
    digitsdir = '../testing/digits'
    clf = pickle.load(open(digits_trained_classifier))

    correct = 0
    total = 0

    for d in xrange(0,10):
        currdir = os.path.join(digitsdir, "%d" % d)
        files = os.listdir(currdir)
        for f in files:
            img = np.uint8(cv2.imread(os.path.join(currdir, f), cv2.IMREAD_GRAYSCALE))
            #TRAINING: fullset.add_info(make_features_array(img), "%d" % d)
            resp = clf.predict(make_features_array(img))
            total += 1
            correct += (1 if (d == int(resp[0])) else 0)

    print clf.n_support_
    print "%d/%d" % (correct, total)
