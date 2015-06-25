#! \usr\bin\python
# -*- coding: utf-8 -*-

import os
import cv2
import math
import pickle
import shutil
import numpy as np
from sklearn import svm

import utils
import imgfun
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_last_digit
from step4 import make_features_array, recognize_digit
from exampleset import ExampleSet

from training_functions import train_plates, train_digits

def classify(fpath):
    img = cv2.imread(fpath)

    plate = extract_plate(img)

    if plate == None:
        print 'placa nao encontrada nessa foto'
        return

    binary_plate = to_binary(plate)

    last_digit_img = extract_last_digit(binary_plate)

    digit = recognize_digit(last_digit_img)

    print "O ultimo digito dessa placa e %s" % (digit)


if __name__ == "__main__":
    import sys
    print len(sys.argv), sys.argv
    if len(sys.argv) < 2:
        print "What do you want?\nUsage: %s <-td, -tp, path>" % sys.argv[1]
        sys.exit(0)
    else:
        for arg in sys.argv[1:]:
            if arg == '-td':
                print "Training digits (step 4)"
                train_digits()
                sys.exit(0)
            elif arg == '-tp':
                print "Training plates (step 1)"
                train_plates()
                sys.exit(0)

    print "I think you want to recognize the last digit from the plate in this picture '%s', am I correct? [Y/n]" % sys.argv[1]
    classify("../images/img_002.jpg")
    classify("../images/img_004.jpg")