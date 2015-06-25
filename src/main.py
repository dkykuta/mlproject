#! \usr\bin\python
# -*- coding: utf-8 -*-

import os
import cv2
import math
import pickle
import shutil
from sklearn import svm

import utils
import imgfun
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_last_digit
from step4 import make_features_array, recognize_digit
from exampleset import ExampleSet

def classify(fpath):
    img = utils.open_gray_image_as_np(fpath)

    plate = extract_plate(img)
    imgfun.save_debug_image(plate, 'plate')

    binary_plate = to_binary(plate)
    imgfun.save_debug_image(binary_plate, 'Bplate')

    last_digit_img = extract_last_digit(binary_plate)
    imgfun.save_debug_image(last_digit_img, 'lastdigit')

    digit = recognize_digit(last_digit_img)

    print "O ultimo digito dessa placa e %s" % (digit)

def train_digits():
    digitsdir = '../digits'
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
            img = utils.open_gray_image_as_np(os.path.join(currdir, f))
            fullset.add_info(make_features_array(img), "%d" % d)

    # essa porcaria ta muito concentrado
    fullset.shuffle()
    # crossvalidation
    error, gamma, C, kernel = fullset.crossvalidation(10)
    print error, gamma, C, kernel

    # salva o classificador escolhido como melhor
    clf = svm.SVC(kernel = kernel, gamma = gamma, C=C)
    clf.fit(fullset.data, fullset.label)
    clf_file = open(outfile, "w")
    pickle.dump(clf, clf_file)
    clf_file.close()



if __name__ == "__main__":
    import sys
    print len(sys.argv), sys.argv
    if len(sys.argv) < 2:
        print "What do you want?\nUsage: ..."
        sys.exit(0)
    else:
        for arg in sys.argv[1:]:
            if arg == '-td':
                print "Training digits (step 4)"
                train_digits()
                sys.exit(0)
            elif arg == '-tp':
                print "Training plates (step 1)"
                sys.exit(0)

    print "I think you want to recognize the last digit from the plate in this picture '%s', am I correct? [Y/n]" % sys.argv[1]
    classify("../images/img_001.jpg")
    classify("../images/img_001.jpg")
    classify("../images/img_001.jpg")
    classify("../images/img_001.jpg")
