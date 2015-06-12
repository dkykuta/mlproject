#! \usr\bin\python
# -*- coding: utf-8 -*-
#
#
# Arquivo que treina o reconhecedor de digitos e salva
#
import utils
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_and_save_all_digits
import imgfun
import os
import shutil
import cv2
import math
from sklearn import svm
import pickle
from step4 import make_features_array

class ExampleSet:
    def __init__(self):
        self.data = []
        self.label = []

    def __len__(self):
        return len(self.data)

    def add_info(self, data, label):
        self.data.append(data)
        self.label.append(label)

    def shuffle(self):
        aux = zip(self.data, self.label)
        import random
        random.shuffle(aux)
        self.data = [x for x,y in aux]
        self.label = [y for x,y in aux]

if __name__ == "__main__":
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

    fullset.shuffle()

    N = len(fullset) 
    step = N/10

    best = (2, -1, -1, "")
    # encontra o melhor conjunto de parametros
    for gamma, C in [(g, c) for g in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.000001] for c in [1., 10., 100., 1000.]]:
        acertos = 0
        for i in xrange(11):
            inf, sup = i * step, min((i+1) * step, N)

            traindataset = fullset.data[0:inf] + fullset.data[sup:N]
            trainlabelset = fullset.label[0:inf] + fullset.label[sup:N]

            validationdataset = fullset.data[inf:sup]
            validationlabelset = fullset.label[inf:sup]

            classifier = svm.SVC(gamma=gamma, C=C)
            classifier.fit(traindataset, trainlabelset)

            ans = classifier.predict(validationdataset)
            acertos += len([x for x in (ans == validationlabelset) if x == True])
        error = float(N - acertos) / N

        if error < best[0]:
            best = (error, gamma, C)
        print "[gamma: %5s C: %5s] %s" % (gamma, C, error)

    # salva o classificador escolhido como melhor
    clf = svm.SVC(gamma = best[1], C=best[2])
    clf.fit(fullset.data, fullset.label)
    clf_file = open(outfile, "w")
    pickle.dump(clf, clf_file)
    clf_file.close()
