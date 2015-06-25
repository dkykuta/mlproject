#! \usr\bin\python
# -*- coding: utf-8 -*-
#
#

from sklearn import svm

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

    def crossvalidation(self, fold=10):
        N = len(self) 
        step = N/fold

        best = (2, -1, -1, "rbf")
        # encontra o melhor conjunto de parametros
        for kernel, gamma, C in [(k, g, c) for k in ["linear", "rbf"] for g in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.000001] for c in [1., 10., 100., 1000.]]:
            acertos = 0
            for i in xrange(fold+1):
                inf, sup = i * step, min((i+1) * step, N)

                traindataset = self.data[0:inf] + self.data[sup:N]
                trainlabelset = self.label[0:inf] + self.label[sup:N]

                validationdataset = self.data[inf:sup]
                validationlabelset = self.label[inf:sup]

                classifier = svm.SVC(kernel=kernel, gamma=gamma, C=C)
                classifier.fit(traindataset, trainlabelset)

                ans = classifier.predict(validationdataset)
                acertos += len([x for x in (ans == validationlabelset) if x == True])
            error = float(N - acertos) / N

            if error < best[0]:
                best = (error, gamma, C, kernel)
            print "[kernel: %10s [gamma %6s, C %6s]] error: %s" % (kernel, gamma, C, error)

        return best
