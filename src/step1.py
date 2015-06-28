# -*- coding: utf-8 -*-

import sys
import cv2
import os
import pickle
import numpy as np
from sklearn import svm
from operator import itemgetter
from matplotlib import pyplot as plt


plates_trained_classifier = '../learned/plates-train'

def cv2_show(img, name='img'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def classify(img, clf):
    img = cv2.resize(img, (300, 100), interpolation = cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    img = np.array(img, dtype='double')
    img = img.flatten()
    
    return int(clf.predict(img)[0])

def extract_plate(bgr_img, fpath = '_'):
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])

    
    #conversao da imagem em tons de cinza
    gray = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
    
    #filtro gaussiano para remover ruidos da camera
    blur = cv2.GaussianBlur(gray,(37,37),0)
    
    #sobelx : derivada na direcao horizontal para detectar contornos verticais (muitos na placa)
    laplacian = cv2.Laplacian(blur,cv2.CV_64F)
    sobelx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=-1)
    sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=-1)
    
    #converte np.float64 para np.uint8 para recuperar ambas derivadas (branco -> preto e preto -> branco)
    abs_sobelx = np.absolute(sobelx)
    sobelx_8u = np.uint8(abs_sobelx)
    
    #threshold da imagem com o operador de Sobel utilizando o algoritmo de Otsu
    ret, thresh = cv2.threshold(sobelx_8u,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # show_cv2(thresh, 'Sobel x filter')
    #cv2.imwrite('thresh.jpg', thresh)
    
    #faz um fechamento (dilatacao seguida de erosao ) para unir os elementos da placa
    closing_kernel =  cv2.getStructuringElement(cv2.MORPH_RECT,(25, 4))
    closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel)

    #encontra os contornos dos elementos brancos da imagem
    contours, _ = cv2.findContours(closing_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    #descarta os contornos muito pequenos
    cnt = [c for c in contours if 9000 <= cv2.contourArea(c) < 25000]

    if not os.path.exists(plates_trained_classifier):
        return "<nao foi treinado ainda>"
    clf = pickle.load(open(plates_trained_classifier))

    plates = []

    for c in cnt:
        x,y,w,h = cv2.boundingRect(c)
#        cv2.rectangle(blabla,(x,y),(x+w,y+h),(255,0,255),5)
        ratio = w/float(h)
        #print ratio
        if 2.5 <= ratio <= 5.0:
            crop = cv2.getRectSubPix(sobelx_8u, (w, h), (x+w/2.0, y+h/2.0))

            if classify(crop,clf) == 1:
                cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(0,255,0),2)
                crop = cv2.getRectSubPix(bgr_img, (w, h), (x+w/2.0, y+h/2.0))
                cr = crop.shape[1] / float(crop.shape[0])
                crop =cv2.resize(crop, (int(100*cr), 100), interpolation = cv2.INTER_CUBIC)
                plates.append((x, crop)) # vamos usar o x depois para ordenar a lista de placas
            else:
                cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(0,25,251),2)


    # ordena a lista de placas da esquerda para a direita
    plates = [p for x, p in sorted(plates, key=itemgetter(0))]
    return [np.uint8(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)) for plate in plates]
