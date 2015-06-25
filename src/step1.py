# -*- coding: utf-8 -*-

import utils
import sys
import cv2
import os
import pickle
import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt

def show_plt(rgb_img):
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([]) 
    plt.show()

def show_cv2(bgr_img, name='img'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,bgr_img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def classify(img, clf):
    img = cv2.resize(img, (300, 100), interpolation = cv2.INTER_CUBIC)
    img = cv2.equalizeHist(img)
    img = np.array(img, dtype='double')
    img = img.flatten()
    
    return int(clf.predict(img)[0])

def extract_plate(bgr_img):
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
    
    show_cv2(thresh, 'Sobel x filter')
    #cv2.imwrite('thresh.jpg', thresh)
    
    #faz um fechamento (dilatacao seguida de erosao ) para unir os elementos da placa
    closing_kernel =  cv2.getStructuringElement(cv2.MORPH_RECT,(23,4))
    closing_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closing_kernel)
    
    show_cv2(closing_img, 'closing')
    #cv2.imwrite('c.jpg', closing_img)
    
    #encontra os contornos dos elementos brancos da imagem
    img = closing_img
    contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #descarta os contornos muito pequenos
    cnt = [c for c in contours if 20000 <= cv2.contourArea(c) < 70000]
    
    #desenha os retangulos retos que cercam os contornos encontrados e salva as subimagens

    plates_trained_classifier = '../learned/plates-train'
    if not os.path.exists(plates_trained_classifier):
        return "<nao foi treinado ainda>"
    clf = pickle.load(open(plates_trained_classifier))

    plates = []
    for c in cnt:
        x,y,w,h = cv2.boundingRect(c)
        ratio = w/float(h)
        #print ratio
        if 2.5 <= ratio <= 5.0:
            crop = cv2.getRectSubPix(sobelx_8u, (w, h), (x+w/2.0, y+h/2.0))
            if classify(crop,clf) == 1:
                cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(0,255,0),2)
                crop = cv2.getRectSubPix(bgr_img, (w, h), (x+w/2.0, y+h/2.0))
                cr = crop.shape[1] / float(crop.shape[0])
                crop =cv2.resize(crop, (int(100*cr), 100), interpolation = cv2.INTER_CUBIC)
                plates.append(crop)
            else:
                cv2.rectangle(bgr_img,(x,y),(x+w,y+h),(0,25,251),2)
    
    #desenha os retangulos rotacionados que cercam os contornos encontrados
    #for c in cnt:
    #    rect = cv2.minAreaRect(c)
    #    _,_,angle = rect
    #    
    #    if(angle >= -90.0 and angle <= -80.0) or (angle >= -10.0 and angle <= -0.0):
    #        box = cv2.cv.BoxPoints(rect)
    #        box = np.int0(box)
    #        #print box
    #        #print rect
    #        xy = [list(p) for p in zip(*box)]
    #        w = max(xy[0]) - min(xy[0])
    #        h = max(xy[1]) - min(xy[1])
    #        ratio = w/float(h)
    #        if 2.5 <= ratio <= 3.5:
    #            #print ratio
    #            cv2.drawContours(bgr_img, [box], 0, (255,0,255),4)
        
    show_cv2(bgr_img, 'detected regions')
    #cv2.imwrite('possible.jpg', bgr_img)
    for i in plates:
        show_cv2(i, 'detected plates')
        return np.uint8(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY))