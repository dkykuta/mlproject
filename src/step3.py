# -*- coding: utf-8 -*-

import operator
import cv2
import imgfun
import numpy as np


def extract_last_digit(former_img):
#    return utils.open_image_as_np('../mockimg/img001_s3.jpg')
    h, w = former_img.shape
    img = np.copy(former_img)

    # percorre a imagem da direita para a esquerda, de cima para baixo
    # a procura de elementos brancos, que indicam um objeto possivelmente
    # interessante. Se suas dimensoes forem plausiveis, entao eh tomado
    # como o ultimo digito
    f = 255
    for j in xrange(w-1, -1, -1):
        for i in xrange(h-1, -1, -1):
            # i é a linha, j é a coluna
            if img[i,j] == 255:
                min_i, max_i, min_j, max_j = bla(img, i, j)
                if 0.35*h <= max_i - min_i <= 0.9 * h and 0.06 * h <= max_j - min_j <= 0.5 * h:
                    return put_in_a_box(former_img[min_i:max_i,min_j:max_j])
    return None

def bla(img, i, j):
    h,w = img.shape
    f = []
    f.append((i, j))
    min_i = i
    max_i = i
    min_j = j
    max_j = j
    while f:
        p = f.pop()
        for viz in ((0,1), (0,-1), (1, 0), (-1, 0)):
            i_, j_ = tuple(map(operator.add, p, viz))
            # se (dentro da imagem) e (branco)
            if (0 <= i_ < h and 0 <= j_ < w) and (img[i_, j_] == 255):
                f.append((i_, j_))
                img[i_, j_] = 1
                if i_ < min_i:
                    min_i = i_
                elif i_ > max_i:
                    max_i = i_

                if j_ < min_j:
                    min_j = j_
                elif j_ > max_j:
                    max_j = j_

    return (min_i, max_i, min_j, max_j)

def put_in_a_box(digit):
    h,w = digit.shape
    side = max(h,w)
    ret = np.zeros((side, side), dtype=np.uint8)

    starting_h = (side - h)/2
    starting_w = (side - w)/2

    ret[starting_h:starting_h+h, starting_w:starting_w+w] = digit
    
    return cv2.resize(ret, (40,40))

def extract_and_save_all_digits(former_img, outdir='../digits/extracted'):
    h, w = former_img.shape
    img = np.copy(former_img)

    # percorre a imagem da direita para a esquerda, de cima para baixo
    # a procura de elementos brancos, que indicam um objeto possivelmente
    # interessante. Se suas dimensoes forem plausiveis, entao eh tomado
    # como o ultimo digito
    f = 255
    for j in xrange(w-1, -1, -1):
        for i in xrange(h-1, -1, -1):
            # i é a linha, j é a coluna
            if img[i,j] == 255:
                min_i, max_i, min_j, max_j = bla(img, i, j)
                if 0.4*h < max_i - min_i < 0.75 * h and 0.06 * h < max_j - min_j < 0.5 * h:
                    imgfun.save_extracted_digit(put_in_a_box(former_img[min_i:max_i,min_j:max_j]), outdir)
    return None


