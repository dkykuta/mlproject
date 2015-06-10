# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np

def np_as_image(nparray):
    x = nparray.astype('B')
    if len(nparray.shape) == 3:
        _, h, w = nparray.shape
        onelayer = False
    else:
        h, w = nparray.shape
        onelayer = True
    # desfaz as trocas feitas em open_image_as_np
    # os eixos estão RGB x largura x altura
    # e serao trocados para: largura x altura x RGB
    if onelayer:
        # mesma coisa que la embaixo, sobre tostring / tobytes
        img = Image.frombytes('L', (w,h), x.tostring())
    else:
        x = np.swapaxes(np.swapaxes(x, 1, 2), 0, 2)
        img = Image.fromstring('RGB', (w,h), x.tostring())
    return img

def open_gray_image_as_np(path):
    img = Image.open(path)
    w, h = img.size
    img = img.convert('L')
    shape = (h, w)
    # era img.tostring() mas tava dando msg de deprecation, mudei
    # para img.tobytes()
    # tem que ficar de olho pra ver se nao deu merda
    ret = np.reshape(np.fromstring(img.tobytes(), 'B', w*h), shape)
    return ret

def open_rgb_image_as_np(path):
    img = Image.open(path)
    w, h = img.size
    img = img.convert('RGB')
    shape = (h, w, 3)
    ret = np.reshape(np.fromstring(img.tostring(), 'B', w*h*3), shape)
    # nesse momento, os eixos são: largura x altura x RGB
    # troca os eixos para ficar RGB x largura x altura
    ret = np.swapaxes(np.swapaxes(ret, 0, 2), 1, 2)
    return ret
