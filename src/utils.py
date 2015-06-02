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
        img = Image.fromstring('L', (w,h), x.tostring())
    else:
        x = np.swapaxes(np.swapaxes(x, 1, 2), 0, 2)
        img = Image.fromstring('RGB', (w,h), x.tostring())
    return img

def open_image_as_np(path):
    img = Image.open(path)
    w, h = img.size
    img = img.convert('RGB')
    shape = (h, w, 3)
    ret = np.reshape(np.fromstring(img.tostring(), 'B', w*h*3), shape)
    # nesse momento, os eixos são: largura x altura x RGB
    # troca os eixos para ficar RGB x largura x altura
    ret = np.swapaxes(np.swapaxes(ret, 0, 2), 1, 2)
    return ret
