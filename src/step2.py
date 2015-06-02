# -*- coding: utf-8 -*-

import utils
import imgfun

def to_binary(former_img):
    return imgfun.autothreshold(former_img)
