#! \usr\bin\python
# -*- coding: utf-8 -*-
#
#
# Arquivo executavel criado para extrair todos os digitos das placas
# Use esse programa para gerar imagens para treino dos digitos
#
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_and_save_all_digits
import imgfun
import os
import shutil
import cv2
import numpy as np

def extract_all_digits_from_plate(path, outdir):
    plate = np.uint8(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    binary_plate = to_binary(plate)
    extract_and_save_all_digits(binary_plate, outdir)

if __name__ == "__main__":
    outdir = '../digits/test/extracted'
#    platedir = '../images/plates/trainingset'

    platedir = '../images/plates/testset'


    if os.path.exists(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    plates = os.listdir(platedir)
    for f in plates:
        ff = os.path.join(platedir, f)
        fname = f.split('.')[0]
        extract_all_digits_from_plate(ff, os.path.join(outdir, fname))
