#! \usr\bin\python
# -*- coding: utf-8 -*-
#
#
# Arquivo executavel criado para extrair todos os digitos das placas
# Use esse programa para gerar imagens para treino dos digitos
#
import utils
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_and_save_all_digits
import imgfun
import os
import shutil

def extract_all_digits_from_plate(path, outdir):
    plate = utils.open_gray_image_as_np(path)
    binary_plate = to_binary(plate)
    extract_and_save_all_digits(binary_plate, outdir)

if __name__ == "__main__":
    outdir = '../digits/extracted'
    platedir = '../plates/trainingset'


    if os.path.exists(outdir):
        shutil.rmtree(outdir, ignore_errors=True)

    plates = os.listdir(platedir)
    for f in plates:
        ff = os.path.join(platedir, f)
        fname = f.split('.')[0]
        extract_all_digits_from_plate(ff, os.path.join(outdir, fname))

#    extract_all_digits_from_plate("../mockimg/img001_s1.jpg", outdir)
#    extract_all_digits_from_plate("../mockimg/img002_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img003_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img004_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img005_s1.jpg")
