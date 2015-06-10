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

if __name__ == "__main__":
    digitsdir = '../digits'
    outdir = '../learned'
    outfile = '%s/digits-train'

    if not os.path.exists(outdir):
        os.mkdirs(outdir)

    if os.path.exists(outfile):
        os.delete(outfile)

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
