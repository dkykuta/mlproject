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


def extract_all_digits_from_plate(path):
    plate = utils.open_gray_image_as_np(path)
    binary_plate = to_binary(plate)
    imgfun.save_debug_image(binary_plate, "Bplate")
    extract_and_save_all_digits(binary_plate)

if __name__ == "__main__":
    extract_all_digits_from_plate("../mockimg/img001_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img002_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img003_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img004_s1.jpg")
#    extract_all_digits_from_plate("../mockimg/img005_s1.jpg")
