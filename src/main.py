#! \usr\bin\python
# -*- coding: utf-8 -*-

import utils
from step1 import extract_plate
from step2 import to_binary
from step3 import extract_last_digit
from step4 import recognize_digit
import imgfun

if __name__ == "__main__":
    img = utils.open_image_as_np("../images/img_001.jpg")

    plate = extract_plate(img)
    print plate.shape

    imgfun.save_debug_image(plate, 'plate')

    binary_plate = to_binary(plate)
    imgfun.save_debug_image(binary_plate, 'Bplate')

    last_digit_img = extract_last_digit(binary_plate)
    digit = recognize_digit(last_digit_img)

    print "O ultimo digito dessa placa e %s" % (digit)
