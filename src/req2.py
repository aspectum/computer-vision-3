import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# PRECISA ARRUMAR O MORPHEUSR.TXT (TC_8 -> TC)

window_size = 1
min_disp = 0
num_disp = 768-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 200,
    speckleRange = 32,
)

# 512 para jadeplant
# stereo = cv2.StereoBM_create(
#     numDisparities = 256,
#     blockSize = 15
# )


def main():
    imL_calib = open('../data/FurukawaPonce/MorpheusL.txt', 'r')
    texto = imL_calib.read()

    start_fc = texto.find('fc = ')
    end_fc = texto.find(']', start_fc)
    fc = np.fromstring(texto[start_fc+5:end_fc].replace('[','').replace(';',' '), dtype=float, sep=' ')
    print('fc = ', fc)

    start_cc = texto.find('c = ')
    end_cc = texto.find(']', start_cc)
    fc = np.fromstring(texto[start_cc+5:end_cc].replace('[','').replace(';',' '), dtype=float, sep=' ')
    print('cc = ', cc)



if __name__  == "__main__":
    main()