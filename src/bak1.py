import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


# MOTO
# window_size = 5
# min_disp = 0
# num_disp = 256-min_disp
# stereo = cv2.StereoSGBM_create(
#     minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = 9,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
#     disp12MaxDiff = 1,
#     uniquenessRatio = 10,
#     speckleWindowSize = 200,
#     speckleRange = 2,
# )

# JADEPLANT
# window_size = 1
# min_disp = 0
# num_disp = 768-min_disp
# stereo = cv2.StereoSGBM_create(
#     minDisparity = min_disp,
#     numDisparities = num_disp,
#     blockSize = 5,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
#     disp12MaxDiff = 1,
#     uniquenessRatio = 10,
#     speckleWindowSize = 200,
#     speckleRange = 2,
# )


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
    im0 = cv2.imread('../data/jadeplant/im0.png', cv2.CV_8UC1)
    im1 = cv2.imread('../data/jadeplant/im1.png', cv2.CV_8UC1)

    disparity = stereo.compute(im0, im1).astype(np.float32)/16.0
    # Isso é o que o req1 quer
    # cada posição de disparity é a distância entre os pixels das 2 imagens
    # os valores -1 eu suponho que sejam os que não tem correspondência
    
    # cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    # cv2.imshow('disparity', disparity)
    # cv2.resizeWindow('disparity', 1000,1000)

    # cv2.waitKey()

    im = plt.imshow(disparity, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

    calib_params = open('../data/jadeplant/calib.txt', 'r')
    texto = calib_params.read()
    start_pos_baseline = texto.find('baseline')
    end_pos_baseline = texto.find('width')
    baseline = float(texto[start_pos_baseline+9:end_pos_baseline-1])
    print(baseline)

    start_pos_doffs = texto.find('doffs')
    end_pos_doffs = start_pos_baseline
    doffs = float(texto[start_pos_doffs+6:end_pos_doffs-1])
    print(doffs)

    start_pos_calib = texto.find('cam0')
    end_pos_calib = texto.find('cam1')
    matrix = texto[start_pos_calib+5:end_pos_calib-1]

    matrix2 = matrix.replace('[','')
    matrix2 = matrix2.replace(']','')
    matrix2 = matrix2.replace(';',' ')
    mat = np.fromstring(matrix2, dtype=float, sep=' ')
    focus = mat[0]
    print(focus)

    disp2depth_factor = baseline * focus

    depth = np.zeros(disparity.shape, dtype=np.uint8)

    furthest = disp2depth_factor/(min(disparity[disparity>0]) + doffs)

    for i in range(len(disparity)):
        for j in range(len(disparity[i])):
            if disparity[i, j] < 0:
                depth[i, j] = 255
            elif disparity[i, j] == 0:
                depth[i, j] = 255
            else:
                depth[i, j] = np.floor((disp2depth_factor/(disparity[i, j] + doffs))*255/furthest)

    im = plt.imshow(depth, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()



if __name__  == "__main__":
    main()