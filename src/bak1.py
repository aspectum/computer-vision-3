import cv2
import numpy as np
import math

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


def find_corresponding_pts(im0, im1, window_size, max_offset):
    h = int(im0.shape[0] - 2 * np.floor(window_size/2))
    w = int(im0.shape[1] - 2 * np.floor(window_size/2))

    base = int(np.floor(window_size/2))
    print(h,w,base)
    for i in range(1000, h - base):
        print(i)
        for j in range(base, w - base):
            # window0_R = np.zeros((window_size, window_size))
            # window0_G = np.zeros((window_size, window_size))
            # window0_B = np.zeros((window_size, window_size))

            window0_R = im0[i-base:i+base, j-base:j+base, 2]
            window0_G = im0[i-base:i+base, j-base:j+base, 1]
            window0_B = im0[i-base:i+base, j-base:j+base, 0]

            best_point = [0, 0]
            best_dist = 999999      # tem um jeito mais elegante de fazer isso com ctz

            for k in range(j, min(j + max_offset, w - base)):
                window1_R = im1[i-base:i+base, k-base:k+base, 2]
                window1_G = im1[i-base:i+base, k-base:k+base, 1]
                window1_B = im1[i-base:i+base, k-base:k+base, 0]

                dif_R = np.absolute(np.subtract(window0_R, window1_R))
                dif_G = np.absolute(np.subtract(window0_G, window1_G))
                dif_B = np.absolute(np.subtract(window0_B, window1_B))

                dif = np.sum(dif_R) + np.sum(dif_G) + np.sum(dif_B)
                if dif < best_dist:
                    best_dist = dif
                    best_point = [i, k]
            print(best_dist)
            # if i != best_point[0] or j != best_point[1]:
            #     print("Ponto: ", i, j, " corresponde a ", best_point)





def main():
    # window_size = input("Digite o tamanho da janela: ")
    window_size = 5
    im0 = cv2.imread('../data/jadeplant/im0.png', cv2.CV_8UC1)
    im1 = cv2.imread('../data/jadeplant/im1.png', cv2.CV_8UC1)

    disparity = stereo.compute(im0, im1).astype(np.float32)/16.0
    # print(np.unique(disparity.astype(int)))
    disparity = (disparity-min_disp)/num_disp
    # disparity = disparity + 16
    
    # disparity = np.floor(disparity * 255 / 256).astype(np.uint8)
    # print(np.unique(disparity))
    

    # find_corresponding_pts(im1, im0, window_size, 500) # imagem mais a esquerda primeiro
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.imshow('disparity', disparity)
    cv2.resizeWindow('disparity', 1000,1000)

    cv2.waitKey()


if __name__  == "__main__":
    main()