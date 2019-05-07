import cv2
import numpy as np
import math
import datetime
from matplotlib import pyplot as plt

def bresenham_alg(x0, y0, x1, y1):
    points_in_line = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points_in_line.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points_in_line.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points_in_line.append((x, y))
    return points_in_line

def dist_eucl(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def calc_disp(im0, im1, window_size, F):
    global disp
    base = int(np.floor(window_size/2))

    h = int(im0.shape[0] - 2 * base)
    w = int(im0.shape[1] - 2 * base)
    c = im0.shape[1]


    disp = np.zeros((h,w))

    for i in range(base, h + base):
        if (i % (h // 10)) == 0:
            print((100*i)//h, "% processado")
        for j in range(base, w + base):
            window0 = im0[i-base:i+base, j-base:j+base]

            best_point = [0, 0]
            best_diff = 999999

            pt = np.array([[i, j]], dtype=int)
            lines = cv2.computeCorrespondEpilines(pt, 1, F)
            line = lines[0].ravel()

            x0, y0 = map(int, [0, -line[2]/line[1]])
            x1, y1 = map(int, [c, -(line[2] + c * line[0]) / line[1]])

            points_list = bresenham_alg(x0, y0, x1, y1)
            points_array = np.asarray(points_list, dtype=int)
            points = points_array[points_array[:, 0] > base]
            points = points[points[:, 0] < h + base]
            points = points[points[:, 1] > base]
            points = points[points[:, 1] < w + base]

            for point in points:
                x = point[0]
                y = point[1]
                window1 = im1[x-base:x+base, y-base:y+base]

                diff = np.sum(np.absolute(window0 - window1))

                if diff < best_diff:
                    best_diff = diff
                    best_point = [x, y]
            disp[i-base,j-base] = dist_eucl(i, j, best_point[0], best_point[1])

    np.savetxt('dispa.txt', disp, fmt='%i')





def main():
    im0 = cv2.imread('data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)
    im1 = cv2.imread('data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1)

    # Matriz fundamental calculada separadamente da seguinte forma (Com os parâmetros igual o req2)
    # Fórmula para calcular F retirada de https://answers.opencv.org/question/118671/compute-fundamental-matrix-from-camera-calibration/?answer=160129#post-id-160129
    # RT = np.concatenate((R1,Tc1), axis=1)
    # proj1 = matrix1 @ RT
    # matrix2 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)
    # RT = np.concatenate((R2,Tc2), axis=1)
    # proj2 = matrix2 @ RT
    # X = np.array([[proj1[1], proj1[2]], [proj1[2], proj1[0]], [proj1[0], proj1[1]]])
    # Y = np.array([[proj2[1], proj2[2]], [proj2[2], proj2[0]], [proj2[0], proj2[1]]])
    # F = np.zeros((3,3))
    # for i in range (3):
    #     for j in range(3):
    #         XY = np.concatenate((X[j], Y[i]), axis=0)
    #         F[i, j] = np.linalg.det(XY)

    F = np.array([[ 3.63851638e+08,  1.01480866e+10,  1.64550533e+13], [-7.96085510e+08,  1.47304655e+09, -2.53412823e+14], [ 7.10124789e+12,  2.34463011e+14 ,-1.31306881e+16]])

    # Medir o tempo que leva pra rodar
    inicio = datetime.datetime.now()

    # Se quiser rodar de fato descomente a chamada da função
    # Aviso: leva MUITO tempo
    # calc_disp(im1, im0, 11, F)

    print(inicio)
    print('DONE')
    print(datetime.datetime.now())

    # Para carregar do arquivo depois
    disp = np.loadtxt('data/output/dispa.txt', dtype=int)

    plt.imshow(disp, cmap='hot')
    plt.colorbar()
    img = plt.gcf()
    img.savefig("data/output/disparidadeMorpheus.png", bbox_inches='tight')


if __name__  == "__main__":
    main()
