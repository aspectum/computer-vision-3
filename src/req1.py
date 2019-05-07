import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Parâmetros para o stereoSGBM da imagem Jadeplant
window_size = 1
min_disp = 0
num_disp = 640-min_disp     # 640 conforme no ndisp no calib.txt
jadeplantSGBM = cv2.StereoSGBM_create(
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

# Parâmetros para o stereoBM da imagem Jadeplant
jadeplantBM = cv2.StereoBM_create(
    numDisparities = 640,   # 640 conforme no ndisp no calib.txt
    blockSize = 15
)

# Parâmetros para o stereoSGBM da imagem Motorcycle
window_size = 5
min_disp = 0
num_disp = 272-min_disp     # O ndisp no calib.txt diz 270, mas tem que ser divisível por 16
motorcycleSGBM = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 9,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 200,
    speckleRange = 2,
)

# Parâmetros para o stereoBM da imagem Motorcycle
motorcycleBM = cv2.StereoBM_create(
    numDisparities = 272,   # O ndisp no calib.txt diz 270, mas tem que ser divisível por 16
    blockSize = 15
)

def plt_show(what, name):
    plt.imshow(what, cmap='hot')
    plt.colorbar()
    img = plt.gcf()
    img.savefig(name, bbox_inches='tight')
    plt.show()
    

def get_calib_params(calib_file):
    calib_params_jadeplant = open(calib_file, 'r')
    texto = calib_params_jadeplant.read()
    start_pos_baseline = texto.find('baseline')
    end_pos_baseline = texto.find('width')
    baseline = float(texto[start_pos_baseline+9:end_pos_baseline-1])

    start_pos_doffs = texto.find('doffs')
    end_pos_doffs = start_pos_baseline
    doffs = float(texto[start_pos_doffs+6:end_pos_doffs-1])

    start_pos_calib = texto.find('cam0')
    end_pos_calib = texto.find('cam1')
    matrix = texto[start_pos_calib+5:end_pos_calib-1]

    matrix2 = matrix.replace('[','')
    matrix2 = matrix2.replace(']','')
    matrix2 = matrix2.replace(';',' ')
    mat = np.fromstring(matrix2, dtype=float, sep=' ')
    focus = mat[0]

    return baseline, doffs, focus

def calc_depth(disp, baseline, doffs, focus):
    disp2depth_factor = baseline * focus

    depth = np.zeros(disp.shape, dtype=np.uint8)

    furthest = disp2depth_factor/(min(disp[disp>0]) + doffs)

    # Fazendo a normalização requerida pelo roteiro
    depth[disp>0] = np.floor((disp2depth_factor/(disp[disp>0] + doffs))*254/furthest)
    depth[disp<=0] = 255
    
    print("Fator de conversão para mm: ", furthest/254)

    return depth

def compare_GT(disp, GT, name):
    highest_GT = np.amax(GT)
    highest_disp = np.amax(disp)

    disp_norm = disp * highest_GT / highest_disp

    error = np.abs(disp_norm - GT).astype(int)
    errors = len(error[error > 2])

    error_percent = 100 * errors / (disp.shape[0] * disp.shape[1])

    cv2.imwrite(name, disp_norm)
    print("Erro com relação ao Ground Truth: ", error_percent, "%")

def main():
    print('Carregando as imagens')
    jadeplant0 = cv2.imread('data/Middlebury/Jadeplant-perfect/im0.png', cv2.CV_8UC1)
    jadeplant1 = cv2.imread('data/Middlebury/Jadeplant-perfect/im1.png', cv2.CV_8UC1)
    motorcycle0 = cv2.imread('data/Middlebury/Motorcycle-perfect/im0.png', cv2.CV_8UC1)
    motorcycle1 = cv2.imread('data/Middlebury/Motorcycle-perfect/im1.png', cv2.CV_8UC1)
    GTjade = cv2.imread('data/Middlebury/Jadeplant-perfect/disp0-n.pgm', cv2.CV_8UC1)
    GTmoto = cv2.imread('data/Middlebury/Motorcycle-perfect/disp0-n.pgm', cv2.CV_8UC1)

    print()
    ###########################
    ######## Jadeplant ########
    ###########################

    print('Imagem Jadeplant')

    print('Computando disparidade com stereoBM')
    dispJadeplantBM = jadeplantBM.compute(jadeplant0, jadeplant1).astype(np.float32)
    # Os valores do stereoBM não são a disparidade em pixels. Tem que fazer alguma normalização.

    plt_show(dispJadeplantBM, "data/output/dispJadeplantBM.png")

    print('Computando disparidade com stereoSGBM')
    dispJadeplantSGBM = jadeplantSGBM.compute(jadeplant0, jadeplant1).astype(np.float32)/16.0
    # Isso é o que o req1 quer
    # cada posição de disparity é a distância entre os pixels das 2 imagens
    # os valores -1 eu suponho que sejam os que não tem correspondência

    plt_show(dispJadeplantSGBM, "data/output/dispJadeplantSGBM.png")

    baseline, doffs, focus = get_calib_params('data/Middlebury/Jadeplant-perfect/calib.txt')

    print('Computando profundidade')
    depthJadeplant = calc_depth(dispJadeplantSGBM, baseline, doffs, focus)

    plt_show(depthJadeplant, "data/output/depthJadeplant.png")

    compare_GT(dispJadeplantSGBM, GTjade, "data/output/dispJadeplantSGBM.pgm")

    print()
    ############################
    ######## Motorcycle ########
    ############################

    print('Imagem Motorcycle')

    print('Computando disparidade com stereoBM')
    dispMotorcycleBM = motorcycleBM.compute(motorcycle0, motorcycle1).astype(np.float32)
    # Os valores do stereoBM não são a disparidade em pixels. Tem que fazer alguma normalização.

    plt_show(dispMotorcycleBM, "data/output/dispMotorcycleBM.png")

    print('Computando disparidade com stereoSGBM')
    dispMotorcycleSGBM = motorcycleSGBM.compute(motorcycle0, motorcycle1).astype(np.float32)/16.0

    plt_show(dispMotorcycleSGBM, "data/output/dispMotorcycleSGBM.png")

    baseline, doffs, focus = get_calib_params('data/Middlebury/Motorcycle-perfect/calib.txt')

    print('Computando profundidade')
    depthMotorcycle = calc_depth(dispMotorcycleSGBM, baseline, doffs, focus)

    plt_show(depthMotorcycle, "data/output/depthMotorcycle.png")

    compare_GT(dispMotorcycleSGBM, GTmoto, "data/output/dispMotorcycleSGBM.pgm")



if __name__  == "__main__":
    main()
