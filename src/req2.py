import numpy as np
import cv2
from matplotlib import pyplot as plt

# Parâmetros do stereoSGBM
window_size = 3
min_disp = -96
num_disp = 128-min_disp
stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8*4*window_size**2,
    P2 = 32*4*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 700,
    speckleRange = 10
)

def plt_show(what, name):
    plt.imshow(what, cmap='hot')
    plt.colorbar()
    img = plt.gcf()
    img.savefig(name, bbox_inches='tight')
    plt.show()

def main():
    # Toda a parte da implementação do SIFT eu peguei do usuário Bilou563 nos fóruns da OpenCV
    # https://answers.opencv.org/question/90742/opencv-depth-map-from-uncalibrated-stereo-system/
    # Acesso em 06/05/2019

    print("Carregando as imagens")

    img1 = cv2.imread('data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)  #queryimage # left image
    img2 = cv2.imread('data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1) #trainimage # right image


    # Ajustando o tamanho de img1 para que as imagens tenham o mesmo tamanho (necessário)
    img1 = img1[:, 0:1300]  # Crop de uma parte que não tem nada (eu vi antes na imagem)
                            # E colocando no mesmo aspect ratio de img2
    img1 = cv2.resize(img1, img2.shape)     # Colocando no mesmo tamanho de img2

    print("Encontrando pontos correspondentes com SIFT")
    #Obtainment of the correspondent point with SIFT
    sift = cv2.xfeatures2d.SIFT_create()

    ###find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    ###FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    ###ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)


    pts1 = np.array(pts1)
    pts2 = np.array(pts2)


    print("Calculando a matriz fundamental com base nos pontos correspondentes")
    #Computation of the fundamental matrix
    F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    # Obtainment of the rectification matrix and use of the warpPerspective to transform them...
    pts1 = pts1[:,:][mask.ravel()==1]
    pts2 = pts2[:,:][mask.ravel()==1]

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
    p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))


    print("Retificando as imagens com base na matriz fundamental")
    retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,img1.shape)

    dst11 = cv2.warpPerspective(img1,rectmat1,img1.shape, cv2.BORDER_ISOLATED)
    dst22 = cv2.warpPerspective(img2,rectmat2,img2.shape, cv2.BORDER_ISOLATED)

    print("Gravando as imagens retificadas")
    # Gravando as imagens retificadas
    cv2.imwrite('data/output/rectifiedMorpheusL.png', dst11)
    cv2.imwrite('data/output/rectifiedMorpheusR.png', dst22)

    print("Calculando a disparidade")
    disp = stereo.compute(dst11.astype(np.uint8), dst22.astype(np.uint8)).astype(np.float32)/16

    plt_show(disp, "data/output/dispMorpheus.png")


    # Histograma que usei para achar os valores 60 e -60 usados
    # para criar disp_filtered logo abaixo
    # É uma tentativa de remover os pontos que ele calculou a
    # disparidade errado/não encontrou match
    # plt.hist(disp.ravel(), 40)
    # plt.title("Histogram with 'auto' bins")
    # hist = plt.gcf()
    # hist.savefig("histMorpheus.png")
    # plt.show()

    disp_filtered = np.where(disp > 60, np.amin(disp), disp)
    disp_filtered = np.where(disp_filtered < -60, -60, disp)

    plt_show(disp_filtered, "data/output/dispMorpheusFiltered.png")

    #########################################
    ######## Cálculo da profundidade ########
    #########################################

    print("Carregando os parâmetros de calibração")
    # Carregando os parâmetros de calibração
    # MorpheusL
    imL_calib = open('data/FurukawaPonce/MorpheusL.txt', 'r')
    texto = imL_calib.read()

    start_f1 = texto.find('fc = ')
    end_f1 = texto.find(']', start_f1)
    f1 = np.fromstring(texto[start_f1+6:end_f1].replace(';',' '), dtype=float, sep=' ')

    start_c1 = texto.find('cc = ')
    end_c1 = texto.find(']', start_c1)
    c1 = np.fromstring(texto[start_c1+6:end_c1].replace(';',' '), dtype=float, sep=' ')

    start_alpha = texto.find('alpha_c = ')
    end_alpha = texto.find(';', start_alpha)
    alpha = float(texto[start_alpha+10:end_alpha])

    start_R = texto.find('R = ')
    end_R = texto.find(']', start_R)
    R1 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
    R1.shape = (3, 3)

    start_Tc = texto.find('Tc = ')
    end_Tc = texto.find(']', start_Tc)
    Tc1 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
    Tc1.shape = (3, 1)

    matrix1 = np.array([[f1[0], alpha, c1[0]], [0, f1[1], c1[1]], [0, 0, 1]], dtype=float)

    # Compensando a matriz de intrínsecos pelo resize da MorpheusL de acordo com:
    # https://dsp.stackexchange.com/a/6098
    # Acesso em 06/05/2019
    scaling_compensation = np.array([[12/13, 0, 1/26], [0, 12/13, 1/26], [0, 0, 1]])
    matrix1 = scaling_compensation @ matrix1

    # MorpheusR
    imR_calib = open('data/FurukawaPonce/MorpheusR.txt', 'r')
    texto = imR_calib.read()

    start_f2 = texto.find('fc = ')
    end_f2 = texto.find(']', start_f2)
    f2 = np.fromstring(texto[start_f2+6:end_f2].replace(';',' '), dtype=float, sep=' ')

    start_c2 = texto.find('cc = ')
    end_c2 = texto.find(']', start_c2)
    c2 = np.fromstring(texto[start_c2+6:end_c2].replace(';',' '), dtype=float, sep=' ')

    start_alpha = texto.find('alpha_c = ')
    end_alpha = texto.find(';', start_alpha)
    alpha = float(texto[start_alpha+10:end_alpha])

    start_R = texto.find('R = ')
    end_R = texto.find(']', start_R)
    R2 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
    R2.shape = (3, 3)

    start_Tc = texto.find('Tc_8 = ')    # Por alguma razão está como 'Tc_8' nessa imagem
    end_Tc = texto.find(']', start_Tc)
    Tc2 = np.fromstring(texto[start_Tc+8:end_Tc].replace(';',' '), dtype=float, sep=' ')
    Tc2.shape = (3, 1)

    matrix2 = np.array([[f2[0], alpha, c1[0]], [0, f2[1], c2[1]], [0, 0, 1]], dtype=float)

    # Ajustando parâmetros
    rvec1, _ = cv2.Rodrigues(R1)
    rvec2, _ = cv2.Rodrigues(R2)

    Tc1.shape = (1, 3)
    Tc2.shape = (1, 3)

    # Calculando o deslocamento e rotação relativos entre as câmeras
    rvec3, tvec3, _, _, _, _, _, _, _, _, = cv2.composeRT(rvec1.ravel(), Tc1.ravel(), rvec2.ravel(), Tc2.ravel())


    print("Calculando profundidade")
    focal_length = np.mean([np.mean(f1),np.mean(f2)])
    doffs = c2[0] - c1[0]               # Diferença no eixo x entre os pontos principais
    baseline = np.linalg.norm(tvec3)    # Distância entre as câmeras
    disp2depth_factor = baseline * focal_length

    # Disparidade em mm(?)
    # Tem algum fator errado aqui porque está muito grande
    depth = np.zeros(disp_filtered.shape, dtype=np.uint8)
    depth = disp2depth_factor/(disp_filtered + doffs)
    # plt_show(depth, "depthMorpheus_mm.png")


    furthest = disp2depth_factor/(np.amin(disp_filtered[disp_filtered > -60]) + doffs)

    depth = disp2depth_factor/(disp_filtered + doffs)
    # Normalizando a profundidade de acordo com o roteiro
    depth[disp_filtered > -60] = np.floor((disp2depth_factor/(disp_filtered[disp_filtered > -60] + doffs))*254/furthest)
    depth[disp_filtered <= -60] = 255

    plt_show(depth, "data/output/depthMorpheus.png")


if __name__  == "__main__":
    main()
