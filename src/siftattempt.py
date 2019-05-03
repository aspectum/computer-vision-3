import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)  #queryimage # left image
img2 = cv2.imread('../data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1) #trainimage # right image

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

#Computation of the fundamental matrix
F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# print(F)


# Obtainment of the rectification matrix and use of the warpPerspective to transform them...
pts1 = pts1[:,:][mask.ravel()==1]
pts2 = pts2[:,:][mask.ravel()==1]

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,img1.shape)

dst11 = cv2.warpPerspective(img1,rectmat1,img1.shape, cv2.BORDER_ISOLATED)
dst22 = cv2.warpPerspective(img2,rectmat2,img2.shape, cv2.BORDER_ISOLATED)

# print(dst11.shape, dst22.shape)
cv2.imwrite('1.png', dst11)
cv2.imwrite('2.png', dst22)

# plt.imshow(dst11, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

# plt.imshow(dst22, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()


#calculation of the disparity
# stereo = cv2.StereoBM_create(numDisparities=64, blockSize=11)
window_size = 3
min_disp = -96
num_disp = 192-min_disp
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
disp = stereo.compute(dst22.astype(np.uint8), dst11.astype(np.uint8)).astype(np.float32)

# print(disp.shape)
# a = input('')
# print(disp[disp>0].shape)
# a = input('')

# (hn, hbins, hpatches) = plt.hist(disp, bins=[-1100, -700, -300, 100, 500, 900, 1300, 1700, 2100])  # arguments are passed to np.histogram

# (hn, hbins, hpatches) = plt.hist(disp.ravel(), 20)  # arguments are passed to np.histogram
# print(hn)
# print(hbins)
# print(hpatches)
# plt.title("Histogram with 'auto' bins")
# hist = plt.gcf()
# hist.savefig("hist.png")
# plt.show()

plt.imshow(disp, cmap='hot')
plt.colorbar()
raw = plt.gcf()
raw.savefig("raw.png")
plt.show()

aaa = np.where(disp > 2000, np.amin(disp), disp)
aaa = np.where(aaa < -1000, -1000, disp)
# print(aaa.shape)

plt.imshow(aaa, cmap='hot')
plt.colorbar()
thrs = plt.gcf()
thrs.savefig("threshold.png")
plt.show()

# focal_length = 6693

# ############## dados das imagens ######################
# imL_calib = open('../data/FurukawaPonce/MorpheusL.txt', 'r')
# texto = imL_calib.read()

# start_fc = texto.find('fc = ')
# end_fc = texto.find(']', start_fc)
# fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')

# start_cc = texto.find('cc = ')
# end_cc = texto.find(']', start_cc)
# cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')

# start_alpha = texto.find('alpha_c = ')
# end_alpha = texto.find(';', start_alpha)
# alpha = float(texto[start_alpha+10:end_alpha])

# start_R = texto.find('R = ')
# end_R = texto.find(']', start_R)
# R1 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
# R1.shape = (3, 3)

# start_Tc = texto.find('Tc = ')
# end_Tc = texto.find(']', start_Tc)
# Tc1 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
# Tc1.shape = (3, 1)

# matrix1 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)

# RT = np.concatenate((R1,Tc1), axis=1)

# proj1 = matrix1 @ RT

# imR_calib = open('../data/FurukawaPonce/MorpheusR.txt', 'r')
# texto = imR_calib.read()

# start_fc = texto.find('fc = ')
# end_fc = texto.find(']', start_fc)
# fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')

# start_cc = texto.find('cc = ')
# end_cc = texto.find(']', start_cc)
# cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')

# start_alpha = texto.find('alpha_c = ')
# end_alpha = texto.find(';', start_alpha)
# alpha = float(texto[start_alpha+10:end_alpha])

# start_R = texto.find('R = ')
# end_R = texto.find(']', start_R)
# R2 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
# R2.shape = (3, 3)

# start_Tc = texto.find('Tc = ')
# end_Tc = texto.find(']', start_Tc)
# Tc2 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
# Tc2.shape = (3, 1)

# matrix2 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)

# RT = np.concatenate((R2,Tc2), axis=1)

# proj2 = matrix2 @ RT

# # X = np.array([[proj1[1], proj1[2]], [proj1[2], proj1[0]], [proj1[0], proj1[1]]])
# # Y = np.array([[proj2[1], proj2[2]], [proj2[2], proj2[0]], [proj2[0], proj2[1]]])

# # F = np.zeros((3,3))

# # for i in range (3):
# #     for j in range(3):
# #         XY = np.concatenate((X[j], Y[i]), axis=0)
# #         F[i, j] = np.linalg.det(XY)

# # print(F)

# rvec1, _ = cv2.Rodrigues(R1)
# rvec2, _ = cv2.Rodrigues(R2)

# Tc1.shape = (1, 3)
# Tc2.shape = (1, 3)

# print(rvec1.ravel().shape)
# print(Tc1.ravel().shape)
# print(rvec2.ravel().shape)
# print(Tc2.ravel().shape)


# rvec3, tvec3, _, _, _, _, _, _, _, _, = cv2.composeRT(rvec1.ravel(), Tc1.ravel(), rvec2.ravel(), Tc2.ravel())

# disp2depth_factor = tvec3 * focal_length

# depth = np.zeros(aaa.shape, dtype=np.uint8)

# doffs = 0   # Ele assume que os pontos principais são no centro em stereoRectifyUncalibrated

# furthest = disp2depth_factor/(np.amax(aaa) + doffs)

# print('Computando profundidade')
# depth[aaa > -1000] = np.floor((disp2depth_factor/(aaa[aaa > -1000] + doffs))*254/furthest)
# depth[disparity <= -1000] = 255

# plt.imshow(depth, cmap='hot')
# plt.colorbar()
# depth = plt.gcf()
# depth.savefig("depth.png")
# plt.show()


# #plot depth by using disparity focal length C1[0,0] from stereo calibration and T[0] the distance between cameras

# # plt.imshow(C1[0,0]*T[0]/(disp),cmap='hot');plt.clim(-0,500);plt.colorbar();plt.show()