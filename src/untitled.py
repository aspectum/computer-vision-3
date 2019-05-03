import cv2
import numpy as np
import math
import matplotlib.pyplot as plt




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




im0 = cv2.imread('../data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)
im1 = cv2.imread('../data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1)

disparity = stereo.compute(im0, im1).astype(np.float32)/16.0


im = plt.imshow(disparity, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()

calib_params = open('../data/Middlebury/Jadeplant-perfect/calib.txt', 'r')
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

depth[disparity>0] = np.floor((disp2depth_factor/(disparity[disparity>0] + doffs))*254/furthest)
depth[disparity<=0] = 255

# for i in range(len(disparity)):
#     for j in range(len(disparity[i])):
#         if disparity[i, j] < 0:
#             depth[i, j] = 255
#         elif disparity[i, j] == 0:
#             depth[i, j] = 255
#         else:
#             depth[i, j] = np.floor((disp2depth_factor/(disparity[i, j] + doffs))*254/furthest)

im = plt.imshow(depth, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()


####################

import numpy as np
import cv2
import matplotlib.pyplot as plt

im0 = cv2.imread('../data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)
im1 = cv2.imread('../data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1)

imL_calib = open('../data/FurukawaPonce/MorpheusL.txt', 'r')
texto = imL_calib.read()

start_fc = texto.find('fc = ')
end_fc = texto.find(']', start_fc)
fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')
print('fc = ', fc)

start_cc = texto.find('cc = ')
end_cc = texto.find(']', start_cc)
cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')
print('cc = ', cc)

start_alpha = texto.find('alpha_c = ')
end_alpha = texto.find(';', start_alpha)
alpha = float(texto[start_alpha+10:end_alpha])
print('alpha_c = ', alpha)

start_R = texto.find('R = ')
end_R = texto.find(']', start_R)
R1 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
R1.shape = (3, 3)
print('R = ', R1)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc1 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc1.shape = (3, 1)
print('Tc = ', Tc1)

matrix1 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)
print('matrix = ', matrix1)

RT = np.concatenate((R1,Tc1), axis=1)
print(RT)

proj1 = matrix1 @ RT

imR_calib = open('../data/FurukawaPonce/MorpheusR.txt', 'r')
texto = imR_calib.read()

start_fc = texto.find('fc = ')
end_fc = texto.find(']', start_fc)
fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')
print('fc = ', fc)

start_cc = texto.find('cc = ')
end_cc = texto.find(']', start_cc)
cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')
print('cc = ', cc)

start_alpha = texto.find('alpha_c = ')
end_alpha = texto.find(';', start_alpha)
alpha = float(texto[start_alpha+10:end_alpha])
print('alpha_c = ', alpha)

start_R = texto.find('R = ')
end_R = texto.find(']', start_R)
R2 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
R2.shape = (3, 3)
print('R = ', R2)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc2 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc2.shape = (3, 1)
print('Tc = ', Tc2)

matrix2 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)
print('matrix = ', matrix2)

RT = np.concatenate((R2,Tc2), axis=1)
print(RT)

proj2 = matrix2 @ RT

X = np.array([[proj1[1], proj1[2]], [proj1[2], proj1[0]], [proj1[0], proj1[1]]])
Y = np.array([[proj2[1], proj2[2]], [proj2[2], proj2[0]], [proj2[0], proj2[1]]])

F = np.zeros((3,3))

for i in range (3):
    for j in range(3):
        XY = np.concatenate((X[j], Y[i]), axis=0)
        F[i, j] = np.linalg.det(XY)

print(F)

rvec1, _ = cv2.Rodrigues(R1)
rvec2, _ = cv2.Rodrigues(R2)

Tc1.shape = (1, 3)
Tc2.shape = (1, 3)

rvec3, tvec3, _, _, _, _, _, _, _, _, = cv2.composeRT(rvec1, Tc1, rvec2, Tc2)

print(rvec3)
print(tvec3)

distCoeffs = np.array([3.5682116529529168e-01, -2.4792040925526120e+00, -2.0346033295904572e-03, 1.9892727029763874e-03, 3.5210739594960989e+00])
#distCoeffs = np.zeros((1,5))
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(matrix1, distCoeffs, matrix2, distCoeffs, im0.shape, rvec3, tvec3)

aaa = cv2.warpPerspective(im0, h1c, im0.shape)
##############
###############
############

# au = norm(extp(Po1(1,1:3)', Po1(3,1:3)'));
# av = norm(extp(Po1(2,1:3)', Po1(3,1:3)'));
au = np.linalg.norm()
% optical centres
c1 = - inv(Po1(:,1:3))*Po1(:,4);
c2 = - inv(Po2(:,1:3))*Po2(:,4);
% retinal planes
fl = Po1(3,1:3)';
fr = Po2(3,1:3)';
nn = extp(fl,fr);
% solve the four systems
A = [ [c1' 1]' [c2' 1]' [nn' 0]' ]';
[U,S,V] = svd(A);
r = 1/(norm(V([1 2 3],4)));
a3 = r * V(:,4);
A = [ [c1' 1]' [c2' 1]' [a3(1:3)' 0]' ]';
[U,S,V] = svd(A);
r = norm(av)/(norm(V([1 2 3],4)));
a2 = r * V(:,4);
A = [ [c1' 1]' [a2(1:3)' 0]' [a3(1:3)' 0]' ]';
[U,S,V] = svd(A);
r = norm(au)/(norm(V([1 2 3],4)));
a1 = r * V(:,4);
A = [ [c2' 1]' [a2(1:3)' 0]' [a3(1:3)' 0]' ]';
[U,S,V] = svd(A);
r = norm(au)/(norm(V([1 2 3],4)));
b1 = r * V(:,4);
% adjustment
H = [
1 0 0
0 1 0
0 0 1 ];
% rectifying projection matrices
Pn1 = H * [ a1 a2 a3 ]';
Pn2 = H * [ b1 a2 a3 ]';
% rectifying image transformation
T1 = Pn1(1:3,1:3)* inv(Po1(1:3,1:3));
T2 = Pn2(1:3,1:3)* inv(Po2(1:3,1:3));

####################



import numpy as np
import cv2
import matplotlib.pyplot as plt

im0 = cv2.imread('../data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)
im1 = cv2.imread('../data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1)

imL_calib = open('../data/FurukawaPonce/MorpheusL.txt', 'r')
texto = imL_calib.read()

start_fc = texto.find('fc = ')
end_fc = texto.find(']', start_fc)
fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')

start_cc = texto.find('cc = ')
end_cc = texto.find(']', start_cc)
cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')

start_alpha = texto.find('alpha_c = ')
end_alpha = texto.find(';', start_alpha)
alpha = float(texto[start_alpha+10:end_alpha])

start_R = texto.find('R = ')
end_R = texto.find(']', start_R)
rotM1 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
rotM1.shape = (3, 3)
print('R = ', rotM1)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc1 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc1.shape = (3, 1)
print('Tc = ', Tc1)

matrix1 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)
print('matrix = ', matrix1)

imR_calib = open('../data/FurukawaPonce/MorpheusR.txt', 'r')
texto = imR_calib.read()

start_fc = texto.find('fc = ')
end_fc = texto.find(']', start_fc)
fc = np.fromstring(texto[start_fc+6:end_fc].replace(';',' '), dtype=float, sep=' ')
print('fc = ', fc)

start_cc = texto.find('cc = ')
end_cc = texto.find(']', start_cc)
cc = np.fromstring(texto[start_cc+6:end_cc].replace(';',' '), dtype=float, sep=' ')
print('cc = ', cc)

start_alpha = texto.find('alpha_c = ')
end_alpha = texto.find(';', start_alpha)
alpha = float(texto[start_alpha+10:end_alpha])
print('alpha_c = ', alpha)

start_R = texto.find('R = ')
end_R = texto.find(']', start_R)
rotM2 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
rotM2.shape = (3, 3)
print('R = ', rotM2)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc2 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc2.shape = (3, 1)
print('Tc = ', Tc2)

matrix2 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)
print('matrix = ', matrix2)

rvec1, _ = cv2.Rodrigues(rotM2)
rvec2, _ = cv2.Rodrigues(rotM2)

Tc1.shape = (1, 3)
Tc2.shape = (1, 3)

t1 = Tc1.ravel()
t2 = Tc2.ravel()
t3 = Tc3.ravel()

r1 = rvec1.ravel()
r2 = rvec2.ravel()

rvec3, tvec3, _, _, _, _, _, _, _, _, = cv2.composeRT(rvec1, t1, rvec2, t2)

print(rvec3)
print(tvec3)


distCoeffs = np.zeros((1, 5))

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(matrix1, distCoeffs, matrix2, distCoeffs, im0.shape, rvec3, tvec3)

aaa = cv2.warpPerspective(im0, R1, im0.shape)

im = plt.imshow(aaa, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()

map1, map2 = cv2.initUndistortRectifyMap(matrix1, distCoeffs, rotM1, matrix1, im0.shape, cv2.CV_32FC1)
aaa = np.zeros(im0.shape)
aaa = cv2.remap(im0, map1, map2, cv2.INTER_NEAREST)

im = plt.imshow(aaa, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()


####################