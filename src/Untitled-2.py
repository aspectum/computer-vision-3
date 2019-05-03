import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../data/FurukawaPonce/MorpheusL.jpg', cv2.CV_8UC1)  #queryimage # left image
img2 = cv2.imread('../data/FurukawaPonce/MorpheusR.jpg', cv2.CV_8UC1) #trainimage # right image

### Imagem da esquerda
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
R1 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
R1.shape = (3, 3)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc1 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc1.shape = (3, 1)

matrix1 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)

scaling_compensation = np.array([[12/13, 0, 1/26], [0, 12/13, 1/26], [0, 0, 1]])
matrix1 = scaling_compensation @ matrix1


### Imagem da direita
imR_calib = open('../data/FurukawaPonce/MorpheusR.txt', 'r')
texto = imR_calib.read()

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
R2 = np.fromstring(texto[start_R+5:end_R].replace(',',' ').replace(';',' '), dtype=float, sep=' ')
R2.shape = (3, 3)

start_Tc = texto.find('Tc = ')
end_Tc = texto.find(']', start_Tc)
Tc2 = np.fromstring(texto[start_Tc+6:end_Tc].replace(';',' '), dtype=float, sep=' ')
Tc2.shape = (3, 1)

matrix2 = np.array([[fc[0], alpha, cc[0]], [0, fc[1], cc[1]], [0, 0, 1]], dtype=float)

rvec1, _ = cv2.Rodrigues(R1)
rvec2, _ = cv2.Rodrigues(R2)

Tc1.shape = (1, 3)
Tc2.shape = (1, 3)

rvec3, tvec3, _, _, _, _, _, _, _, _, = cv2.composeRT(rvec1.ravel(), Tc1.ravel(), rvec2.ravel(), Tc2.ravel())

R3, _ = cv2.Rodrigues(rvec3)

t1 = Tc1.ravel()
t2 = Tc2.ravel()
t3 = tvec3.ravel()

print(t1, t2, t3)

e1 = t3 / np.linalg.norm(t3)
e2 = np.transpose([-t3[1], t3[0], 0]) / math.sqrt(t3[0]**2 + t3[1]**2)
e3 = np.cross(e1, e2)

print(e1)
print(e2)
print(e3)

R_rect = np.array([e1, e2, e3])

print(R_rect)

R_l = R_rect
R_r = R3 @ R_rect

newL = np.zeros(2*img1.shape)
newR = np.zeros(2*img2.shape)

pl_temp = 