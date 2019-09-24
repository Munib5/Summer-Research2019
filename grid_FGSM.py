import cv2
import numpy as np

img_orig_6_0 = cv2.imread('orig_6_0.png')
img_orig_5_0 = cv2.imread('orig_5_0.png')
img_orig_4_0 = cv2.imread('orig_4_0.png')
img_orig_3_0 = cv2.imread('orig_3_0.png')
img_orig_2_0 = cv2.imread('orig_2_0.png')
img_orig_1_0 = cv2.imread('orig_1_0.png')
img_orig_0_0 = cv2.imread('orig_0_0.png')

img_orig_6_0 = cv2.resize(img_orig_6_0, (0, 0), None, 2, 2)
img_orig_5_0 = cv2.resize(img_orig_5_0, (0, 0), None, 2, 2)
img_orig_4_0 = cv2.resize(img_orig_4_0, (0, 0), None, 2, 2)
img_orig_3_0 = cv2.resize(img_orig_3_0, (0, 0), None, 2, 2)
img_orig_2_0 = cv2.resize(img_orig_2_0, (0, 0), None, 2, 2)
img_orig_1_0 = cv2.resize(img_orig_1_0, (0, 0), None, 2, 2)
img_orig_0_0 = cv2.resize(img_orig_0_0, (0, 0), None, 2, 2)

img_orig_4_1 = cv2.imread('orig_4_1.png')
img_orig_3_1 = cv2.imread('orig_3_1.png')
img_orig_2_1 = cv2.imread('orig_2_1.png')
img_orig_1_1 = cv2.imread('orig_1_1.png')
img_orig_0_1 = cv2.imread('orig_0_1.png')

img_orig_4_2 = cv2.imread('orig_4_2.png')
img_orig_3_2 = cv2.imread('orig_3_2.png')
img_orig_2_2 = cv2.imread('orig_2_2.png')
img_orig_1_2 = cv2.imread('orig_1_2.png')
img_orig_0_2 = cv2.imread('orig_0_2.png')

img_orig_4_3 = cv2.imread('orig_4_3.png')
img_orig_3_3 = cv2.imread('orig_3_3.png')
img_orig_2_3 = cv2.imread('orig_2_3.png')
img_orig_1_3 = cv2.imread('orig_1_3.png')
img_orig_0_3 = cv2.imread('orig_0_3.png')

img_orig_4_4 = cv2.imread('orig_4_4.png')
img_orig_3_4 = cv2.imread('orig_3_4.png')
img_orig_2_4 = cv2.imread('orig_2_4.png')
img_orig_1_4 = cv2.imread('orig_1_4.png')
img_orig_0_4 = cv2.imread('orig_0_4.png')

img_adv_6_0 = cv2.imread('adv_6_0.png')
img_adv_5_0 = cv2.imread('adv_5_0.png')
img_adv_4_0 = cv2.imread('adv_4_0.png')
img_adv_3_0 = cv2.imread('adv_3_0.png')
img_adv_2_0 = cv2.imread('adv_2_0.png')
img_adv_1_0 = cv2.imread('adv_1_0.png')
img_adv_0_0 = cv2.imread('adv_0_0.png')

img_adv_6_0 = cv2.resize(img_adv_6_0, (0, 0), None, 2, 2)
img_adv_5_0 = cv2.resize(img_adv_5_0, (0, 0), None, 2, 2)
img_adv_4_0 = cv2.resize(img_adv_4_0, (0, 0), None, 2, 2)
img_adv_3_0 = cv2.resize(img_adv_3_0, (0, 0), None, 2, 2)
img_adv_2_0 = cv2.resize(img_adv_2_0, (0, 0), None, 2, 2)
img_adv_1_0 = cv2.resize(img_adv_1_0, (0, 0), None, 2, 2)
img_adv_0_0 = cv2.resize(img_adv_0_0, (0, 0), None, 2, 2)

img_adv_4_1 = cv2.imread('adv_4_1.png')
img_adv_3_1 = cv2.imread('adv_3_1.png')
img_adv_2_1 = cv2.imread('adv_2_1.png')
img_adv_1_1 = cv2.imread('adv_1_1.png')
img_adv_0_1 = cv2.imread('adv_0_1.png')

img_adv_4_2 = cv2.imread('adv_4_2.png')
img_adv_3_2 = cv2.imread('adv_3_2.png')
img_adv_2_2 = cv2.imread('adv_2_2.png')
img_adv_1_2 = cv2.imread('adv_1_2.png')
img_adv_0_2 = cv2.imread('adv_0_2.png')

img_adv_4_3 = cv2.imread('adv_4_3.png')
img_adv_3_3 = cv2.imread('adv_3_3.png')
img_adv_2_3 = cv2.imread('adv_2_3.png')
img_adv_1_3 = cv2.imread('adv_1_3.png')
img_adv_0_3 = cv2.imread('adv_0_3.png')

img_adv_4_4 = cv2.imread('adv_4_4.png')
img_adv_3_4 = cv2.imread('adv_3_4.png')
img_adv_2_4 = cv2.imread('adv_2_4.png')
img_adv_1_4 = cv2.imread('adv_1_4.png')
img_adv_0_4 = cv2.imread('adv_0_4.png')

numpy_horizontal_1 = np.hstack((img_orig_6_0, img_orig_5_0, img_orig_4_0,
img_orig_3_0,
img_orig_2_0,
img_orig_1_0,
img_orig_0_0))

numpy_horizontal_2 = np.hstack((img_adv_6_0, img_adv_5_0, img_adv_4_0,
img_adv_3_0,
img_adv_2_0,
img_adv_1_0,
img_adv_0_0))

numpy_firstset = np.vstack((numpy_horizontal_1,numpy_horizontal_2))
numpy_horizontal_1_concat = np.concatenate((numpy_horizontal_1, numpy_horizontal_2), axis = 0)

cv2.imshow('7 different noise values (FGSM): 1, 0.85, 0.7, 0.5, 0.2, 0.1, 0', numpy_horizontal_1_concat)

cv2.waitKey()
