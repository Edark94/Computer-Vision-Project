import cv2
import numpy as np
from pythonRLSA import rlsa
from scipy.ndimage import label

img = cv2.cvtColor(cv2.imread('0019.jpg'), cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

img_rlsa_horizontal = rlsa.rlsa(thresh, True, False, 5)
img_rlsa_vertical = rlsa.rlsa(img_rlsa_horizontal, False, True, 20)

cv2.imwrite('placeholder.jpg', img_rlsa_vertical)

opening = cv2.morphologyEx(img_rlsa_vertical, cv2.MORPH_OPEN, np.ones((3,3), np.int), iterations = 2)

cv2.imwrite('written_0019.jpg', opening)

sure_bg = cv2.dilate(opening, None, iterations=5)
sure_bg = sure_bg - cv2.erode(sure_bg, None)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
dist_transform = ((dist_transform - dist_transform.min()) / (dist_transform.max() - dist_transform.min()) * 255).astype(np.uint8)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

lbl, ncc = label(dist_transform)
lbl = lbl * lbl * (255 / (ncc+1))
lbl[sure_bg == 255] = 255
lbl = lbl.astype(np.int32)

img_3C = cv2.imread('0019.jpg')
cv2.watershed(img_3C, lbl)
lbl[lbl == -1] = 0
lbl = lbl.astype(np.uint8)
result = 255 - lbl


cv2.imwrite('written_0019_02.jpg', result)

result[result != 255] = 0
result = cv2.dilate(result, None)
img_3C[result == 255] = (0, 0, 255)
cv2.imwrite('written_0019_03.jpg', img_3C)
