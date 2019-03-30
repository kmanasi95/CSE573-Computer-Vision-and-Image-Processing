#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 23:22:29 2018

@author: manasikulkarni
"""

import cv2
import numpy as np

readImage = cv2.imread('/Users/manasikulkarni/Desktop/Task3/pos_11.jpg',cv2.IMREAD_GRAYSCALE)
template = cv2.imread('/Users/manasikulkarni/Desktop/template.png',cv2.IMREAD_GRAYSCALE)

resized_image = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
w, h= resized_image.shape[::-1]

mainImageBlur = cv2.GaussianBlur(readImage,(3,3),0)
templateImageBlur = cv2.GaussianBlur(resized_image,(3,3),0)

main_Laplacian = cv2.Laplacian(mainImageBlur,0)
template_Laplacian = cv2.Laplacian(templateImageBlur,0)
res = cv2.matchTemplate(main_Laplacian.astype(np.uint8),template_Laplacian.astype(np.uint8),cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(readImage, top_left, bottom_right, 255, 2)

cv2.imwrite('cursorDetection.jpg', readImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
