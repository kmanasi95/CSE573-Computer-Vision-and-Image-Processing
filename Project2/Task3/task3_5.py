#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:19:37 2018

@author: manasikulkarni
"""

import cv2
from scipy.stats import multivariate_normal

X = [[5.9, 3.2],[4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5, 4.2],[5.0, 3.0],[4.9, 3.1],[6.7, 3.1],[5.1, 3.8],[6.0, 3.0]]

mean1 = [6.2, 3.2]
mean2 = [6.6, 3.7]
mean3 = [6.5, 3.0]

cov1 = [[0.5, 0],[0, 0.5]]
cov2 = [[0.5, 0],[0, 0.5]]
cov3 = [[0.5, 0],[0, 0.5]]
     
pi1 = 1/3
pi2 = 1/3
pi3 = 1/3

N1 = multivariate_normal.pdf(X, mean1, cov1)
N2 = multivariate_normal.pdf(X, mean2, cov2)
N3 = multivariate_normal.pdf(X, mean3, cov3)

Numerator1 = N1 * pi1
Numerator2 = N2 * pi2
Numerator3 = N3 * pi3

Denominator = (pi1 * N1) + (pi2 * N2) + (pi3 * N3)

ResponsibilityMat1 = Numerator1 / Denominator
ResponsibilityMat2 = Numerator2 / Denominator
ResponsibilityMat3 = Numerator3 / Denominator

def ReComputeMeanValues():
    list1 = []
    list2 = []
    for i in X:
        list1.append(i[0])
        list2.append(i[1])

    a = []
    b = []
    c = []
    d = []
    e = []
    f = []
    for i in range(0,10):
        a.append(list1[i] * ResponsibilityMat1[i])
        b.append(list2[i] * ResponsibilityMat1[i])
        c.append(list1[i] * ResponsibilityMat2[i])
        d.append(list2[i] * ResponsibilityMat2[i])
        e.append(list1[i] * ResponsibilityMat3[i])
        f.append(list2[i] * ResponsibilityMat3[i])

    mean1_new = [sum(a) / sum(ResponsibilityMat1),sum(b) / sum(ResponsibilityMat1)]
    mean2_new = [sum(c) / sum(ResponsibilityMat2),sum(d) / sum(ResponsibilityMat2)]
    mean3_new = [sum(e) / sum(ResponsibilityMat3),sum(f) / sum(ResponsibilityMat3)]

    print("Recomputed Mean1 : ", mean1_new)
    print("Recomputed Mean2 : ", mean2_new)
    print("Recomputed Mean3 : ", mean3_new)