#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:16:11 2018

@author: manasikulkarni

Referenced websites : 
1. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
2. https://www.programcreek.com/python/example/89385/cv2.StereoBM_create
"""

import cv2
import numpy as np
#import random

UBIT = 'mkulkarn'
np.random.seed(sum([ord(c) for c in UBIT]))
     
img1 = cv2.imread("/Users/manasikulkarni/Desktop/tsucuba_left.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/Users/manasikulkarni/Desktop/tsucuba_right.png", cv2.IMREAD_GRAYSCALE)
height1, width1 = img1.shape
height2, width2 = img2.shape
    
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None) 
kp2, des2 = sift.detectAndCompute(img2, None)  

def drawKeypointsForBothImages():
    img1_kp=cv2.drawKeypoints(img1,kp1,img1)
    img2_kp=cv2.drawKeypoints(img2,kp2,img2)
    cv2.imwrite('task2_sift1.jpg',img1_kp)
    cv2.imwrite('task2_sift2.jpg',img2_kp)

def matchKeypoints():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    img1_kp=cv2.drawKeypoints(img1,kp1,img1)
    img2_kp=cv2.drawKeypoints(img2,kp2,img2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
        
        img3 = cv2.drawMatches(img1_kp,kp1,img2_kp,kp2,good,None,flags=2)
        cv2.imwrite('task2_matches_knn.jpg',img3)
        
def computeFundamentalMatrix():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print("Fundamental Matrix : ", F)

def drawlines(image1, image2, lines, pts1, pts2):
    r, c = image1.shape
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1] ])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
        image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
        image1 = cv2.circle(image1, tuple(pt1), 5, color, -1)
        image2 = cv2.circle(image2, tuple(pt2), 5, color, -1)
    return image1, image2

def drawEpilines():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    matches_random = np.random.permutation(matches)[:10]

    good_random = []
    pts1_rand = []
    pts2_rand = []
    for m,n in matches_random:
        good_random.append(m)
        pts1_rand.append(kp1[m.queryIdx].pt)
        pts2_rand.append(kp2[m.trainIdx].pt)
        
    pts1_rand = np.int32(pts1_rand)
    pts2_rand = np.int32(pts2_rand)
    F1, mask1 = cv2.findFundamentalMat(pts1_rand,pts2_rand,cv2.FM_LMEDS)

    linesLeft = cv2.computeCorrespondEpilines(pts2_rand.reshape(-1, 1, 2), 2, F1)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, linesLeft, pts1_rand, pts2_rand)

    linesRight = cv2.computeCorrespondEpilines(pts1_rand.reshape(-1, 1, 2), 1, F1)
    linesRight = linesRight.reshape(-1, 3) 
    img7, img8 = drawlines(img2, img1, linesRight, pts2_rand, pts1_rand)

    cv2.imwrite('task2_epi_left.jpg',img5)
    cv2.imwrite('task2_epi_right.jpg',img7)

def computeDisparityMap():
    stereobm = cv2.StereoBM_create(numDisparities=64, blockSize=31)
    stereobm.setSpeckleWindowSize(100)
    stereobm.setSpeckleRange(20)
    disparity = stereobm.compute(img1, img2)

    cv2.imwrite('task2_disparity.jpg', (disparity/2048) * 255)

drawKeypointsForBothImages()
matchKeypoints()
computeFundamentalMatrix()
drawEpilines()
computeDisparityMap()
