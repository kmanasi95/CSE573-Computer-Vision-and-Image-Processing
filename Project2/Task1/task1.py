#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:02:04 2018

@author: manasikulkarni

Referenced websites : 
1. https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
"""
import cv2
import numpy as np

UBIT = 'mkulkarn'
np.random.seed(sum([ord(c) for c in UBIT]))

mountain1_color = cv2.imread("/Users/manasikulkarni/Desktop/mountain1.jpg")
mountain2_color = cv2.imread("/Users/manasikulkarni/Desktop/mountain2.jpg")
 
mountain1 = cv2.imread("/Users/manasikulkarni/Desktop/mountain1.jpg", cv2.IMREAD_GRAYSCALE)
mountain2 = cv2.imread("/Users/manasikulkarni/Desktop/mountain2.jpg", cv2.IMREAD_GRAYSCALE)
    
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(mountain1, None) 
kp2, des2 = sift.detectAndCompute(mountain2, None)  
 
def drawKeypointsForBothImages():
    mountain1_kp=cv2.drawKeypoints(mountain1,kp1,mountain1)
    mountain2_kp=cv2.drawKeypoints(mountain2,kp2,mountain2)

    cv2.imwrite("task1_sift1.jpg", mountain1_kp)
    cv2.imwrite("task1_sift2.jpg", mountain2_kp)

def matchKeyPoints():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    mountain1_kp=cv2.drawKeypoints(mountain1,kp1,mountain1)
    mountain2_kp=cv2.drawKeypoints(mountain2,kp2,mountain2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    img3 = cv2.drawMatchesKnn(mountain1_kp,kp1,mountain2_kp,kp2,matches,None,flags=2)
    
    cv2.imwrite("task1_matches_knn.jpg", img3)
    
def computeHomographyMatrix():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
    print("Homography Matrix : ", H)
    
def match10Inliers():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    mountain1_kp=cv2.drawKeypoints(mountain1,kp1,mountain1)
    mountain2_kp=cv2.drawKeypoints(mountain2,kp2,mountain2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
           
    a = np.random.permutation(good)[:10]
    img4 = cv2.drawMatches(mountain1_kp,kp1,mountain2_kp,kp2,a,None,flags=0)
    
    cv2.imwrite("task1_matches.jpg",img4)

def Warping():
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
      
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    h1,w1,c1 = mountain1_color.shape
    h2,w2,c2 = mountain2_color.shape
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts1_new = cv2.perspectiveTransform(pts1, H)
    finalPoints = np.concatenate((pts2, pts1_new), axis=0)
    [xmin, ymin] = np.int32(finalPoints.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(finalPoints.max(axis=0).ravel() + 0.5)
    T = [-xmin,-ymin]
    HomoT = np.array([[1,0,T[0]],[0,1,T[1]],[0,0,1]])

    result = cv2.warpPerspective(mountain1_color, HomoT.dot(H), (xmax-xmin, ymax-ymin))
    result[T[1]:h2+T[1],T[0]:w2+T[0]] = mountain2_color
    cv2.imwrite("task1_pano.jpg", result)

drawKeypointsForBothImages()
matchKeyPoints()
computeHomographyMatrix()
match10Inliers()
Warping()
