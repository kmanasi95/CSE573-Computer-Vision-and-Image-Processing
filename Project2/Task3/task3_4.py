#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:33:55 2018

@author: manasikulkarni

Referenced websites:
1. https://stackoverflow.com/questions/26765875/python-pil-compare-colors/26768008
"""

import cv2
import numpy as np
import math
from numpy import mean

UBIT = 'mkulkarn'
np.random.seed(sum([ord(c) for c in UBIT]))

img1 = cv2.imread("/Users/manasikulkarni/Desktop/baboon.jpg")
img_copy3 = cv2.imread("/Users/manasikulkarni/Desktop/baboon.jpg")
img_copy5 = cv2.imread("/Users/manasikulkarni/Desktop/baboon.jpg")
img_copy10 = cv2.imread("/Users/manasikulkarni/Desktop/baboon.jpg")
img_copy20 = cv2.imread("/Users/manasikulkarni/Desktop/baboon.jpg")
height, width, channels = img1.shape

centroids3 = np.array([[0 for i in range(3)] for j in range(3)],dtype = np.int)
centroids5 = np.array([[0 for i in range(3)] for j in range(5)],dtype = np.int)
centroids10 = np.array([[0 for i in range(3)] for j in range(10)],dtype = np.int)
centroids20 = np.array([[0 for i in range(3)] for j in range(20)],dtype = np.int)

for i in range(0,3):
    rand1 = np.random.uniform(0,width-2,1)
    rand2 = np.random.uniform(0,height-2,1)
    
    centroids3[i] = img1[int(rand1[0])][int(rand2[0])]
    
for i in range(0,5):
    rand1 = np.random.uniform(0,width-2,1)
    rand2 = np.random.uniform(0,height-2,1)
    centroids5[i] = img1[int(rand1[0])][int(rand2[0])]
    
for i in range(0,10):
    rand1 = np.random.uniform(0,width-2,1)
    rand2 = np.random.uniform(0,height-2,1)
    centroids10[i] = img1[int(rand1[0])][int(rand2[0])]
    
for i in range(0,20):
    rand1 = np.random.uniform(0,width-2,1)
    rand2 = np.random.uniform(0,height-2,1)
    centroids20[i] = img1[int(rand1[0])][int(rand2[0])]
    
def checkLuminance(point):
    return (0.299 * point[0] + 0.587 * point[1] + 0.114 * point[2])
    
def checkSimilarity(pt1, pt2, threshold):
    return abs(checkLuminance(pt1) - checkLuminance(pt2)) < threshold

threshold = 10

def callColor3():
    Color3(centroids3[0],centroids3[1],centroids3[2])
    
def callColor5():
    Color5(centroids5[0],centroids5[1],centroids5[2],centroids5[3],centroids5[4])
    
def callColor10():
    Color10(centroids10[0],centroids10[1],centroids10[2],centroids10[3],centroids10[4],centroids10[5],centroids10[6],centroids10[7],centroids10[8],centroids10[9])
    
def callColor20():
    Color20(centroids20[0], centroids20[1], centroids20[2], centroids20[3], centroids20[4], centroids20[5], centroids20[6], centroids20[7], centroids20[8], centroids20[9], centroids20[10],centroids20[11],centroids20[12],centroids20[13],centroids20[14],centroids20[15],centroids20[16],centroids20[17],centroids20[18],centroids20[19])
    
list1 = []
list2 = []
list3 = []   
def Color3(centroids31, centroids32, centroids33):
    for x in range(0, height):
        for y in range(0, width):
            euc1 = math.sqrt((img1[x][y][0] - centroids31[0])*(img1[x][y][0] - centroids31[0]) + (img1[x][y][1] - centroids31[1])*(img1[x][y][1] - centroids31[1]) + (img1[x][y][2] - centroids31[2])*(img1[x][y][2] - centroids31[2]))
            euc2 = math.sqrt((img1[x][y][0] - centroids32[0])*(img1[x][y][0] - centroids32[0]) + (img1[x][y][1] - centroids32[1])*(img1[x][y][1] - centroids32[1]) + (img1[x][y][2] - centroids32[2])*(img1[x][y][2] - centroids32[2]))
            euc3 = math.sqrt((img1[x][y][0] - centroids33[0])*(img1[x][y][0] - centroids33[0]) + (img1[x][y][1] - centroids33[1])*(img1[x][y][1] - centroids33[1]) + (img1[x][y][2] - centroids33[2])*(img1[x][y][2] - centroids33[2]))
            if(euc1 == min(euc1, euc2, euc3)):
                r1,g1,b1 = centroids31
                img_copy3[x][y] = (r1,g1,b1)
                list1.append(img1[x][y])
            elif(euc2 == min(euc1, euc2, euc3)):
                r1,g1,b1 = centroids32
                img_copy3[x][y] = (r1,g1,b1)
                list2.append(img1[x][y])
            elif(euc3 == min(euc1, euc2, euc3)):
                r1,g1,b1 = centroids33
                img_copy3[x][y] = (r1,g1,b1)
                list3.append(img1[x][y])

    ReComputeCentroids3(img_copy3, list1,list2, list3, centroids3[0],centroids3[1],centroids3[2])

centroids31_old = np.array([0 for i in range(3)],dtype = np.int)
centroids32_old = np.array([0 for i in range(3)],dtype = np.int)
centroids33_old = np.array([0 for i in range(3)],dtype = np.int)
def ReComputeCentroids3(img_copy3, list1,list2, list3, centroids31,centroids32,centroids33):
    centroids31_old[0] = centroids31[0]
    centroids31_old[1] = centroids31[1]
    centroids31_old[2] = centroids31[2]
    centroids32_old[0] = centroids32[0]
    centroids32_old[1] = centroids32[1]
    centroids32_old[2] = centroids32[2]
    centroids33_old[0] = centroids33[0]
    centroids33_old[1] = centroids33[1]
    centroids33_old[2] = centroids33[2]

    a = mean(list1,axis=0)
    centroids31 = a  
    
    b = mean(list2,axis=0)
    centroids32 = b
    
    c = mean(list3,axis=0)
    centroids33 = c  
    
    if not checkSimilarity(centroids31_old, centroids31, threshold) or not(checkSimilarity(centroids32_old, centroids32, threshold)) or not(checkSimilarity(centroids33_old, centroids33, threshold)):
        Color3(centroids31,centroids32,centroids33)
        return
           
    cv2.imwrite("task3_baboon_3.jpg", img_copy3)

list51 = []
list52 = []
list53 = []
list54 = []
list55 = []   
def Color5(centroids51, centroids52, centroids53, centroids54, centroids55):
    for x in range(0, height):
        for y in range(0, width):
            euc1 = math.sqrt((img1[x][y][0] - centroids51[0])*(img1[x][y][0] - centroids51[0]) + (img1[x][y][1] - centroids51[1])*(img1[x][y][1] - centroids51[1]) + (img1[x][y][2] - centroids51[2])*(img1[x][y][2] - centroids51[2]))
            euc2 = math.sqrt((img1[x][y][0] - centroids52[0])*(img1[x][y][0] - centroids52[0]) + (img1[x][y][1] - centroids52[1])*(img1[x][y][1] - centroids52[1]) + (img1[x][y][2] - centroids52[2])*(img1[x][y][2] - centroids52[2]))
            euc3 = math.sqrt((img1[x][y][0] - centroids53[0])*(img1[x][y][0] - centroids53[0]) + (img1[x][y][1] - centroids53[1])*(img1[x][y][1] - centroids53[1]) + (img1[x][y][2] - centroids53[2])*(img1[x][y][2] - centroids53[2]))
            euc4 = math.sqrt((img1[x][y][0] - centroids54[0])*(img1[x][y][0] - centroids54[0]) + (img1[x][y][1] - centroids54[1])*(img1[x][y][1] - centroids54[1]) + (img1[x][y][2] - centroids54[2])*(img1[x][y][2] - centroids54[2]))
            euc5 = math.sqrt((img1[x][y][0] - centroids55[0])*(img1[x][y][0] - centroids55[0]) + (img1[x][y][1] - centroids55[1])*(img1[x][y][1] - centroids55[1]) + (img1[x][y][2] - centroids55[2])*(img1[x][y][2] - centroids55[2]))
            
            if(euc1 == min(euc1, euc2, euc3, euc4, euc5)):
                r1,g1,b1 = centroids51
                img_copy5[x][y] = (r1,g1,b1)
                list51.append(img1[x][y])
            elif(euc2 == min(euc1, euc2, euc3, euc4, euc5)):
                r1,g1,b1 = centroids52
                img_copy5[x][y] = (r1,g1,b1)
                list52.append(img1[x][y])
            elif(euc3 == min(euc1, euc2, euc3, euc4, euc5)):
                r1,g1,b1 = centroids53
                img_copy5[x][y] = (r1,g1,b1)
                list53.append(img1[x][y])
            elif(euc4 == min(euc1, euc2, euc3, euc4, euc5)):
                r1,g1,b1 = centroids54
                img_copy5[x][y] = (r1,g1,b1)
                list54.append(img1[x][y])
            elif(euc5 == min(euc1, euc2, euc3, euc4, euc5)):
                r1,g1,b1 = centroids55
                img_copy5[x][y] = (r1,g1,b1)
                list55.append(img1[x][y])

    ReComputeCentroids5(img_copy5, list51,list52, list53, list54, list55, centroids51,centroids52,centroids53,centroids54,centroids55)

centroids51_old = np.array([0 for i in range(5)],dtype = np.int)
centroids52_old = np.array([0 for i in range(5)],dtype = np.int)
centroids53_old = np.array([0 for i in range(5)],dtype = np.int) 
centroids54_old = np.array([0 for i in range(5)],dtype = np.int) 
centroids55_old = np.array([0 for i in range(5)],dtype = np.int)    
def ReComputeCentroids5(img_copy5, list51,list52, list53, list54, list55, centroids51,centroids52,centroids53,centroids54,centroids55):
    centroids51_old[0] = centroids51[0]
    centroids51_old[1] = centroids51[1]
    centroids51_old[2] = centroids51[2]
    centroids52_old[0] = centroids52[0]
    centroids52_old[1] = centroids52[1]
    centroids52_old[2] = centroids52[2]
    centroids53_old[0] = centroids53[0]
    centroids53_old[1] = centroids53[1]
    centroids53_old[2] = centroids53[2]
    centroids54_old[0] = centroids54[0]
    centroids54_old[1] = centroids54[1]
    centroids54_old[2] = centroids54[2]
    centroids55_old[0] = centroids55[0]
    centroids55_old[1] = centroids55[1]
    centroids55_old[2] = centroids55[2]
    
    a = mean(list51,axis=0)
    centroids51 = a   
    
    b = mean(list52,axis=0)
    centroids52 = b  
    
    c = mean(list53,axis=0)
    centroids53 = c   
    
    d = mean(list54,axis=0)
    centroids54 = d   
    
    e = mean(list55,axis=0)
    centroids55 = e   
    
    if not checkSimilarity(centroids51_old, centroids51, threshold) or not(checkSimilarity(centroids52_old, centroids52, threshold)) or not(checkSimilarity(centroids53_old, centroids53, threshold)) or not(checkSimilarity(centroids54_old, centroids54, threshold)) or not(checkSimilarity(centroids55_old, centroids55, threshold)):
        Color5(centroids51,centroids52,centroids53,centroids54,centroids55)
        return
       
    cv2.imwrite("task3_baboon_5.jpg", img_copy5)


list101 = []
list102 = []
list103 = []
list104 = []
list105 = []
list106 = []
list107 = []
list108 = []
list109 = []
list100 = []   
def Color10(centroids101, centroids102, centroids103, centroids104, centroids105, centroids106, centroids107, centroids108, centroids109, centroids100):
    for x in range(0, height):
        for y in range(0, width):
            euc1 = math.sqrt((img1[x][y][0] - centroids101[0])*(img1[x][y][0] - centroids101[0]) + (img1[x][y][1] - centroids101[1])*(img1[x][y][1] - centroids101[1]) + (img1[x][y][2] - centroids101[2])*(img1[x][y][2] - centroids101[2]))
            euc2 = math.sqrt((img1[x][y][0] - centroids102[0])*(img1[x][y][0] - centroids102[0]) + (img1[x][y][1] - centroids102[1])*(img1[x][y][1] - centroids102[1]) + (img1[x][y][2] - centroids102[2])*(img1[x][y][2] - centroids102[2]))
            euc3 = math.sqrt((img1[x][y][0] - centroids103[0])*(img1[x][y][0] - centroids103[0]) + (img1[x][y][1] - centroids103[1])*(img1[x][y][1] - centroids103[1]) + (img1[x][y][2] - centroids103[2])*(img1[x][y][2] - centroids103[2]))
            euc4 = math.sqrt((img1[x][y][0] - centroids104[0])*(img1[x][y][0] - centroids104[0]) + (img1[x][y][1] - centroids104[1])*(img1[x][y][1] - centroids104[1]) + (img1[x][y][2] - centroids104[2])*(img1[x][y][2] - centroids104[2]))
            euc5 = math.sqrt((img1[x][y][0] - centroids105[0])*(img1[x][y][0] - centroids105[0]) + (img1[x][y][1] - centroids105[1])*(img1[x][y][1] - centroids105[1]) + (img1[x][y][2] - centroids105[2])*(img1[x][y][2] - centroids105[2]))
            euc6 = math.sqrt((img1[x][y][0] - centroids106[0])*(img1[x][y][0] - centroids106[0]) + (img1[x][y][1] - centroids106[1])*(img1[x][y][1] - centroids106[1]) + (img1[x][y][2] - centroids106[2])*(img1[x][y][2] - centroids106[2]))
            euc7 = math.sqrt((img1[x][y][0] - centroids107[0])*(img1[x][y][0] - centroids107[0]) + (img1[x][y][1] - centroids107[1])*(img1[x][y][1] - centroids107[1]) + (img1[x][y][2] - centroids107[2])*(img1[x][y][2] - centroids107[2]))
            euc8 = math.sqrt((img1[x][y][0] - centroids108[0])*(img1[x][y][0] - centroids108[0]) + (img1[x][y][1] - centroids108[1])*(img1[x][y][1] - centroids108[1]) + (img1[x][y][2] - centroids108[2])*(img1[x][y][2] - centroids108[2]))
            euc9 = math.sqrt((img1[x][y][0] - centroids109[0])*(img1[x][y][0] - centroids109[0]) + (img1[x][y][1] - centroids109[1])*(img1[x][y][1] - centroids109[1]) + (img1[x][y][2] - centroids109[2])*(img1[x][y][2] - centroids109[2]))
            euc0 = math.sqrt((img1[x][y][0] - centroids100[0])*(img1[x][y][0] - centroids100[0]) + (img1[x][y][1] - centroids100[1])*(img1[x][y][1] - centroids100[1]) + (img1[x][y][2] - centroids100[2])*(img1[x][y][2] - centroids100[2]))
            if(euc1 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids101
                img_copy10[x][y] = (r1,g1,b1)
                list101.append(img1[x][y])
            elif(euc2 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids102
                img_copy10[x][y] = (r1,g1,b1)
                list102.append(img1[x][y])
            elif(euc3 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids103
                img_copy10[x][y] = (r1,g1,b1)
                list103.append(img1[x][y])
                #print(img1[x][y]) 
            elif(euc4 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids104
                img_copy10[x][y] = (r1,g1,b1)
                list104.append(img1[x][y])
            elif(euc5 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids105
                img_copy10[x][y] = (r1,g1,b1)
                list105.append(img1[x][y])
            elif(euc6 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids106
                img_copy10[x][y] = (r1,g1,b1)
                list106.append(img1[x][y])
            elif(euc7 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids107
                img_copy10[x][y] = (r1,g1,b1)
                list107.append(img1[x][y])
            elif(euc8 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids108
                img_copy10[x][y] = (r1,g1,b1)
                list108.append(img1[x][y])
            elif(euc9 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids109
                img_copy10[x][y] = (r1,g1,b1)
                list109.append(img1[x][y])
            elif(euc0 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc0)):
                r1,g1,b1 = centroids100
                img_copy10[x][y] = (r1,g1,b1)
                list100.append(img1[x][y])

    ReComputeCentroids10(img_copy10, list101,list102, list103, list104, list105, list106, list107, list108, list109, list100, centroids101,centroids102,centroids103,centroids104,centroids105,centroids106,centroids107,centroids108,centroids109,centroids100)

centroids101_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids102_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids103_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids104_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids105_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids106_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids107_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids108_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids109_old = np.array([0 for i in range(10)],dtype = np.int)    
centroids100_old = np.array([0 for i in range(10)],dtype = np.int)    
def ReComputeCentroids10(img_copy10, list101,list102, list103, list104, list105, list106, list107, list108, list109, list100, centroids101,centroids102,centroids103,centroids104,centroids105,centroids106,centroids107,centroids108,centroids109,centroids100):
    centroids101_old[0] = centroids101[0]
    centroids101_old[1] = centroids101[1]
    centroids101_old[2] = centroids101[2]
    centroids102_old[0] = centroids102[0]
    centroids102_old[1] = centroids102[1]
    centroids102_old[2] = centroids102[2]
    centroids103_old[0] = centroids103[0]
    centroids103_old[1] = centroids103[1]
    centroids103_old[2] = centroids103[2]
    centroids104_old[0] = centroids104[0]
    centroids104_old[1] = centroids104[1]
    centroids104_old[2] = centroids104[2]
    centroids105_old[0] = centroids105[0]
    centroids105_old[1] = centroids105[1]
    centroids105_old[2] = centroids105[2]
    centroids106_old[0] = centroids106[0]
    centroids106_old[1] = centroids106[1]
    centroids106_old[2] = centroids106[2]
    centroids107_old[0] = centroids107[0]
    centroids107_old[1] = centroids107[1]
    centroids107_old[2] = centroids107[2]
    centroids108_old[0] = centroids108[0]
    centroids108_old[1] = centroids108[1]
    centroids108_old[2] = centroids108[2]
    centroids109_old[0] = centroids109[0]
    centroids109_old[1] = centroids109[1]
    centroids109_old[2] = centroids109[2]
    centroids100_old[0] = centroids100[0]
    centroids100_old[1] = centroids100[1]
    centroids100_old[2] = centroids100[2]
    
    a1 = mean(list101,axis=0)
    centroids101 = a1
    
    a2 = mean(list102,axis=0)
    centroids102 = a2   
    
    a3 = mean(list103,axis=0)
    centroids103 = a3  
    
    a4 = mean(list104,axis=0)
    centroids104 = a4  
    
    a5 = mean(list105,axis=0)
    centroids105 = a5   
    
    a6 = mean(list106,axis=0)
    centroids106 = a6   
    
    a7 = mean(list107,axis=0)
    centroids107 = a7   
    
    a8 = mean(list108,axis=0)
    centroids108 = a8   
    
    a9 = mean(list109,axis=0)
    centroids109 = a9   
    
    a0 = mean(list100,axis=0)
    centroids100 = a0 
    
    if not checkSimilarity(centroids101_old, centroids101, threshold) or not(checkSimilarity(centroids102_old, centroids102, threshold)) or not(checkSimilarity(centroids103_old, centroids103, threshold)) or not(checkSimilarity(centroids104_old, centroids104, threshold)) or not(checkSimilarity(centroids105_old, centroids105, threshold)) or not(checkSimilarity(centroids106_old, centroids106, threshold)) or not(checkSimilarity(centroids107_old, centroids107, threshold)) or not(checkSimilarity(centroids108_old, centroids108, threshold)) or not(checkSimilarity(centroids109_old, centroids109, threshold)) or not(checkSimilarity(centroids100_old, centroids100, threshold)):
        Color10(centroids101,centroids102,centroids103,centroids104,centroids105,centroids106,centroids107,centroids108,centroids109,centroids100)
        return
        
    cv2.imwrite("task3_baboon_10.jpg", img_copy10)

list201 = []
list202 = []
list203 = []
list204 = []
list205 = []
list206 = []
list207 = []
list208 = []
list209 = []
list210 = []
list211 = []
list212 = []
list213 = []
list214 = []
list215 = []
list216 = []
list217 = []
list218 = []
list219 = []
list220 = []
def Color20(centroids201, centroids202, centroids203, centroids204, centroids205, centroids206, centroids207, centroids208, centroids209, centroids210, centroids211,centroids212,centroids213,centroids214,centroids215,centroids216,centroids217,centroids218,centroids219,centroids220):
    for x in range(0, height):
        for y in range(0, width):
            euc1 = math.sqrt((img1[x][y][0] - centroids201[0])*(img1[x][y][0] - centroids201[0]) + (img1[x][y][1] - centroids201[1])*(img1[x][y][1] - centroids201[1]) + (img1[x][y][2] - centroids201[2])*(img1[x][y][2] - centroids201[2]))
            euc2 = math.sqrt((img1[x][y][0] - centroids202[0])*(img1[x][y][0] - centroids202[0]) + (img1[x][y][1] - centroids202[1])*(img1[x][y][1] - centroids202[1]) + (img1[x][y][2] - centroids202[2])*(img1[x][y][2] - centroids202[2]))
            euc3 = math.sqrt((img1[x][y][0] - centroids203[0])*(img1[x][y][0] - centroids203[0]) + (img1[x][y][1] - centroids203[1])*(img1[x][y][1] - centroids203[1]) + (img1[x][y][2] - centroids203[2])*(img1[x][y][2] - centroids203[2]))
            euc4 = math.sqrt((img1[x][y][0] - centroids204[0])*(img1[x][y][0] - centroids204[0]) + (img1[x][y][1] - centroids204[1])*(img1[x][y][1] - centroids204[1]) + (img1[x][y][2] - centroids204[2])*(img1[x][y][2] - centroids204[2]))
            euc5 = math.sqrt((img1[x][y][0] - centroids205[0])*(img1[x][y][0] - centroids205[0]) + (img1[x][y][1] - centroids205[1])*(img1[x][y][1] - centroids205[1]) + (img1[x][y][2] - centroids205[2])*(img1[x][y][2] - centroids205[2]))
            euc6 = math.sqrt((img1[x][y][0] - centroids206[0])*(img1[x][y][0] - centroids206[0]) + (img1[x][y][1] - centroids206[1])*(img1[x][y][1] - centroids206[1]) + (img1[x][y][2] - centroids206[2])*(img1[x][y][2] - centroids206[2]))
            euc7 = math.sqrt((img1[x][y][0] - centroids207[0])*(img1[x][y][0] - centroids207[0]) + (img1[x][y][1] - centroids207[1])*(img1[x][y][1] - centroids207[1]) + (img1[x][y][2] - centroids207[2])*(img1[x][y][2] - centroids207[2]))
            euc8 = math.sqrt((img1[x][y][0] - centroids208[0])*(img1[x][y][0] - centroids208[0]) + (img1[x][y][1] - centroids208[1])*(img1[x][y][1] - centroids208[1]) + (img1[x][y][2] - centroids208[2])*(img1[x][y][2] - centroids208[2]))
            euc9 = math.sqrt((img1[x][y][0] - centroids209[0])*(img1[x][y][0] - centroids209[0]) + (img1[x][y][1] - centroids209[1])*(img1[x][y][1] - centroids209[1]) + (img1[x][y][2] - centroids209[2])*(img1[x][y][2] - centroids209[2]))
            euc10 = math.sqrt((img1[x][y][0] - centroids210[0])*(img1[x][y][0] - centroids210[0]) + (img1[x][y][1] - centroids210[1])*(img1[x][y][1] - centroids210[1]) + (img1[x][y][2] - centroids210[2])*(img1[x][y][2] - centroids210[2]))
            euc11 = math.sqrt((img1[x][y][0] - centroids211[0])*(img1[x][y][0] - centroids211[0]) + (img1[x][y][1] - centroids211[1])*(img1[x][y][1] - centroids211[1]) + (img1[x][y][2] - centroids211[2])*(img1[x][y][2] - centroids211[2]))
            euc12 = math.sqrt((img1[x][y][0] - centroids212[0])*(img1[x][y][0] - centroids212[0]) + (img1[x][y][1] - centroids212[1])*(img1[x][y][1] - centroids212[1]) + (img1[x][y][2] - centroids212[2])*(img1[x][y][2] - centroids212[2]))
            euc13 = math.sqrt((img1[x][y][0] - centroids213[0])*(img1[x][y][0] - centroids213[0]) + (img1[x][y][1] - centroids213[1])*(img1[x][y][1] - centroids213[1]) + (img1[x][y][2] - centroids213[2])*(img1[x][y][2] - centroids213[2]))
            euc14 = math.sqrt((img1[x][y][0] - centroids214[0])*(img1[x][y][0] - centroids214[0]) + (img1[x][y][1] - centroids214[1])*(img1[x][y][1] - centroids214[1]) + (img1[x][y][2] - centroids214[2])*(img1[x][y][2] - centroids214[2]))
            euc15 = math.sqrt((img1[x][y][0] - centroids215[0])*(img1[x][y][0] - centroids215[0]) + (img1[x][y][1] - centroids215[1])*(img1[x][y][1] - centroids215[1]) + (img1[x][y][2] - centroids215[2])*(img1[x][y][2] - centroids215[2]))
            euc16 = math.sqrt((img1[x][y][0] - centroids216[0])*(img1[x][y][0] - centroids216[0]) + (img1[x][y][1] - centroids216[1])*(img1[x][y][1] - centroids216[1]) + (img1[x][y][2] - centroids216[2])*(img1[x][y][2] - centroids216[2]))
            euc17 = math.sqrt((img1[x][y][0] - centroids217[0])*(img1[x][y][0] - centroids217[0]) + (img1[x][y][1] - centroids217[1])*(img1[x][y][1] - centroids217[1]) + (img1[x][y][2] - centroids217[2])*(img1[x][y][2] - centroids217[2]))
            euc18 = math.sqrt((img1[x][y][0] - centroids218[0])*(img1[x][y][0] - centroids218[0]) + (img1[x][y][1] - centroids218[1])*(img1[x][y][1] - centroids218[1]) + (img1[x][y][2] - centroids218[2])*(img1[x][y][2] - centroids218[2]))
            euc19 = math.sqrt((img1[x][y][0] - centroids219[0])*(img1[x][y][0] - centroids219[0]) + (img1[x][y][1] - centroids219[1])*(img1[x][y][1] - centroids219[1]) + (img1[x][y][2] - centroids219[2])*(img1[x][y][2] - centroids219[2]))
            euc20 = math.sqrt((img1[x][y][0] - centroids220[0])*(img1[x][y][0] - centroids220[0]) + (img1[x][y][1] - centroids220[1])*(img1[x][y][1] - centroids220[1]) + (img1[x][y][2] - centroids220[2])*(img1[x][y][2] - centroids220[2]))
            
            if(euc1 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids201
                img_copy20[x][y] = (r1,g1,b1)
                list201.append(img1[x][y])
            elif(euc2 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids202
                img_copy20[x][y] = (r1,g1,b1)
                list202.append(img1[x][y])
            elif(euc3 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids203
                img_copy20[x][y] = (r1,g1,b1)
                list203.append(img1[x][y])
                #print(img1[x][y]) 
            elif(euc4 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids204
                img_copy20[x][y] = (r1,g1,b1)
                list204.append(img1[x][y])
            elif(euc5 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids205
                img_copy20[x][y] = (r1,g1,b1)
                list205.append(img1[x][y])
            elif(euc6 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids206
                img_copy20[x][y] = (r1,g1,b1)
                list206.append(img1[x][y])
            elif(euc7 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids207
                img_copy20[x][y] = (r1,g1,b1)
                list207.append(img1[x][y])
            elif(euc8 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids208
                img_copy20[x][y] = (r1,g1,b1)
                list208.append(img1[x][y])
            elif(euc9 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids209
                img_copy20[x][y] = (r1,g1,b1)
                list209.append(img1[x][y])
            elif(euc10 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids210
                img_copy20[x][y] = (r1,g1,b1)
                list210.append(img1[x][y])
            elif(euc11 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids211
                img_copy20[x][y] = (r1,g1,b1)
                list211.append(img1[x][y])
            elif(euc12 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids212
                img_copy20[x][y] = (r1,g1,b1)
                list212.append(img1[x][y])
            elif(euc13 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids213
                img_copy20[x][y] = (r1,g1,b1)
                list213.append(img1[x][y])
            elif(euc14 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids214
                img_copy20[x][y] = (r1,g1,b1)
                list214.append(img1[x][y])
            elif(euc15 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids215
                img_copy20[x][y] = (r1,g1,b1)
                list215.append(img1[x][y])
            elif(euc16 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids216
                img_copy20[x][y] = (r1,g1,b1)
                list216.append(img1[x][y])
            elif(euc17 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids217
                img_copy20[x][y] = (r1,g1,b1)
                list217.append(img1[x][y])
            elif(euc18 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids218
                img_copy20[x][y] = (r1,g1,b1)
                list218.append(img1[x][y])
            elif(euc19 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids219
                img_copy20[x][y] = (r1,g1,b1)
                list219.append(img1[x][y])
            elif(euc20 == min(euc1, euc2, euc3, euc4, euc5, euc6, euc7, euc8, euc9, euc10,euc11,euc12,euc13,euc14,euc15,euc16,euc17,euc18,euc19,euc20)):
                r1,g1,b1 = centroids220
                img_copy20[x][y] = (r1,g1,b1)
                list220.append(img1[x][y])
                
    ReComputeCentroids20(img_copy20, list201,list202, list203, list204, list205, list206, list207, list208, list209, list210,list211,list212,list213,list214,list215,list216,list217,list218,list219,list220,centroids201,centroids202,centroids203,centroids204,centroids205,centroids206,centroids207,centroids208,centroids209,centroids210,centroids211,centroids212,centroids213,centroids214,centroids215,centroids216,centroids217,centroids218,centroids219,centroids220)

centroids201_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids202_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids203_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids204_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids205_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids206_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids207_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids208_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids209_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids210_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids211_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids212_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids213_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids214_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids215_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids216_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids217_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids218_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids219_old = np.array([0 for i in range(20)],dtype = np.int) 
centroids220_old = np.array([0 for i in range(20)],dtype = np.int) 
def ReComputeCentroids20(img_copy20, list201,list202, list203, list204, list205, list206, list207, list208, list209, list210,list211,list212,list213,list214,list215,list216,list217,list218,list219,list220,centroids201,centroids202,centroids203,centroids204,centroids205,centroids206,centroids207,centroids208,centroids209,centroids210,centroids211,centroids212,centroids213,centroids214,centroids215,centroids216,centroids217,centroids218,centroids219,centroids220):
    centroids201_old[0] = centroids201[0]
    centroids201_old[1] = centroids201[1]
    centroids201_old[2] = centroids201[2]
    centroids202_old[0] = centroids202[0]
    centroids202_old[1] = centroids202[1]
    centroids202_old[2] = centroids202[2]
    centroids203_old[0] = centroids203[0]
    centroids203_old[1] = centroids203[1]
    centroids203_old[2] = centroids203[2]
    centroids204_old[0] = centroids204[0]
    centroids204_old[1] = centroids204[1]
    centroids204_old[2] = centroids204[2]
    centroids205_old[0] = centroids205[0]
    centroids205_old[1] = centroids205[1]
    centroids205_old[2] = centroids205[2]
    centroids206_old[0] = centroids206[0]
    centroids206_old[1] = centroids206[1]
    centroids206_old[2] = centroids206[2]
    centroids207_old[0] = centroids207[0]
    centroids207_old[1] = centroids207[1]
    centroids207_old[2] = centroids207[2]
    centroids208_old[0] = centroids208[0]
    centroids208_old[1] = centroids208[1]
    centroids208_old[2] = centroids208[2]
    centroids209_old[0] = centroids209[0]
    centroids209_old[1] = centroids209[1]
    centroids209_old[2] = centroids209[2]
    centroids210_old[0] = centroids210[0]
    centroids210_old[1] = centroids210[1]
    centroids210_old[2] = centroids210[2]
    centroids211_old[0] = centroids211[0]
    centroids211_old[1] = centroids211[1]
    centroids211_old[2] = centroids211[2]
    centroids212_old[0] = centroids212[0]
    centroids212_old[1] = centroids212[1]
    centroids212_old[2] = centroids212[2]
    centroids213_old[0] = centroids213[0]
    centroids213_old[1] = centroids213[1]
    centroids213_old[2] = centroids213[2]
    centroids214_old[0] = centroids214[0]
    centroids214_old[1] = centroids214[1]
    centroids214_old[2] = centroids214[2]
    centroids215_old[0] = centroids215[0]
    centroids215_old[1] = centroids215[1]
    centroids215_old[2] = centroids215[2]
    centroids216_old[0] = centroids216[0]
    centroids216_old[1] = centroids216[1]
    centroids216_old[2] = centroids216[2]
    centroids217_old[0] = centroids217[0]
    centroids217_old[1] = centroids217[1]
    centroids217_old[2] = centroids217[2]
    centroids218_old[0] = centroids218[0]
    centroids218_old[1] = centroids218[1]
    centroids218_old[2] = centroids218[2]
    centroids219_old[0] = centroids219[0]
    centroids219_old[1] = centroids219[1]
    centroids219_old[2] = centroids219[2]
    centroids220_old[0] = centroids220[0]
    centroids220_old[1] = centroids220[1]
    centroids220_old[2] = centroids220[2]
    
    a1 = mean(list201,axis=0)
    centroids201 = a1
    
    a2 = mean(list202,axis=0)
    centroids202 = a2   
    
    a3 = mean(list203,axis=0)
    centroids203 = a3  
    
    a4 = mean(list204,axis=0)
    centroids204 = a4  
    
    a5 = mean(list205,axis=0)
    centroids205 = a5   
    
    a6 = mean(list206,axis=0)
    centroids206 = a6   
    
    a7 = mean(list207,axis=0)
    centroids207 = a7   
    
    a8 = mean(list208,axis=0)
    centroids208 = a8   
    
    a9 = mean(list209,axis=0)
    centroids209 = a9   
    
    a10 = mean(list210,axis=0)
    centroids210 = a10 
    
    a11 = mean(list211,axis=0)
    centroids211 = a11
    
    a12 = mean(list212,axis=0)
    centroids212 = a12   
    
    a13 = mean(list213,axis=0)
    centroids213 = a13  
    
    a14 = mean(list214,axis=0)
    centroids214 = a14  
    
    a15 = mean(list215,axis=0)
    centroids215 = a15   
    
    a16 = mean(list216,axis=0)
    centroids216 = a16   
    
    a17 = mean(list217,axis=0)
    centroids217 = a17   
    
    a18 = mean(list218,axis=0)
    centroids218 = a18   
    
    a19 = mean(list219,axis=0)
    centroids219 = a19   
    
    a20 = mean(list220,axis=0)
    centroids220 = a20 
    
    if not checkSimilarity(centroids201_old, centroids201, threshold) or not(checkSimilarity(centroids202_old, centroids202, threshold)) or not(checkSimilarity(centroids203_old, centroids203, threshold)) or not(checkSimilarity(centroids204_old, centroids204, threshold)) or not(checkSimilarity(centroids205_old, centroids205, threshold)) or not(checkSimilarity(centroids206_old, centroids206, threshold)) or not(checkSimilarity(centroids207_old, centroids207, threshold)) or not(checkSimilarity(centroids208_old, centroids208, threshold)) or not(checkSimilarity(centroids209_old, centroids209, threshold)) or not(checkSimilarity(centroids210_old, centroids210, threshold)) or not(checkSimilarity(centroids211_old, centroids211, threshold)) or not(checkSimilarity(centroids212_old, centroids212, threshold)) or not(checkSimilarity(centroids213_old, centroids213, threshold)) or not(checkSimilarity(centroids214_old, centroids214, threshold)) or not(checkSimilarity(centroids215_old, centroids215, threshold)) or not(checkSimilarity(centroids216_old, centroids216, threshold)) or not(checkSimilarity(centroids217_old, centroids217, threshold)) or not(checkSimilarity(centroids218_old, centroids218, threshold)) or not(checkSimilarity(centroids219_old, centroids219, threshold)) or not(checkSimilarity(centroids220_old, centroids220, threshold)):
        Color20(centroids201, centroids202, centroids203, centroids204, centroids205, centroids206, centroids207, centroids208, centroids209, centroids210, centroids211,centroids212,centroids213,centroids214,centroids215,centroids216,centroids217,centroids218,centroids219,centroids220)
        return
         
    cv2.imwrite("task3_baboon_20.jpg", img_copy20)