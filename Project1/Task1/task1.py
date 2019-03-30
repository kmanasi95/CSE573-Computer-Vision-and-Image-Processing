#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:44:38 2018

@author: manasikulkarni
"""

import cv2
import math
import numpy

readImage = cv2.imread("/Users/manasikulkarni/Desktop/task1.png",cv2.IMREAD_GRAYSCALE)
imgArray = numpy.asarray(readImage)
row = len(imgArray)
col = len(imgArray[0])

newimgX = numpy.array([[0 for i in range(int(col+2))] for j in range(int(row+2) )],dtype = numpy.float)
newimgY = numpy.array([[0 for i in range(int(col+2))] for j in range(int(row+2) )],dtype = numpy.float)
newimgCommon = numpy.array([[0 for i in range(int(col+2))] for j in range(int(row+2) )],dtype = numpy.float)

for i in range(0,row+1):
    for j in range(0,col+1):
        if i == 0 or j==0 or i >= col or j >= row :
            newimgX[i,j] = 0
            newimgY[i,j] = 0
            newimgCommon[i,j] = 0
        else: 
            newimgX[i,j] = readImage[i-1,j-1]
            newimgY[i,j] = readImage[i-1,j-1]
            newimgCommon[i,j] = readImage[i-1,j-1]
            
pixel = readImage[0,0]
        
def pixel00(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return (-1 *(pixel))
    elif val == 1:
        return (-1 *(pixel))
    
    
def pixel01(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return (-2 * (pixel))
    elif val == 1:
        return 0
    
def pixel02(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return (-1 * (pixel))
    elif val == 1:
        return pixel
    
def pixel10(i,j,val):
    if val == 0:
        return 0
    elif val == 1:
        return (-2 * (pixel))

def pixel11(i,j,val):
    if val == 0:
        return 0
    elif val == 1:
        return 0
    
def pixel12(i,j,val):
    if val == 0:
        return 0
    elif val == 1:
        return (2 * (pixel))
    
def pixel20(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return pixel
    elif val == 1:
        return (-1 * (pixel))
    
def pixel21(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return (2 * (pixel))
    elif val == 1:
        return 0
      
def pixel22(i,j,val):
    pixel = readImage[i,j]
    if val == 0:
        return  pixel
    elif val == 1:
        return  pixel
    
for x in range(1, row-1):  
    for y in range(1, col-1): 

        gradientX = 0
        gradientY = 0

        gradientX += pixel00(x-1,y-1,0)
        gradientX += pixel10(x-1,y,0)
        gradientX += pixel20(x-1,y+1,0)
        gradientX += pixel01(x,y-1,0)
        gradientX += pixel11(x,y,0)
        gradientX += pixel21(x,y+1,0)
        gradientX += pixel02(x+1,y-1,0)
        gradientX += pixel12(x+1,y,0)
        gradientX += pixel22(x+1,y+1,0)
        
        gradientY += pixel00(x-1,y-1,1)
        gradientY += pixel10(x-1,y,1)
        gradientY += pixel20(x-1,y+1,1)
        gradientY += pixel01(x,y-1,1)
        gradientY += pixel11(x,y,1)
        gradientY += pixel21(x,y+1,1)
        gradientY += pixel02(x+1,y-1,1)
        gradientY += pixel12(x+1,y,1)
        gradientY += pixel22(x+1,y+1,1)

        lengthX = gradientX
        if lengthX < 0 :
            lengthX = 0
        
        lengthY = gradientY
        if lengthY < 0 :
            lengthY = 0
            
        lengthCommon = math.sqrt((lengthX * lengthX)+(lengthY * lengthY))
        
        lengthX = int(lengthX/800 * 255)
        
        lengthY = int(lengthY/400 * 255)    
        
        newimgX[x,y] = lengthX
        newimgY[x,y] = lengthY
        newimgCommon[x,y] = lengthCommon
  
cv2.imwrite('imageX.png', newimgX)
cv2.imwrite('imageY.png', newimgY)
cv2.imwrite('imageCommon.png', newimgCommon)
cv2.waitKey(0)
cv2.destroyAllWindows()
