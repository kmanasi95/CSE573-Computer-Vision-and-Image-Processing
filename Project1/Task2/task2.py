#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:00:48 2018

@author: manasikulkarni
"""

import cv2
import math
import numpy

def Octave1Computation():
    readImage = cv2.imread("/Users/manasikulkarni/Desktop/task2.jpg",cv2.IMREAD_GRAYSCALE)
    height, width = readImage.shape

    newImage1 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    newImage2 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    newImage3 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    newImage4 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    newImage5 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    
    dogOctave1Image1 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    dogOctave1Image2 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    dogOctave1Image3 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)
    dogOctave1Image4 = numpy.array([[0 for i in range(int(width))] for j in range(int(height))],dtype = numpy.float)

    octave1 = [1 / math.sqrt(2) , 1 , math.sqrt(2) , 2 , 2 * math.sqrt(2)]

    gaussianSum1 = 0.0
    gaussianSum2 = 0.0
    gaussianSum3 = 0.0
    gaussianSum4 = 0.0
    gaussianSum5 = 0.0             
    
    gaussianKernel1 = []
    gaussianKernel2 = []
    gaussianKernel3 = []
    gaussianKernel4 = []
    gaussianKernel5 = []

    for x in range(-3,4):
        for y in range(3,-4,-1):
            gaussianResult1 = float(1/float(2 * math.pi * octave1[0] * octave1[0])) * numpy.exp(-((float(x * x + y * y))/(2 * octave1[0] * octave1[0])))
            gaussianSum1 += gaussianResult1
            gaussianKernel1.append(gaussianResult1)
            
            gaussianResult2 = float(1/float(2 * math.pi * octave1[1] * octave1[1])) * numpy.exp(-((float(x * x + y * y))/(2 * octave1[1] * octave1[1])))
            gaussianSum2 += gaussianResult2
            gaussianKernel2.append(gaussianResult2)
            
            gaussianResult3 = float(1/float(2 * math.pi * octave1[2] * octave1[2])) * numpy.exp(-((float(x * x + y * y))/(2 * octave1[2] * octave1[2])))
            gaussianSum3 += gaussianResult3
            gaussianKernel3.append(gaussianResult3)
            
            gaussianResult4 = float(1/float(2 * math.pi * octave1[3] * octave1[3])) * numpy.exp(-((float(x * x + y * y))/(2 * octave1[3] * octave1[3])))
            gaussianSum4 += gaussianResult4
            gaussianKernel4.append(gaussianResult4)
            
            gaussianResult5 = float(1/float(2 * math.pi * octave1[4] * octave1[4])) * numpy.exp(-((float(x * x + y * y))/(2 * octave1[4] * octave1[4])))
            gaussianSum5 += gaussianResult5
            gaussianKernel5.append(gaussianResult5)
            
    for i in range(0,len(gaussianKernel1)):
        gaussianKernel1[i] = gaussianKernel1[i] / float(gaussianSum1)
        gaussianKernel2[i] = gaussianKernel2[i] / float(gaussianSum2)
        gaussianKernel3[i] = gaussianKernel3[i] / float(gaussianSum3)
        gaussianKernel4[i] = gaussianKernel4[i] / float(gaussianSum4)
        gaussianKernel5[i] = gaussianKernel5[i] / float(gaussianSum5)
        
    for x in range(3, height-3):
        for y in range(3, width-3):
            GResult1 = 0
            GResult2 = 0
            GResult3 = 0
            GResult4 = 0
            GResult5 = 0
            var = 0
            
            for i in range(-3,4):
                for j in range(-3,4):
                    pixel = readImage[x+i,y+j]
                    GResult1 += pixel * gaussianKernel1[var]
                    GResult2 += pixel * gaussianKernel2[var]
                    GResult3 += pixel * gaussianKernel3[var]
                    GResult4 += pixel * gaussianKernel4[var]
                    GResult5 += pixel * gaussianKernel5[var]
                    
                    var = var + 1
                              
            newImage1[x,y] = GResult1
            newImage2[x,y] = GResult2
            newImage3[x,y] = GResult3
            newImage4[x,y] = GResult4
            newImage5[x,y] = GResult5
            
            dogOctave1Image1[x,y] = (newImage2[x,y] - newImage1[x,y]) * 255
            dogOctave1Image2[x,y] = (newImage3[x,y] - newImage2[x,y]) * 255
            dogOctave1Image3[x,y] = (newImage4[x,y] - newImage3[x,y]) * 255
            dogOctave1Image4[x,y] = (newImage5[x,y] - newImage4[x,y]) * 255
            
    cv2.imwrite('image1octave1.png', newImage1)
    cv2.imwrite('image2octave1.png', newImage2)
    cv2.imwrite('image3octave1.png', newImage3)
    cv2.imwrite('image4octave1.png', newImage4)
    cv2.imwrite('image5octave1.png', newImage5)
    
    cv2.imwrite('dog1octave1.png', dogOctave1Image1)
    cv2.imwrite('dog2octave1.png', dogOctave1Image2)
    cv2.imwrite('dog3octave1.png', dogOctave1Image3)
    cv2.imwrite('dog4octave1.png', dogOctave1Image4)
    
    keyPointsList = []
    
    for x in range(3, height-3):
        for y in range(3, width-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave1Image2[x,y] > dogOctave1Image2[x+i,y+j] and dogOctave1Image2[x,y] > dogOctave1Image1[x,y] and dogOctave1Image2[x,y] > dogOctave1Image1[x+i,y+j] and dogOctave1Image2[x,y] > dogOctave1Image3[x,y] and dogOctave1Image2[x,y] > dogOctave1Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave1Image2[x,y] < dogOctave1Image2[x+i,y+j] and dogOctave1Image2[x,y] < dogOctave1Image1[x,y] and dogOctave1Image2[x,y] < dogOctave1Image1[x+i,y+j] and dogOctave1Image2[x,y] < dogOctave1Image3[x,y] and dogOctave1Image2[x,y] < dogOctave1Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            elif isMinimum:
                keyPointsList.append((y,x))
        
    for i in range(0,len(keyPointsList)):
        cv2.circle(readImage,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints1octave1.png', readImage)
    
    keyPointsList = []
    
    for x in range(3, height-3):
        for y in range(3, width-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave1Image3[x,y] > dogOctave1Image3[x+i,y+j] and dogOctave1Image3[x,y] > dogOctave1Image2[x,y] and dogOctave1Image3[x,y] > dogOctave1Image2[x+i,y+j] and dogOctave1Image3[x,y] > dogOctave1Image4[x,y] and dogOctave1Image3[x,y] > dogOctave1Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave1Image3[x,y] < dogOctave1Image3[x+i,y+j] and dogOctave1Image3[x,y] < dogOctave1Image2[x,y] and dogOctave1Image3[x,y] < dogOctave1Image2[x+i,y+j] and dogOctave1Image3[x,y] < dogOctave1Image4[x,y] and dogOctave1Image3[x,y] < dogOctave1Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            if isMinimum:
                keyPointsList.append((y,x))
                        
    for i in range(0,len(keyPointsList)):
        cv2.circle(readImage,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints2octave1.png', readImage)
            
def Octave2Computation(): 
    readImage = cv2.imread("/Users/manasikulkarni/Desktop/task2.jpg",cv2.IMREAD_GRAYSCALE)
    height, width = readImage.shape
    
    imageBy2 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    imageBy21 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    imageBy22 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    imageBy23 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    imageBy24 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    imageBy25 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2))],dtype = numpy.float)
    
    dogOctave2Image1 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2) )],dtype = numpy.float)
    dogOctave2Image2 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2) )],dtype = numpy.float)
    dogOctave2Image3 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2) )],dtype = numpy.float)
    dogOctave2Image4 = numpy.array([[0 for i in range(int(width/2))] for j in range(int(height/2) )],dtype = numpy.float)

    for x in range(0,int(height/2)):
        for y in range(0,int(width/2)):
            imageBy2[x,y] = readImage[x*2,y*2]
            
    octave2 = [math.sqrt(2) , 2 , 2 * math.sqrt(2) , 4 , 4 * math.sqrt(2)]

    gaussianSum1 = 0.0
    gaussianSum2 = 0.0
    gaussianSum3 = 0.0
    gaussianSum4 = 0.0
    gaussianSum5 = 0.0             
    
    gaussianKernel1 = []
    gaussianKernel2 = []
    gaussianKernel3 = []
    gaussianKernel4 = []
    gaussianKernel5 = []

    for x in range(-3,4):
        for y in range(3,-4,-1):
            gaussianResult1 = float(1/float(2 * math.pi * octave2[0] * octave2[0])) * numpy.exp(-((float(x * x + y * y))/(2 * octave2[0] * octave2[0])))
            gaussianSum1 += gaussianResult1
            gaussianKernel1.append(gaussianResult1)
            
            gaussianResult2 = float(1/float(2 * math.pi * octave2[1] * octave2[1])) * numpy.exp(-((float(x * x + y * y))/(2 * octave2[1] * octave2[1])))
            gaussianSum2 += gaussianResult2
            gaussianKernel2.append(gaussianResult2)
            
            gaussianResult3 = float(1/float(2 * math.pi * octave2[2] * octave2[2])) * numpy.exp(-((float(x * x + y * y))/(2 * octave2[2] * octave2[2])))
            gaussianSum3 += gaussianResult3
            gaussianKernel3.append(gaussianResult3)
            
            gaussianResult4 = float(1/float(2 * math.pi * octave2[3] * octave2[3])) * numpy.exp(-((float(x * x + y * y))/(2 * octave2[3] * octave2[3])))
            gaussianSum4 += gaussianResult4
            gaussianKernel4.append(gaussianResult4)
            
            gaussianResult5 = float(1/float(2 * math.pi * octave2[4] * octave2[4])) * numpy.exp(-((float(x * x + y * y))/(2 * octave2[4] * octave2[4])))
            gaussianSum5 += gaussianResult5
            gaussianKernel5.append(gaussianResult5)
            
    for i in range(0,len(gaussianKernel1)):
        gaussianKernel1[i] = gaussianKernel1[i] / float(gaussianSum1)
        gaussianKernel2[i] = gaussianKernel2[i] / float(gaussianSum2)
        gaussianKernel3[i] = gaussianKernel3[i] / float(gaussianSum3)
        gaussianKernel4[i] = gaussianKernel4[i] / float(gaussianSum4)
        gaussianKernel5[i] = gaussianKernel5[i] / float(gaussianSum5)
        
    for x in range(3, int(height/2)-3):
        for y in range(3, int(width/2)-3):
            GResult1 = 0
            GResult2 = 0
            GResult3 = 0
            GResult4 = 0
            GResult5 = 0
            var = 0
            
            for i in range(-3,4):
                for j in range(-3,4):
                    pixel = imageBy2[x+i,y+j]
                    GResult1 += pixel * gaussianKernel1[var]
                    GResult2 += pixel * gaussianKernel2[var]
                    GResult3 += pixel * gaussianKernel3[var]
                    GResult4 += pixel * gaussianKernel4[var]
                    GResult5 += pixel * gaussianKernel5[var]
                    
                    var = var + 1
            
            imageBy21[x,y] = GResult1
            imageBy22[x,y] = GResult2
            imageBy23[x,y] = GResult3
            imageBy24[x,y] = GResult4
            imageBy25[x,y] = GResult5
            
            dogOctave2Image1[x,y] = ((imageBy22[x,y] - imageBy21[x,y])*255)
            dogOctave2Image2[x,y] = ((imageBy23[x,y] - imageBy22[x,y])*255)
            dogOctave2Image3[x,y] = ((imageBy24[x,y] - imageBy23[x,y])*255)
            dogOctave2Image4[x,y] = ((imageBy25[x,y] - imageBy24[x,y])*255)
    
    cv2.imwrite('image1octave2.png', imageBy21)
    cv2.imwrite('image2octave2.png', imageBy22)
    cv2.imwrite('image3octave2.png', imageBy23)
    cv2.imwrite('image4octave2.png', imageBy24)
    cv2.imwrite('image5octave2.png', imageBy25)
    
    cv2.imwrite('dog1octave2.png', dogOctave2Image1)
    cv2.imwrite('dog2octave2.png', dogOctave2Image2)
    cv2.imwrite('dog3octave2.png', dogOctave2Image3)
    cv2.imwrite('dog4octave2.png', dogOctave2Image4)
    
    keyPointsList = []
    
    for x in range(3, int(height/2)-3):
        for y in range(3, int(width/2)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave2Image2[x,y] > dogOctave2Image2[x+i,y+j] and dogOctave2Image2[x,y] > dogOctave2Image1[x,y] and dogOctave2Image2[x,y] > dogOctave2Image1[x+i,y+j] and dogOctave2Image2[x,y] > dogOctave2Image3[x,y] and dogOctave2Image2[x,y] > dogOctave2Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave2Image2[x,y] < dogOctave2Image2[x+i,y+j] and dogOctave2Image2[x,y] < dogOctave2Image1[x,y] and dogOctave2Image2[x,y] < dogOctave2Image1[x+i,y+j] and dogOctave2Image2[x,y] < dogOctave2Image3[x,y] and dogOctave2Image2[x,y] < dogOctave2Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            elif isMinimum:
                keyPointsList.append((y,x))
        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy2,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints1octave2.png', imageBy2)
    
    keyPointsList = []
    
    for x in range(3, int(height/2)-3):
        for y in range(3, int(width/2)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave2Image3[x,y] > dogOctave2Image3[x+i,y+j] and dogOctave2Image3[x,y] > dogOctave2Image2[x,y] and dogOctave2Image3[x,y] > dogOctave2Image2[x+i,y+j] and dogOctave2Image3[x,y] > dogOctave2Image4[x,y] and dogOctave2Image3[x,y] > dogOctave2Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave2Image3[x,y] < dogOctave2Image3[x+i,y+j] and dogOctave2Image3[x,y] < dogOctave2Image2[x,y] and dogOctave2Image3[x,y] < dogOctave2Image2[x+i,y+j] and dogOctave2Image3[x,y] < dogOctave2Image4[x,y] and dogOctave2Image3[x,y] < dogOctave2Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            if isMinimum:
                keyPointsList.append((y,x))
                        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy2,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints2octave2.png', imageBy2)
    
            
def Octave3Computation(): 
    readImage = cv2.imread("/Users/manasikulkarni/Desktop/task2.jpg",cv2.IMREAD_GRAYSCALE)
    height, width = readImage.shape

    imageBy4 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    imageBy41 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    imageBy42 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    imageBy43 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    imageBy44 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    imageBy45 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    
    dogOctave3Image1 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    dogOctave3Image2 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    dogOctave3Image3 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)
    dogOctave3Image4 = numpy.array([[0 for i in range(int(width/4 ))] for j in range(int(height/4) )],dtype = numpy.float)

    for x in range(0,int(height/4)):
        for y in range(0,int(width/4)):
            imageBy4[x,y] = readImage[x*4,y*4]
            
    octave3 = [2 * math.sqrt(2) , 4 , 4 * math.sqrt(2) , 8 , 8 * math.sqrt(2)]

    gaussianSum1 = 0.0
    gaussianSum2 = 0.0
    gaussianSum3 = 0.0
    gaussianSum4 = 0.0
    gaussianSum5 = 0.0             
    
    gaussianKernel1 = []
    gaussianKernel2 = []
    gaussianKernel3 = []
    gaussianKernel4 = []
    gaussianKernel5 = []

    for x in range(-3,4):
        for y in range(3,-4,-1):
            gaussianResult1 = float(1/float(2 * math.pi * octave3[0] * octave3[0])) * numpy.exp(-((float(x * x + y * y))/(2 * octave3[0] * octave3[0])))
            gaussianSum1 += gaussianResult1
            gaussianKernel1.append(gaussianResult1)
            
            gaussianResult2 = float(1/float(2 * math.pi * octave3[1] * octave3[1])) * numpy.exp(-((float(x * x + y * y))/(2 * octave3[1] * octave3[1])))
            gaussianSum2 += gaussianResult2
            gaussianKernel2.append(gaussianResult2)
            
            gaussianResult3 = float(1/float(2 * math.pi * octave3[2] * octave3[2])) * numpy.exp(-((float(x * x + y * y))/(2 * octave3[2] * octave3[2])))
            gaussianSum3 += gaussianResult3
            gaussianKernel3.append(gaussianResult3)
            
            gaussianResult4 = float(1/float(2 * math.pi * octave3[3] * octave3[3])) * numpy.exp(-((float(x * x + y * y))/(2 * octave3[3] * octave3[3])))
            gaussianSum4 += gaussianResult4
            gaussianKernel4.append(gaussianResult4)
            
            gaussianResult5 = float(1/float(2 * math.pi * octave3[4] * octave3[4])) * numpy.exp(-((float(x * x + y * y))/(2 * octave3[4] * octave3[4])))
            gaussianSum5 += gaussianResult5
            gaussianKernel5.append(gaussianResult5)
            
    for i in range(0,len(gaussianKernel1)):
        gaussianKernel1[i] = gaussianKernel1[i] / float(gaussianSum1)
        gaussianKernel2[i] = gaussianKernel2[i] / float(gaussianSum2)
        gaussianKernel3[i] = gaussianKernel3[i] / float(gaussianSum3)
        gaussianKernel4[i] = gaussianKernel4[i] / float(gaussianSum4)
        gaussianKernel5[i] = gaussianKernel5[i] / float(gaussianSum5)
        
    for x in range(3, int(height/4)-3):
        for y in range(3, int(width/4)-3):
            GResult1 = 0
            GResult2 = 0
            GResult3 = 0
            GResult4 = 0
            GResult5 = 0
            var = 0
            
            for i in range(-3,4):
                for j in range(-3,4):
                    pixel = imageBy4[x+i,y+j]
                    GResult1 += pixel * gaussianKernel1[var]
                    GResult2 += pixel * gaussianKernel2[var]
                    GResult3 += pixel * gaussianKernel3[var]
                    GResult4 += pixel * gaussianKernel4[var]
                    GResult5 += pixel * gaussianKernel5[var]
                    
                    var = var + 1
                              
            imageBy41[x,y] = GResult1
            imageBy42[x,y] = GResult2
            imageBy43[x,y] = GResult3
            imageBy44[x,y] = GResult4
            imageBy45[x,y] = GResult5
            
            dogOctave3Image1[x,y] = ((imageBy42[x,y] - imageBy41[x,y]) * 255)
            dogOctave3Image2[x,y] = ((imageBy43[x,y] - imageBy42[x,y]) * 255)
            dogOctave3Image3[x,y] = ((imageBy44[x,y] - imageBy43[x,y]) * 255)
            dogOctave3Image4[x,y] = ((imageBy45[x,y] - imageBy44[x,y]) * 255)
    
    cv2.imwrite('image1octave3.png', imageBy41)
    cv2.imwrite('image2octave3.png', imageBy42)
    cv2.imwrite('image3octave3.png', imageBy43)
    cv2.imwrite('image4octave3.png', imageBy44)
    cv2.imwrite('image5octave3.png', imageBy45)
    
    cv2.imwrite('dog1octave3.png', dogOctave3Image1)
    cv2.imwrite('dog2octave3.png', dogOctave3Image2)
    cv2.imwrite('dog3octave3.png', dogOctave3Image3)
    cv2.imwrite('dog4octave3.png', dogOctave3Image4)
    
    keyPointsList = []
    
    for x in range(3, int(height/4)-3):
        for y in range(3, int(width/4)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave3Image2[x,y] > dogOctave3Image2[x+i,y+j] and dogOctave3Image2[x,y] > dogOctave3Image1[x,y] and dogOctave3Image2[x,y] > dogOctave3Image1[x+i,y+j] and dogOctave3Image2[x,y] > dogOctave3Image3[x,y] and dogOctave3Image2[x,y] > dogOctave3Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave3Image2[x,y] < dogOctave3Image2[x+i,y+j] and dogOctave3Image2[x,y] < dogOctave3Image1[x,y] and dogOctave3Image2[x,y] < dogOctave3Image1[x+i,y+j] and dogOctave3Image2[x,y] < dogOctave3Image3[x,y] and dogOctave3Image2[x,y] < dogOctave3Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            elif isMinimum:
                keyPointsList.append((y,x))
        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy4,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints1octave3.png', imageBy4)
    
    keyPointsList = []
    
    for x in range(3, int(height/4)-3):
        for y in range(3, int(width/4)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave3Image3[x,y] > dogOctave3Image3[x+i,y+j] and dogOctave3Image3[x,y] > dogOctave3Image2[x,y] and dogOctave3Image3[x,y] > dogOctave3Image2[x+i,y+j] and dogOctave3Image3[x,y] > dogOctave3Image4[x,y] and dogOctave3Image3[x,y] > dogOctave3Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave3Image3[x,y] < dogOctave3Image3[x+i,y+j] and dogOctave3Image3[x,y] < dogOctave3Image2[x,y] and dogOctave3Image3[x,y] < dogOctave3Image2[x+i,y+j] and dogOctave3Image3[x,y] < dogOctave3Image4[x,y] and dogOctave3Image3[x,y] < dogOctave3Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            if isMinimum:
                keyPointsList.append((y,x))
                        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy4,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints2octave3.png', imageBy4)
    
            
def Octave4Computation():   
    readImage = cv2.imread("/Users/manasikulkarni/Desktop/task2.jpg",cv2.IMREAD_GRAYSCALE)
    height, width = readImage.shape

    imageBy8 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    imageBy81 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    imageBy82 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    imageBy83 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    imageBy84 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    imageBy85 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    
    dogOctave4Image1 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    dogOctave4Image2 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    dogOctave4Image3 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)
    dogOctave4Image4 = numpy.array([[0 for i in range(int(width/8 ))] for j in range(int(height/8) )],dtype = numpy.float)

    for x in range(0,int(height/8)):
        for y in range(0,int(width/8)):
            imageBy8[x,y] = readImage[x*8,y*8]
            
    octave4 = [4 * math.sqrt(2) , 8 , 8 * math.sqrt(2) , 16 , 16 * math.sqrt(2)]

    gaussianSum1 = 0.0
    gaussianSum2 = 0.0
    gaussianSum3 = 0.0
    gaussianSum4 = 0.0
    gaussianSum5 = 0.0             
    
    gaussianKernel1 = []
    gaussianKernel2 = []
    gaussianKernel3 = []
    gaussianKernel4 = []
    gaussianKernel5 = []

    for x in range(-3,4):
        for y in range(3,-4,-1):
            gaussianResult1 = float(1/float(2 * math.pi * octave4[0] * octave4[0])) * numpy.exp(-((float(x * x + y * y))/(2 * octave4[0] * octave4[0])))
            gaussianSum1 += gaussianResult1
            gaussianKernel1.append(gaussianResult1)
            
            gaussianResult2 = float(1/float(2 * math.pi * octave4[1] * octave4[1])) * numpy.exp(-((float(x * x + y * y))/(2 * octave4[1] * octave4[1])))
            gaussianSum2 += gaussianResult2
            gaussianKernel2.append(gaussianResult2)
            
            gaussianResult3 = float(1/float(2 * math.pi * octave4[2] * octave4[2])) * numpy.exp(-((float(x * x + y * y))/(2 * octave4[2] * octave4[2])))
            gaussianSum3 += gaussianResult3
            gaussianKernel3.append(gaussianResult3)
            
            gaussianResult4 = float(1/float(2 * math.pi * octave4[3] * octave4[3])) * numpy.exp(-((float(x * x + y * y))/(2 * octave4[3] * octave4[3])))
            gaussianSum4 += gaussianResult4
            gaussianKernel4.append(gaussianResult4)
            
            gaussianResult5 = float(1/float(2 * math.pi * octave4[4] * octave4[4])) * numpy.exp(-((float(x * x + y * y))/(2 * octave4[4] * octave4[4])))
            gaussianSum5 += gaussianResult5
            gaussianKernel5.append(gaussianResult5)
            
    for i in range(0,len(gaussianKernel1)):
        gaussianKernel1[i] = gaussianKernel1[i] / float(gaussianSum1)
        gaussianKernel2[i] = gaussianKernel2[i] / float(gaussianSum2)
        gaussianKernel3[i] = gaussianKernel3[i] / float(gaussianSum3)
        gaussianKernel4[i] = gaussianKernel4[i] / float(gaussianSum4)
        gaussianKernel5[i] = gaussianKernel5[i] / float(gaussianSum5)
        
    for x in range(3, int(height/8)-3):
        for y in range(3, int(width/8)-3):
            GResult1 = 0
            GResult2 = 0
            GResult3 = 0
            GResult4 = 0
            GResult5 = 0
            var = 0
            
            for i in range(-3,4):
                for j in range(-3,4):
                    pixel = imageBy8[x+i,y+j]
                    GResult1 += pixel * gaussianKernel1[var]
                    GResult2 += pixel * gaussianKernel2[var]
                    GResult3 += pixel * gaussianKernel3[var]
                    GResult4 += pixel * gaussianKernel4[var]
                    GResult5 += pixel * gaussianKernel5[var]
                    
                    var = var + 1
                              
            imageBy81[x,y] = GResult1
            imageBy82[x,y] = GResult2
            imageBy83[x,y] = GResult3
            imageBy84[x,y] = GResult4
            imageBy85[x,y] = GResult5
            
            dogOctave4Image1[x,y] = (imageBy82[x,y] - imageBy81[x,y]) * 255
            dogOctave4Image2[x,y] = (imageBy83[x,y] - imageBy82[x,y]) * 255
            dogOctave4Image3[x,y] = (imageBy84[x,y] - imageBy83[x,y]) * 255
            dogOctave4Image4[x,y] = (imageBy85[x,y] - imageBy84[x,y]) * 255
    
    cv2.imwrite('image1octave4.png', imageBy81)
    cv2.imwrite('image2octave4.png', imageBy82)
    cv2.imwrite('image3octave4.png', imageBy83)
    cv2.imwrite('image4octave4.png', imageBy84)
    cv2.imwrite('image5octave4.png', imageBy85)
    
    cv2.imwrite('dog1octave4.png', dogOctave4Image1)
    cv2.imwrite('dog2octave4.png', dogOctave4Image2)
    cv2.imwrite('dog3octave4.png', dogOctave4Image3)
    cv2.imwrite('dog4octave4.png', dogOctave4Image4)
    
    keyPointsList = []
    
    for x in range(3, int(height/8)-3):
        for y in range(3, int(width/8)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave4Image2[x,y] > dogOctave4Image2[x+i,y+j] and dogOctave4Image2[x,y] > dogOctave4Image1[x,y] and dogOctave4Image2[x,y] > dogOctave4Image1[x+i,y+j] and dogOctave4Image2[x,y] > dogOctave4Image3[x,y] and dogOctave4Image2[x,y] > dogOctave4Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave4Image2[x,y] < dogOctave4Image2[x+i,y+j] and dogOctave4Image2[x,y] < dogOctave4Image1[x,y] and dogOctave4Image2[x,y] < dogOctave4Image1[x+i,y+j] and dogOctave4Image2[x,y] < dogOctave4Image3[x,y] and dogOctave4Image2[x,y] < dogOctave4Image3[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            elif isMinimum:
                keyPointsList.append((y,x))
        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy8,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints1octave4.png', imageBy8)
    
    keyPointsList = []
    
    for x in range(3, int(height/8)-3):
        for y in range(3, int(width/8)-3):
            isMaximum = True
            isMinimum = True
            for i in range(-1,2):
                for j in range(-1,2):
                    if dogOctave4Image3[x,y] > dogOctave4Image3[x+i,y+j] and dogOctave4Image3[x,y] > dogOctave4Image2[x,y] and dogOctave4Image3[x,y] > dogOctave4Image2[x+i,y+j] and dogOctave4Image3[x,y] > dogOctave4Image4[x,y] and dogOctave4Image3[x,y] > dogOctave4Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMaximum = True
                        isMinimum = False
                        break
                    else:
                        isMaximum = False
                        isMinimum = False
                        break
                    
                    if dogOctave4Image3[x,y] < dogOctave4Image3[x+i,y+j] and dogOctave4Image3[x,y] < dogOctave4Image2[x,y] and dogOctave4Image3[x,y] < dogOctave4Image2[x+i,y+j] and dogOctave4Image3[x,y] < dogOctave4Image4[x,y] and dogOctave4Image3[x,y] < dogOctave4Image4[x+i,y+j]:
                        if x==x+i and y==y+j:
                            break
                        isMinimum = True
                        isMaximum = False
                        break
                    else:
                        isMinimum = False
                        isMaximum = False
                        break
                    
            if isMaximum:
                keyPointsList.append((y,x))
            if isMinimum:
                keyPointsList.append((y,x))
                        
    for i in range(0,len(keyPointsList)):
        cv2.circle(imageBy8,(keyPointsList[i][0],keyPointsList[i][1]), 1, (255,255,255), -1)
        
    cv2.imwrite('keypoints2octave4.png', imageBy8)
    
    
def main():
    Octave1Computation()
    Octave2Computation()
    Octave3Computation()
    Octave4Computation()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
main()