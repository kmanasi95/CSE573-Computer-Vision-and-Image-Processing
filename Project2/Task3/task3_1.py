#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:18:40 2018

@author: manasikulkarni
"""

import math
import matplotlib.pyplot as plt
import numpy as np

X = [[5.9, 3.2],[4.6, 2.9],[6.2, 2.8],[4.7, 3.2],[5.5, 4.2],[5.0, 3.0],[4.9, 3.1],[6.7, 3.1],[5.1, 3.8],[6.0, 3.0]]

m1 = [6.2, 3.2]
m2 = [6.6, 3.7]
m3 = [6.5, 3.0]

colors = ["r", "g", "b"]

def FirstCluster():
    list1 = []
    list2 = []
    list3 = []
    m1_cluster = []
    m2_cluster = []
    m3_cluster = []

    for i in range(0,10):
        euc = math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))

        if(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[0]):
            m1_cluster = (X[i][0], X[i][1])
            list1.append(m1_cluster)  
        elif(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[1]):
            m2_cluster = (X[i][0], X[i][1])
            list2.append(m2_cluster)  
        elif(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[2]):
            m3_cluster = (X[i][0], X[i][1])
            list3.append(m3_cluster)

    plt.scatter(m1[0], m1[1], s = 30, color = colors[0], marker = "o")
    plt.scatter(m2[0], m2[1], s = 30, color = colors[1], marker = "o")
    plt.scatter(m3[0], m3[1], s = 30, color = colors[2], marker = "o")
    plt.text(m1[0], m1[1], "  (" + str(m1[0]) + "," + str(m1[1]) +")")
    plt.text(m2[0], m2[1], "  (" + str(m2[0]) + "," + str(m2[1]) +")")
    plt.text(m3[0], m3[1], "  (" + str(m3[0]) + "," + str(m3[1]) +")")
              
    for features1 in list1:
        plt.scatter(features1[0], features1[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[0])
        plt.text(features1[0], features1[1], "  (" + str(features1[0]) + "," + str(features1[1]) +")")
    for features2 in list2:
        plt.scatter(features2[0], features2[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[1])
        plt.text(features2[0], features2[1], "  (" + str(features2[0]) + "," + str(features2[1]) +")")
    for features3 in list3:
        plt.scatter(features3[0], features3[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[2])
        plt.text(features3[0], features3[1], "  (" + str(features3[0]) + "," + str(features3[1]) +")")
            
    plt.savefig("task3_iter1_a.jpg", dpi = 300)
    plt.close()
    
    list1.append(tuple(m1))
    list2.append(tuple(m2))
    list3.append(tuple(m3))
    
    ReComputeCentroid(list1, list2, list3, m1, m2, m3, False, False)
    
m1_old = [0,0]
m2_old = [0,0]
m3_old = [0,0] 
def ReComputeCentroid(list1, list2, list3, m1, m2, m3, flag, figStatus):
    list1.sort(key=lambda t: t[0])
    list2.sort(key=lambda t: t[0])
    list3.sort(key=lambda t: t[0])
    
    m1_old[0] = np.round(m1[0],2)
    m1_old[1] = np.round(m1[1],2)
    m2_old[0] = np.round(m2[0],2)
    m2_old[1] = np.round(m2[1],2)
    m3_old[0] = np.round(m3[0],2)
    m3_old[1] = np.round(m3[1],2)
  
    a = np.mean(list1,axis=0)
    m1[0] = np.round(a[0],1)
    m1[1] = np.round(a[1],1)
    
    b = np.mean(list2,axis=0)
    m2[0] = np.round(b[0],1)
    m2[1] = np.round(b[1],1)
    
    c = np.mean(list3,axis=0)
    m3[0] = np.round(c[0],1)
    m3[1] = np.round(c[1],1)
     
    for i in list1:
        if(np.array_equal(i,m1)):
            list1.remove(i)
    for j in list2:
        if(np.array_equal(j,m2)):
            list2.remove(j)
    for k in list3:
        if(np.array_equal(k,m3)):
            list3.remove(k)
     
    plt.scatter(m1[0], m1[1], s = 30, color = colors[0], marker = "o")
    plt.scatter(m2[0], m2[1], s = 30, color = colors[1], marker = "o")
    plt.scatter(m3[0], m3[1], s = 30, color = colors[2], marker = "o")
    plt.text(m1[0], m1[1], "  (" + str(m1[0]) + "," + str(m1[1]) +")")
    plt.text(m2[0], m2[1], "  (" + str(m2[0]) + "," + str(m2[1]) +")")
    plt.text(m3[0], m3[1], "  (" + str(m3[0]) + "," + str(m3[1]) +")")
        
    for features1 in list1:
        plt.scatter(features1[0], features1[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[0])
        plt.text(features1[0], features1[1], "  (" + str(features1[0]) + "," + str(features1[1]) +")")
    for features2 in list2:
        plt.scatter(features2[0], features2[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[1])
        plt.text(features2[0], features2[1], "  (" + str(features2[0]) + "," + str(features2[1]) +")")
    for features3 in list3:
        plt.scatter(features3[0], features3[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[2])
        plt.text(features3[0], features3[1], "  (" + str(features3[0]) + "," + str(features3[1]) +")")
    
    if(figStatus):
        plt.savefig("task3_iter2_b.jpg", dpi = 300)
        plt.close()
    else:
        plt.savefig("task3_iter1_b.jpg", dpi = 300)
        plt.close()
        
    if(not flag):  
        ReClustering(m1, m2, m3, m1_old, m2_old, m3_old)   
    
def ReClustering(m1, m2, m3, m1_old, m2_old, m3_old):
    list1 = []
    list2 = []
    list3 = []
    m1_cluster = []
    m2_cluster = []
    m3_cluster = []
    
    X.append(np.round(m1_old,1))
    X.append(np.round(m2_old,1))
    X.append(np.round(m3_old,1))
    
    for i in range(0,13):
        euc = math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))

        if(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[0]):
            m1_cluster = (X[i][0], X[i][1])
            list1.append(m1_cluster)
        elif(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[1]):
            m2_cluster = (X[i][0], X[i][1])
            list2.append(m2_cluster)
        elif(min(math.sqrt(((X[i][0] - m1[0])*(X[i][0] - m1[0])) + ((X[i][1] - m1[1])*(X[i][1] - m1[1]))),math.sqrt(((X[i][0] - m2[0])*(X[i][0] - m2[0])) + ((X[i][1] - m2[1])*(X[i][1] - m2[1]))), math.sqrt(((X[i][0] - m3[0])*(X[i][0] - m3[0])) + ((X[i][1] - m3[1])*(X[i][1] - m3[1])))) == euc[2]):
            m3_cluster = (X[i][0], X[i][1])
            list3.append(m3_cluster)
    
    for i in list1:
        if(np.array_equal(i,m1)):
            list1.remove(i)
    for j in list2:
        if(np.array_equal(j,m2)):
            list2.remove(j)
    for k in list3:
        if(np.array_equal(k,m3)):
            list3.remove(k)

    plt.scatter(m1[0], m1[1], s = 30, color = colors[0], marker = "o")
    plt.scatter(m2[0], m2[1], s = 30, color = colors[1], marker = "o")
    plt.scatter(m3[0], m3[1], s = 30, color = colors[2], marker = "o")
    plt.text(m1[0], m1[1], "  (" + str(m1[0]) + "," + str(m1[1]) +")")
    plt.text(m2[0], m2[1], "  (" + str(m2[0]) + "," + str(m2[1]) +")")
    plt.text(m3[0], m3[1], "  (" + str(m3[0]) + "," + str(m3[1]) +")")
    
    for features1 in list1:
        plt.scatter(features1[0], features1[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[0])
        plt.text(features1[0], features1[1], "  (" + str(features1[0]) + "," + str(features1[1]) +")")
    for features2 in list2:
        plt.scatter(features2[0], features2[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[1])
        plt.text(features2[0], features2[1], "  (" + str(features2[0]) + "," + str(features2[1]) +")")
    for features3 in list3:
        plt.scatter(features3[0], features3[1], s = 30, marker = "^", facecolors='none', edgecolors=colors[2])
        plt.text(features3[0], features3[1], "  (" + str(features3[0]) + "," + str(features3[1]) +")")
    
    plt.savefig("task3_iter2_a.jpg", dpi = 300)
    plt.close()
    list1.append(tuple(m1))
    list2.append(tuple(m2))
    list3.append(tuple(m3))
    ReComputeCentroid(list1, list2, list3, m1, m2, m3, True, True)