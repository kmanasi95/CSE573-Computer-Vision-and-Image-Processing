#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 18:26:11 2018

@author: manasikulkarni
"""

import cv2
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

eruptions = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(1))
waiting = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(2))

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def callGMM(): 
    X = []
    for i in range(len(eruptions)):
        X.append((eruptions[i],waiting[i]))

    mean1 = [4.0, 81]
    mean2 = [2.0, 57]
    mean3 = [4.0, 71]

    cov1 = [[1.30, 13.98],[13.98, 184.82]]
    cov2 = [[1.30, 13.98],[13.98, 184.82]]
    cov3 = [[1.30, 13.98],[13.98, 184.82]]
     
    pi1 = 0.33
    pi2 = 0.33
    pi3 = 0.33
    for x in range (0,5):
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

        list1 = []
        list2 = []
        for i in eruptions:
            list1.append(i)
        for j in waiting:
            list2.append(j)

        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        for i in range(0,272):
            a.append(list1[i] * ResponsibilityMat1[i])
            b.append(list2[i] * ResponsibilityMat1[i])
            c.append(list1[i] * ResponsibilityMat2[i])
            d.append(list2[i] * ResponsibilityMat2[i])
            e.append(list1[i] * ResponsibilityMat3[i])
            f.append(list2[i] * ResponsibilityMat3[i])

        mean1 = [sum(a) / sum(ResponsibilityMat1),sum(b) / sum(ResponsibilityMat1)]
        mean2 = [sum(c) / sum(ResponsibilityMat2),sum(d) / sum(ResponsibilityMat2)]
        mean3 = [sum(e) / sum(ResponsibilityMat3),sum(f) / sum(ResponsibilityMat3)]

        a1 = []
        b1 = []
        c1 = []
        d1 = []
        e1 = []
        f1 = []
        for i in range(0,272):
            a1.append(ResponsibilityMat1[i] * ((list1[i] - mean1[0])**2))
            b1.append(ResponsibilityMat1[i] * ((list2[i] - mean1[1])**2))
            c1.append(ResponsibilityMat2[i] * ((list1[i] - mean2[0])**2))
            d1.append(ResponsibilityMat2[i] * ((list2[i] - mean2[1])**2))
            e1.append(ResponsibilityMat3[i] * ((list1[i] - mean3[0])**2))
            f1.append(ResponsibilityMat3[i] * ((list2[i] - mean3[1])**2))
        
    
        S1 = [[sum(a1) / sum(ResponsibilityMat1)], [sum(b1) / sum(ResponsibilityMat1)]]
        S2 = [sum(c1) / sum(ResponsibilityMat2), sum(d1) / sum(ResponsibilityMat2)]
        S3 = [sum(e1) / sum(ResponsibilityMat3), sum(f1) / sum(ResponsibilityMat3)]
        

        pi1 = sum(ResponsibilityMat1) / len(X)
        pi2 = sum(ResponsibilityMat2) / len(X)
        pi3 = sum(ResponsibilityMat3) / len(X)
        
        x1, y1 = np.transpose(X)
        plt.plot(x1, y1, 'ro')
        
        plot_cov_ellipse(np.cov(X,rowvar = False), mean1, nstd=1.5, alpha = 0.5, color = 'r')
        plot_cov_ellipse(np.cov(X,rowvar = False), mean2, nstd=1.5, alpha = 0.5, color = 'g')
        plot_cov_ellipse(np.cov(X,rowvar = False), mean3, nstd=1.5, alpha = 0.5, color = 'b')
        plt.savefig("task3_gmm_iter"+str(x)+".jpg", dpi = 300)
        plt.close()
          
    print("Recomputed Means : ", mean1, mean2, mean3)
    print("Recomputed Weights : ", pi1, pi2, pi3)
    
