# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 16:51:31 2021

@author: Vito
"""

import math

def calcSBDncc(x,y,s):
    assert len(x)==len(y)
    assert isinstance(s,int)
    length_ = len(x)
    pow_x = 0
    pow_y = 0
    for i in range(length_):
        pow_x += math.pow(x[i],2)
        pow_y += math.pow(y[i],2)
    dist_x =math.pow(pow_x,0.5)
    dist_y =math.pow(pow_y,0.5)
    dist_xy = dist_x*dist_y
    ccs = 0
    for j in range(length_-s):
         ccs +=  x[j+s]*y[j]
    ncc = ccs/dist_xy
    return ncc

def SBD(x,y,s=None):
    assert len(x)==len(y)
    if  s==None:
        length_ = len(x)
        ncc_list = []
        for s in range(length_-1):
            ncc_list.append(calcSBDncc(x,y,s))
        ncc = max(ncc_list)
        sbd = 1 - ncc
    else:
        ncc = calcSBDncc(x,y,s)
        sbd = 1 - ncc
    return sbd