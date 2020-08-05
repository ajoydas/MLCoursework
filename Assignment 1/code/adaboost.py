#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 02:53:44 2018

@author: ajoy
"""

import numpy as np
import pandas as pd
from numpy import log

eps = np.finfo(float).eps

def weighted_majority():
    pass


    
def Adaboost(examples, attributes, weak_learner, K):
    N = examples.shape[0]
    w = [1/N for x in range(N)]
    h = list()
    z = list()
    Class = examples.keys()[-1]

    for k in range(K):
        data = examples.sample(n = 10, weights = w, replace=True)
        h[k] = weak_learner(0, 1, data, attributes)
        error = 0
        for j in range(N):
            xj = examples.iloc[[j]].loc[:,examples.columns != Class]
            yj = examples.iloc[[2]].loc[:,examples.columns == Class]
            print(xj)
            print(yj)
            if predict(h[k], xj) !=  yj:
                error += w[j]
        
        if error > .5:
            k -=1
            continue
        
        for j in range(N):
            xj = examples.iloc[[j]].loc[:,examples.columns != Class]
            yj = examples.iloc[[2]].loc[:,examples.columns == Class]
            if predict(h[k], xj) ==  yj:
                w[j] =  w[j] * error / (1 - error)
            
            ## normalize
            z[k] = log[(1-error)/error]
            
            pprint(h[k])
            print(z[k])
            
        return h, z



w = [1/10 for x in range(10)]