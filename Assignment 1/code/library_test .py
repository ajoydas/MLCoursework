#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 19:57:00 2018

@author: ajoy
"""

from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
X_scaled                                          


X_scaled.mean(axis=0)
X_scaled.std(axis=0)


import pandas as pd

arr = np.arange(10)
s = pd.DataFrame(arr)

w = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
ws = s.sample(n = 10, weights = w, replace=True)

for i in range(10):
    count = 0
    for ind, j in ws.iterrows():
#        print(j)
        if j[0] == i:
            count +=1
    print(str(i)+' : '+ str(count))
            


columns=['Taste','Temperature','Texture','Eat']
columns.remove('Taste')

columns.remove('Temperature')
