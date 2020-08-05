# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from numpy import log2 as ln
eps = np.finfo(float).eps


df_full = pd.read_csv('test/data.csv', header=None)

# make a smaller dataset 
#df = df_full.sample(frac=0.1, replace=True).reset_index(drop=True)

df = df_full.iloc[0:2000,:]
# split to train, test, validation
df_indx = df[0]
# remove first column
df = df.iloc[:,1:]

N = df.shape[0]
M = df.shape[1]

# reset column index 
df.columns = [x for x in range(M)]

# gen test & validation
test = 99* np.ones((N,M))
validation = 99* np.ones((N,M))
    
df = df.values
for i in range(N):
    for j in range(M):
        val = df[i,j]
        if val != 99:
             np.random.seed(i*N+j)
             prob = np.random.uniform(0,1,1)
             if prob >= 0.8:
                 test[i,j] = val
                 df[i,j] = 99

x_valid = df.copy()
for i in range(N):
    for j in range(M):
        val = df[i,j]
        if val != 99:
             np.random.seed(2*(i*N+j))
             prob = np.random.uniform(0,1,1)
             if prob >= 0.75:
                 validation[i,j] = val
                 df[i,j] = 99
                 
print(len(np.where( validation != 99)[0]))
print(len(np.where( test != 99)[0]))
print(len(np.where( df != 99)[0]))


cols = [[] for x in range(M)]
rows = []
for i in range(N):
    row = []
    for j in range(M):
        if df[i,j] != 99:
            cols[j].append(i)
            row.append(j)
    
    rows.append(row)


# rmse calculation
def risk(real, predict):
    sum = 0 
    count = 0
    for i in range(N):
        for j in range(M):
            if real[i,j] != 99:
                sum += (real[i,j]-predict[i,j])**2
                count += 1
    
    return np.sqrt(sum/count)
    
def risk_x(real, predict):
    sum = 0 
    count = 0
    for j in range(M):
        for i in cols[j]:
            sum += (real[i,j]-predict[i,j])**2
        
        count += len(cols[j])
        
    return np.sqrt(sum/count)

# train als
def train(x, K, lambda_u, lambda_v):
    N = x.shape[0]
    M = x.shape[1]

    u = np.random.uniform(0,1,(K, N))
    v = np.zeros((K, M))
    
    prev = 10**10
    
#    for i in range(5):
    while(True):
        for m in range(M):
            sum1 = np.zeros((K, K))
            sum2 = np.zeros((K, 1))
            for n in cols[m]:
                sum1 += np.matmul(u[:,n].reshape(K,1),u[:,n].T.reshape(1,K))
                sum2 += (x[n,m] * u[:,n]).reshape((K,1))
            
            val = np.linalg.inv(sum1 + lambda_v * np.identity(K))
            v[:,m] = np.matmul(val, sum2).reshape(K)
            
        for n in range(N):
            sum1 = np.zeros((K, K))
            sum2 = np.zeros((K, 1))
            for m in rows[n]:
                sum1 += np.matmul(v[:,m].reshape(K,1),v[:,m].T.reshape(1,K))
                sum2 += (x[n,m] * v[:,m].reshape(K,1))
            
            val = np.linalg.inv(sum1 + lambda_u * np.identity(K))
            u[:,n] = np.matmul(val, sum2).reshape(K)
        
        # empirical risk 
        predict = np.matmul(u.T, v)
        
        rmse = risk_x(x, predict)
        print(rmse)
        if(prev - rmse) < 0.1:
            break
        prev = rmse
        
    return predict

# make als 
x = df
K_arr = [5,10,20]
lambda_u_arr = [0.01, 0.1, 1, 10]
lambda_v_arr = [0.01, 0.1, 1, 10]

best_K = 0
best_lambda_u = 0
best_lambda_v = 0
min_error = 10**10

# select lambda, k
for K in K_arr:
    for lambda_u in lambda_u_arr:
        for lambda_v in lambda_v_arr:        
            predict = train(x, K, lambda_u, lambda_v)
            
            valid_err = risk(validation, predict)
            print(valid_err)
            
            if(valid_err < min_error):
                min_error = valid_err
                best_K = K
                best_lambda_u = lambda_u
                best_lambda_v = lambda_v
                

predict = train(x_valid, best_K, best_lambda_u, best_lambda_v)


# save model in file
np.save('predict.npy', predict)
#np.save('lambda_u.npy', lambda_u)
#np.save('lambda_v.npy', lambda_v)
#np.save('K.npy', K)


# open model from file
predict = np.load('predict.npy')
#lambda_u = np.load('lambda_u.npy')
#lambda_v = np.load('lambda_v.npy', )
#K = np.load('K.npy')

# test using test split
print(risk(test, predict))    


#test_data = pd.read_csv('data.csv', header=None)








