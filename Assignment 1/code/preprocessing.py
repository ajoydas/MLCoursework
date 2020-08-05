#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:07:56 2018

@author: ajoy
"""

import pandas as pd
import numpy as np
eps = np.finfo(float).eps





df = pd.read_csv('online1_data.csv')

names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','output']
df1 = pd.read_csv('datasets/adultdataset/adult.data', names = names, index_col = False, skipinitialspace=True)

df2 = pd.read_csv('datasets/creditcard.csv')

df3 = pd.read_csv('datasets/CustomerChurn.csv')


df3 = df3.drop('customerID', axis=1)
df3 = df3.replace(r'^\s+$', np.nan, regex=True) 
df3['TotalCharges'] = df3['TotalCharges'].astype(float)
df3 = df3.fillna(df3.mean())
df3.isnull().values.any()

for key in df3.keys():
#    print(np.unique(df3[key]))
    print(key+' '+str(len(np.unique(df3[key]))))

#sorted = df3.sort_values(['tenure'])

def check_gini(N, le_yes, le_no, gt_yes, gt_no, mn, gini, curr_val):
    le_N = le_no + le_yes + eps
    gt_N = gt_no + gt_yes + eps
            
    val1 = 1 - (le_yes/le_N)**2 - (le_no/le_N)**2
    val2 = 1 - (gt_yes/gt_N)**2 - (gt_no/gt_N)**2

    val1 = (le_N/N) * val1
    val2 = (gt_N/N) * val2
    
    new_gini = val1 + val2
    
    if new_gini < gini:
        gini = new_gini
        mn = curr_val
    return mn, gini
    
def binarization(df, col, Class, yes, clvalues):
    N = df.shape[0]    
    sorted = df.sort_values([col])
    uq = np.unique(sorted[col])

    le_yes = 0
    le_no = 0
    
    indx = 0
    no = N - yes
    
    mn = -1
    gini = 1
    for index, r in sorted.iterrows():
        if r[col] != uq[indx]:
            mn, gini = check_gini(N, le_yes, le_no, yes-le_yes, 
                                  no-le_no, mn, gini, uq[indx])
            indx += 1
            
        if r[Class] == clvalues[0]:
            le_yes += 1
        else:
            le_no += 1
            
#        print(mn)
#        print(gini)    
        
    mn, gini = check_gini(N, le_yes, le_no, yes-le_yes, 
                                  no-le_no, mn, gini, uq[indx])
    
    
    return (df[col] >= mn).astype(int)

def binarization_all(df):
    Class = df.keys()[-1]
    clvalues = np.unique(df[Class])
    yes = df[df[Class] == clvalues[0]].shape[0]

    for key in df.keys():
        if df.dtypes[key] == float or df.dtypes[key] == int:
            df[key] = binarization(df, key, Class, yes, clvalues)
#            break
    return df

    
df3['tenure'] = binarization(df3, 'tenure')
df3['MonthlyCharges'] = binarization(df3, 'MonthlyCharges')
df3['TotalCharges'] = binarization(df3, 'TotalCharges')



df2_0 = df2[df2['Class']==0]
df2_1 = df2[df2['Class']==1]    

df_0 = df2_0.sample(n = 50000)
df2 = pd.concat([df_0, df2_1]).sort_index().reset_index() 



df2 = df2.drop('Time', axis=1)
#df2 = df2.replace(r'^\s+$', np.nan, regex=True) 
#df2['TotalCharges'] = df3['TotalCharges'].astype(float)
df2 = df2.fillna(df3.mean())
df2.isnull().values.any()

def split(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def check_values(df):
    for key in df.keys():
    #    print(key)
        print(np.unique(df[key]))
        print(key+' '+ str(df.dtypes[key])+' '+str(len(np.unique(df[key]))))


        
        
df2 = binarization_all(df2)
df2[df2['V2'] == 0].shape[0]
df2[df2['V2'] == 1].shape[0]

df_back = df2
df2 = df2.drop('Time', axis=1)
#df2 = df2.replace(r'^\s+$', np.nan, regex=True) 
#df2['TotalCharges'] = df3['TotalCharges'].astype(float)
df2 = df2.fillna(df3.mean())
df2.isnull().values.any()




df1 = pd.read_csv('datasets/adultdataset/adult.data', names = names, index_col = False, skipinitialspace=True)

check_values(df1)
#df1['State'] = df1['State'].str.strip()
df1 = df1.replace(r'^[\s?]+$', np.nan, regex=True)
df1['native-country'] = (df1['native-country'] == 'United-States')
df1['output'] = (df1['output'] == '>50K')

df1 = df1.fillna(df1.mean())
df1 = df1.fillna(df1.mode().iloc[0])

df1 = binarization_all(df1)
X_train, X_test, y_train, y_test = split(df1)

X_train[df1.keys()[-1]] = y_train
X_test[df1.keys()[-1]] = y_test




















