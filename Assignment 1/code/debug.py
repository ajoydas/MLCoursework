
import numpy as np
import pandas as pd
eps = np.finfo(float).eps


df = pd.read_csv('binarazation.csv')
uq = np.unique(df['income'])

def gini_impurity(df, col, Class, val):
    N = df.shape[0]
    le = df[df[col] <= val]
    gt = df[df[col] > val]
    
    le_N = le.shape[0] + eps
    gt_N = gt.shape[0] + eps
    
    clValues = np.unique(df[Class])
    
    val1 = 1
    val2 = 1
    
    for cl in clValues:
        
        le_cl = le[df[Class] == cl].shape[0]
        val1 -= (le_cl/le_N)**2
        
        gt_cl = gt[df[Class] == cl].shape[0]
        val2 -= (gt_cl/gt_N)**2

    val1 = (le_N/N) * val1
    val2 = (gt_N/N) * val2
    
    gini = val1 + val2
    print(gini)
    
    return gini

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
    
def binarization(df, col):
    Class = df.keys()[-1]
    clvalues = np.unique(df[Class])
    N = df.shape[0]
    
    sorted = df.sort_values([col]).copy()
    uq = np.unique(sorted[col])

    le_yes = 0
    le_no = 0
    
    indx = 0
    yes = sorted[sorted[Class] == clvalues[0]].shape[0]
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
            
        print(mn)
        print(gini)    
        
    mn, gini = check_gini(N, le_yes, le_no, yes-le_yes, 
                                  no-le_no, mn, gini, uq[indx])
    
    
    return (df[col] >= mn).astype(int)
df['income'] = binarization(df, 'income')

    


























