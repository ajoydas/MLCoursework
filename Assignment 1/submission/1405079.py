#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:00:04 2018

@author: ajoy
"""
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log
import pprint
from sklearn.metrics import accuracy_score
from collections import Counter
from numpy import log10 
from sklearn.metrics import confusion_matrix

def find_entropy(df, Class):
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/df.shape[0]
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
#def find_entropy_attribute(df,attribute, Class):
#  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
#  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
#  entropy2 = 0
#  
#  for variable in variables:
#      entropy = 0
#      for target_variable in target_variables:
#          search = df[df[attribute]==variable]
#          num = search[df[Class] ==target_variable].shape[0]
#          den = search.shape[0]
#          fraction = num/(den+eps)
#          entropy += -fraction*log(fraction+eps)
#      fraction2 = den/df.shape[0]
#      entropy2 += -fraction2*entropy
#      
#  return abs(entropy2)

def find_entropy_attribute(df,attribute, Class):
  target_variables = uniques[Class] 
  variables = uniques[attribute]
  entropy2 = 0
  N = df.shape[0]
  
  counts = {}
  for variable in variables:
      var = {}
      for target_variable in target_variables:
          var[target_variable] = 0
      counts[variable] = var
          
  for index, r in df.iterrows():
      counts[r[attribute]][r[Class]] +=1
  
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = counts[variable][target_variable]
          
          den = 0
          for i in target_variables:
              den += counts[variable][i]
          
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      
      fraction2 = den/N
      entropy2 += -fraction2*entropy
      
      
  return abs(entropy2)



def predict(tree, inst):
    for nodes in tree.keys():        
#        print(nodes)
        value = inst[nodes]
#        print(value)
        tree = tree[nodes][value]
        prediction = 0
            
        if type(tree) is dict:
            prediction = predict(tree, inst)
        else:
            prediction = tree
            break;                            
        
    return prediction


def find_max_IG(df, attributes, Class):
    min_key = None
    min_ig = np.inf
    IG = []
    for key in attributes:
#        print('calculating... '+ key)
        ig = find_entropy_attribute(df,key, Class)
        if ig < min_ig:
            min_key = key
            min_ig = ig
    
        IG.append(ig)
#    print(IG)   
    return min_key


def plurality_value(clValues):
    count = Counter()
    for i in clValues:
        count[i] += 1
    return count.most_common(1)[0][0]

#plurality_value(df3.iloc[:,-1])


def Decision_Tree_Learning(depth, max_depth, df, attributes, tree=None):
    depth += 1 
    Class = df.keys()[-1]
    
    node = find_max_IG(df, attributes, Class)
#    print('Found.. '+ node)
    attValue = uniques[node]
    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
    for value in attValue:
        subtable = df[df[node] == value].reset_index(drop=True)
        if subtable.shape[0] == 0: # case 1
            tree[node][value] = plurality_value(df.iloc[:,-1])
        
        else:
            clValues = subtable[Class].unique()                        
            
            if len(clValues)==1: # case 2
                tree[node][value] = clValues[0]                                                    
            else:
                child_attr = attributes.copy()
                child_attr.remove(node)
                
                if len(child_attr) == 0 or max_depth == depth: # case 3
                    tree[node][value] = plurality_value(subtable.iloc[:,-1])
                else:
                    tree[node][value] = Decision_Tree_Learning(depth+1,max_depth,subtable, child_attr) # case 4                
    
    return tree
    
    
#columns=['Taste','Temperature','Texture','Eat']
#df = pd.DataFrame(dataset,columns=columns)
#
#attributes = columns.copy()
#attributes.remove('Eat')
##attributes.pop('Eat')
#
#tree2 = Decision_Tree_Learning(0, 100, df, attributes)
#pprint.pprint(tree2)


def test_accuracy(tree, df):
    predictions = []
    for index, r in df.iterrows():
        predictions.append(predict(tree, r))
    score = accuracy_score(df.iloc[:,-1], predictions)
    print(score) 


def metrics(tree, df):
    y_true = df.iloc[:,-1]
    y_pred = []
    for index, r in df.iterrows():
        y_pred.append(predict(tree, r))
   
    score = accuracy_score(y_true, y_pred)
    
    y_pos = 0
    for y in y_true:
        if y:
            y_pos +=1
    y_neg = len(y_true) - y_pos
    
    print(y_pos)
    print(y_neg)
#    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true,y_pred, labels=[0,1]).ravel()

    true_pos_rate =  tp / y_pos
    true_neg_rate = tn / y_neg
    precision = tp / (tp+fp)
    fdr = fp / (tp+fp)
    f1 = 2 / ((1/true_pos_rate) + (1/precision))

    print("accuracy: "+ str(score))
    print("true_pos_rate: "+ str(true_pos_rate))
    print("true_neg_rate: "+ str(true_neg_rate))
    print("precision: "+ str(precision))
    print("fdr: "+ str(fdr))
    print("f1: "+ str(f1))
    
    
def test_accuracy_adaboost(df,h,z):
    predictions = []
    for index, r in df.iterrows():
        val = 0
        for i in range(len(h)):
            if predict(h[i], r) == 0:
                val += z[i]
            else:
                val -= z[i]
            
        if val >= 0:
            predictions.append(0)
        else:
            predictions.append(1)
    
    score = accuracy_score(df.iloc[:,-1], predictions)
    print(score) 

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

def cat_code(df):
    for key in df.keys():
     if df.dtypes[key] != float and df.dtypes[key] != int:
        print(key)
        df[key] = df[key].astype('category')
#        df[key] = df[key].apply(lambda x: x.cat.codes)
    return df

def unique_list(df):
    uni = {}
    for key in df.keys():
        uni[key] = df[key].unique()
    return uni

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


def dataset1():
    names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','output']
    df1 = pd.read_csv('datasets/adultdataset/adult.data', names = names, index_col = False, skipinitialspace=True)
#    check_values(df1)
    df1 = df1.replace(r'^[\s?]+$', np.nan, regex=True)
    df1['native-country'] = (df1['native-country'] == 'United-States')
    df1['output'] = (df1['output'] == '>50K')

    df1 = df1.fillna(df1.mean())
    df1 = df1.fillna(df1.mode().iloc[0])
    
    df1 = binarization_all(df1)
    
    df1 = cat_code(df1)
    cat_columns = df1.select_dtypes(['category']).columns
    df1[cat_columns] = df1[cat_columns].apply(lambda x: x.cat.codes)

    X_train, X_test, y_train, y_test = split(df1)
    X_train[df1.keys()[-1]] = y_train
    X_test[df1.keys()[-1]] = y_test
        
    return  df1, X_train, X_test

def dataset2():
    df2 = pd.read_csv('datasets/creditcard.csv')
    df2_0 = df2[df2['Class']==0]
    df2_1 = df2[df2['Class']==1]    
    
    df_0 = df2_0.sample(n = 50000)
    df2 = pd.concat([df_0, df2_1]).sort_index().reset_index() 

    df2 = df2.drop('Time', axis=1)

    df2 = df2.fillna(df2.mean())
    df2.isnull().values.any()
    df2 = binarization_all(df2)

    df2 = cat_code(df2)
    cat_columns = df2.select_dtypes(['category']).columns
    df2[cat_columns] = df2[cat_columns].apply(lambda x: x.cat.codes)
    
    X_train, X_test, y_train, y_test = split(df2)
    X_train[df2.keys()[-1]] = y_train
    X_test[df2.keys()[-1]] = y_test
        
    return  df2, X_train, X_test


def dataset3():
    df3 = pd.read_csv('datasets/CustomerChurn.csv', )
    df3 = df3.drop('customerID', axis=1)
    df3 = df3.replace(r'^\s+$', np.nan, regex=True) 
    df3['TotalCharges'] = df3['TotalCharges'].astype(float)
    df3['Churn'] = (df3['Churn'] == 'Yes')

    df3 = df3.fillna(df3.mean())
    df3.isnull().values.any()
    
    
    df3 = binarization_all(df3)
    df3 = cat_code(df3)
    cat_columns = df3.select_dtypes(['category']).columns
    df3[cat_columns] = df3[cat_columns].apply(lambda x: x.cat.codes)

    X_train, X_test, y_train, y_test = split(df3)
    X_train[df3.keys()[-1]] = y_train
    X_test[df3.keys()[-1]] = y_test
        
    return  df3, X_train, X_test


######### Added new dataset  #########
    
def dataset4():
    
    df3 = pd.read_csv('data/crx.data',header=None,skipinitialspace=True)
    
    df3 = df3.replace(r'^[\s?]+$', np.nan, regex=True)
    df3[1] = df3[1].astype(float)
    df3[13] = df3[13].astype(float)
    
    df3[15] = (df3[15] == '+')
    
    df3 = df3.fillna(df3.mean())
    df3.isnull().values.any()
    
    nanlist = [0,1,3,4,5,6]
    for i in nanlist:
        df3[i] = df3[i].fillna(df3[i].mode().iloc[0])
        print(df3[i].isnull().values.any())
    
    df3.columns=['A1', 'A2', 'A3','A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
    #df3[1] = df3[1].fillna(df3[1].mean())
    
    df3 = binarization_all(df3)
    df3 = cat_code(df3)
    cat_columns = df3.select_dtypes(['category']).columns
    df3[cat_columns] = df3[cat_columns].apply(lambda x: x.cat.codes)
    
    X_train, X_test, y_train, y_test = split(df3)
    X_train[df3.keys()[-1]] = y_train
    X_test[df3.keys()[-1]] = y_test

    return  df3, X_train, X_test


######### Added new dataset  #########

df, train, test = dataset4()  # Changed to new dataset
attributes = list(df.columns)
attributes.remove(df.keys()[-1])

uniques =  unique_list(df)

tree2 = Decision_Tree_Learning(0, 100, train, attributes)
pprint.pprint(tree1)

metrics(tree3, train)
metrics(tree3, test)

metrics(tree1, train)
metrics(tree1, test)

metrics(tree2, train)
metrics(tree2, test)


def Adaboost(examples, attributes, weak_learner, K):
    N = examples.shape[0]
    w = [1/N for x in range(N)]
    h = list()
    z = list()
    Class = examples.keys()[-1]

    for k in range(K):
        print("Sampling...")
        data = examples.sample(n = N, weights = w, replace=True).reset_index()
        print("Learning...")

        h.append(weak_learner(0, 1, data, attributes))   # Changed to depth 2
        
        print("Learned...")
        error = 0
        j = 0
        for indx, r in examples.iterrows():
            if predict(h[k], r) !=  r[-1]:
                error += w[j]
            j += 1
        print(error)
        if error > .5:
            k -=1
            continue
        
        j = 0
        for indx, r in examples.iterrows():
            if predict(h[k], r) ==  r[-1]:
                w[j] =  w[j] * error / (1 - error)
            j += 1
        ## normalize
        z.append(log10((1-error)/(error+eps)))

        if error == 0:
            return h, z
        
        sum_w = sum(w)
        w = [float(i)/(sum_w + eps) for i in w]
            
        pprint.pprint(h[k])
        print(z[k])
        
    return h, z



#columns=['Taste','Temperature','Texture','Eat']
#df = pd.DataFrame(dataset,columns=columns)
#
#attributes = columns.copy()
#attributes.remove('Eat')

df, train, test = dataset3()  # Changed to new dataset
attributes = list(df.columns)
attributes.remove(df.keys()[-1])
uniques =  unique_list(df)

h, z = Adaboost(train, attributes, Decision_Tree_Learning, 5)
test_accuracy_adaboost(train,h,z)
test_accuracy_adaboost(test,h,z)















    