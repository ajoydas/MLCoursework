import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_spd_matrix
from numpy import log2 as ln
eps = np.finfo(float).eps


df = pd.read_csv('data.txt', header=None, delimiter = '\s*')
df = df.replace(r'^[\s?]+$', np.nan, regex=True)
df.isnull().values.any()

X = df.values
x = X.T
N = X.shape[0]
D = X.shape[1]
K = 2

#mu = []
#for i in range(D):
#    mu.append([X[:,i].mean()])
#scatter_matrix = np.zeros((D,D))
#for i in range(N):
#    scatter_matrix += 1/N* (x[:,i].reshape(D,1) - mu).dot((x[:,i].reshape(D,1) - mu).T)
#    

cov_mat = np.cov(x)

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

#for i in eig_pairs:
#    print(i[0])    
matrix_w = np.hstack((eig_pairs[0][1].reshape(D,1), eig_pairs[1][1].reshape(D,1)))
transformed = matrix_w.T.dot(x)


plt.plot(transformed[0], transformed[1], 'o', markersize=7, color='black', alpha=0.5)
plt.xlabel('PC-1')
plt.ylabel('PC-2')
plt.title('Transformed samples')
plt.show()

X = transformed.T
D = X.shape[1]
K = 3
n_itr = 100

#np.random.seed(0)
#mu = np.random.rand(K,D)

#mu = np.zeros((K, D))
#S = []
#for i in range(K):
#    np.random.seed(0)
#    randint = np.random.randint(0, N-1)
#    while randint in S:
#        np.random.seed(i)
#        randint = np.random.randint(0, N-1)    
#    
#    S.append(randint)
#    mu[i] = X[randint]
    
mu = [[0,3],[4,-2],[7,3]]

m = make_spd_matrix(D,k)
#m = np.array([[1,0.1],[0.1,1.0]])
cov = [m]*K
w = [1.0/K]*K

def loglikely():
    logl = 0
    for i in range(N):
        val = 0
        for k in range(K):
            Nk = multivariate_normal.pdf(X[i], mean=mu[k], cov=cov[k]);
            val += w[k]* Nk
        
        logl += np.log(val)
    
    print("Log Likelyhood: "+str(logl))
    return logl

 
def converged():
    global prev
    logl = loglikely()
    if abs(logl-prev) <= 0.01:
        return True
    prev = logl
    return False

def draw():
    
    classes = []
    for k in range(K):
        classes.append([])
    
   
    for i in range(N):
        mxVal = -1
        cls = 0
        for k in range(K):
            if P[i][k] > mxVal:
                mxVal = P[i][k]
                cls = k
        
        classes[cls].append(X[i])
        
    for k in range(K):
        classes[k] = np.array(classes[k])
        
    colors = ['red', 'blue', 'green']
    
    for k in range(K):
        color = colors[k%len(colors)]
        if classes[k].shape[0]>0:
            plt.scatter(classes[k][:, 0], classes[k][:, 1], color=color)
        
        plt.scatter(mu[k][0],mu[k][1], s=500 ,c=color, marker="d", alpha=0.5)

    plt.show()


loglikely()    
covs = []
prev = 0.0
m = 0
for e in range(n_itr):
    
    P = np.array([([0.0]*K)]*N)
    for i in range(N):
        val = 0.0
        for k in range(K):
            mul = w[k]*multivariate_normal.pdf(X[i], mu[k], cov[k]);
            P[i][k] =  mul
            val += mul
#            print(str(Nk)+' --- '+str(P[i][k])+'----- '+ str(w[k]* Nk))
            
        P[i] /= val 
        
    for k in range(K):
        val = np.zeros(D)
        for i in range(N):
            val += P[i][k]*X[i]
        
        mu[k] = val/(P[:,k].sum())

    
    for k in range(K):
        val = np.zeros((D,D))
        for i in range(N):
            m = X[i]-mu[k]
            val += P[i][k]* np.dot(np.transpose([m]), [m])
        
        cov[k] = val/np.sum(P[:,k])
        
    
    for k in range(K):
        w[k] = P[:,k].sum()/N

#    print(mu)
#    covs.append(cov)    

    draw()

    if converged():
        break 
        









