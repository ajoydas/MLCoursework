mean = [0, 0]
cov = [[1, 0], [0, 100]]  # diagonal covariance

import matplotlib.pyplot as plt
x, y = np.random.multivariate_normal(mean, cov, 50).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5);


X = np.hstack((x.reshape(50,1),y.reshape(50,1)))
y = multivariate_normal.pdf(X, mean=mean, cov=cov);


fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x, y)





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X[:,0], X[:,1], y.reshape(50,1), cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()



