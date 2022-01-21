import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

N=20

a=1
b=-2

e=0.5


x=np.zeros((N, 2)) #N samples, 2 features

x[:, 0]=10.0*(np.random.rand((N))-0.5)
x[:, 1]=a*x[:, 0]+b+e*(np.random.rand((N))-0.5)

fig=plt.figure(figsize=(4, 4))
plt.plot(x[:, 0], x[:, 1], '.', )
plt.xlim([-5, 5])
plt.ylim([-5+b, 5+b])
dx=-1
dy=1
plt.arrow(0, b, dx, dy, head_width=0.1)
plt.arrow(0, b, dy, -dx, head_width=0.1)
plt.savefig('pca01_01.png', bbox_inches='tight',pad_inches = 0)
plt.show()

pca = PCA(2)
#pca_x = pca.fit_transform(x)
pca.fit(x)
pca_x = pca.transform(x)

fig=plt.figure()
plt.plot(pca_x[:, 0], pca_x[:, 1], '.')
plt.xlim([-8, 8])
plt.ylim([-0.2, 0.2])
plt.savefig('pca01_02.png', bbox_inches='tight',pad_inches = 0)
plt.show()


pca_x_reduced=pca_x.copy()
pca_x_reduced[:, 1]=0
x_reduced=pca.inverse_transform(pca_x_reduced)


fig=plt.figure()
plt.plot(pca_x[:, 0], pca_x[:, 1], '.')
plt.plot(pca_x_reduced[:, 0], pca_x_reduced[:, 1], '.')
plt.xlim([-8, 8])
plt.ylim([-0.2, 0.2])
plt.savefig('pca01_03.png', bbox_inches='tight',pad_inches = 0)
plt.show()


fig=plt.figure(figsize=(4, 4))
plt.plot(x[:, 0], x[:, 1], '.', )
plt.plot(x_reduced[:, 0], x_reduced[:, 1], '.', )
plt.xlim([-5, 5])
plt.ylim([-5+b, 5+b])
#dx=-1
#dy=1
#plt.arrow(0, b, dx, dy, head_width=0.1)
#plt.arrow(0, b, dy, -dx, head_width=0.1)
plt.savefig('pca01_04.png', bbox_inches='tight',pad_inches = 0)
plt.show()
