import numpy as np
import matplotlib.pyplot as plt

N=100

t=np.arange(0, 2*np.pi, 2*np.pi/N)
x=np.sin(2*t)
y=np.cos(t)


fig=plt.figure(figsize=(16, 8))
plt.plot(x)
plt.plot(y, '--')
plt.savefig("03_01.png", bbox_inches='tight',pad_inches = 0)
plt.show()

fig=plt.figure(figsize=(16, 8))
plt.plot(t, x)
plt.plot(t, y, 'm:')
plt.savefig("03_02.png", bbox_inches='tight',pad_inches = 0)
plt.show()

fig=plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.savefig("03_03.png", bbox_inches='tight',pad_inches = 0)
plt.show()

fig=plt.figure(figsize=(8, 8))
plt.plot(x, y, 'k.')
plt.grid(b=True, linestyle=':')
plt.savefig("03_04.png")
plt.show()

data={'a': x, 'b':y, 's':40*t+1, 'c': np.random.randint(0, 50, size=x.shape)}
plt.figure(figsize=(8,8))
plt.scatter('a', 'b', s='s', c='c', data=data)
plt.xlabel(r'$\sin\,2t$')
plt.ylabel(r'$\cos\,t$')
plt.savefig("03_05.png", bbox_inches='tight',pad_inches = 0)
plt.show()


names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(figsize=(16, 8))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
plt.savefig("03_06.png", bbox_inches='tight',pad_inches = 0)
plt.show()


YT=np.reshape(y, (1, N))
X=np.reshape(x, (N, 1))
A=X+YT
del X, YT
plt.imshow(A, cmap='jet')
plt.colorbar()
plt.savefig("03_07.png", bbox_inches='tight',pad_inches = 0)
plt.show()


fig=plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
X, Y = np.meshgrid(t, t)
surf=ax.plot_surface(X, Y, A, cmap='jet')
fig.colorbar(surf, shrink=0.5)
plt.savefig("03_08.png", bbox_inches='tight',pad_inches = 0)
plt.show()

