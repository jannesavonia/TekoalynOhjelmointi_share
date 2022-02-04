import matplotlib.pyplot as plt
import numpy as np

step=0.1

x=np.arange(-2, 2+step, step)

X, Y = np.meshgrid(x, x)
f_xy=X**2-2*X*Y-1


ax = plt.axes(projection='3d')
surf=ax.plot_surface(X, Y, f_xy, cmap='jet')
plt.savefig('fxy.png')

