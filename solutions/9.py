import matplotlib.pyplot as plt
import numpy as np

step=0.1

x=np.arange(-2, 2+step, step)
fx=x**2 -2*x -1

plt.plot(x, fx)
plt.savefig('fx.png')
