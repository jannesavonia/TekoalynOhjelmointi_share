import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 5, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin, label=r'sinikayra, eli $y = \sin\, x$')
plt.plot(x, y_cos, 'k', label=r'$y = \cos\, x$')

plt.legend()
plt.show()
#plt.savefig('pyplot.png')
