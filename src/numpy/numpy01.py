import numpy as np
import matplotlib.pyplot as plt

N_points=50

#Generate N_points in [0, 2*pi[
x=np.linspace(0, 2*np.pi, N_points, endpoint=False)
y_sin=np.sin(x)
y_cos=np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.show()
