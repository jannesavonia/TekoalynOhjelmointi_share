import numpy as np

#Luodaan vektorit matriiseina (2d-taulukoina)
x=np.array([1, 2], ndmin=2)
y=np.array([-1, 2], ndmin=2)

print('x shape =', np.shape(x))
print('y shape =', np.shape(y))

print('x^T y =\n', x.transpose()@y)
