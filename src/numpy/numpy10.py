import numpy as np

x=np.array([1, 2])
y=np.array([-1, 2])

print('x shape =', np.shape(x))
print('y shape =', np.shape(y))

#Ei toimi, pitaisi tulla 2x2 matriisi!
print('x^T y =', x.transpose()@y)
