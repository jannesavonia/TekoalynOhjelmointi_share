import numpy as np

def unit(v):
    return (1/np.linalg.norm(v))*v


x=np.array([2, 1, 3])
y=np.array([1, 1, 1])

print('Pistetulo on', x@y, '.')


x_=unit(x)
y_=unit(y)
d=x_@y_
phi=np.arccos(d)

print('Kulma on', phi, 'radiaania.')
