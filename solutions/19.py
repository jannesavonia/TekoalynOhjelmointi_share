import numpy as np

def unit(v):
    return (1/np.linalg.norm(v))*v


x=np.array([1, 1, 0])
y=np.array([0, 0, 1])

print('Pistetulo on', x@y, '.')


x_=unit(x)
y_=unit(y)
d=x_@y_
phi=np.arccos(d)

print('Kulma on', phi, 'radiaania.')
