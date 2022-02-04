import numpy as np

x=np.array([1, -1, 0])
y=np.array([3, -2, 0])
z=np.array([1, 2])
w=np.array([-2, -1])

def unit(v):
    return (1/np.linalg.norm(v))*v

print('x ->', unit(x))
print('y ->', unit(y))
print('z ->', unit(z))
print('w ->', unit(w))

