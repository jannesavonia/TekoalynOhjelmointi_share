import numpy as np

x=np.array([1, -1, 0])
y=np.array([3, -2, 0])
z=np.array([1, 2])
w=np.array([-2, -1])

print('x+y = ', x+y)
print('x-y = ', x-y)
print('x+w ei ole määritelty')
print('z+w = ', z+w)
print('y-y = ', y-y)
print()
print('|x+y| =', np.linalg.norm(x+y))
print('|x-y| =', np.linalg.norm(x-y))
print('|x|+|w| =', np.linalg.norm(x)+np.linalg.norm(w))
print('|z|+w ei ole määritelty')
print('|z|+|w| =', np.linalg.norm(x)+np.linalg.norm(w))
print('|y|y =', np.linalg.norm(y)*y)

