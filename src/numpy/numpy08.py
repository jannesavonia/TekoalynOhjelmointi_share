import numpy as np

x=np.array([1, 2, 1])
y=np.array([-2, 1, 0])

print('x.y =', x@y)
print('||x|| =', np.linalg.norm(x))
print('||y|| =', np.linalg.norm(y))
print('(x, y) =', np.arccos(
  x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y)))
)
