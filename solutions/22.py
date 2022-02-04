import numpy as np

def proj(x, y):
    return ((x@y)/(y@y))*y

x=np.array([-1, 1])
y=np.array([2, -1])


print('x:n projektio y:lle on', proj(x, y))
