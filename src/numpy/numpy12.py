import numpy as np

A=np.array([[0, 1, 2],[3, 4, 5]])
print(A)

x=A.reshape((6,))
print(x)

B=A.reshape((3,2))
print(B)
