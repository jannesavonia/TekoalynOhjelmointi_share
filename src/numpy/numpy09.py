import numpy as np

A=np.array([[1, 2],[3, 4],[5, 6]])
B=np.array([[0, 1],[1, 0]])
x=np.array([1, 2])

print(A@B@x)
