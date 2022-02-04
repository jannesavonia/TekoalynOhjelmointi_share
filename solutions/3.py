import numpy as np

A=np.array([[1, 2, -1],
            [0, 1,  0],
            [0, 1, 1]])

print('A^-1 =\n', np.linalg.inv(A))
