import numpy as np

A=np.array([[2, 4],
            [1, 1],
            [4, 2]])

B=np.array([[1, 2],
            [2, 3]])

print('AB =\n', A@B)
print('(B^T A^T)^T =\n', (B.transpose()@A.transpose()).transpose())
